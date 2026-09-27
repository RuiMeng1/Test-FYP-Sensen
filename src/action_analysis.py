"""Optional offline SlowFast pass. Run python -m src.action_analysis --help."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import tempfile
import time
from datetime import datetime, timezone

from src.action_events import fingerprint, source_signature
from src.action_labels import LABELS

# AVA uses one-based label IDs; output columns are zero-based.


class SlowFast:
    def __init__(self, device):
        import torch
        try:
            from pytorchvideo.models.hub import slowfast_r50_detection
        except ImportError as error:
            raise RuntimeError('Install optional dependencies: pip install -r requirements-actions.txt') from error
        self.torch = torch
        self.device = device
        self.model = slowfast_r50_detection(pretrained=True).eval().to(device)

    def predict(self, frames, boxes):
        import numpy as np
        import cv2
        torch = self.torch
        height, width = frames[0].shape[:2]
        scale = 256 / min(height, width)
        new_w, new_h = round(width * scale), round(height * scale)
        rgb = np.stack([cv2.cvtColor(cv2.resize(f, (new_w, new_h)), cv2.COLOR_BGR2RGB) for f in frames])
        clip = torch.from_numpy(rgb).permute(3, 0, 1, 2).float() / 255
        clip = (clip - 0.45) / 0.225
        slow = clip[:, torch.linspace(0, 31, 8).long()]
        rois = torch.tensor([[0, b[0]*new_w/width, b[1]*new_h/height,
                              b[2]*new_w/width, b[3]*new_h/height] for b in boxes], dtype=torch.float32)
        with torch.inference_mode():
            scores = self.model([slow.unsqueeze(0).to(self.device), clip.unsqueeze(0).to(self.device)], rois.to(self.device))
        return scores.cpu().numpy()


def read_clip(cap, center, fps, total):
    """32 frames across ~2.13 seconds, preserving the checkpoint's time scale."""
    import cv2
    import numpy as np
    indices = np.linspace(max(0, center-fps*32/30), min(total-1, center+fps*32/30), 32).round().astype(int)
    result = []
    for index in indices:
        if int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) != index:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f'Could not decode frame {index}')
        result.append(frame)
    return result


def merge_samples(samples, step):
    """Merge adjacent positive samples, never bridging unobserved windows."""
    groups = {}
    events = []
    for sample in sorted(samples, key=lambda x: x['time']):
        key = (sample['camera_id'], sample['person_id'], sample['action'])
        previous = groups.get(key)
        if previous and sample['time'] - previous['end_seconds'] <= step * 1.05:
            previous['end_seconds'] = sample['time']
            previous['samples'] += 1
            if sample['score'] > previous['model_score']:
                previous.update(model_score=sample['score'], evidence_frame=sample['frame'], evidence_bbox=sample['bbox'])
        else:
            previous = dict(person_id=sample['person_id'], camera_id=sample['camera_id'],
                            action=sample['action'], start_seconds=sample['time'], end_seconds=sample['time'],
                            model_score=sample['score'], samples=1, review_status='unreviewed',
                            evidence_frame=sample['frame'], evidence_bbox=sample['bbox'])
            groups[key] = previous
            events.append(previous)
    # Require repeated posture observations, but preserve brief possible falls.
    return [e for e in events if e['samples'] >= 2 or e['action'] == 'possible_fall']


def run(args, model=None):
    started = time.perf_counter()
    import cv2
    from src.config import DATA_DIR
    output = Path(args.output_dir)
    log = output / 'tracking_log.json'
    data = json.loads(log.read_text())
    model = model or SlowFast(args.device)
    sources, samples, coverage = [], [], []
    for cid, meta in data['cameras'].items():
        if args.camera is not None and int(cid) != args.camera:
            continue
        source = Path(args.data_dir or DATA_DIR) / (meta.get('raw_video_file') or meta['video_file'].replace('_tracked.mp4', '.mp4'))
        signature = source_signature(source)
        sources.append(signature)
        cap = cv2.VideoCapture(str(source))
        try:
            fps = float(meta['fps'])
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not cap.isOpened() or total <= 0 or fps <= 0:
                raise RuntimeError(f'Cannot read camera {cid}')
            duration = total/fps
            end = min(duration, args.end) if args.end is not None else duration
            if args.start >= end:
                raise ValueError(f'Start time is outside camera {cid}')
            detections = data.get('detections', {}).get(str(cid), {})
            checked = ambiguous = missing = 0
            time_s = args.start
            while time_s < end:
                frame_idx = min(total-1, round(time_s*fps))
                found = detections.get(str(frame_idx), [])
                counts = Counter(d['gid'] for d in found)
                ambiguous += sum(n > 1 for n in counts.values())
                valid = [d for d in found if d['gid'] >= 0 and counts[d['gid']] == 1
                         and d['bbox'][2] > d['bbox'][0] and d['bbox'][3] > d['bbox'][1]]
                if valid:
                    frames = read_clip(cap, frame_idx, fps, total)
                    scores = model.predict(frames, [d['bbox'] for d in valid])
                    if scores.shape != (len(valid), 80):
                        raise RuntimeError('Unexpected model output; expected 80 AVA action columns')
                    checked += 1
                    for detection, score in zip(valid, scores):
                        for label, action in LABELS.items():
                            value = float(score[label-1])
                            if math.isfinite(value) and value >= args.threshold:
                                samples.append(dict(person_id=detection['gid'], camera_id=int(cid),
                                                    time=round(frame_idx/fps, 3), frame=frame_idx,
                                                    bbox=detection['bbox'], action=action, score=value))
                else:
                    missing += 1
                print(f'Camera {cid}: {time_s:.1f}/{end:.1f}s', flush=True)
                time_s += args.step
            coverage.append(dict(camera_id=int(cid), start_seconds=args.start, end_seconds=end,
                                 analyzed_windows=checked, skipped_windows=missing, ambiguous_ids_skipped=ambiguous))
            if source_signature(source) != signature:
                raise RuntimeError('Source video changed during analysis; results discarded')
        finally:
            cap.release()
    if not coverage:
        raise ValueError('No matching cameras')
    if fingerprint(json.loads(log.read_text())) != fingerprint(data):
        raise RuntimeError('Tracking log changed during analysis; results discarded')
    result = dict(schema_version=1, tracking_fingerprint=fingerprint(data), sources=sources,
                  generated_at=datetime.now(timezone.utc).isoformat(), model='slowfast_r50_detection/AVA',
                  status='Completed for listed coverage only. Predictions are unreviewed; absence is not proof of no action.',
                  coverage=coverage, threshold=args.threshold, sample_step_seconds=args.step,
                  processing_seconds=round(time.perf_counter()-started, 2),
                  supported_actions=list(LABELS.values()), events=merge_samples(samples, args.step))
    # Publish only a complete result, leaving the previous sidecar intact on failure.
    with tempfile.NamedTemporaryFile(mode='w', dir=output, suffix='.json', delete=False) as file:
        json.dump(result, file, indent=2)
        temp_path = Path(file.name)
    temp_path.replace(output / 'action_events.json')
    print(f"Saved {len(result['events'])} events to {output / 'action_events.json'}")
    print(f"Analyzed {sum(c['analyzed_windows'] for c in coverage)} windows in {result['processing_seconds']}s. "
          'Restart review to load these results.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='output')
    parser.add_argument('--data-dir')
    parser.add_argument('--camera', type=int)
    parser.add_argument('--start', type=float, default=0)
    parser.add_argument('--end', type=float)
    parser.add_argument('--step', type=float, default=1)
    parser.add_argument('--threshold', type=float, default=0.6)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    if not all(math.isfinite(v) for v in [args.start, args.step, args.threshold]) or args.start < 0 or args.step <= 0 or not 0 < args.threshold <= 1:
        parser.error('Require start >= 0, step > 0, and 0 < threshold <= 1')
    if args.end is not None and (not math.isfinite(args.end) or args.end <= args.start):
        parser.error('end must be greater than start')
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(1, f'Action analysis failed: {error}\n')


if __name__ == '__main__':
    main()
