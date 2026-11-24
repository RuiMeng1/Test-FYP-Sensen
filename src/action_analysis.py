"""Offline office activity suggestions from tracked boxes and sampled video frames."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import tempfile
import threading
import time

from src.action_events import fingerprint, source_signature
from src.action_labels import LABELS


def merge_samples(samples, step):
    """Keep repeated observations without filling gaps in the footage."""
    groups, events = {}, []
    for sample in sorted(samples, key=lambda item: item['time']):
        key = (sample['camera_id'], sample['person_id'], sample['action'])
        previous = groups.get(key)
        if previous and 0 < sample['time'] - previous['end_seconds'] <= step * 1.05:
            previous['end_seconds'] = sample['time']
            previous['samples'] += 1
            if sample['score'] > previous['model_score']:
                previous.update(model_score=sample['score'], evidence_frame=sample['frame'],
                                evidence_bbox=sample['bbox'])
        else:
            previous = dict(person_id=sample['person_id'], camera_id=sample['camera_id'],
                            action=sample['action'], start_seconds=sample['time'],
                            end_seconds=sample['time'], model_score=sample['score'],
                            samples=1, review_status='unreviewed',
                            evidence_frame=sample['frame'], evidence_bbox=sample['bbox'])
            groups[key] = previous
            events.append(previous)
    return [event for event in events if event['samples'] >= 2]


def movement_speed(previous, current, elapsed):
    """Ground-point travel in person heights per second for a fixed camera."""
    if elapsed <= 0:
        return 0.0
    height = max(1, (previous[3] - previous[1] + current[3] - current[1]) / 2)
    distance = math.hypot((current[0] + current[2] - previous[0] - previous[2]) / 2,
                          current[3] - previous[3])
    return distance / height / elapsed


def _analyze_camera(cid, meta, data, args, model, data_dir):
    """Analyze one camera with a model owned by the calling worker."""
    import cv2
    source = Path(data_dir) / (meta.get('raw_video_file') or meta['video_file'].replace('_tracked.mp4', '.mp4'))
    signature = source_signature(source)
    samples = []
    cap = cv2.VideoCapture(str(source))
    try:
        fps = float(meta['fps'])
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or total <= 0 or fps <= 0:
            raise RuntimeError(f'Cannot read camera {cid}')
        duration = total / fps
        end = min(duration, args.end) if args.end is not None else duration
        if args.start >= end:
            raise ValueError(f'Start time is outside camera {cid}')
        detections = data.get('detections', {}).get(str(cid), {})
        checked = ambiguous = missing = 0
        last_seen = {}
        time_s = args.start
        while time_s < end:
            frame_idx = min(total - 1, round(time_s * fps))
            found = detections.get(str(frame_idx), [])
            counts = Counter(d['gid'] for d in found if d['gid'] >= 0)
            ambiguous += sum(n > 1 for n in counts.values())
            valid = [d for d in found if d['gid'] >= 0 and counts[d['gid']] == 1
                     and d['bbox'][2] > d['bbox'][0] and d['bbox'][3] > d['bbox'][1]]
            if valid:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f'Could not decode camera {cid} frame {frame_idx}')
                observations = model.predict(frame, [d['bbox'] for d in valid])
                if len(observations) != len(valid):
                    raise RuntimeError('Office model returned the wrong number of observations')
                checked += 1
                for detection, observation in zip(valid, observations):
                    gid, box = detection['gid'], detection['bbox']
                    seconds = round(frame_idx / fps, 3)
                    posture = observation.get('posture')
                    if posture in {'sitting', 'standing'}:
                        samples.append(dict(person_id=gid, camera_id=int(cid), time=seconds,
                                            frame=frame_idx, bbox=box, action=posture,
                                            score=float(observation['posture_score'])))
                    if observation.get('phone_score') is not None:
                        samples.append(dict(person_id=gid, camera_id=int(cid), time=seconds,
                                            frame=frame_idx, bbox=box, action='possible_phone_use',
                                            score=float(observation['phone_score'])))
                    prior = last_seen.get(gid)
                    if posture == 'standing' and prior and 0 < seconds - prior[0] <= args.step * 1.1:
                        if movement_speed(prior[1], box, seconds - prior[0]) >= args.walk_speed:
                            samples.append(dict(person_id=gid, camera_id=int(cid), time=seconds,
                                                frame=frame_idx, bbox=box, action='walking',
                                                score=float(observation['posture_score'])))
                    last_seen[gid] = (seconds, box)
            else:
                missing += 1
            print(f'Camera {cid}: {time_s:.1f}/{end:.1f}s', flush=True)
            time_s += args.step
        coverage = dict(camera_id=int(cid), start_seconds=args.start, end_seconds=end,
                        analyzed_windows=checked, skipped_windows=missing,
                        ambiguous_ids_skipped=ambiguous)
        if source_signature(source) != signature:
            raise RuntimeError('Source video changed during analysis; results discarded')
        return signature, samples, coverage
    finally:
        cap.release()


def run(args, model=None):
    """Write a complete sidecar only after every selected camera succeeds."""
    from src.config import DATA_DIR
    from src.office_vision import OfficeVision

    started = time.perf_counter()
    output = Path(args.output_dir)
    log = output / 'tracking_log.json'
    data = json.loads(log.read_text())
    selected = [(cid, meta) for cid, meta in data['cameras'].items()
                if args.camera is None or int(cid) == args.camera]
    if not selected:
        raise ValueError('No matching cameras')
    device = getattr(args, 'device', 'cpu')
    requested = getattr(args, 'workers', None)
    workers = min(len(selected), 1 if model is not None or device == 'mps'
                  else (requested or (2 if device == 'cpu' else 1)))
    print(f'Analyzing {len(selected)} camera(s) with {workers} office worker(s)', flush=True)
    local = threading.local()

    def analyze(item):
        if model is None:
            if not hasattr(local, 'model'):
                local.model = OfficeVision(device, getattr(args, 'pose_model', 'yolov8n-pose.pt'),
                                           getattr(args, 'object_model', 'yolov8n.pt'),
                                           getattr(args, 'object_size', 1920))
            current_model = local.model
        else:
            current_model = model
        cid, meta = item
        return _analyze_camera(cid, meta, data, args, current_model, args.data_dir or DATA_DIR)

    if workers == 1:
        per_camera = [analyze(item) for item in selected]
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='Office') as pool:
            per_camera = list(pool.map(analyze, selected))
    sources = [signature for signature, _, _ in per_camera]
    samples = [sample for _, camera_samples, _ in per_camera for sample in camera_samples]
    coverage = [camera_coverage for _, _, camera_coverage in per_camera]
    if fingerprint(json.loads(log.read_text())) != fingerprint(data):
        raise RuntimeError('Tracking log changed during analysis; results discarded')
    result = dict(schema_version=2, tracking_fingerprint=fingerprint(data), sources=sources,
                  generated_at=datetime.now(timezone.utc).isoformat(),
                  model=f'office-pose+objects/{getattr(args, "pose_model", "yolov8n-pose.pt")}+'
                        f'{getattr(args, "object_model", "yolov8n.pt")}',
                  status='Office activity suggestions are unreviewed. Missing events do not prove inactivity.',
                  coverage=coverage, sample_step_seconds=args.step, walk_speed=args.walk_speed,
                  processing_seconds=round(time.perf_counter() - started, 2),
                  supported_actions=list(LABELS.values()), events=merge_samples(samples, args.step))
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
    parser.add_argument('--walk-speed', type=float, default=0.25,
                        help='Minimum ground-point travel in person heights per second')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    parser.add_argument('--workers', type=int, default=None,
                        help='Concurrent camera analyses (default: 2 on CPU, 1 on GPU)')
    parser.add_argument('--pose-model', default='yolov8n-pose.pt')
    parser.add_argument('--object-model', default='yolov8n.pt')
    parser.add_argument('--object-size', type=int, default=1920,
                        help='Object inference image size; larger helps with small phones but runs slower')
    args = parser.parse_args()
    if not all(math.isfinite(v) for v in [args.start, args.step, args.walk_speed]) or args.start < 0 or args.step <= 0 or args.walk_speed <= 0:
        parser.error('Require start >= 0, step > 0, and walk-speed > 0')
    if args.end is not None and (not math.isfinite(args.end) or args.end <= args.start):
        parser.error('end must be greater than start')
    if args.object_size < 640:
        parser.error('object-size must be at least 640')
    if args.workers is not None and args.workers < 1:
        parser.error('--workers must be at least 1')
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(1, f'Office activity analysis failed: {error}\n')


if __name__ == '__main__':
    main()
