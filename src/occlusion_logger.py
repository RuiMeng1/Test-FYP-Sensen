"""Diagnostic heuristics only; never modify detections or identity assignment."""
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def area(b):
    return max(0, b[2]-b[0])*max(0, b[3]-b[1])


def overlap(a, b):
    intersection = area((max(a[0], b[0]), max(a[1], b[1]),
                         min(a[2], b[2]), min(a[3], b[3])))
    return intersection/max(1, min(area(a), area(b)))


class OcclusionLogger:
    def __init__(self, directory, camera_id, source, fps):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
        self.path = directory / f'{stamp}-cam{camera_id}-{uuid4().hex[:8]}.jsonl'
        self.camera_id, self.source, self.fps = camera_id, str(source), fps
        self.tracks = {}
        self.emit('session_start', 0, thresholds=dict(overlap_fraction=.35,
                  height_ratio=.65, gap_min_seconds=.2, return_window_seconds=3))

    def emit(self, kind, frame, **fields):
        record = dict(event=kind, camera_id=self.camera_id, source=self.source,
                      frame=int(frame), seconds=round(frame/self.fps, 3),
                      review_status='unreviewed', **fields)
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record)+'\n')

    def update(self, frame, detections, identities):
        current = {int(d.local_id): d for d in detections}
        window = round(3*self.fps)
        for tid, d in current.items():
            box = [int(v) for v in d.bbox]
            gid = int(identities.get(tid, -1))
            old = self.tracks.get(tid)
            if old is None:
                candidates = [dict(local_id=k, global_id=v['gid'], bbox=v['box'])
                              for k, v in self.tracks.items()
                              if k not in current and 0 < frame-v['frame'] <= window
                              and overlap(box, v['box']) >= .35]
                if candidates:
                    self.emit('possible_reappearance_new_track', frame, local_id=tid,
                              global_id=gid, bbox=box, candidates=candidates,
                              note='Spatial candidates only; same person not established')
            elif frame-old['frame'] >= max(2, round(.2*self.fps)):
                self.emit('track_reappeared', frame, local_id=tid, global_id=gid,
                          previous_global_id=old['gid'], bbox=box,
                          last_seen_frame=old['frame'], gap_seconds=(frame-old['frame'])/self.fps,
                          prior_occlusion_evidence=old['reasons'],
                          note='Tracker continuity, not independently verified identity')
            height = max(1, box[3]-box[1])
            baseline = old['height'] if old else height
            ratio = height/baseline
            others = [int(o.local_id) for o in detections
                      if o.local_id != tid and overlap(box, o.bbox) >= .35]
            reasons = []
            if others:
                reasons.append('person_box_overlap')
            if ratio < .65:
                reasons.append('height_shrinkage')
            if getattr(d, 'partially_occluded', False):
                reasons.append('detector_partial_occlusion_flag')
            before = old['reasons'] if old else []
            if reasons != before:
                self.emit('suspected_occlusion' if reasons else 'occlusion_cues_cleared',
                          frame, local_id=tid, global_id=gid, bbox=box,
                          reasons=reasons, overlapping_local_ids=others,
                          height_ratio=round(ratio, 3), baseline_height=baseline,
                          detection_confidence=float(d.confidence))
            self.tracks[tid] = dict(frame=frame, box=box, gid=gid, reasons=reasons,
                                   height=baseline if reasons else .95*baseline+.05*height,
                                   missing=False)
        for tid, old in list(self.tracks.items()):
            gap = frame-old['frame']
            if tid not in current and gap >= max(2, round(.2*self.fps)) and not old['missing']:
                self.emit('track_missing', frame, local_id=tid, global_id=old['gid'],
                          last_seen_frame=old['frame'], bbox=old['box'],
                          prior_occlusion_evidence=old['reasons'],
                          note='Could also be an exit or missed detection')
                old['missing'] = True
            if gap > window:
                del self.tracks[tid]

    def finish(self, frame):
        self.emit('session_end', frame)
