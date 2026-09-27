"""Lightweight action sidecar loading; no ML imports in review mode."""
import hashlib
import json
import math
from pathlib import Path


def fingerprint(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source_signature(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def load_actions(data, output_dir):
    path = Path(output_dir) / 'action_events.json'
    if not path.exists():
        return {'events': [], 'status': 'Action analysis has not been run.'}
    try:
        result = json.loads(path.read_text())
        if not isinstance(result, dict):
            raise ValueError('invalid sidecar')
        if result.get('schema_version') != 1 or result.get('tracking_fingerprint') != fingerprint(data):
            raise ValueError('tracking data changed')
        for source in result['sources']:
            if source_signature(source['path']) != source:
                raise ValueError('source footage changed')
        if not isinstance(result['events'], list):
            raise ValueError('invalid events')
        for event in result['events']:
            if not isinstance(event, dict) or not isinstance(event['action'], str):
                raise ValueError('invalid event')
            if str(event['person_id']) not in {str(gid) for gid in data.get('persons', {})}:
                raise ValueError('unknown person')
            if str(event['camera_id']) not in {str(cid) for cid in data.get('cameras', {})}:
                raise ValueError('unknown camera')
            for key in ('start_seconds', 'end_seconds', 'model_score', 'evidence_frame'):
                if not math.isfinite(float(event[key])) or float(event[key]) < 0:
                    raise ValueError('invalid event timing or score')
            if event['end_seconds'] < event['start_seconds'] or not isinstance(event['review_status'], str):
                raise ValueError('invalid interval')
        if not isinstance(result['status'], str):
            raise ValueError('invalid status')
        return result
    except (OSError, ValueError, KeyError, TypeError):
        return {'events': [], 'status': 'Action results unavailable or stale. Run action analysis again.'}


def timestamp(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    return f'{minutes}:{seconds:02d}'
