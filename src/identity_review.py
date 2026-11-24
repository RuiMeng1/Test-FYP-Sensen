"""Apply review corrections to the tracking log and its activity sidecar."""

import copy
import json
from pathlib import Path
import tempfile

from src.action_events import fingerprint, load_actions
from src.tracking_log import Appearance, saved_detection_counts


class TrackingChanged(ValueError):
    """The reviewer is looking at an older version of the tracking log."""


def atomic_json(path, data):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False, suffix='.json', encoding='utf-8') as handle:
            temporary = Path(handle.name)
            json.dump(data, handle, indent=2)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def combine_appearances(appearances, cameras):
    """Count overlapping intervals on one camera as a single sighting."""
    joined = []
    for interval in sorted(appearances, key=lambda a: (a['camera_id'], a['first_frame'])):
        previous = joined[-1] if joined else None
        cid = interval['camera_id']
        if previous and previous['camera_id'] == cid and interval['first_frame'] <= previous['last_frame'] + 1:
            fps = cameras.get(str(cid), {}).get('fps', 25.0)
            joined[-1] = Appearance(cid, previous['first_frame'], max(previous['last_frame'], interval['last_frame']), fps).to_dict()
        else:
            joined.append(interval)
    return joined


def correct_identities(data, source, target=None):
    """Delete or merge an identity and compact the surviving IDs from zero."""
    updated = copy.deepcopy(data)
    persons = updated['persons']
    if str(source) not in persons or (target is not None and (target == source or str(target) not in persons)):
        raise ValueError('Choose existing person IDs')
    dropped = persons.pop(str(source))
    if target is not None:
        kept = persons[str(target)]
        kept['appearances'] = combine_appearances(kept.get('appearances', []) + dropped.get('appearances', []),
                                                  updated['cameras'])
        kept['cameras_seen'] = sorted(set(kept.get('cameras_seen', [])) | set(dropped.get('cameras_seen', [])))
        kept['total_detections'] = kept.get('total_detections', 0) + dropped.get('total_detections', 0)
        images = [image for person in (kept, dropped)
                  for image in person.get('thumbnails_b64') or [person.get('thumbnail_b64')] if image]
        kept['thumbnails_b64'] = list(dict.fromkeys(images))[:3]
        kept['thumbnail_b64'] = kept['thumbnails_b64'][0] if images else None

    remapped = {old: new for new, old in enumerate(sorted(map(int, persons)))}
    updated['persons'] = {str(new): dict(persons[str(old)], global_id=new) for old, new in remapped.items()}
    boxes = updated.get('detections', {})
    for frames in boxes.values():
        for frame, sightings in list(frames.items()):
            target_visible = target is not None and any(box.get('gid', box.get('global_id')) == target for box in sightings)
            corrected = []
            for box in sightings:
                visible = box.get('gid', box.get('global_id'))
                if visible == source and (target is None or target_visible):
                    continue
                if visible == source:
                    visible = target
                if visible in remapped:
                    box['gid'] = remapped[visible]
                    if 'global_id' in box:
                        box['global_id'] = box['gid']
                    corrected.append(box)
                elif isinstance(visible, int) and visible < 0:
                    box['gid'] = visible
                    corrected.append(box)
            if corrected:
                frames[frame] = corrected
            else:
                del frames[frame]

    for gid, count in saved_detection_counts(boxes).items():
        updated['persons'][str(gid)]['total_detections'] = count
    updated['identities_reviewed'] = True
    return updated, remapped


def save_identity_correction(output, data, source, target=None):
    """Save the correction and carry over valid activity for unaffected tracks."""
    log = output / 'tracking_log.json'
    revision = fingerprint(data)
    try:
        current = json.loads(log.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise TrackingChanged('Tracking data is being updated. Restart review before editing IDs.') from exc
    if fingerprint(current) != revision:
        raise TrackingChanged('Tracking data changed. Restart review before editing IDs.')
    updated, remapped = correct_identities(data, source, target)
    kind = 'merge' if target is not None else 'delete'
    backup = output / f'tracking_log.before-{kind}-{revision[:12]}.json'
    if not backup.exists():
        atomic_json(backup, data)
    actions = load_actions(data, output)
    if actions.get('schema_version') == 2:
        action_backup = output / f'action_events.before-{kind}-{revision[:12]}.json'
        if not action_backup.exists():
            atomic_json(action_backup, actions)
    atomic_json(log, updated)

    activity_note = ''
    if actions.get('schema_version') == 2:
        actions['tracking_fingerprint'] = fingerprint(updated)
        actions['events'] = [dict(event, person_id=remapped[int(event['person_id'])]) for event in actions['events']
                             if int(event['person_id']) in remapped and int(event['person_id']) != target]
        if target is not None:
            activity_note = 'Rerun activity analysis for the merged person.'
            actions['status'] += ' ' + activity_note
        try:
            atomic_json(output / 'action_events.json', actions)
        except OSError:
            # A sidecar with the old fingerprint cannot be mistaken for the new IDs.
            activity_note = 'The activity file could not be updated. Rerun activity analysis.'
    return updated, remapped, activity_note
