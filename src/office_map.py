"""Build recording-relative movement tracks for the web review map."""

import json
import math
from pathlib import Path
import re


def camera_stem(camera):
    if camera.get('raw_video_file'):
        return Path(camera['raw_video_file']).stem
    return Path(camera['video_file']).stem.removesuffix('_tracked')


def camera_label(stem):
    name = re.sub(r'_\d{3}$', '', re.sub(r'^\d+-', '', stem))
    return name.replace('-', ' ').replace('_', ' ').strip().title() or stem


def schematic(cameras):
    columns = min(4, max(1, math.ceil(math.sqrt(len(cameras)))))
    rows = max(1, math.ceil(len(cameras) / columns))
    floor = dict(id='zones', label='Camera zones', elevation=0,
                 bounds=[0, 0, columns * 6 + 1, rows * 5 + 1])
    zones, placement = [], {}
    for index, cid in enumerate(sorted(cameras, key=int)):
        stem = camera_stem(cameras[cid])
        zone = dict(id=cid, label=camera_label(stem), floor='zones',
                    bounds=[1 + index % columns * 6, 1 + index // columns * 5, 5, 4])
        zones.append(zone)
        placement[stem] = dict(zone=cid)
    return dict(name='Office camera zones', floors=[floor], zones=zones, cameras=placement)


def coordinate_list(values, length):
    if not isinstance(values, list) or len(values) != length or any(
        type(value) not in (int, float) or not math.isfinite(value) for value in values
    ):
        raise ValueError(f'Expected {length} finite coordinates')
    return values


def bounds(values):
    result = coordinate_list(values, 4)
    if min(result[2:]) <= 0:
        raise ValueError('Map rectangles need positive width and depth')
    return result


def cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def quadrilateral(points):
    if not isinstance(points, list) or len(points) != 4:
        raise ValueError('Calibration requires four corresponding floor points')
    points = [coordinate_list(point, 2) for point in points]
    turns = [cross(points[i - 2], points[i - 1], point) for i, point in enumerate(points)]
    if min(turns) <= 0 <= max(turns):
        raise ValueError('Calibration points must form a convex quadrilateral in boundary order')
    return points


def within_rectangle(point, rectangle):
    x, y, width, depth = rectangle
    return x <= point[0] <= x + width and y <= point[1] <= y + depth


def calibration_transform(calibration, zone):
    import cv2
    import numpy as np

    image, world = (quadrilateral(calibration[key]) for key in ('image', 'world'))
    if not all(within_rectangle(point, [0, 0, 1, 1]) for point in image):
        raise ValueError('Calibration image coordinates must be normalized to [0, 1]')
    if not all(within_rectangle(point, zone['bounds']) for point in world):
        raise ValueError('Calibration world coordinates must sit within the camera zone')
    matrix = cv2.getPerspectiveTransform(np.array(image, np.float32), np.array(world, np.float32))
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError('Calibration points do not define a valid projection')
    return image, matrix.tolist()


def load_office_map(cameras, path=None):
    """Return the public layout and internal camera projection lookup."""
    config = json.loads(Path(path).read_text(encoding='utf-8')) if path else schematic(cameras)
    return build_office_map(cameras, config, is_schematic=path is None)


def camera_placement(item, floors, zones):
    """Validate the camera pin and the area where its sightings are shown."""
    zone = item.get('zone')
    if zone is not None and (not isinstance(zone, str) or zone not in zones):
        raise ValueError('Choose a coverage area on this map')
    position = item.get('position')
    floor = item.get('floor', zones[zone]['floor'] if zone else None)
    if position is None:
        floor = None
    else:
        position = coordinate_list(position, 2)
        if not isinstance(floor, str) or floor not in floors or not within_rectangle(position, floors[floor]['bounds']):
            raise ValueError('Camera pins must be inside the selected floor')
    heading = coordinate_list([item.get('heading', 0)], 1)[0]
    if not 0 <= heading < 360:
        raise ValueError('Camera direction must be between 0 and 359 degrees')
    return dict(zone=zone, floor=floor, position=position, heading=heading)


def build_office_map(cameras, config, *, is_schematic=False, overrides=None):
    floors = {}
    for item in config['floors']:
        floor = dict(id=item['id'], label=item['label'], bounds=bounds(item['bounds']),
                     elevation=coordinate_list([item['elevation']], 1)[0])
        if floor['id'] in floors:
            raise ValueError(f"Duplicate floor ID {floor['id']}")
        floors[floor['id']] = floor
    if not floors:
        raise ValueError('An office map requires at least one floor')

    zones = {}
    for item in config['zones']:
        zone = dict(id=item['id'], label=item['label'], floor=item['floor'], bounds=bounds(item['bounds']))
        if zone['id'] in zones:
            raise ValueError(f"Duplicate zone ID {zone['id']}")
        x, y, width, depth = zone['bounds']
        floor = floors[zone['floor']]
        if not all(within_rectangle(point, floor['bounds']) for point in ((x, y), (x + width, y + depth))):
            raise ValueError(f"Zone {zone['id']} extends beyond its floor")
        zones[zone['id']] = zone

    pins, lookup = {}, {}
    overrides = overrides or {}
    for cid, camera in cameras.items():
        stem = camera_stem(camera)
        original = config['cameras'].get(stem, {})
        edit = overrides.get(stem, original)
        pin = camera_placement(edit, floors, zones)
        zone = zones.get(pin['zone'])
        # A calibration remains valid when a pin or direction is corrected within
        # the same map coverage. Reassigning coverage drops the old projection.
        transform = None
        if zone and pin['zone'] == original.get('zone') and 'calibration' in original:
            transform = calibration_transform(original['calibration'], zone)
        lookup[cid] = (zone, transform)
        pins[cid] = dict(**pin, label=camera_label(stem),
                         filename=camera.get('raw_video_file') or camera['video_file'],
                         calibrated=transform is not None)

    public = dict(name=config.get('name', 'Office map'), schematic=is_schematic,
                  approximate=config.get('approximate', False), floors=list(floors.values()),
                  zones=list(zones.values()), cameras=pins)
    return public, lookup


def floor_position(box, width, height, transform):
    if not (transform and width > 0 and height > 0):
        return None
    image, matrix = transform
    point = ((box[0] + box[2]) / (2 * width), box[3] / height)
    sides = [cross(image[i - 1], p, point) for i, p in enumerate(image)]
    if min(sides) < -1e-8 and max(sides) > 1e-8:
        return None
    x, y = point
    wx, wy, w = (row[0] * x + row[1] * y + row[2] for row in matrix)
    if abs(w) < 1e-10:
        return None
    return [round(wx / w, 3), round(wy / w, 3)]


def overview_frames(tracking, placements, media, start, end):
    """Return map observations in a short playback window, indexed by camera frame."""
    saved = tracking.get('detections', {})
    persons = tracking['persons']
    identities = {int(gid) for gid in persons}
    result = {}
    for cid, camera in tracking['cameras'].items():
        fps = camera['fps']
        if fps <= 0 or start >= camera['total_frames'] / fps:
            continue
        first = int(start * fps)
        last = min(camera['total_frames'] - 1, math.ceil(end * fps))
        _, transform = placements[cid]
        width = camera.get('width') or media.get(cid, {}).get('width', 0)
        height = camera.get('height') or media.get(cid, {}).get('height', 0)
        frames = result[cid] = {}
        if cid not in saved:
            # Older logs only have appearance intervals. Those can locate rooms only.
            for gid, person in persons.items():
                for appearance in person.get('appearances', []):
                    if str(appearance['camera_id']) == cid:
                        seen = range(max(first, appearance['first_frame']), min(last, appearance['last_frame']) + 1)
                        for number in seen:
                            frames.setdefault(number, set()).add(int(gid))
            result[cid] = {number: [[gid] for gid in sorted(ids)] for number, ids in frames.items()}
            continue

        detections = saved[cid]
        for number in range(first, last + 1):
            unique = {}
            for box in detections.get(str(number), []):
                gid = box['gid']
                if gid in identities:
                    unique[gid] = None if gid in unique else box['bbox']
            sightings = []
            for gid, box in unique.items():
                if box is not None:
                    position = floor_position(box, width, height, transform)
                    sightings.append([gid, *(position or [])])
            if sightings:
                frames[number] = sightings
    return result


def movement_history(person_id, cameras, frames_by_camera, appearances, placements, media):
    """Keep gaps and camera changes separate, even if both views cover one zone."""
    duration = max((c['total_frames'] / c['fps'] for c in cameras.values() if c['fps'] > 0), default=0)
    point_interval = max(.25, duration / 15000)
    segments = []
    for cid, camera in cameras.items():
        fps = camera['fps']
        if fps <= 0:
            continue
        zone, transform = placements[cid]
        source = media.get(cid, {})
        width = camera.get('width') or source.get('width', 0)
        height = camera.get('height') or source.get('height', 0)
        zone_id = zone['id'] if zone else None

        def segment(first, last, points):
            return dict(camera=cid, zone=zone_id, start=round(first / fps, 3),
                        end=round((last + 1) / fps, 3), points=points)

        if cid not in frames_by_camera:
            segments.extend(segment(item['first_frame'], item['last_frame'], [])
                            for item in appearances if str(item['camera_id']) == cid)
            continue

        matches = []
        for number, boxes in frames_by_camera[cid].items():
            found = [box['bbox'] for box in boxes if box['gid'] == person_id]
            if found:
                matches.append((int(number), found[0] if len(found) == 1 else None))
        matches.sort()
        max_gap = max(1, round(fps * .35))
        first, last, points, last_point = None, None, [], None

        def finish():
            if first is not None:
                if last_point is not None and points[-1][0] < last_point[0]:
                    points.append(last_point)
                segments.append(segment(first, last, points))

        for frame, box in matches:
            position = floor_position(box, width, height, transform) if box is not None else None
            if first is not None and (box is None or frame - last > max_gap or bool(points) != bool(position)):
                finish()
                first, points, last_point = None, [], None
            if box is None:
                continue
            if first is None:
                first = frame
            last = frame
            if position is not None:
                last_point = [round(frame / fps, 3), *position]
                if not points or last_point[0] - points[-1][0] >= point_interval:
                    points.append(last_point)
        finish()
    return dict(segments=sorted(segments, key=lambda item: (item['start'], item['camera'])),
                point_interval=round(point_interval, 2))


def office_chat_context(layout, tracking, placements, media, view, requested_ids):
    """Summarize the current map and selected routes for one assistant turn."""
    floors = {floor['id']: floor['label'] for floor in layout['floors']}
    areas = {zone['id']: {'area': zone['label'], 'floor': floors[zone['floor']]}
             for zone in layout['zones']}
    cameras = {}
    for cid, camera in layout['cameras'].items():
        pin = camera['position']
        cameras[cid] = dict(filename=camera['filename'], coverage=areas.get(camera['zone']),
                            pin=dict(floor=floors[camera['floor']], xy=pin,
                                     heading_degrees=camera['heading']) if pin is not None else None,
                            calibrated=camera['calibrated'])

    raw_cameras = tracking['cameras']
    duration = max((camera['total_frames'] / camera['fps'] for camera in raw_cameras.values()
                    if camera['fps'] > 0), default=0)
    playback = {}
    seconds = view.get('seconds')
    if type(seconds) in (int, float) and 0 <= seconds <= duration:
        playback['seconds'] = round(seconds, 3)
    persons = tracking['persons']
    followed = str(view.get('followed'))
    if followed in persons:
        playback['followed_person'] = int(followed)
    focus = str(view.get('camera'))
    if focus in cameras:
        playback['focused_camera'] = int(focus)

    routes = {}
    # Prioritize explicit IDs over the followed person and bound the evidence per turn.
    ids = dict.fromkeys([*(str(gid) for gid in requested_ids), followed])
    for gid in [gid for gid in ids if gid in persons][:3]:
        intervals = movement_history(int(gid), raw_cameras, tracking.get('detections', {}),
                                     persons[gid].get('appearances', []), placements, media)['segments']

        def sighting(segment):
            return dict(camera=int(segment['camera']), start_seconds=segment['start'],
                        end_seconds=segment['end'], coverage=areas.get(segment['zone']))

        # Keep the start and end of long histories and spread the samples across the recording.
        max_intervals = 60
        included = (intervals if len(intervals) <= max_intervals else
                    [intervals[round(i * (len(intervals) - 1) / (max_intervals - 1))]
                     for i in range(max_intervals)])
        at_playhead = []
        if 'seconds' in playback:
            t = playback['seconds']
            for segment in intervals:
                if segment['start'] <= t < segment['end']:
                    current = sighting(segment)
                    if segment['points']:
                        point = min(segment['points'], key=lambda p: abs(p[0] - t))
                        current['nearest_estimated_floor_sample'] = dict(seconds=point[0], xy=point[1:])
                    at_playhead.append(current)
        routes[gid] = dict(observed_intervals=len(intervals), chronological_sightings=[sighting(s) for s in included],
                           sightings_at_playhead=at_playhead)

    return dict(name=layout['name'], schematic=layout['schematic'], approximate=layout['approximate'],
                floors=list(floors.values()), areas=list(areas.values()), cameras=cameras,
                recording_duration_seconds=round(duration, 3), playback=playback, routes=routes)
