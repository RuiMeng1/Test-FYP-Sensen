"""Independent recorded-footage review. python -m src.web_review"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import secrets
import threading
from functools import wraps
from fractions import Fraction

from flask import Flask, abort, jsonify, request, send_file
from markdown_it import MarkdownIt
from src.action_events import fingerprint
from src.action_labels import LABELS
from src.office_map import build_office_map, camera_stem, movement_history, office_chat_context, overview_frames, schematic
from src.tracking_log import saved_detection_counts
from src.ai_assistant import AIAssistant, parse_requested_ids
from src.identity_review import TrackingChanged, atomic_json, save_identity_correction


CHAT_MARKDOWN = MarkdownIt('commonmark', {'html': False}).enable('table').disable('image')


def overlay_boxes(frames, start, end, fps):
    """Bridge brief gaps only when the same unique ID exists on both sides."""
    max_gap = max(1, min(8, round(fps * .32)))
    result = {str(frame): [dict(box) for box in frames[str(frame)]]
              for frame in range(start, end + 1) if str(frame) in frames}
    by_id = {}
    for frame in range(max(0, start - max_gap - 1), end + max_gap + 2):
        boxes = frames.get(str(frame), [])
        counts = {}
        for box in boxes:
            counts[box['gid']] = counts.get(box['gid'], 0) + 1
        for box in boxes:
            if box['gid'] >= 0 and counts[box['gid']] == 1:
                by_id.setdefault(box['gid'], []).append((frame, box['bbox']))
    for gid, observations in by_id.items():
        for (before, old), (after, new) in zip(observations, observations[1:]):
            gap = after - before - 1
            if gap < 1 or gap > max_gap:
                continue
            if any(sum(box['gid'] == gid for box in frames.get(str(frame), [])) > 1
                   for frame in range(before + 1, after)):
                continue
            height = max(old[3] - old[1], new[3] - new[1], 1)
            travel = math.hypot((new[0] + new[2] - old[0] - old[2]) / 2,
                                (new[1] + new[3] - old[1] - old[3]) / 2)
            if travel > 1.5 * height:
                continue
            for frame in range(max(start, before + 1), min(end, after - 1) + 1):
                current = result.setdefault(str(frame), [])
                if any(box['gid'] == gid for box in current):
                    continue
                blend = (frame - before) / (after - before)
                bbox = [round(a + (b - a) * blend) for a, b in zip(old, new)]
                # An observed Unknown box takes precedence over an overlapping guess.
                x1, y1, x2, y2 = bbox
                unknown_boxes = (box['bbox'] for box in current if box['gid'] < 0)
                if any(max(x1, left) < min(x2, right) and max(y1, top) < min(y2, bottom)
                       for left, top, right, bottom in unknown_boxes):
                    continue
                current.append({'gid': gid, 'bbox': bbox, 'interpolated': True})
    return result


def prepare_video(source, cache, fps=None):
    """Build a 720p H.264 fast-start proxy once. Original footage is untouched."""
    import av
    stat = source.stat()
    key = hashlib.sha256(f'{source.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:720-v2:{fps}'.encode()).hexdigest()
    target = cache / f'{key}.mp4'
    if target.exists():
        return target
    temporary = target.with_suffix('.partial.mp4')
    print(f'[Web review] Preparing {source.name} (one-time browser video cache)…', flush=True)
    try:
        with av.open(str(source)) as incoming:
            stream = incoming.streams.video[0]
            rate = Fraction(str(fps)).limit_denominator(1000000) if fps else stream.average_rate or Fraction(25)
            if rate <= 0:
                raise ValueError('Video FPS must be positive')
            scale = min(1, 1280 / stream.width, 720 / stream.height)
            width, height = max(2, int(stream.width*scale)//2*2), max(2, int(stream.height*scale)//2*2)
            with av.open(str(temporary), 'w', options={'movflags': '+faststart'}) as outgoing:
                encoded = outgoing.add_stream('libx264', rate=rate)
                encoded.width, encoded.height, encoded.pix_fmt = width, height, 'yuv420p'
                encoded.options = {'preset': 'veryfast', 'crf': '22'}
                encoded.codec_context.gop_size = max(1, round(float(rate)))
                for frame_number, frame in enumerate(incoming.decode(stream)):
                    scaled = frame.reformat(width, height, 'yuv420p')
                    # The tracking log uses frame_index / FPS, not source PTS.
                    # Some cameras repeat timestamps or start with near-duplicates.
                    scaled.pts, scaled.time_base = frame_number, 1 / rate
                    for packet in encoded.encode(scaled):
                        outgoing.mux(packet)
                for packet in encoded.encode():
                    outgoing.mux(packet)
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def create_app(output_dir, data_dir, prepare=True, office_map_path=None):
    output, data_dir = Path(output_dir).resolve(), Path(data_dir).resolve()
    log = output / 'tracking_log.json'
    data = json.loads(log.read_text())
    sketch = (json.loads(Path(office_map_path).read_text(encoding='utf-8'))
              if office_map_path else schematic(data['cameras']))
    map_key = fingerprint(sketch)
    pins_file = output / 'office_camera_placements.json'

    def read_pins():
        try:
            saved = json.loads(pins_file.read_text(encoding='utf-8'))
        except FileNotFoundError:
            return {}
        if not isinstance(saved, dict):
            raise ValueError('Invalid saved camera layout')
        return saved

    def build_layout(edits):
        layout, lookup = build_office_map(data['cameras'], sketch, is_schematic=office_map_path is None,
                                          overrides=edits)
        layout['revision'] = fingerprint([map_key, edits])
        return layout, lookup

    office_map, placements = build_layout(read_pins().get(map_key, {}))
    assistant = AIAssistant(data, output)
    lock = threading.Lock()
    ai_lock = threading.Lock()
    token = secrets.token_urlsafe(32)
    revision = fingerprint(data)
    assets = Path(__file__).parent / 'web_review_assets'
    app = Flask(__name__, static_folder=str(assets), static_url_path='/assets')
    app.config['MAX_CONTENT_LENGTH'] = 64 * 1024

    def assistant_operation(fn):
        @wraps(fn)
        def serialized(*args, **kwargs):
            if not ai_lock.acquire(blocking=False):
                return jsonify(error='The assistant is busy. Try again shortly.'), 409
            try:
                return fn(*args, **kwargs)
            finally:
                ai_lock.release()
        return serialized
    media = {}
    cache = output / 'web_video_cache'
    cache.mkdir(exist_ok=True)
    for cid, camera in data['cameras'].items():
        raw = data_dir / (camera.get('raw_video_file') or camera['video_file'].replace('_tracked.mp4', '.mp4'))
        tracked = output / camera['video_file']
        if data.get('identities_reviewed') and tracked.is_file() and not raw.is_file():
            media[cid] = {'available': False, 'requires_original': True}
            continue
        source = raw if raw.is_file() else tracked
        if not source.is_file():
            media[cid] = {'available': False}
            continue
        import av
        with av.open(str(source)) as container:
            stream = container.streams.video[0]
            width, height = stream.width, stream.height
        media[cid] = {'available': True, 'raw': source == raw, 'width': width, 'height': height,
                      'path': prepare_video(source, cache, camera['fps']) if prepare else source}

    @app.before_request
    def local_request():
        if request.host.split(':')[0] not in {'localhost', '127.0.0.1'}:
            abort(403)
        if request.method == 'POST' and not secrets.compare_digest(request.headers.get('X-Review-Token', ''), token):
            abort(403)

    @app.after_request
    def headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        if request.path.startswith('/api/'):
            response.headers['Cache-Control'] = 'no-store'
        return response

    @app.get('/')
    def index():
        return send_file(assets / 'index.html')

    @app.get('/api/state')
    def state():
        with lock:
            settings = list(LABELS.values())
            try:
                settings = json.loads((output/'action_display_settings.json').read_text())['actions']
            except (OSError, ValueError, KeyError):
                pass
            settings = [action for action in settings if action in LABELS.values()] if isinstance(settings, list) else list(LABELS.values())
            cameras = {cid: {**meta, **{k: v for k, v in media[cid].items() if k != 'path'}}
                       for cid, meta in data['cameras'].items()}
            return jsonify(token=token, revision=revision, cameras=cameras, persons=data['persons'],
                           office_map=office_map,
                           person_counts=saved_detection_counts(data.get('detections', {})),
                           actions=assistant.action_data, labels=list(LABELS.values()), selected=settings)

    @app.get('/api/office-map/<int:person_id>')
    def person_movement(person_id):
        with lock:
            person = data['persons'].get(str(person_id))
            if person is None:
                abort(404)
            cameras, frames = data['cameras'], data.get('detections', {})
            history_revision = revision
            history_placements, map_revision = placements, office_map['revision']
        history = movement_history(person_id, cameras, frames, person.get('appearances', []), history_placements, media)
        return jsonify(revision=history_revision, map_revision=map_revision, **history)

    @app.get('/api/office-map/overview/<int:chunk>')
    def map_overview(chunk):
        with lock:
            tracking = dict(data)
            overview_placements = placements
            tracking_revision, map_revision = revision, office_map['revision']
        frames = overview_frames(tracking, overview_placements, media, chunk * 10, (chunk + 1) * 10)
        return jsonify(revision=tracking_revision, map_revision=map_revision, frames=frames)

    @app.post('/api/office-map/placements')
    def save_camera_placements():
        nonlocal office_map, placements
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or not isinstance(body.get('cameras'), dict):
            return jsonify(error='Expected camera placements'), 400
        edits = body['cameras']
        with lock:
            if any(cid not in data['cameras'] or not isinstance(pin, dict) for cid, pin in edits.items()):
                return jsonify(error='Choose cameras from the current recording'), 400
            try:
                saved = read_pins()
                overrides = saved.get(map_key, {})
                if (body.get('revision') != office_map['revision'] or
                        fingerprint([map_key, overrides]) != office_map['revision']):
                    return jsonify(error='The camera layout changed. Reload before saving again.'), 409
                for cid, pin in edits.items():
                    overrides[camera_stem(data['cameras'][cid])] = {
                        key: pin.get(key) for key in ('zone', 'floor', 'position', 'heading')
                    }
                updated, lookup = build_layout(overrides)
                saved[map_key] = overrides
                atomic_json(pins_file, saved)
            except (TypeError, ValueError, KeyError) as exc:
                return jsonify(error=str(exc)), 400
            except OSError:
                return jsonify(error='Could not save the camera layout to the output folder'), 500
            office_map, placements = updated, lookup
        return jsonify(office_map=updated)

    @app.get('/video/<int:camera_id>')
    def video(camera_id):
        item = media.get(str(camera_id), {})
        if not item.get('available'):
            abort(404)
        return send_file(item['path'], mimetype='video/mp4', conditional=True, max_age=3600)

    @app.get('/api/boxes/<int:camera_id>/<int:chunk>')
    def boxes(camera_id, chunk):
        # Small on-demand chunks keep per-frame boxes out of the startup payload.
        with lock:
            cid = str(camera_id)
            if cid not in data['cameras']:
                abort(404)
            fps = data['cameras'][cid]['fps']
            frames = data.get('detections', {}).get(cid, {})
            start, end = int(chunk * 10 * fps), int((chunk + 1) * 10 * fps) + 1
            return jsonify(overlay_boxes(frames, start, end, fps))

    @app.post('/api/settings')
    def settings():
        body = request.get_json() or {}
        selected = body.get('actions')
        if not isinstance(selected, list) or any(x not in LABELS.values() for x in selected):
            abort(400)
        with lock:
            atomic_json(output/'action_display_settings.json', {'actions': selected})
        return jsonify(ok=True)

    @app.post('/api/chat')
    def chat():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            abort(400)
        prompt, view = body.get('prompt', ''), body.get('view', {})
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 10000 or not isinstance(view, dict):
            abort(400)
        if not ai_lock.acquire(blocking=False):
            return jsonify(error='Another operation is running. Try again shortly.'), 409
        try:
            with lock:
                chat_layout, chat_placements = office_map, placements
                tracking = dict(data)
            context = office_chat_context(chat_layout, tracking, chat_placements, media, view,
                                          parse_requested_ids(prompt))
            reply = assistant.ask(prompt, office_context=context)
            return jsonify(reply=reply, reply_html=CHAT_MARKDOWN.render(reply),
                           prompt_html=CHAT_MARKDOWN.render(prompt),
                           evidence=assistant.get_visual_evidence(prompt, reply))
        except Exception:
            app.logger.exception('Assistant request failed')
            return jsonify(error='Assistant request failed. Check your API configuration and server console.'), 502
        finally:
            ai_lock.release()

    @app.post('/api/chat/clear')
    @assistant_operation
    def clear():
        with lock:
            assistant.clear_history()
        return jsonify(ok=True)

    @app.get('/api/report.pdf')
    @assistant_operation
    def report():
        # Reuse the existing Pillow report implementation without creating Tk.
        from src.review_ui import ReviewUI
        with lock:
            renderer = ReviewUI.__new__(ReviewUI)
            renderer.persons = {int(k): v for k, v in data['persons'].items()}
            renderer.ai_assistant = assistant
            pages = renderer._build_person_summary_pages()
            if not pages:
                return jsonify(error='No people to export'), 400
            document = io.BytesIO()
            pages[0].save(document, 'PDF', resolution=150, save_all=True, append_images=pages[1:])
            document.seek(0)
        return send_file(document, mimetype='application/pdf', as_attachment=True, download_name='sensen-person-summary.pdf')

    def correct_person(merge):
        nonlocal revision
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            abort(400)
        try:
            source = int(str(body.get('source')))
            target = int(str(body.get('target'))) if merge else None
        except ValueError:
            abort(400)
        with lock:
            if body.get('revision') != revision:
                return jsonify(error='Tracking data changed. Reload before editing IDs.'), 409
            if any(item.get('available') and not item.get('raw') for item in media.values()):
                return jsonify(error='Open review with the original videos to edit IDs that are baked into the tracked copies.'), 409
            try:
                updated, remapped, activity_note = save_identity_correction(output, data, source, target)
            except TrackingChanged as exc:
                return jsonify(error=str(exc)), 409
            except ValueError as exc:
                return jsonify(error=str(exc)), 400
            except OSError:
                return jsonify(error='Could not save the identity correction.'), 500
            data.clear()
            data.update(updated)
            revision = fingerprint(data)
            assistant.refresh_reviewed_identities(remapped)
        return jsonify(ok=True, kept_id=remapped.get(target), activity_note=activity_note)

    @app.post('/api/merge')
    @assistant_operation
    def merge():
        return correct_person(merge=True)

    @app.post('/api/delete-id')
    @assistant_operation
    def delete_id():
        return correct_person(merge=False)

    return app


def main():
    from src.config import BASE_DIR, DATA_DIR, OUTPUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default=str(OUTPUT_DIR))
    parser.add_argument('--data-dir', default=str(DATA_DIR))
    parser.add_argument('--port', type=int, default=5002)
    default_map = BASE_DIR / 'config' / 'office_map.json'
    parser.add_argument('--office-map', type=Path, default=default_map if default_map.is_file() else None,
                        help='Building layout JSON (default: config/office_map.json when present)')
    args = parser.parse_args()
    try:
        app = create_app(args.output_dir, args.data_dir, office_map_path=args.office_map)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f'Could not open web review: {exc}\n')
    print(f'Web review ready: http://127.0.0.1:{args.port}', flush=True)
    app.run(host='127.0.0.1', port=args.port, threaded=True, debug=False)


if __name__ == '__main__':
    main()
