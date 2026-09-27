"""Independent recorded-footage review. python -m src.web_review"""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import secrets
import threading
import tempfile
from functools import wraps
from fractions import Fraction

from flask import Flask, abort, jsonify, request, send_file
from src.action_events import fingerprint, load_actions
from src.action_labels import LABELS
from src.ai_assistant import AIAssistant


def atomic_json(path, data):
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False, suffix='.json') as handle:
        json.dump(data, handle, indent=2)
        temporary = Path(handle.name)
    temporary.replace(path)


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


def create_app(output_dir, data_dir, prepare=True):
    output, data_dir = Path(output_dir).resolve(), Path(data_dir).resolve()
    log = output / 'tracking_log.json'
    data = json.loads(log.read_text())
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
            cameras = {cid: {**meta, **{k: v for k, v in media[cid].items() if k != 'path'}}
                       for cid, meta in data['cameras'].items()}
            return jsonify(token=token, revision=revision, cameras=cameras, persons=data['persons'],
                           actions=assistant.action_data, labels=list(LABELS.values()), selected=settings)

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
            return jsonify({str(f): frames[str(f)] for f in range(int(chunk*10*fps), int((chunk+1)*10*fps)+1)
                            if str(f) in frames})

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
        prompt = (request.get_json() or {}).get('prompt', '')
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 10000:
            abort(400)
        if not ai_lock.acquire(blocking=False):
            return jsonify(error='Another operation is running. Try again shortly.'), 409
        try:
            reply = assistant.ask(prompt)
            return jsonify(reply=reply, evidence=assistant.get_visual_evidence(prompt, reply))
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

    @app.post('/api/merge')
    @assistant_operation
    def merge():
        nonlocal revision
        body = request.get_json() or {}
        source, target = str(body.get('source')), str(body.get('target'))
        with lock:
            if body.get('revision') != revision or fingerprint(json.loads(log.read_text())) != revision:
                return jsonify(error='Tracking data changed. Restart web review before merging.'), 409
            if source == target or source not in data['persons'] or target not in data['persons']:
                abort(400)
            if any(not item.get('raw', False) for item in media.values()):
                return jsonify(error='Original videos are required to redraw merged IDs correctly.'), 409
            updated = copy.deepcopy(data)
            src, dst = updated['persons'][source], updated['persons'][target]
            dst['appearances'] = sorted(dst.get('appearances', [])+src.get('appearances', []),
                                        key=lambda a: (a['camera_id'], a['first_frame']))
            dst['cameras_seen'] = sorted(set(dst.get('cameras_seen', [])) | set(src.get('cameras_seen', [])))
            dst['total_detections'] = dst.get('total_detections', 0)+src.get('total_detections', 0)
            thumbnails = []
            for person in (dst, src):
                for image in person.get('thumbnails_b64') or [person.get('thumbnail_b64')]:
                    if image and image not in thumbnails:
                        thumbnails.append(image)
            dst['thumbnails_b64'] = thumbnails[:3]
            dst['thumbnail_b64'] = thumbnails[0] if thumbnails else None
            for frames in updated.get('detections', {}).values():
                for detections in frames.values():
                    for detection in detections:
                        if detection.get('gid') == int(source) or detection.get('global_id') == int(source):
                            detection['gid'] = detection['global_id'] = int(target)
            del updated['persons'][source]
            backup = output / f'tracking_log.before-merge-{revision[:12]}.json'
            if not backup.exists():
                atomic_json(backup, data)
            atomic_json(log, updated)
            data.clear()
            data.update(updated)
            revision = fingerprint(data)
            assistant._profiles = assistant._load_profiles()
            assistant.action_data = load_actions(data, output)
            assistant.clear_history()
        return jsonify(ok=True)

    return app


def main():
    from src.config import DATA_DIR, OUTPUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default=str(OUTPUT_DIR))
    parser.add_argument('--data-dir', default=str(DATA_DIR))
    parser.add_argument('--port', type=int, default=5002)
    args = parser.parse_args()
    app = create_app(args.output_dir, args.data_dir)
    print(f'Web review ready: http://127.0.0.1:{args.port}', flush=True)
    app.run(host='127.0.0.1', port=args.port, threaded=True, debug=False)


if __name__ == '__main__':
    main()
