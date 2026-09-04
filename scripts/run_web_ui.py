"""
run_web_ui.py — Launch the Web Review Dashboard & OpenAI AI Assistant server.

Reads output/tracking_log.json and serves the interactive Web Dashboard on port 5000:
- Live Person Logger
- OpenAI AI Assistant (spatiotemporal & clothing queries)
- Multi-camera video preview

Usage:
    python scripts/run_web_ui.py
"""

import json
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask, Response, render_template, request, jsonify, send_from_directory

from src.pipeline import get_color_hex
from src.ai_assistant import query_tracking_ai, describe_person_thumbnail
from src import config

OUTPUT_DIR = ROOT / "output"
LOG_PATH = OUTPUT_DIR / "tracking_log.json"


app = Flask(
    __name__,
    template_folder=str(ROOT / "src" / "templates"),
    static_folder=str(ROOT / "src" / "static"),
    static_url_path="/static",
)


@app.route("/videos/<path:filename>")
def serve_video(filename: str):
    """Serve recorded MP4 files from output/ directory."""
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/api/camera_videos")
def camera_videos():
    """Returns mapping of {cam_id: filename} for tracked videos in output/."""
    video_map = {}
    if LOG_PATH.exists():
        try:
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            for cid_str, meta in data.get("cameras", {}).items():
                cid = int(cid_str)
                vf = meta.get("video_file", "")
                if vf and (OUTPUT_DIR / vf).exists():
                    video_map[cid] = vf
        except Exception:
            pass

    if not video_map:
        for cid in range(1, 5):
            candidates = list(OUTPUT_DIR.glob(f"cam{cid}_*tracked.mp4"))
            if candidates:
                video_map[cid] = candidates[0].name

    return jsonify(video_map)



@app.route("/")
def index():
    # Load camera count from tracking_log.json if available
    camera_ids = [1, 2, 3, 4]
    if LOG_PATH.exists():
        try:
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                log_data = json.load(f)
            cams = log_data.get("cameras", {})
            if cams:
                camera_ids = [int(k) for k in cams.keys()]
        except Exception:
            pass

    return render_template("index.html", camera_ids=sorted(camera_ids))


@app.route("/video_feed/<int:cam_id>")
def video_feed(cam_id: int):
    """Streams MJPEG frames from the tracked video files in output/."""
    import cv2
    import time

    def generate_frames():
        # Find matching tracked video in output/
        video_candidates = list(OUTPUT_DIR.glob(f"cam{cam_id}_*tracked.mp4")) + list(OUTPUT_DIR.glob(f"*cam{cam_id}*tracked.mp4"))
        if not video_candidates:
            video_candidates = sorted(OUTPUT_DIR.glob("*.mp4"))

        if not video_candidates:
            return

        video_path = str(video_candidates[0])

        while True:
            cap = cv2.VideoCapture(video_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ret:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + jpeg.tobytes()
                        + b"\r\n"
                    )
                time.sleep(0.04)  # ~25 fps cap
            cap.release()

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/persons")
def persons_sse():

    """Yields SSE stream reading from tracking_log.json for the review dashboard."""
    import time

    def sse_generator():
        while True:
            persons = []
            total_tracked = 0
            if LOG_PATH.exists():
                try:
                    with open(LOG_PATH, "r", encoding="utf-8") as f:
                        log_data = json.load(f)

                    cameras_meta = log_data.get("cameras", {})
                    persons_dict = log_data.get("persons", {})
                    total_tracked = len(persons_dict)

                    for gid_str, pinfo in persons_dict.items():
                        gid = int(gid_str)
                        cams_seen = pinfo.get("cameras_seen", [])

                        # Enrich each appearance with fps from camera metadata
                        raw_appearances = pinfo.get("appearances", [])
                        enriched = []
                        for app in raw_appearances:
                            cam_meta = cameras_meta.get(str(app.get("camera_id", "")), {})
                            fps = cam_meta.get("fps", 25.0)
                            enriched.append({**app, "fps": fps})

                        persons.append({
                            "global_id": gid,
                            "color_hex": get_color_hex(gid),
                            "hit_count": pinfo.get("total_detections", 0),
                            "cameras_seen": cams_seen,
                            "active_cameras": cams_seen,
                            "is_active": True,
                            "thumbnail_b64": pinfo.get("thumbnail_b64"),
                            "visual_attributes": pinfo.get("visual_attributes"),
                            "appearances": enriched,
                        })


                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"

            payload = json.dumps({
                "persons": persons,
                "total_tracked": total_tracked,
            })
            yield f"data: {payload}\n\n"
            time.sleep(1.0)

    return Response(
        sse_generator(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/ai/chat", methods=["POST"])
def ai_chat():
    data = request.get_json() or {}
    prompt = data.get("prompt", "").strip()
    api_key = data.get("api_key") or config.OPENAI_API_KEY

    if not prompt:
        return jsonify({"error": "Prompt cannot be empty"}), 400

    log_data = {}
    if LOG_PATH.exists():
        try:
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                log_data = json.load(f)
        except Exception as e:
            print(f"[WebUI] Warning reading {LOG_PATH}: {e}")

    response_text = query_tracking_ai(
        user_prompt=prompt,
        log_data=log_data,
        api_key=api_key,
    )
    return jsonify({"response": response_text})


@app.route("/api/ai/describe-person", methods=["POST"])
def describe_person():
    data = request.get_json() or {}
    gid = data.get("global_id")
    api_key = data.get("api_key") or config.OPENAI_API_KEY

    if gid is None:
        return jsonify({"error": "global_id is required"}), 400

    if not LOG_PATH.exists():
        return jsonify({"error": "tracking_log.json not found"}), 404

    with open(LOG_PATH, "r", encoding="utf-8") as f:
        log_data = json.load(f)

    persons = log_data.get("persons", {})
    pinfo = persons.get(str(gid))

    if not pinfo or not pinfo.get("thumbnail_b64"):
        return jsonify({"error": f"Thumbnail not found for person ID {gid}"}), 404

    try:
        desc = describe_person_thumbnail(pinfo["thumbnail_b64"], api_key=api_key)
        pinfo["visual_attributes"] = desc

        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2)

        return jsonify({"global_id": gid, "visual_attributes": desc})
    except Exception as err:
        return jsonify({"error": str(err)}), 500


def main():
    port = 5001

    print(f"\n╔══════════════════════════════════════════════════════════╗")
    print(f"║  Sensen Web Review Dashboard & OpenAI AI Assistant       ║")
    print(f"╠══════════════════════════════════════════════════════════╣")
    print(f"║  ✦ Dashboard running at: http://localhost:{port}          ║")
    print(f"║  ✦ Press Ctrl+C in terminal to stop                       ║")
    print(f"╚══════════════════════════════════════════════════════════╝\n")

    webbrowser.open(f"http://localhost:{port}")

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
