"""
web_server.py — Flask web UI for live multi-camera person tracking.

Endpoints:
  GET /                    — Main dashboard (HTML page)
  GET /video_feed/<cam_id> — MJPEG stream for a specific camera
  GET /api/persons         — SSE stream of real-time person logger data (JSON)

The server reads from shared data structures populated by CameraWorkers:
  - latest_frames:  {cam_id: np.ndarray}   — latest annotated frame per camera
  - gallery:        GlobalGallery           — full person identity store
  - active_ids:     pipeline.active_ids_by_camera — who's visible right now
"""

from __future__ import annotations

import json
import queue
import threading
import time
from typing import Dict, Optional

import cv2
import numpy as np
from flask import Flask, Response, render_template, request, jsonify

from src.pipeline import active_ids_by_camera, get_color_hex
from src.ai_assistant import query_tracking_ai, describe_person_thumbnail
from src import config
from pathlib import Path


class TrackingWebServer:
    """
    Wraps a Flask app that serves the tracking dashboard.

    Parameters
    ----------
    gallery     : The shared GlobalGallery instance.
    n_cameras   : Total number of camera streams.
    port        : HTTP port to listen on.
    """

    def __init__(self, gallery, n_cameras: int, port: int = 5000):
        self.gallery = gallery
        self.n_cameras = n_cameras
        self.port = port

        # Shared frame buffer: {cam_id: np.ndarray (BGR)}
        # Written by CameraWorkers via push_frame(), read by MJPEG generator.
        self._latest_frames: Dict[int, np.ndarray] = {}
        self._frame_lock = threading.Lock()

        # Flask app
        self.app = Flask(
            __name__,
            template_folder=str(
                __import__("pathlib").Path(__file__).parent / "templates"
            ),
        )
        self._register_routes()

        self._thread: Optional[threading.Thread] = None

    # ── Frame ingestion (called by CameraWorkers) ─────────────────────────────

    def push_frame(self, cam_id: int, frame: np.ndarray) -> None:
        """Store the latest annotated frame for a camera (thread-safe)."""
        with self._frame_lock:
            self._latest_frames[cam_id] = frame

    # ── Routes ────────────────────────────────────────────────────────────────

    def _register_routes(self) -> None:
        app = self.app

        @app.route("/")
        def index():
            camera_ids = list(range(1, self.n_cameras + 1))
            return render_template("index.html", camera_ids=camera_ids)

        @app.route("/video_feed/<int:cam_id>")
        def video_feed(cam_id: int):
            return Response(
                self._mjpeg_generator(cam_id),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @app.route("/api/persons")
        def persons_sse():
            return Response(
                self._sse_generator(),
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

            # Load tracking log context from disk if available
            log_path = config.OUTPUT_DIR / "tracking_log.json"
            log_data = {}
            if log_path.exists():
                try:
                    with open(log_path, "r", encoding="utf-8") as f:
                        log_data = json.load(f)
                except Exception as e:
                    print(f"[WebServer] Warning reading {log_path}: {e}")

            # If tracking log wasn't on disk, construct from live gallery entries
            if not log_data.get("persons"):
                entries = self.gallery.get_all_entries_snapshot()
                persons = {}
                for entry in entries:
                    gid = entry["global_id"]
                    persons[str(gid)] = {
                        "global_id": gid,
                        "thumbnail_b64": entry.get("thumbnail_b64"),
                        "total_detections": entry.get("hit_count", 0),
                        "cameras_seen": entry.get("cameras_seen", []),
                        "visual_attributes": entry.get("visual_attributes"),
                        "appearances": [],
                    }
                log_data = {
                    "cameras": {
                        str(i): {"fps": 25.0, "total_frames": 0, "video_file": f"cam{i}"}
                        for i in range(1, self.n_cameras + 1)
                    },
                    "persons": persons,
                }

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

            entries = self.gallery.get_all_entries_snapshot()
            entry = next((e for e in entries if e["global_id"] == int(gid)), None)

            if not entry or not entry.get("thumbnail_b64"):
                return jsonify({"error": f"Thumbnail not found for person ID {gid}"}), 404

            try:
                desc = describe_person_thumbnail(entry["thumbnail_b64"], api_key=api_key)
                entry["visual_attributes"] = desc

                # Also update tracking_log.json on disk if present
                log_path = config.OUTPUT_DIR / "tracking_log.json"
                if log_path.exists():
                    with open(log_path, "r", encoding="utf-8") as f:
                        log_data = json.load(f)
                    if str(gid) in log_data.get("persons", {}):
                        log_data["persons"][str(gid)]["visual_attributes"] = desc
                        with open(log_path, "w", encoding="utf-8") as f:
                            json.dump(log_data, f, indent=2)

                return jsonify({"global_id": gid, "visual_attributes": desc})
            except Exception as err:
                return jsonify({"error": str(err)}), 500


    # ── MJPEG generator ───────────────────────────────────────────────────────

    def _mjpeg_generator(self, cam_id: int):
        """Yields MJPEG frames for the given camera."""
        while True:
            with self._frame_lock:
                frame = self._latest_frames.get(cam_id)

            if frame is not None:
                ret, jpeg = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
                )
                if ret:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + jpeg.tobytes()
                        + b"\r\n"
                    )
            time.sleep(0.033)  # ~30 fps cap

    # ── SSE generator ─────────────────────────────────────────────────────────

    def _sse_generator(self):
        """Yields SSE events with person logger data every ~500ms."""
        while True:
            try:
                entries = self.gallery.get_all_entries_snapshot()

                # Build set of all currently active global IDs (across all cameras)
                all_active: Dict[int, list] = {}  # gid -> [cam_ids where active]
                for cam_id, gids in dict(active_ids_by_camera).items():
                    for gid in gids:
                        all_active.setdefault(gid, []).append(cam_id)

                persons = []
                for entry in entries:
                    gid = entry["global_id"]
                    active_cams = sorted(all_active.get(gid, []))
                    persons.append({
                        "global_id": gid,
                        "color_hex": get_color_hex(gid),
                        "hit_count": entry["hit_count"],
                        "cameras_seen": entry["cameras_seen"],
                        "active_cameras": active_cams,
                        "is_active": len(active_cams) > 0,
                        "thumbnail_b64": entry["thumbnail_b64"],
                    })

                payload = json.dumps({
                    "persons": persons,
                    "total_tracked": len(entries),
                })

                yield f"data: {payload}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

            time.sleep(0.5)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the Flask server in a background daemon thread."""
        self._thread = threading.Thread(
            target=self._run_server, name="WebServer", daemon=True
        )
        self._thread.start()

    def _run_server(self) -> None:
        import logging

        # Suppress Flask's verbose request logging
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.WARNING)

        print(f"\n[WebUI] ✦ Dashboard running at  http://localhost:{self.port}\n")
        self.app.run(
            host="0.0.0.0",
            port=self.port,
            debug=False,
            use_reloader=False,
            threaded=True,
        )
