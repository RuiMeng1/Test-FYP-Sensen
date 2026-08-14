"""
dispatcher.py — Round-robin scheduler that keeps cameras frame-synchronized.

Problem solved:
  With 3 separate YOLO instances racing for the MPS lock, whichever camera
  grabs the lock first runs far ahead in the video — cam2 at frame 600 while
  cam1 is only at frame 150.

Solution:
  N detectors (one per camera) run in strict round-robin inside a single
  dispatcher thread.  Only one camera's YOLO runs at a time (single-threaded
  MPS), so no malloc corruption.  Each camera keeps its own ByteTrack state
  (correct per-camera tracking).  Round-robin ordering keeps all cameras
  advancing at the same pace.

Key design decision — why N detectors and not 1?
  YOLO's ByteTrack uses `persist=True`, which maintains tracker state across
  frames.  If one shared model handles cam1 frame then cam2 frame, ByteTrack
  thinks cam2 is the next frame of the same video — completely wrong tracks.
  Each camera MUST have its own detector instance.
"""

from __future__ import annotations

import queue
import threading
from typing import List

import numpy as np

from src.detector import Detection, PersonDetector


class DetectionDispatcher:
    """
    Round-robin frame dispatcher: one detector per camera, serialized execution.

    - ONE thread runs all detections sequentially (no concurrent MPS calls).
    - Cameras are served in strict round-robin: cam0 → cam1 → cam2 → cam0 …
    - Each camera blocks on `detect()` until its turn comes around.
    - Naturally keeps all cameras within ±1 frame of each other.

    Usage (CameraWorker):
        detections = dispatcher.detect(cam_idx=0, frame=frame, camera_id=1)
    """

    def __init__(self, n_cameras: int, device: str, model_name: str = "yolov8n.pt"):
        self._n = n_cameras

        # ONE detector per camera — preserves independent ByteTrack state
        print(f"[Dispatcher] Loading {n_cameras} YOLO detector(s) on {device.upper()} …")
        self._detectors: List[PersonDetector] = [
            PersonDetector(model_name=model_name, device=device)
            for _ in range(n_cameras)
        ]

        # Per-camera queues (maxsize=1 = backpressure: fast cameras wait for dispatcher)
        self._in_q:  List[queue.Queue] = [queue.Queue(maxsize=1) for _ in range(n_cameras)]
        self._out_q: List[queue.Queue] = [queue.Queue(maxsize=1) for _ in range(n_cameras)]

        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._run, name="Dispatcher", daemon=True)
        self._thread.start()

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, cam_idx: int, frame: np.ndarray, camera_id: int) -> List[Detection]:
        """
        Submit a frame and block until detections come back.
        Called from each CameraWorker thread.
        """
        self._in_q[cam_idx].put((frame, camera_id))
        return self._out_q[cam_idx].get()

    def stop(self) -> None:
        self._stop.set()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run(self) -> None:
        """
        Dispatcher loop: round-robin over cameras.
        Blocks on each camera's in-queue so the loop naturally paces with workers.
        """
        while not self._stop.is_set():
            for i in range(self._n):
                try:
                    frame, camera_id = self._in_q[i].get(timeout=0.5)
                except queue.Empty:
                    continue   # Camera i is done or slow — skip this slot
                # Use camera i's own detector to keep ByteTrack state correct
                dets = self._detectors[i].detect(frame, camera_id)
                self._out_q[i].put(dets)

