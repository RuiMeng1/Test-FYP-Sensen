"""Bounded, frame-paced YOLO detection across camera streams.

Each camera has its own detector because ByteTrack keeps state across frames.
The dispatcher collects at most one frame per camera per round and runs those
detectors on a small thread pool. MPS stays serial for thread safety.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import queue
import threading
from typing import List

import numpy as np

from src.detector import Detection, PersonDetector


class DetectionDispatcher:
    """Serve camera frames in rounds with a bounded number of YOLO calls."""

    def __init__(
        self,
        n_cameras: int,
        device: str,
        model_name: str = "yolov8n.pt",
        max_workers: int | None = None,
    ):
        if n_cameras < 1:
            raise ValueError("n_cameras must be at least 1")
        if max_workers is not None and max_workers < 1:
            raise ValueError("max_workers must be at least 1")

        self.n_workers = min(n_cameras, 1 if device == "mps" else (max_workers or (2 if device == "cpu" else 1)))
        print(f"[Dispatcher] Loading {n_cameras} YOLO detector(s) on {device.upper()} "
              f"with {self.n_workers} inference worker(s) …")
        self._detectors: List[PersonDetector] = [
            PersonDetector(model_name=model_name, device=device)
            for _ in range(n_cameras)
        ]
        self._in_q: List[queue.Queue] = [queue.Queue(maxsize=1) for _ in range(n_cameras)]
        self._out_q: List[queue.Queue] = [queue.Queue(maxsize=1) for _ in range(n_cameras)]
        self._active = set(range(n_cameras))
        self._active_lock = threading.Lock()
        self._stop = threading.Event()
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._run, name="Dispatcher", daemon=True)
        self._thread.start()

    def detect(self, cam_idx: int, frame: np.ndarray, camera_id: int) -> List[Detection]:
        """Submit one frame and wait for its camera's result."""
        if not 0 <= cam_idx < len(self._detectors):
            raise IndexError(f"Invalid camera slot {cam_idx}")
        while not self._stop.is_set():
            try:
                self._in_q[cam_idx].put((frame, camera_id), timeout=0.1)
                break
            except queue.Full:
                pass
        while not self._stop.is_set():
            try:
                return self._out_q[cam_idx].get(timeout=0.1)
            except queue.Empty:
                pass
        if self._error is not None:
            raise RuntimeError("Detection worker failed") from self._error
        raise RuntimeError("Detection dispatcher stopped")

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def camera_done(self, cam_idx: int) -> None:
        """Stop waiting for frames after a camera has finished its video."""
        with self._active_lock:
            self._active.discard(cam_idx)

    def _run(self) -> None:
        try:
            with ThreadPoolExecutor(max_workers=self.n_workers, thread_name_prefix="YOLO") as pool:
                while not self._stop.is_set():
                    # A camera can submit only one frame until its result is
                    # returned. This keeps its ByteTrack calls ordered.
                    batch = []
                    with self._active_lock:
                        active = sorted(self._active)
                    if not active:
                        return
                    for cam_idx in active:
                        if self._stop.is_set():
                            return
                        try:
                            frame, camera_id = self._in_q[cam_idx].get(timeout=0.1)
                        except queue.Empty:
                            continue
                        batch.append((cam_idx, frame, camera_id))
                    futures = [
                        (cam_idx, pool.submit(self._detectors[cam_idx].detect, frame, camera_id))
                        for cam_idx, frame, camera_id in batch
                    ]
                    # Return a whole round together so a fast camera cannot
                    # start multiple new rounds while another is still detecting.
                    results = [(cam_idx, future.result()) for cam_idx, future in futures]
                    for cam_idx, detections in results:
                        self._out_q[cam_idx].put(detections)
        except Exception as exc:
            self._error = exc
            self._stop.set()
