"""
detector.py — YOLOv8 person detector with ByteTrack for per-camera local tracking.

Each CameraWorker creates its own PersonDetector instance because
YOLO's .track(persist=True) maintains internal tracker state per model instance.
Sharing one instance across threads would mix up tracker IDs between cameras.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# PyTorch MPS is NOT safe for concurrent multi-threaded use.
# All PersonDetector instances share this lock so only one camera
# runs MPS inference at a time. (Other cameras do CPU/IO work meanwhile.)
_MPS_LOCK = threading.Lock()


@dataclass
class Detection:
    """A single detected and locally-tracked person in one frame."""
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2) pixel coords
    local_id: int                      # ByteTrack local ID (per camera, resets on new video)
    confidence: float                  # YOLO detection confidence
    camera_id: int                     # Which camera this belongs to


class PersonDetector:
    """
    Wraps YOLOv8-nano with ByteTrack for person detection and local tracking.
    One instance per camera stream.
    """

    def __init__(self, model_name: str = "yolov8n.pt", device: str = "cpu"):
        from ultralytics import YOLO
        self._model = YOLO(model_name)
        self.device = device
        print(f"[Detector] YOLOv8 loaded on {device.upper()}")

    def detect(self, frame: np.ndarray, camera_id: int) -> List[Detection]:
        """
        Run detection + ByteTrack on a single frame.

        Args:
            frame:      BGR numpy array (OpenCV format)
            camera_id:  Camera index for labelling detections

        Returns:
            List of Detection objects for every tracked person in the frame.
        """
        track_kwargs = dict(
            source=frame,
            persist=True,           # Maintain tracker state across frames
            classes=[0],            # Class 0 = person only
            conf=0.40,              # Minimum detection confidence
            iou=0.50,               # NMS IoU threshold
            tracker="bytetrack.yaml",
            verbose=False,
            device=self.device,
        )

        # Serialize MPS access — concurrent MPS from multiple threads causes
        # malloc heap corruption on macOS (Metal GPU resource contention).
        if self.device == "mps":
            with _MPS_LOCK:
                results = self._model.track(**track_kwargs)
        else:
            results = self._model.track(**track_kwargs)

        detections: List[Detection] = []
        for r in results:
            if r.boxes is None or r.boxes.id is None:
                continue
            for box in r.boxes:
                if box.id is None:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                # Clamp to frame bounds
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    local_id=int(box.id.item()),
                    confidence=float(box.conf.item()),
                    camera_id=camera_id,
                ))

        return detections
