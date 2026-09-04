"""
tracking_log.py — Thread-safe tracking log that records per-person appearance timelines.

During processing, CameraWorkers call `record_frame()` each frame to report
which global IDs are currently visible. The log automatically builds
per-person appearance timelines (continuous stretches on a specific camera).

After processing, call `save()` to write the log as JSON for the review UI.

Data model:
  An "appearance" = a continuous stretch of frames where a person is visible
  on a specific camera. If they disappear for >GAP_THRESHOLD frames and
  reappear, it starts a new appearance.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# If a person disappears for more than this many frames, the next sighting
# starts a new appearance entry (rather than extending the current one).
GAP_THRESHOLD = 30


@dataclass
class Appearance:
    """A continuous stretch of visibility on one camera."""
    camera_id: int
    first_frame: int
    last_frame: int
    fps: float = 25.0

    @property
    def duration_seconds(self) -> float:
        return (self.last_frame - self.first_frame) / self.fps if self.fps > 0 else 0.0

    def first_timestamp(self) -> str:
        return _frames_to_timestamp(self.first_frame, self.fps)

    def last_timestamp(self) -> str:
        return _frames_to_timestamp(self.last_frame, self.fps)

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "first_timestamp": self.first_timestamp(),
            "last_timestamp": self.last_timestamp(),
            "duration_seconds": round(self.duration_seconds, 1),
        }


def _frames_to_timestamp(frame: int, fps: float) -> str:
    """Convert frame index to a human-readable timestamp (M:SS)."""
    if fps <= 0:
        return "0:00"
    total_seconds = int(frame / fps)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes}:{seconds:02d}"


class TrackingLog:
    """
    Thread-safe log of per-person appearance timelines across cameras.

    Usage (during processing):
        log = TrackingLog()
        log.register_camera(1, fps=25.0, total_frames=2785, video_file="cam1_tracked.mp4")
        # Each frame:
        log.record_frame(camera_id=1, frame_idx=100, visible_gids={0, 2, 5})

    Usage (after processing):
        log.save("output/tracking_log.json", gallery)

    Usage (review UI):
        data = TrackingLog.load("output/tracking_log.json")
    """

    def __init__(self):
        self._lock = threading.Lock()

        # Camera metadata: {cam_id: {fps, total_frames, video_file}}
        self._cameras: Dict[int, dict] = {}

        # Per-person, per-camera: the current (open) appearance being extended
        # Key: (global_id, camera_id) → Appearance
        self._open_appearances: Dict[Tuple[int, int], Appearance] = {}

        # Per-person: list of all completed + open appearances
        # Key: global_id → [Appearance, ...]
        self._appearances: Dict[int, List[Appearance]] = {}

        # Track last frame each person was seen per camera (for gap detection)
        self._last_seen: Dict[Tuple[int, int], int] = {}

        # Per-frame visibility record for the review UI
        # Stored as: {frame_idx: {cam_id: set of gids}}
        # To keep memory reasonable, we store per-camera frame maps
        # {cam_id: {frame_idx: set of gids}}
        self._frame_visibility: Dict[int, Dict[int, Set[int]]] = {}

        # Per-camera, per-frame bounding boxes: {cam_id: {frame_idx: [{"gid": gid, "bbox": [x1,y1,x2,y2]}]}}
        self._detections: Dict[int, Dict[int, List[dict]]] = {}

    def register_camera(
        self,
        cam_id: int,
        fps: float,
        total_frames: int,
        video_file: str,
        raw_video_file: str = "",
    ) -> None:
        """Register a camera's metadata."""
        with self._lock:
            self._cameras[cam_id] = {
                "fps": fps,
                "total_frames": total_frames,
                "video_file": video_file,
                "raw_video_file": raw_video_file,
            }
            self._frame_visibility[cam_id] = {}
            self._detections[cam_id] = {}

    def record_detections(
        self,
        camera_id: int,
        frame_idx: int,
        detections: List[Tuple[int, List[int]]],
    ) -> None:
        """Record per-frame bounding boxes: list of (gid, [x1, y1, x2, y2])."""
        with self._lock:
            if detections:
                self._detections.setdefault(camera_id, {})[frame_idx] = [
                    {"gid": gid, "bbox": bbox} for gid, bbox in detections
                ]

    def record_frame(self, camera_id: int, frame_idx: int, visible_gids: Set[int]) -> None:
        """
        Record which global IDs are visible in a specific camera at a specific frame.
        Called once per frame per camera from CameraWorker threads.
        """
        with self._lock:
            cam_meta = self._cameras.get(camera_id, {})
            fps = cam_meta.get("fps", 25.0)

            # Store frame visibility (sampled every 5 frames to save memory)
            if frame_idx % 5 == 0:
                self._frame_visibility.setdefault(camera_id, {})[frame_idx] = set(visible_gids)

            for gid in visible_gids:
                key = (gid, camera_id)
                last = self._last_seen.get(key, -GAP_THRESHOLD - 1)
                gap = frame_idx - last

                if gap > GAP_THRESHOLD:
                    # Start a new appearance
                    app = Appearance(
                        camera_id=camera_id,
                        first_frame=frame_idx,
                        last_frame=frame_idx,
                        fps=fps,
                    )
                    self._open_appearances[key] = app
                    self._appearances.setdefault(gid, []).append(app)
                else:
                    # Extend current appearance
                    if key in self._open_appearances:
                        self._open_appearances[key].last_frame = frame_idx

                self._last_seen[key] = frame_idx

    def get_visible_gids_at_frame(self, camera_id: int, frame_idx: int) -> Set[int]:
        """
        Return the set of global IDs visible on a camera at (or near) a given frame.
        Uses the nearest sampled frame for efficiency.
        """
        with self._lock:
            cam_frames = self._frame_visibility.get(camera_id, {})
            if not cam_frames:
                return set()
            sampled_idx = (frame_idx // 5) * 5
            return cam_frames.get(sampled_idx, set())

    def save(self, output_path: str, gallery) -> None:
        """
        Save the tracking log as JSON, enriched with gallery data and per-frame detections.
        """
        import base64
        import cv2
        from src.pipeline import get_color_hex

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            # Build person data from gallery + appearances
            persons = {}
            all_gids = set(self._appearances.keys())

            # Also include gallery entries that might not have appearances
            snapshot = gallery.get_all_entries_snapshot()
            for entry in snapshot:
                all_gids.add(entry["global_id"])

            for gid in sorted(all_gids):
                # Find gallery entry for thumbnail
                gallery_entry = next(
                    (e for e in snapshot if e["global_id"] == gid), None
                )

                appearances = self._appearances.get(gid, [])
                total_detections = sum(
                    a.last_frame - a.first_frame + 1 for a in appearances
                )

                persons[str(gid)] = {
                    "global_id": gid,
                    "color_hex": get_color_hex(gid),
                    "thumbnail_b64": gallery_entry["thumbnail_b64"] if gallery_entry else None,
                    "total_detections": gallery_entry["hit_count"] if gallery_entry else total_detections,
                    "cameras_seen": gallery_entry["cameras_seen"] if gallery_entry else sorted(set(a.camera_id for a in appearances)),
                    "appearances": [a.to_dict() for a in appearances],
                }

            # Convert detections dictionary to JSON serializable keys
            detections_json = {
                str(cid): {str(f): dets for f, dets in frames.items()}
                for cid, frames in self._detections.items()
            }

            data = {
                "cameras": {str(k): v for k, v in self._cameras.items()},
                "persons": persons,
                "detections": detections_json,
            }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"[TrackingLog] Saved tracking log to: {path}")

    @staticmethod
    def load(path: str) -> dict:
        """Load a tracking log JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
