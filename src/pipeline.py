"""
pipeline.py — Single-camera processing worker.

Each CameraWorker runs in its own thread:
  1. Reads frames from the video file
  2. Sends the frame to the shared DetectionDispatcher (round-robin, synchronized)
  3. Every reid_interval frames: extracts TransReID embeddings for all detected persons
  4. Queries the shared GlobalGallery for global IDs
  5. Annotates frames with large, visible bounding boxes and global IDs
  6. Pushes annotated frames to a shared display queue (for live preview)
  7. Writes annotated frames to an output .mp4 file

Synchronization note:
  Detection now goes through DetectionDispatcher which processes cameras in
  round-robin order, keeping all cameras at the same frame position in their videos.
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.config import GALLERY_TTL_FRAMES, OUTPUT_DIR, EXCLUSION_ZONES
from src.detector import Detection
from src.gallery import GlobalGallery
from src.reid_model import TransReIDEncoder

# ── Active-person state shared with the web UI ────────────────────────────────
# Dict of {camera_id: set of global_ids currently visible}.
# Written by CameraWorkers each frame, read by the Flask server.
active_ids_by_camera: Dict[int, set] = {}


# ── Colour Normalisation ──────────────────────────────────────────────────────
def normalize_color(image: np.ndarray) -> np.ndarray:
    """
    Normalises the colour of an image using the Gray World assumption.
    This helps to correct colour casts caused by different camera sensors/lighting,
    ensuring that colours are consistent across multiple cameras for ReID.
    """
    means = np.mean(image, axis=(0, 1))
    if np.any(means == 0):
        return image
    gray_mean = np.mean(means)
    scales = gray_mean / means
    return np.clip(image * scales, 0, 255).astype(np.uint8)


# ── Colour palette for global IDs ────────────────────────────────────────────
_PALETTE: List[Tuple[int, int, int]] = [
    (60, 180, 255),  (50, 220,  50),  (255,  80,  80),  (255, 200,  50),
    (200,  80, 255),  (50, 200, 200),  (255, 140,  50),  (100, 255, 100),
    ( 80,  80, 255),  (255,  80, 200),  (50, 255, 170),  (255, 170,  80),
    (170,  50, 255),  ( 80, 255,  80),  (255,  50, 130),  (130, 255,  50),
    ( 50, 130, 255),  (255, 255,  80),  ( 80, 255, 255),  (255,  80, 255),
]

def _color(global_id: int) -> Tuple[int, int, int]:
    """Return the BGR colour for a global ID.  Also used by the web UI."""
    return _PALETTE[global_id % len(_PALETTE)]


def get_color_hex(global_id: int) -> str:
    """Return the CSS hex colour (RGB) for a global ID."""
    b, g, r = _color(global_id)
    return f"#{r:02x}{g:02x}{b:02x}"


class CameraWorker:
    """
    Processes one video stream in a background thread.

    Parameters
    ----------
    video_path    : Path to the .mp4 file.
    camera_id     : Integer camera label (1-based, for display).
    cam_idx       : 0-based index into the dispatcher's camera slots.
    dispatcher    : Shared DetectionDispatcher (round-robin single YOLO).
    reid_model    : Shared TransReID encoder (thread-safe via internal lock).
    gallery       : Shared GlobalGallery (thread-safe).
    display_queue : Optional queue for pushing annotated frames to the display loop.
    reid_interval : Extract Re-ID features every N frames.
    """

    def __init__(
        self,
        video_path:    Path,
        camera_id:     int,
        cam_idx:       int,
        dispatcher,                         # DetectionDispatcher
        reid_model:    TransReIDEncoder,
        gallery:       GlobalGallery,
        display_queue: Optional[queue.Queue] = None,
        reid_interval: int = 5,
        tracking_log=None,                  # Optional TrackingLog for review UI
    ):
        self.video_path    = video_path
        self.camera_id     = camera_id
        self.cam_idx       = cam_idx
        self.dispatcher    = dispatcher
        self.reid_model    = reid_model
        self.gallery       = gallery
        self.display_queue = display_queue
        self.reid_interval = reid_interval
        self.tracking_log  = tracking_log

        self.output_path   = OUTPUT_DIR / f"{video_path.stem}_tracked.mp4"
        self.is_done       = False
        self.has_started   = False
        self.fps_stat      = 0.0
        self.total_frames  = 0
        self.frame_idx     = 0

        self._stop_event   = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Cache: local_id → global_id between Re-ID intervals
        self._id_cache: Dict[int, int] = {}
        # local track ID -> last frame offered as a thumbnail candidate
        self._last_thumbnail_frame: Dict[int, int] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name=f"Cam{self.camera_id}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread:
            self._thread.join(timeout=timeout)

    @property
    def progress_pct(self) -> float:
        if self.total_frames <= 0:
            return 0.0
        return 100.0 * self.frame_idx / self.total_frames

    # ── Main processing loop ──────────────────────────────────────────────────

    def _run(self) -> None:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            print(f"[Cam{self.camera_id}] ERROR: Cannot open {self.video_path}")
            self.has_started = True
            self.is_done = True
            return

        self.has_started = True

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Register camera in the tracking log
        if self.tracking_log is not None:
            self.tracking_log.register_camera(
                self.camera_id, fps=src_fps,
                total_frames=self.total_frames,
                video_file=self.output_path.name,
                raw_video_file=self.video_path.name,
            )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        from src.occlusion_logger import OcclusionLogger
        try:
            occlusion_logger = OcclusionLogger(
                OUTPUT_DIR / 'occlusion_logs', self.camera_id, self.video_path, src_fps)
            print(f'[Cam{self.camera_id}] Occlusion diagnostics: {occlusion_logger.path}')
        except OSError as exc:
            print(f'[Cam{self.camera_id}] Occlusion logging unavailable: {exc}')
            occlusion_logger = None
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.output_path), fourcc, src_fps, (width, height))

        t_start = time.time()
        print(
            f"[Cam{self.camera_id}] ▶  {self.video_path.name}  "
            f"({width}×{height} @ {src_fps:.1f} fps | {self.total_frames} frames)"
        )

        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            # ── Detection via shared dispatcher (round-robin, synchronized) ──
            detections = self.dispatcher.detect(self.cam_idx, frame, self.camera_id)

            # ── Normalise Colour ──
            # Normalise the frame's colours to correct camera-specific tints.
            # Doing this after detection preserves YOLO's accuracy while 
            # providing consistent colours for Re-ID crops and the final output.
            frame = normalize_color(frame)

            # ── Exclusion Zone Filtering ──
            if self.video_path.name in EXCLUSION_ZONES:
                filtered = []
                for d in detections:
                    x1, y1, x2, y2 = d.bbox
                    cx, cy = (x1 + x2) / 2.0, float(y2)  # bottom-center point
                    in_zone = False
                    for poly in EXCLUSION_ZONES[self.video_path.name]:
                        pts = np.array(poly, np.int32)
                        if cv2.pointPolygonTest(pts, (cx, cy), False) >= 0:
                            in_zone = True
                            break
                    if not in_zone:
                        filtered.append(d)
                detections = filtered

            # ── Re-ID every N frames (or immediately for newly appeared tracks) ──
            has_unmapped = any(d.local_id not in self._id_cache for d in detections)
            if (self.frame_idx % self.reid_interval == 0 or has_unmapped) and detections:
                crops_to_extract = []
                valid_indices = set()

                for idx, d in enumerate(detections):
                    x1, y1, x2, y2 = d.bbox
                    bw = x2 - x1
                    bh = y2 - y1
                    # Quality gate: clean bounding box with sufficient resolution
                    # Preserve the local/global mapping during suspected
                    # occlusion, but skip embeddings AND thumbnail candidates.
                    # The gallery retains known IDs when passed embedding=None.
                    if not d.partially_occluded and bw >= 28 and bh >= 60 and d.confidence >= 0.40:
                        crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                        if crop.size > 0:
                            crops_to_extract.append(crop)
                            valid_indices.add(idx)

                extracted_embs = (
                    self.reid_model.extract_batch(crops_to_extract)
                    if crops_to_extract
                    else []
                )

                batch_items: List[Tuple[int, Optional[np.ndarray]]] = []
                local_id_to_crop = {}
                emb_ptr = 0
                for idx, d in enumerate(detections):
                    if idx in valid_indices:
                        batch_items.append((d.local_id, extracted_embs[emb_ptr]))
                        local_id_to_crop[d.local_id] = crops_to_extract[emb_ptr]
                        emb_ptr += 1
                    else:
                        batch_items.append((d.local_id, None))

                # ── Intra-Frame Duplicate / Reflection Suppression ─────────────
                # If two valid detections in the same frame have highly similar features,
                # they are likely the same person (e.g. a reflection or duplicate box).
                # We suppress the one with lower YOLO confidence by dropping its embedding.
                for i in range(len(batch_items)):
                    tid_i, emb_i = batch_items[i]
                    if emb_i is None:
                        continue
                    for j in range(i + 1, len(batch_items)):
                        tid_j, emb_j = batch_items[j]
                        if emb_j is None:
                            continue
                        
                        sim = float(np.dot(emb_i, emb_j))
                        if sim > 0.85:  # 0.82 → 0.85: only suppress genuine reflections/duplicates
                            conf_i = detections[i].confidence
                            conf_j = detections[j].confidence
                            if conf_i < conf_j:
                                batch_items[i] = (tid_i, None)
                                emb_i = None
                                break  # 'i' is suppressed, stop comparing it
                            else:
                                batch_items[j] = (tid_j, None)

                matched = self.gallery.match_batch(self.camera_id, self.frame_idx, batch_items)
                self._id_cache.update(matched)

                # Periodically offer crops so the gallery can retain three
                # strong, varied views instead of only the track's first frame.
                thumbnail_interval = max(5, self.reid_interval)
                for tid, gid in matched.items():
                    last_thumbnail_frame = self._last_thumbnail_frame.get(tid)
                    is_thumbnail_due = (
                        last_thumbnail_frame is None
                        or self.frame_idx - last_thumbnail_frame >= thumbnail_interval
                    )
                    if tid in local_id_to_crop and is_thumbnail_due:
                        self.gallery.add_thumbnail_to_entry(gid, local_id_to_crop[tid], self.camera_id)
                        self._last_thumbnail_frame[tid] = self.frame_idx

            # ── Periodic gallery cleanup ─────────────────────────────────────
            if self.frame_idx % 60 == 0:
                self.gallery.cleanup(self.frame_idx)

            # ── Annotate ─────────────────────────────────────────────────────
            # Diagnostics are isolated from identity matching and appearance updates.
            if occlusion_logger is not None:
                try:
                    occlusion_logger.update(self.frame_idx, detections, self._id_cache)
                except OSError as exc:
                    print(f'[Cam{self.camera_id}] Occlusion logging disabled: {exc}')
                    occlusion_logger = None
            annotated = self._draw(frame, detections)

            # ── Update active-person state ─────────────────────────────────────
            visible_gids = set()
            for det in detections:
                gid = self._id_cache.get(det.local_id, -1)
                if gid >= 0:
                    visible_gids.add(gid)
            active_ids_by_camera[self.camera_id] = visible_gids

            # ── Record in tracking log ────────────────────────────────────────
            if self.tracking_log is not None:
                self.tracking_log.record_frame(
                    self.camera_id, self.frame_idx, visible_gids
                )
                frame_dets = []
                for det in detections:
                    gid = self._id_cache.get(det.local_id, -1)
                    if gid >= 0:
                        frame_dets.append((gid, [int(v) for v in det.bbox]))
                self.tracking_log.record_detections(
                    self.camera_id, self.frame_idx, frame_dets
                )

            # ── Write to file ─────────────────────────────────────────────────
            writer.write(annotated)

            # ── Push to display queue ─────────────────────────────────────────
            if self.display_queue is not None:
                try:
                    self.display_queue.put_nowait((self.camera_id, annotated.copy()))
                except queue.Full:
                    pass

            # ── Stats ─────────────────────────────────────────────────────────
            self.frame_idx += 1
            elapsed = time.time() - t_start
            self.fps_stat = self.frame_idx / elapsed if elapsed > 0 else 0.0

            if self.frame_idx % 150 == 0:
                print(
                    f"[Cam{self.camera_id}]  "
                    f"{self.frame_idx:>5}/{self.total_frames}  "
                    f"({self.progress_pct:5.1f}%)  "
                    f"{self.fps_stat:.1f} fps  |  "
                    f"gallery: {self.gallery.active_count} persons"
                )

        cap.release()
        writer.release()
        if occlusion_logger is not None:
            try:
                occlusion_logger.finish(self.frame_idx)
            except OSError as exc:
                print(f'[Cam{self.camera_id}] Could not finalize occlusion log: {exc}')
        self.is_done = True
        print(
            f"[Cam{self.camera_id}] ✓  Done — {self.fps_stat:.1f} fps avg  →  {self.output_path}"
        )

    # ── Annotation ────────────────────────────────────────────────────────────

    def _draw(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            gid   = self._id_cache.get(det.local_id, -1)
            color = _color(gid) if gid >= 0 else (140, 140, 140)

            # ── Thick bounding box ────────────────────────────────────────────
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)

            # ── Large ID label ────────────────────────────────────────────────
            id_text   = f"ID:{gid:03d}" if gid >= 0 else f"L:{det.local_id}"
            id_font   = cv2.FONT_HERSHEY_DUPLEX
            id_scale  = 1.6          # Large and readable
            id_thick  = 2
            (tw, th), bl = cv2.getTextSize(id_text, id_font, id_scale, id_thick)

            pad = 8
            # Position: above the box, clamped to frame top
            lbl_x1 = x1
            lbl_y2 = y1
            lbl_y1 = max(0, y1 - th - bl - pad * 2)
            lbl_x2 = x1 + tw + pad * 2

            # Solid coloured background for the label
            cv2.rectangle(out, (lbl_x1, lbl_y1), (lbl_x2, lbl_y2), color, -1)
            # Black border for contrast
            cv2.rectangle(out, (lbl_x1, lbl_y1), (lbl_x2, lbl_y2), (0, 0, 0), 1)
            # White text on coloured background
            cv2.putText(
                out, id_text,
                (lbl_x1 + pad, lbl_y2 - bl - pad),
                id_font, id_scale, (255, 255, 255), id_thick, cv2.LINE_AA
            )

            # ── Confidence badge — small tag at bottom-left of box ────────────
            conf_text = f"{det.confidence:.0%}"
            cv2.putText(
                out, conf_text,
                (x1 + 4, y2 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA
            )

        # ── HUD: semi-transparent bar at top of frame ─────────────────────────
        hud = (
            f"  CAM {self.camera_id}  |  "
            f"Frame {self.frame_idx}/{self.total_frames}  "
            f"({self.progress_pct:.1f}%)  |  "
            f"{self.fps_stat:.1f} fps  |  "
            f"{len(detections)} person(s)"
        )
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, 42), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.60, out, 0.40, 0, out)
        cv2.putText(
            out, hud, (6, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 140), 2, cv2.LINE_AA
        )

        return out
