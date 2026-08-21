"""
main.py — Entry point for the multi-camera person tracking pipeline.

Usage:
  # Default: all .mp4 in data/, live preview + save to output/
  python -m src.main

  # Pick specific videos
  python -m src.main --videos data/cam1_20260507_160822.mp4 data/cam2_20260507_160823.mp4

  # Headless (no preview — SSH / servers)
  python -m src.main --no-preview

  # Less frequent Re-ID (faster, slightly less cross-camera accuracy)
  python -m src.main --reid-interval 10

  # Force device
  python -m src.main --device cpu

Press  q  or  ESC  in the preview window to stop early.

Display layout (2 × N grid):
  ┌──────────┬──────────┐
  │  Cam 1   │  Cam 2   │
  ├──────────┼──────────┤
  │  Cam 3   │  Cam 4   │   ← empty tile if odd number of cameras
  └──────────┴──────────┘
"""

from __future__ import annotations

import queue
import time
from typing import Dict, List, Optional

import cv2
import numpy as np


# ── Layout constants ──────────────────────────────────────────────────────────
GRID_COLS = 2          # Always 2 columns (2 × N format)
TILE_W    = 640        # Width of each camera tile in pixels
TILE_H    = 360        # Height of each camera tile in pixels
DIVIDER   = (50, 50, 50)   # Grid line colour (dark grey, BGR)


# ── Tile helper ───────────────────────────────────────────────────────────────

def _tile_frames(
    frames: Dict[int, np.ndarray],
    tile_w: int = TILE_W,
    tile_h: int = TILE_H,
) -> np.ndarray:
    """
    Composite camera frames into a 2-column grid.

    For 3 cameras:          For 4 cameras:
      [ Cam1 | Cam2 ]         [ Cam1 | Cam2 ]
      [ Cam3 | ---- ]         [ Cam3 | Cam4 ]
    """
    sorted_ids = sorted(frames.keys())
    n    = len(sorted_ids)
    cols = GRID_COLS
    rows = (n + cols - 1) // cols      # ceil(n / 2)

    grid = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

    for idx, cam_id in enumerate(sorted_ids):
        row, col = divmod(idx, cols)
        resized = cv2.resize(frames[cam_id], (tile_w, tile_h))
        y1, y2 = row * tile_h, (row + 1) * tile_h
        x1, x2 = col * tile_w, (col + 1) * tile_w
        grid[y1:y2, x1:x2] = resized

    # Draw grid dividers
    for c in range(1, cols):
        cv2.line(grid, (c * tile_w, 0), (c * tile_w, rows * tile_h), DIVIDER, 2)
    for r in range(1, rows):
        cv2.line(grid, (0, r * tile_h), (cols * tile_w, r * tile_h), DIVIDER, 2)

    # Label empty tile if odd camera count
    if n % cols != 0:
        last_row = (n // cols)
        last_col = n % cols
        ex1 = last_col * tile_w
        ey1 = last_row * tile_h
        ex2 = ex1 + tile_w
        ey2 = ey1 + tile_h
        cv2.rectangle(grid, (ex1 + 2, ey1 + 2), (ex2 - 2, ey2 - 2), (30, 30, 30), -1)
        cv2.putText(
            grid, "No feed", (ex1 + tile_w // 2 - 50, ey1 + tile_h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 2, cv2.LINE_AA
        )

    return grid


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    from src.video_loader import parse_args, get_video_paths
    from src.config import DEVICE, REID_DEVICE, REID_EVERY_N_FRAMES, TRANSREID_WEIGHTS
    from src.config import REID_THRESHOLD, GALLERY_TTL_FRAMES, YOLO_MODEL_NAME
    from src.config import REID_THRESHOLD_MATRIX
    from src.dispatcher import DetectionDispatcher
    from src.reid_model import TransReIDEncoder
    from src.gallery import GlobalGallery
    from src.pipeline import CameraWorker

    args          = parse_args()
    video_paths   = get_video_paths(args)
    device        = args.device or DEVICE
    reid_interval = args.reid_interval or REID_EVERY_N_FRAMES
    show_preview  = not args.no_preview

    # Reid device: CPU when YOLO runs on MPS (MPS can't be used from threads)
    reid_dev = REID_DEVICE if device == DEVICE else ("cpu" if device == "mps" else device)

    # ── Banner ────────────────────────────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       Multi-Camera Person Tracking  (YOLOv8 + TransReID) ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Detect device : {device.upper()}")
    print(f"  ReID device   : {reid_dev.upper()}  "
          f"{'(MPS → CPU for thread safety)' if device == 'mps' else ''}")
    print(f"  Cameras       : {len(video_paths)}")
    print(f"  Grid layout   : {GRID_COLS} columns × {(len(video_paths)+1)//2} rows")
    print(f"  Re-ID every   : {reid_interval} frames")
    print(f"  Live preview  : {'ON  (press q / ESC to stop)' if show_preview else 'OFF'}")
    print(f"  Output dir    : output/")
    print()

    # ── Shared models ─────────────────────────────────────────────────────────
    n_cameras = len(video_paths)

    print(f"[Setup] Creating detection dispatcher ({n_cameras} camera(s), {device.upper()}) …")
    dispatcher = DetectionDispatcher(
        n_cameras=n_cameras,
        device=device,
        model_name=YOLO_MODEL_NAME,
    )

    print("[Setup] Loading TransReID encoder …")
    reid_model = TransReIDEncoder(weights_path=TRANSREID_WEIGHTS, device=reid_dev)

    print("[Setup] Initialising global gallery …")
    # Build a camera_id -> video stem map so the gallery can resolve per-pair thresholds
    camera_name_map = {idx + 1: vp.stem for idx, vp in enumerate(video_paths)}
    gallery = GlobalGallery(
        reid_threshold=REID_THRESHOLD,
        ttl_frames=GALLERY_TTL_FRAMES,
        threshold_matrix=REID_THRESHOLD_MATRIX,
        camera_name_map=camera_name_map,
    )

    # ── Display queue ─────────────────────────────────────────────────────────
    disp_q: Optional[queue.Queue] = (
        queue.Queue(maxsize=n_cameras * 4) if show_preview else None
    )

    # ── Camera workers ────────────────────────────────────────────────────────
    workers: List[CameraWorker] = []
    for idx, vp in enumerate(video_paths):
        w = CameraWorker(
            video_path    = vp,
            camera_id     = idx + 1,
            cam_idx       = idx,           # 0-based dispatcher slot
            dispatcher    = dispatcher,
            reid_model    = reid_model,
            gallery       = gallery,
            display_queue = disp_q,
            reid_interval = reid_interval,
        )
        workers.append(w)

    # ── Create window BEFORE threads (avoids Metal/Cocoa init race on macOS) ──
    n_rows = (n_cameras + GRID_COLS - 1) // GRID_COLS
    win_w  = TILE_W * GRID_COLS
    win_h  = TILE_H * n_rows

    if show_preview:
        cv2.namedWindow("Multi-Camera Tracker", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Multi-Camera Tracker", win_w, win_h)
        cv2.waitKey(1)

    print(f"\n[Main] Starting {n_cameras} worker thread(s) …\n")
    for w in workers:
        w.start()

    latest_frames: Dict[int, np.ndarray] = {}

    # ── Display / wait loop ───────────────────────────────────────────────────
    try:
        while True:
            all_started = all(w.has_started for w in workers)
            all_done    = all_started and all(w.is_done for w in workers)

            if show_preview and disp_q is not None:
                # Drain display queue
                for _ in range(n_cameras * 2):
                    try:
                        cam_id, frame = disp_q.get_nowait()
                        latest_frames[cam_id] = frame
                    except queue.Empty:
                        break

                if latest_frames:
                    grid = _tile_frames(latest_frames, TILE_W, TILE_H)
                    cv2.imshow("Multi-Camera Tracker", grid)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("\n[Main] Stopping on user request …")
                    for w in workers:
                        w.stop()
                    break
            else:
                time.sleep(0.05)

            if all_done:
                if show_preview:
                    print("\n[Main] All cameras finished. Press any key to close preview.")
                    while True:
                        if latest_frames:
                            cv2.imshow(
                                "Multi-Camera Tracker",
                                _tile_frames(latest_frames, TILE_W, TILE_H)
                            )
                        if cv2.waitKey(100) != -1:
                            break
                break

    except KeyboardInterrupt:
        print("\n[Main] Keyboard interrupt — stopping workers …")
        for w in workers:
            w.stop()
    finally:
        dispatcher.stop()
        for w in workers:
            w.join(timeout=10)
        if show_preview:
            cv2.destroyAllWindows()

    # ── Generate Summary Report ───────────────────────────────────────────────
    gallery.generate_html_report("output")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║  Processing complete                                 ║")
    print("╠══════════════════════════════════════════════════════╣")
    for w in workers:
        status = "✓" if w.is_done else "✗"
        name   = w.output_path.name
        print(f"║  {status} Cam{w.camera_id}  {name:<30} {w.fps_stat:5.1f} fps  ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"\nAnnotated videos saved to:  output/\n")


if __name__ == "__main__":
    main()
