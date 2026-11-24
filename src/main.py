"""
main.py — Entry point for the multi-camera person tracking pipeline.

Two-phase workflow:
  Phase 1: Pre-process all videos headlessly (no UI, maximum speed)
  Phase 2: Open interactive review dashboard with playback + person logger

Usage:
  # Full pipeline: process → review
  python -m src.main --all-id-cameras

  # Pick specific videos
  python -m src.main --videos data/cam1.mp4 data/cam2.mp4 --new-id-cameras cam1

  # Skip the review UI (headless processing only)
  python -m src.main --use-saved-id-cameras --no-preview

  # Only open the review UI (skip processing, use existing output)
  python -m src.main --review-only

  # Force device
  python -m src.main --all-id-cameras --device cpu
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import cv2
import numpy as np


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    from src.video_loader import parse_args, get_video_paths, print_camera_list, resolve_new_id_cameras
    from src.config import OUTPUT_DIR, PIPELINE_SETTINGS_PATH
    from src.tracking_log import TrackingLog

    args          = parse_args()
    show_preview  = not args.no_preview
    review_only   = args.review_only

    # ── Review-only mode: skip processing, open review UI directly ────────────
    if review_only:
        log_path = OUTPUT_DIR / "tracking_log.json"
        if not log_path.exists():
            print(f"[Main] ERROR: No tracking log found at {log_path}")
            print("       Run processing first: python -m src.main --all-id-cameras")
            return

        print(f"[Main] Loading tracking log from {log_path} …")
        tracking_data = TrackingLog.load(str(log_path))

        from src.review_ui import ReviewUI
        ui = ReviewUI(tracking_data=tracking_data, output_dir=str(OUTPUT_DIR))
        ui.run()
        return

    video_paths = get_video_paths(args)
    if args.list_cameras:
        print_camera_list(video_paths)
        return
    try:
        new_id_cameras = resolve_new_id_cameras(args, video_paths, PIPELINE_SETTINGS_PATH)
    except ValueError as exc:
        print_camera_list(video_paths)
        raise SystemExit(str(exc)) from None

    # ── Phase 1: Pre-process (headless) ───────────────────────────────────────
    from src.config import DEVICE, REID_DEVICE, REID_EVERY_N_FRAMES, TRANSREID_WEIGHTS
    from src.config import REID_THRESHOLD, GALLERY_TTL_FRAMES, YOLO_MODEL_NAME
    from src.config import REID_THRESHOLD_MATRIX
    from src.dispatcher import DetectionDispatcher
    from src.reid_model import TransReIDEncoder
    from src.gallery import GlobalGallery
    from src.pipeline import CameraWorker

    device        = args.device or DEVICE
    reid_interval = args.reid_interval or REID_EVERY_N_FRAMES

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
    id_sources = ('All cameras' if new_id_cameras is None else
                  ', '.join(video_paths[number - 1].name for number in sorted(new_id_cameras)))
    print(f"  New IDs from  : {id_sources}")
    print(f"  Re-ID every   : {reid_interval} frames")
    print(f"  Review UI     : {'AFTER PROCESSING' if show_preview else 'OFF'}")
    print(f"  Output dir    : output/")
    print()

    # ── Shared models ─────────────────────────────────────────────────────────
    n_cameras = len(video_paths)

    print(f"[Setup] Creating detection dispatcher ({n_cameras} camera(s), {device.upper()}) …")
    dispatcher = DetectionDispatcher(
        n_cameras=n_cameras,
        device=device,
        model_name=YOLO_MODEL_NAME,
        max_workers=args.detector_workers,
    )
    print(f"[Setup] Running {dispatcher.n_workers} YOLO detection worker(s)")

    print("[Setup] Loading TransReID encoder …")
    reid_model = TransReIDEncoder(weights_path=TRANSREID_WEIGHTS, device=reid_dev)

    print("[Setup] Initialising global gallery …")
    camera_name_map = {idx + 1: vp.stem for idx, vp in enumerate(video_paths)}
    gallery = GlobalGallery(
        reid_threshold=REID_THRESHOLD,
        ttl_frames=GALLERY_TTL_FRAMES,
        threshold_matrix=REID_THRESHOLD_MATRIX,
        camera_name_map=camera_name_map,
        new_id_camera_ids=new_id_cameras,
    )

    # ── Tracking log ──────────────────────────────────────────────────────────
    tracking_log = TrackingLog()

    # ── Camera workers (no display queue — headless processing) ────────────────
    workers: List[CameraWorker] = []
    for idx, vp in enumerate(video_paths):
        w = CameraWorker(
            video_path    = vp,
            camera_id     = idx + 1,
            cam_idx       = idx,
            dispatcher    = dispatcher,
            reid_model    = reid_model,
            gallery       = gallery,
            display_queue = None,    # No live display — maximum speed
            reid_interval = reid_interval,
            tracking_log  = tracking_log,
        )
        workers.append(w)

    print(f"\n[Main] Starting {n_cameras} worker thread(s) (headless) …\n")
    t_start = time.time()
    for w in workers:
        w.start()

    # ── Wait for completion ───────────────────────────────────────────────────
    interrupted = False
    try:
        while True:
            all_started = all(w.has_started for w in workers)
            all_done    = all_started and all(w.is_done for w in workers)
            if all_done or any(w.error for w in workers):
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        interrupted = True
        print("\n[Main] Keyboard interrupt — stopping workers …")
        for w in workers:
            w.stop()

    # ── Cleanup ───────────────────────────────────────────────────────────────
    dispatcher.stop()
    for w in workers:
        w.join(timeout=10)

    if interrupted:
        print("[Main] Processing interrupted; tracking log was not saved")
        return

    failures = [(w.camera_id, w.error) for w in workers if w.error is not None]
    if failures:
        cameras = ", ".join(str(camera_id) for camera_id, _ in failures)
        raise RuntimeError(f"Processing failed on camera(s) {cameras}; tracking log was not saved") from failures[0][1]

    elapsed = time.time() - t_start

    # ── Save tracking log ─────────────────────────────────────────────────────
    log_path = OUTPUT_DIR / "tracking_log.json"
    tracking_log.save(str(log_path), gallery)

    # ── Generate HTML Report ──────────────────────────────────────────────────
    gallery.generate_html_report("output")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Processing complete                                     ║")
    print("╠══════════════════════════════════════════════════════════╣")
    for w in workers:
        status = "✓" if w.is_done else "✗"
        name   = w.output_path.name
        print(f"║  {status} Cam{w.camera_id}  {name:<30} {w.fps_stat:5.1f} fps  ║")
    print(f"║                                                          ║")
    print(f"║  Total time: {elapsed:.1f}s                                       ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\nAnnotated videos saved to:  output/")
    print(f"Tracking log saved to:      {log_path}\n")

    # ── Phase 2: Open review UI ───────────────────────────────────────────────
    if show_preview:
        print("[Main] Opening review dashboard …\n")
        try:
            tracking_data = TrackingLog.load(str(log_path))
            from src.review_ui import ReviewUI
            ui = ReviewUI(tracking_data=tracking_data, output_dir=str(OUTPUT_DIR))
            ui.run()
        except (ImportError, ModuleNotFoundError) as e:
            print(f"[Main] Note: Tkinter desktop UI unavailable ({e}).")
            print(f"[Main] ✦ To open the Web Review Dashboard & OpenAI AI Assistant, run:\n")
            print("       python -m src.web_review\n")




if __name__ == "__main__":
    main()
