"""
playback.py — Replay annotated output videos in the same tiled live-preview layout.

Usage:
  # Auto-load all *_tracked.mp4 from output/
  python scripts/playback.py

  # Pick specific files
  python scripts/playback.py --videos output/cam1_tracked.mp4 output/cam2_tracked.mp4

  # Loop continuously
  python scripts/playback.py --loop

  # Start from a specific time (seconds)
  python scripts/playback.py --seek 30

Controls:
  q / ESC      — quit
  SPACE        — pause / resume
  ← / →        — seek ±5 s
  [ / ]        — seek ±30 s
  0-9          — jump to 0%-90% of video
  +/-          — speed up / slow down (0.25× – 4×)
"""

from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

# ── Layout — mirrors src/main.py exactly ─────────────────────────────────────
GRID_COLS = 2
TILE_W    = 640
TILE_H    = 360
DIVIDER   = (50, 50, 50)

# ── Speed steps ───────────────────────────────────────────────────────────────
SPEEDS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
DEFAULT_SPEED_IDX = 3   # 1.0×


# ── Tile helper (identical logic to main.py) ──────────────────────────────────

def _tile_frames(
    frames: Dict[int, np.ndarray],
    tile_w: int = TILE_W,
    tile_h: int = TILE_H,
) -> np.ndarray:
    sorted_ids = sorted(frames.keys())
    n    = len(sorted_ids)
    cols = GRID_COLS
    rows = (n + cols - 1) // cols

    grid = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

    for idx, cam_id in enumerate(sorted_ids):
        row, col = divmod(idx, cols)
        resized = cv2.resize(frames[cam_id], (tile_w, tile_h))
        y1, y2 = row * tile_h, (row + 1) * tile_h
        x1, x2 = col * tile_w, (col + 1) * tile_w
        grid[y1:y2, x1:x2] = resized

    # Grid dividers
    for c in range(1, cols):
        cv2.line(grid, (c * tile_w, 0), (c * tile_w, rows * tile_h), DIVIDER, 2)
    for r in range(1, rows):
        cv2.line(grid, (0, r * tile_h), (cols * tile_w, r * tile_h), DIVIDER, 2)

    # Empty tile label
    if n % cols != 0:
        last_row = n // cols
        last_col = n % cols
        ex1 = last_col * tile_w
        ey1 = last_row * tile_h
        ex2 = ex1 + tile_w
        ey2 = ey1 + tile_h
        cv2.rectangle(grid, (ex1 + 2, ey1 + 2), (ex2 - 2, ey2 - 2), (30, 30, 30), -1)
        cv2.putText(
            grid, "No feed", (ex1 + tile_w // 2 - 50, ey1 + tile_h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 2, cv2.LINE_AA,
        )

    return grid


# ── HUD overlay ──────────────────────────────────────────────────────────────

def _draw_hud(
    grid: np.ndarray,
    pos_sec: float,
    dur_sec: float,
    speed: float,
    paused: bool,
    n_cams: int,
) -> np.ndarray:
    """Burn a progress bar + info strip onto the bottom of the composite grid."""
    h, w = grid.shape[:2]
    BAR_H = 28
    out = grid.copy()

    # Dark strip behind HUD
    cv2.rectangle(out, (0, h - BAR_H), (w, h), (20, 20, 20), -1)

    # Progress bar fill
    pct = min(pos_sec / dur_sec, 1.0) if dur_sec > 0 else 0.0
    bar_w = int(w * pct)
    cv2.rectangle(out, (0, h - BAR_H), (bar_w, h - BAR_H + 4), (0, 200, 100), -1)

    # Time text
    def fmt(s: float) -> str:
        m = int(s) // 60
        return f"{m:02d}:{int(s) % 60:02d}"

    state_icon = "⏸" if paused else "▶"
    label = (
        f"  {state_icon}  {fmt(pos_sec)} / {fmt(dur_sec)}"
        f"    {speed:.2f}×"
        f"    {n_cams} cam{'s' if n_cams > 1 else ''}"
        f"    [q] quit  [SPACE] pause  [←→] ±5s  [+/-] speed"
    )
    cv2.putText(
        out, label, (6, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA,
    )

    return out


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Replay tracked output videos in tiled live-preview mode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--videos", nargs="+", metavar="PATH",
        help="Specific *_tracked.mp4 files to play. Default: all in output/",
    )
    p.add_argument(
        "--output-dir", default="output",
        help="Directory to search for *_tracked.mp4 (default: output/)",
    )
    p.add_argument(
        "--loop", action="store_true",
        help="Loop playback continuously until q/ESC.",
    )
    p.add_argument(
        "--seek", type=float, default=0.0, metavar="SECONDS",
        help="Start playback from this timestamp (seconds).",
    )
    p.add_argument(
        "--speed", type=float, default=1.0, metavar="MULTIPLIER",
        help="Initial playback speed (default: 1.0).",
    )
    return p.parse_args()


def get_video_paths(args: argparse.Namespace) -> List[Path]:
    if args.videos:
        paths = [Path(v) for v in args.videos]
        missing = [p for p in paths if not p.exists()]
        if missing:
            print(f"[Playback] ERROR: File(s) not found: {missing}", file=sys.stderr)
            sys.exit(1)
        return sorted(paths)

    out_dir = Path(args.output_dir)
    if not out_dir.exists():
        print(f"[Playback] ERROR: Output directory not found: {out_dir}", file=sys.stderr)
        sys.exit(1)

    paths = sorted(out_dir.glob("*_tracked.mp4"))
    if not paths:
        print(
            f"[Playback] No *_tracked.mp4 found in '{out_dir}/'. "
            "Run the pipeline first, or pass --videos explicitly.",
            file=sys.stderr,
        )
        sys.exit(1)
    return paths


# ── Main playback loop ────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    video_paths = get_video_paths(args)
    n_cams = len(video_paths)

    # Pick nearest valid speed
    speed_idx = min(
        range(len(SPEEDS)), key=lambda i: abs(SPEEDS[i] - args.speed)
    )

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       Multi-Camera Playback  (tracked output)            ║")
    print("╚══════════════════════════════════════════════════════════╝")
    for i, vp in enumerate(video_paths):
        print(f"  Cam {i+1}: {vp}")
    print(f"  Speed : {SPEEDS[speed_idx]}×")
    print(f"  Loop  : {'yes' if args.loop else 'no'}")
    print()

    # Window dimensions
    n_rows = (n_cams + GRID_COLS - 1) // GRID_COLS
    win_w  = TILE_W * GRID_COLS
    win_h  = TILE_H * n_rows + 28   # +28 for HUD bar

    WIN_NAME = "Multi-Camera Playback"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, win_w, win_h)

    while True:   # outer loop — supports --loop
        caps: List[cv2.VideoCapture] = []
        fpss: List[float]             = []
        frame_counts: List[int]       = []

        for vp in video_paths:
            cap = cv2.VideoCapture(str(vp))
            if not cap.isOpened():
                print(f"[Playback] ERROR: Cannot open {vp}", file=sys.stderr)
                sys.exit(1)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            fc  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            caps.append(cap)
            fpss.append(fps)
            frame_counts.append(fc)

        # Seek to --seek offset (use camera 0's FPS as reference)
        ref_fps = fpss[0]
        if args.seek > 0:
            seek_frame = int(args.seek * ref_fps)
            for cap, fps in zip(caps, fpss):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.seek * fps))

        max_frames = max(frame_counts)
        dur_sec    = max_frames / ref_fps

        paused    = False
        last_tick = time.perf_counter()

        latest_frames: Dict[int, Optional[np.ndarray]] = {}

        try:
            while True:
                speed = SPEEDS[speed_idx]

                if not paused:
                    now   = time.perf_counter()
                    delta = now - last_tick
                    last_tick = now

                    # How many frames to advance at current speed
                    frame_step = max(1, round(delta * ref_fps * speed))

                    all_done = True
                    for cam_idx, (cap, fps) in enumerate(zip(caps, fpss)):
                        cam_id = cam_idx + 1
                        ok = True
                        frame = None
                        for _ in range(frame_step):
                            ret, f = cap.read()
                            if ret:
                                frame = f
                            else:
                                ok = False
                                break
                        if frame is not None:
                            latest_frames[cam_id] = frame
                            all_done = False
                        elif cam_id not in latest_frames:
                            # Haven't got even one frame yet
                            all_done = False

                    if all_done:
                        break  # inner — restart if --loop, else exit
                else:
                    last_tick = time.perf_counter()   # don't accumulate while paused

                # Render
                if latest_frames:
                    # Filter out None values
                    display = {k: v for k, v in latest_frames.items() if v is not None}
                    if display:
                        # Position from cam 0
                        pos_frame = caps[0].get(cv2.CAP_PROP_POS_FRAMES)
                        pos_sec   = pos_frame / ref_fps

                        grid = _tile_frames(display, TILE_W, TILE_H)
                        grid = _draw_hud(grid, pos_sec, dur_sec, speed, paused, n_cams)
                        cv2.imshow(WIN_NAME, grid)

                # Throttle: target ~60 Hz UI updates, let frame_step handle speed
                wait_ms = max(1, int(1000 / (ref_fps * speed * 2)))
                key = cv2.waitKey(wait_ms) & 0xFF

                if key in (ord("q"), 27):           # q / ESC — quit
                    print("\n[Playback] Quit.")
                    for cap in caps:
                        cap.release()
                    cv2.destroyAllWindows()
                    return

                elif key == ord(" "):               # SPACE — pause / resume
                    paused = not paused
                    last_tick = time.perf_counter()
                    print(f"[Playback] {'Paused' if paused else 'Resumed'}")

                elif key == 81 or key == ord("["):  # ← or [ — seek -5s / -30s
                    delta_s = -30.0 if key == ord("[") else -5.0
                    for cap, fps in zip(caps, fpss):
                        new_pos = max(0, cap.get(cv2.CAP_PROP_POS_FRAMES) + delta_s * fps)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                    last_tick = time.perf_counter()

                elif key == 83 or key == ord("]"):  # → or ] — seek +5s / +30s
                    delta_s = 30.0 if key == ord("]") else 5.0
                    for cap, fps in zip(caps, fpss):
                        new_pos = min(
                            cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1,
                            cap.get(cv2.CAP_PROP_POS_FRAMES) + delta_s * fps,
                        )
                        cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                    last_tick = time.perf_counter()

                elif key in range(ord("0"), ord("9") + 1):   # 0-9 — jump to %
                    pct = (key - ord("0")) / 10.0
                    for cap, fc in zip(caps, frame_counts):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fc * pct))
                    last_tick = time.perf_counter()

                elif key in (ord("+"), ord("=")):   # + — speed up
                    speed_idx = min(speed_idx + 1, len(SPEEDS) - 1)
                    print(f"[Playback] Speed: {SPEEDS[speed_idx]}×")

                elif key == ord("-"):               # - — slow down
                    speed_idx = max(speed_idx - 1, 0)
                    print(f"[Playback] Speed: {SPEEDS[speed_idx]}×")

        finally:
            for cap in caps:
                cap.release()

        if not args.loop:
            break

        print("[Playback] Loop — restarting …")

    cv2.destroyAllWindows()
    print("[Playback] Done.")


if __name__ == "__main__":
    main()
