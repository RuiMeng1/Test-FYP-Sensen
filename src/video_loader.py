"""
video_loader.py — Discover video files and parse CLI arguments.

Usage examples:
  # Use all .mp4 files in /data (default)
  python src/main.py

  # Manually select specific videos
  python src/main.py --videos data/cam1_20260507_160822.mp4 data/cam2_20260507_160823.mp4

  # Disable live preview
  python src/main.py --no-preview

  # Process faster (less accurate Re-ID)
  python src/main.py --reid-interval 10
"""

import argparse
import sys
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-Camera Person Tracking  (YOLOv8 + TransReID)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--videos", "-v",
        nargs="+",
        type=str,
        metavar="PATH",
        help=(
            "Manually specify one or more video file paths. "
            "If omitted, all *.mp4 files in --data-dir are used."
        ),
    )
    parser.add_argument(
        "--data-dir", "-d",
        type=str,
        default=None,
        metavar="DIR",
        help="Override the data directory (default: data/ next to project root).",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable the live preview window (output videos are still saved).",
    )
    parser.add_argument(
        "--reid-interval",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Extract Re-ID features every N frames. "
            "Higher = faster but less accurate cross-camera matching. "
            "Default: value in config.py (5)."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Override compute device (default: auto-detected).",
    )
    return parser.parse_args()


def get_video_paths(args: argparse.Namespace) -> List[Path]:
    """
    Return an ordered list of video paths to process.
    Validates that all files exist before returning.
    """
    # ── Manual selection ────────────────────────────────────────────────────
    if args.videos:
        paths = [Path(v) for v in args.videos]
        missing = [p for p in paths if not p.exists()]
        if missing:
            print("[VideoLoader] ERROR: The following files were not found:")
            for m in missing:
                print(f"  {m}")
            sys.exit(1)
        print(f"[VideoLoader] Using {len(paths)} manually selected video(s).")
        return paths

    # ── Auto-discovery ───────────────────────────────────────────────────────
    from src.config import DATA_DIR
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR

    if not data_dir.exists():
        print(f"[VideoLoader] ERROR: data directory not found: {data_dir}")
        sys.exit(1)

    # Only top-level .mp4 files (not recursive) — use camfootage/ manually if needed
    paths = sorted(data_dir.glob("*.mp4"))

    if not paths:
        print(f"[VideoLoader] ERROR: No *.mp4 files found in {data_dir}")
        print("[VideoLoader] Tip: use --videos to specify files directly.")
        sys.exit(1)

    print(f"[VideoLoader] Auto-discovered {len(paths)} video(s) in {data_dir}:")
    for p in paths:
        size_mb = p.stat().st_size / (1024 ** 2)
        print(f"  • {p.name}  ({size_mb:.1f} MB)")

    return paths
