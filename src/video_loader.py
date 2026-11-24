"""
video_loader.py — Discover video files and parse CLI arguments.

Usage examples:
  # Use all .mp4 files in /data (default)
  python -m src.main --all-id-cameras

  # Manually select specific videos
  python -m src.main --videos data/cam1.mp4 data/cam2.mp4 --new-id-cameras cam1

  # Disable live preview
  python -m src.main --all-id-cameras --no-preview

  # Process faster (less accurate Re-ID)
  python -m src.main --all-id-cameras --reid-interval 10
"""

import argparse
import json
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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--review-only",
        action="store_true",
        help="Skip video processing and open the review dashboard with existing output.",
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
    parser.add_argument(
        "--detector-workers",
        type=int,
        default=None,
        metavar="N",
        help="Maximum concurrent YOLO detections (default: 2 on CPU, 1 on GPU).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5001,
        metavar="PORT",
        help="Port for the web dashboard (default: 5000).",
    )
    mode.add_argument('--new-id-cameras', nargs='+', metavar='CAMERA',
                      help='Allow new IDs only on these camera filenames, file stems, or run numbers.')
    mode.add_argument('--all-id-cameras', action='store_true',
                      help='Allow every camera to create new IDs.')
    mode.add_argument('--use-saved-id-cameras', action='store_true',
                      help='Use the camera selection saved from scripts/configure.py.')
    mode.add_argument('--list-cameras', action='store_true',
                      help='Show the numbered input filenames and exit.')
    args = parser.parse_args()
    if args.detector_workers is not None and args.detector_workers < 1:
        parser.error("--detector-workers must be at least 1")
    return args


def resolve_new_id_cameras(args, video_paths: List[Path], settings_path: Path) -> set[int] | None:
    """Resolve the explicit run policy. None allows every camera to create IDs."""
    stems = {path.stem: number for number, path in enumerate(video_paths, 1)}
    if len(stems) != len(video_paths):
        raise ValueError('Each input video must have a unique filename stem.')
    if args.all_id_cameras:
        return None
    if args.use_saved_id_cameras:
        try:
            selection = json.loads(settings_path.read_text(encoding='utf-8'))['new_id_cameras']
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError('Save a New Person IDs selection in python scripts/configure.py first.') from exc
        if selection is None:
            return None
        if not isinstance(selection, list) or not selection or not all(isinstance(s, str) for s in selection):
            raise ValueError('The saved selection is invalid. Choose cameras and save again in the config UI.')
        missing = [stem for stem in selection if stem not in stems]
        if missing:
            raise ValueError(f'Saved cameras are missing from this run: {", ".join(missing)}. '
                             'Use --new-id-cameras to select cameras for these inputs.')
        return {stems[stem] for stem in selection}

    if not args.new_id_cameras:
        raise ValueError('Choose --new-id-cameras CAMERA ..., --all-id-cameras, or --use-saved-id-cameras. '
                         'For camera previews, run python scripts/configure.py.')
    names = {path.name: number for number, path in enumerate(video_paths, 1)}
    numbers = {str(number): number for number in range(1, len(video_paths) + 1)}
    result = set()
    for camera in args.new_id_cameras:
        number = numbers.get(camera) or names.get(camera) or stems.get(camera)
        if number is None:
            raise ValueError(f'Unknown camera {camera!r}. Use a filename, file stem, or run number shown above.')
        result.add(number)
    return result


def print_camera_list(video_paths: List[Path]) -> None:
    rows = [f'  {number:>2}  {path.name}' for number, path in enumerate(video_paths, 1)]
    print('\n'.join(['Input cameras in run order:', *rows]), flush=True)


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
    paths = sorted(path for path in data_dir.glob("*.mp4") if path.is_file())

    if not paths:
        print(f"[VideoLoader] ERROR: No *.mp4 files found in {data_dir}")
        print("[VideoLoader] Tip: use --videos to specify files directly.")
        sys.exit(1)

    print(f"[VideoLoader] Auto-discovered {len(paths)} video(s) in {data_dir}:")
    for p in paths:
        size_mb = p.stat().st_size / (1024 ** 2)
        print(f"  • {p.name}  ({size_mb:.1f} MB)")

    return paths
