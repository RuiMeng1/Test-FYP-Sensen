"""
download_weights.py — Download TransReID pretrained weights for person Re-ID.

The official TransReID model was pretrained on MSMT17 (the hardest/most
generalised ReID benchmark). These weights give the best cross-camera
person matching accuracy.

Usage:
    python scripts/download_weights.py

The file (~350 MB) is saved to:
    models/transreid_vitbase_msmt17.pth
"""

import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"

# ── Known direct download sources ────────────────────────────────────────────
# The official weights are hosted on Google Drive by the TransReID authors.
# We use gdown to handle the Drive download flow.
#
# Source: https://github.com/heshuting555/TransReID (README → Model Zoo)
# MSMT17 ViT-Base/16 (SIE + JPM, stride 16)
GDRIVE_FILE_ID  = "1UTfsBm-WA4us7G27ypgtXTsLZCVpg5Dh"
OUTPUT_FILENAME = "transreid_vitbase_msmt17.pth"


def check_gdown() -> bool:
    try:
        import gdown   # noqa: F401
        return True
    except ImportError:
        return False


def download_with_gdown(file_id: str, dest: Path) -> bool:
    import gdown
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"[Download] Downloading TransReID weights from Google Drive …")
    print(f"[Download] → {dest}")
    try:
        gdown.download(url, str(dest), quiet=False)
        return dest.exists() and dest.stat().st_size > 1_000_000
    except Exception as e:
        print(f"[Download] gdown failed: {e}")
        return False


def print_manual_instructions(dest: Path) -> None:
    print()
    print("━" * 65)
    print("  Manual download instructions")
    print("━" * 65)
    print()
    print("  1. Visit the TransReID Model Zoo:")
    print("     https://github.com/heshuting555/TransReID#model-zoo")
    print()
    print("  2. Download the MSMT17 ViT-Base/16 checkpoint")
    print("     (labelled 'ViT-B/16 + SIE + JPM, stride 16')")
    print()
    print(f"  3. Save the file to:")
    print(f"     {dest}")
    print()
    print("  Then re-run:  python src/main.py")
    print()
    print("  Without weights the pipeline still works using ImageNet-")
    print("  pretrained ViT — good enough for testing the system.")
    print("━" * 65)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / OUTPUT_FILENAME

    if dest.exists():
        size_mb = dest.stat().st_size / (1024 ** 2)
        print(f"[Download] Weights already present: {dest}  ({size_mb:.1f} MB)")
        print("[Download] Delete the file and re-run to force re-download.")
        return

    if not check_gdown():
        print("[Download] 'gdown' is not installed.")
        print("[Download] Install it with:  pip install gdown")
        print_manual_instructions(dest)
        sys.exit(1)

    success = download_with_gdown(GDRIVE_FILE_ID, dest)

    if success:
        size_mb = dest.stat().st_size / (1024 ** 2)
        print(f"\n[Download] ✓ Success! Saved to {dest}  ({size_mb:.1f} MB)")
        print("[Download] You can now run:  python src/main.py")
    else:
        print("\n[Download] ✗ Automatic download failed.")
        print_manual_instructions(dest)
        sys.exit(1)


if __name__ == "__main__":
    main()
