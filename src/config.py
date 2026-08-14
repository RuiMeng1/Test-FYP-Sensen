"""
config.py — Central configuration for the tracking pipeline.
Adjust thresholds, paths, and device settings here.
"""

from pathlib import Path
import torch

# ── Exclusion Zones ──────────────────────────────────────────────────────────
# Dictionary mapping video filenames to a list of polygons.
# Each polygon is a list of (x,y) coordinates.
# Detections whose bottom-center falls inside these polygons will be ignored.
EXCLUSION_ZONES = {
    # "cam1_20260507_160822.mp4": [
    #     [(100, 100), (200, 100), (200, 200), (100, 200)]
    # ],
    "cam1_20260507_160822.mp4": [
        [(273, 48), (388, 1065), (27, 1059), (11, 40)]
    ],
    "cam2_20260507_160823.mp4": [
        [(94, 132), (191, 407), (380, 1060), (6, 716), (9, 209)]
    ],
}

# ── Directory Paths ─────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
DATA_DIR    = BASE_DIR / "data"
OUTPUT_DIR  = BASE_DIR / "output"
MODELS_DIR  = BASE_DIR / "models"

# ── Model Paths ──────────────────────────────────────────────────────────────
YOLO_MODEL_NAME     = "yolov8n.pt"          # auto-downloaded by ultralytics
TRANSREID_WEIGHTS   = MODELS_DIR / "transreid_vitbase_msmt17.pth"

# ── Re-ID Settings ───────────────────────────────────────────────────────────
# Cosine similarity threshold for cross-camera "same person" match.
# DINOv2 (current fallback): 0.65  — DINOv2 has higher baseline similarity
#   between any two person crops than a ReID-trained model, so 0.65 is tight
#   enough to avoid false cross-camera merges.
# TransReID pretrained weights: 0.75  — features are more discriminative;
#   you can raise the threshold once weights are downloaded.
# Lower = more likely to link same person; Higher = fewer false links.
REID_THRESHOLD      = 0.65

REID_EVERY_N_FRAMES = 5      # Extract Re-ID features every N frames (CPU speed trade-off)
                              # Lower = more accurate, slower. Increase to 10+ on slow machines.

# Frames before a gallery entry expires (person not seen for this long → ID released).
# At 10 fps this is 50 seconds — enough for a person to leave and re-enter the scene.
GALLERY_TTL_FRAMES  = 50000

# ── Display ──────────────────────────────────────────────────────────────────
TILE_W = 640   # Width of each camera tile in the preview grid
TILE_H = 360   # Height of each camera tile in the preview grid

# ── Device Selection ─────────────────────────────────────────────────────────
def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    # Apple Silicon GPU (M1/M2/M3)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# Device used for YOLOv8 detection (supports MPS/CUDA/CPU)
DEVICE = _select_device()

# Device used for TransReID Re-ID model.
# IMPORTANT: PyTorch MPS does NOT support multi-threaded use — it can only run
# on the main thread. Since Re-ID runs inside camera worker threads, we must
# use CPU (or CUDA if available). This is a known macOS/MPS limitation.
# See: https://github.com/pytorch/pytorch/issues/77764
if DEVICE == "mps":
    REID_DEVICE = "cpu"
else:
    REID_DEVICE = DEVICE
