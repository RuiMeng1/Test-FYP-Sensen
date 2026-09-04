"""
config.py — Central configuration for the tracking pipeline.
Adjust thresholds, paths, and device settings here.
"""

from pathlib import Path
import json
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

# Per-camera-pair similarity thresholds (overrides REID_THRESHOLD for specific pairs).
# Keys are frozensets of the two video stem names (order-independent).
# Cameras with very different angles / orientations (e.g. portrait cam3 vs
# landscape cam1/cam2) require a lower threshold to avoid false negatives
# caused by natural embedding drift across extreme viewpoints.
#
# NOTE: These defaults are overridden at runtime by config/pipeline_settings.json
#       if the file exists (edit via: python scripts/configure.py).
REID_THRESHOLD_MATRIX: dict = {
    frozenset({"cam1_20260507_160822", "cam2_20260507_160823"}): 0.65,  # similar angles
    frozenset({"cam1_20260507_160822", "cam3_20260507_160825"}): 0.55,  # portrait vs landscape
    frozenset({"cam2_20260507_160823", "cam3_20260507_160825"}): 0.55,  # portrait vs landscape
}

# Gallery exemplar bank settings.
REID_MAX_EXEMPLARS  = 8      # Max stored appearance vectors per person (front, back, side, etc.)
                              # More exemplars = more viewpoint coverage, slightly more memory.
REID_EMA_ALPHA      = 0.15   # EMA blend factor when refining a near-duplicate exemplar.
                              # Higher = faster adaptation; lower = more stable.

REID_EVERY_N_FRAMES = 15     # Extract Re-ID features every N frames (CPU speed trade-off)
                              # Lower = more accurate, slower. Increase to 10+ on slow machines.

# Frames before a gallery entry expires (person not seen for this long → ID released).
# At 10 fps this is 50 seconds — enough for a person to leave and re-enter the scene.
GALLERY_TTL_FRAMES  = 50000


# ── Runtime settings loader ───────────────────────────────────────────────────
# Overrides the defaults above with values saved by the configuration UI.
# Run `python scripts/configure.py` to open the UI and save your settings.

def _load_runtime_settings() -> None:
    """Load config/pipeline_settings.json and override module-level variables."""
    settings_path = BASE_DIR / "config" / "pipeline_settings.json"
    if not settings_path.exists():
        return  # No UI config yet — use built-in defaults

    global REID_THRESHOLD, REID_THRESHOLD_MATRIX, REID_MAX_EXEMPLARS
    global REID_EMA_ALPHA, REID_EVERY_N_FRAMES, GALLERY_TTL_FRAMES

    try:
        with open(settings_path, "r") as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"[Config] Warning: could not load {settings_path}: {e}")
        return

    if "global_threshold" in cfg:
        REID_THRESHOLD = float(cfg["global_threshold"])

    if "reid_every_n_frames" in cfg:
        REID_EVERY_N_FRAMES = int(cfg["reid_every_n_frames"])

    if "gallery_ttl_frames" in cfg:
        GALLERY_TTL_FRAMES = int(cfg["gallery_ttl_frames"])

    if "max_exemplars" in cfg:
        REID_MAX_EXEMPLARS = int(cfg["max_exemplars"])

    if "ema_alpha" in cfg:
        REID_EMA_ALPHA = float(cfg["ema_alpha"])

    # pair_thresholds stored as "stemA|stemB" -> float (stems sorted alphabetically)
    if "pair_thresholds" in cfg:
        REID_THRESHOLD_MATRIX = {
            frozenset(key.split("|", 1)): float(val)
            for key, val in cfg["pair_thresholds"].items()
        }
        if REID_THRESHOLD_MATRIX:
            print(f"[Config] Loaded {len(REID_THRESHOLD_MATRIX)} per-pair threshold(s) "
                  f"from {settings_path.name}")


_load_runtime_settings()

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
