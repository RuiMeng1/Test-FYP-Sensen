# Multi-Camera Person Tracking Pipeline

Real-time multi-camera person detection and re-identification using **YOLOv8** (bounding boxes + local tracking) and **TransReID** (cross-camera global identity matching).

---

## Architecture

```
data/
  cam1_*.mp4 ──┐
  cam2_*.mp4 ──┤──► VideoLoader ──► CameraWorker threads (one per video)
  cam3_*.mp4 ──┘
                         │
                  [YOLOv8-nano]         person detection + ByteTrack
                  bounding boxes + local IDs
                         │
                  [Person crops]
                         │
                  [TransReID ViT-Base]  feature extraction (768-dim)
                         │
                  [GlobalGallery]       cosine similarity cross-camera matching
                         │
               ┌─────────┴─────────┐
         [Live preview]      [output/*.mp4]
         tiled OpenCV window  annotated video files
```

---

## Setup

### 1. Create and activate the virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> First run downloads YOLOv8-nano weights (~6 MB) automatically.

### 3. Download TransReID pretrained weights (optional but recommended)

```bash
python scripts/download_weights.py
```

The script downloads `transreid_vitbase_msmt17.pth` (~350 MB) to `models/`.
Without it the pipeline falls back to **ImageNet-pretrained ViT** — still functional for testing, but cross-camera matching is less accurate.

---

## Running

```bash
# Activate venv first
source .venv/bin/activate

# Default: process all 3 videos in data/, show live preview + save output
python -m src.main

# Manually select specific videos
python -m src.main --videos data/cam1_20260507_160822.mp4 data/cam2_20260507_160823.mp4

# Headless (no preview window — useful for SSH / servers)
python -m src.main --no-preview

# Faster on slow CPU (Re-ID every 10 frames instead of 5)
python -m src.main --reid-interval 10

# Force a specific device
python -m src.main --device cpu
python -m src.main --device mps    # Apple Silicon GPU
python -m src.main --device cuda   # NVIDIA GPU (when available)
```

**Live preview controls:**
- `q` or `ESC` — stop processing and exit
- The window shows all cameras tiled side-by-side

---

## Output

Annotated videos are saved to `output/`:

```
output/
  cam1_20260507_160822_tracked.mp4
  cam2_20260507_160823_tracked.mp4
  cam3_20260507_160825_tracked.mp4
```

Each video shows:
- **Bounding boxes** around every detected person
- **Global ID** (e.g. `ID:003`) — stable and consistent across all cameras
- **HUD overlay** — camera number, frame progress, FPS, person count

---

## Performance Guide

| Hardware | Expected FPS | Notes |
|---|---|---|
| Mac CPU (Intel) | ~1–3 fps | Use `--reid-interval 10` or higher |
| Mac CPU (Apple M-series) | ~4–8 fps | Try `--device mps` for ~2× speedup |
| NVIDIA GPU | 25–60+ fps | Use `--device cuda` |

> **Tip for CPU:** Use `--reid-interval 15` to skip Re-ID most frames. Detection still runs every frame so bounding boxes are always accurate; only cross-camera ID matching is reduced.

---

## Configuration

Edit [`src/config.py`](src/config.py) to change:

| Setting | Default | Description |
|---|---|---|
| `REID_THRESHOLD` | `0.65` | Cosine similarity for "same person". Raise if false matches occur. |
| `REID_EVERY_N_FRAMES` | `5` | Default Re-ID interval |
| `GALLERY_TTL_FRAMES` | `200` | Frames before an unseen person is removed from gallery |
| `TILE_W / TILE_H` | `640 / 360` | Preview tile size per camera |

---

## Enabling Your Home GPU Later

When you have access to your NVIDIA machine:

```bash
# Uninstall CPU torch, install CUDA version (e.g. CUDA 12.1)
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Run with CUDA
python -m src.main --device cuda
```

---

## Project Structure

```
Test-FYP-Sensen/
├── .venv/                      virtual environment
├── data/                       input video files
│   ├── cam1_20260507_160822.mp4
│   ├── cam2_20260507_160823.mp4
│   └── cam3_20260507_160825.mp4
├── models/                     downloaded model weights
│   └── transreid_vitbase_msmt17.pth
├── output/                     annotated output videos (created on first run)
├── scripts/
│   └── download_weights.py     TransReID weight downloader
├── src/
│   ├── __init__.py
│   ├── config.py               paths, thresholds, device selection
│   ├── video_loader.py         video discovery + CLI argument parser
│   ├── detector.py             YOLOv8 + ByteTrack person detector
│   ├── reid_model.py           TransReID ViT-Base feature extractor
│   ├── gallery.py              cross-camera global ID gallery
│   ├── pipeline.py             per-camera worker thread
│   └── main.py                 entry point + live preview loop
├── requirements.txt
└── README.md
```
