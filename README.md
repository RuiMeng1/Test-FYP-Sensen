# Multi-Camera Person Tracking Pipeline

Real-time multi-camera person detection and re-identification using **YOLOv8** (bounding boxes + local tracking) and **PersonViT** (cross-camera global identity matching).

---

## Quick Start

### 1. Setup

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note:** First run will auto-download YOLOv8 weights (~6 MB) and the PersonViT MSMT17 model.

### 2. Configure Settings (UI)

Instead of editing code, you can use the built-in web UI to configure thresholds, especially for cameras with extreme angle differences (like portrait vs landscape).

```bash
python scripts/configure.py
```
This will open your browser at `http://localhost:7700`.
- Adjust **Per-Camera-Pair Thresholds** (lower thresholds like 0.55 are recommended for cameras with very different angles to avoid false negatives).
- Click **Save Settings**. (Settings are saved to `config/pipeline_settings.json`).
- Stop the server with `Ctrl+C` in your terminal.

### 3. Run Pipeline

```bash
# Default: process all videos in data/, show live preview, and save to output/
python -m src.main
```

**Live preview controls:**
- `q` or `ESC` — stop processing and exit
- The window shows all cameras tiled side-by-side.

---

## Advanced Usage & CLI Flags

```bash
# Manually select specific videos instead of auto-loading all in data/
python -m src.main --videos data/cam1.mp4 data/cam2.mp4

# Headless (no preview window — useful for SSH / remote servers)
python -m src.main --no-preview

# Performance tweak: extract Re-ID features less often (e.g., every 10 frames)
python -m src.main --reid-interval 10

# Force a specific device (auto-detects by default)
python -m src.main --device cpu
python -m src.main --device mps    # Apple Silicon GPU
python -m src.main --device cuda   # NVIDIA GPU (when available)
```

---

## Output

Annotated videos and reports are saved to `output/`:

```
output/
  cam1_20260507_160822_tracked.mp4
  cam2_20260507_160823_tracked.mp4
  id_summary_report.html
```

Each video shows:
- **Bounding boxes** around every detected person.
- **Global ID** (e.g. `ID:003`) — stable and consistent across all cameras.
- **HUD overlay** — camera number, frame progress, FPS, person count.

---

## Architecture Overview

```text
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
                  [PersonViT (MSMT17)]  feature extraction (384-dim)
                         │
                  [GlobalGallery]       cosine similarity cross-camera matching
                                        (supports per-pair thresholds & exemplar banks)
                         │
               ┌─────────┴─────────┐
         [Live preview]      [output/*.mp4]
         tiled OpenCV window  annotated video files
```

---

## Directory Structure

```text
Test-FYP-Sensen/
├── .venv/                      # Python virtual environment
├── config/
│   └── pipeline_settings.json  # Saved from scripts/configure.py
├── data/                       # Input video files (*.mp4)
├── models/                     # Downloaded model weights (PersonViT)
├── output/                     # Annotated output videos & HTML reports
├── scripts/
│   └── configure.py            # Local web server for configuration UI
├── src/
│   ├── config.py               # Hardcoded defaults & settings loader
│   ├── detector.py             # YOLOv8 + ByteTrack person detector
│   ├── dispatcher.py           # Syncs frames across multiple cameras
│   ├── gallery.py              # Cross-camera global ID gallery & exemplar banks
│   ├── pipeline.py             # Per-camera worker thread logic
│   ├── reid_model.py           # PersonViT feature extractor
│   └── main.py                 # Entry point + live preview loop
├── pyrightconfig.json          # IDE configuration for type-checking
├── requirements.txt            # Python dependencies
└── README.md                   # You are here
```
