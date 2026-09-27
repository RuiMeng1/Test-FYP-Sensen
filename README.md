# Multi-Camera Person Tracking Pipeline

Real-time multi-camera person detection and re-identification using **YOLOv8** (bounding boxes + local tracking) and **PersonViT** (cross-camera global identity matching).

---

## Quick Start

### 1. Setup

> **⚠️ Requires Python 3.10.** Using Python 3.9 or earlier will cause errors. Check your version with `python3 --version`.

Create and activate a virtual environment using **Python 3.10**, then install dependencies:

```bash
# macOS / Linux
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Windows
py -3.10 -m venv .venv
.venv\Scripts\activate
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
# Default: process all videos in data/, then open the interactive review dashboard
python -m src.main
```

The pipeline runs in **two phases**:
1. **Processing** — All videos are processed headlessly at maximum speed. Annotated output videos and a `tracking_log.json` are saved to `output/`.
2. **Review** — An interactive tkinter dashboard opens automatically for post-processing review.

---

## Advanced Usage & CLI Flags

```bash
# Manually select specific videos instead of auto-loading all in data/
python -m src.main --videos data/cam1.mp4 data/cam2.mp4

# Skip the review UI (headless processing only)
python -m src.main --no-preview

# Skip processing and open the review dashboard with existing output
python -m src.main --review-only

# Performance tweak: extract Re-ID features less often (e.g., every 10 frames)
python -m src.main --reid-interval 10

# Force a specific device (auto-detects by default)
python -m src.main --device cpu
python -m src.main --device mps    # Apple Silicon GPU
python -m src.main --device cuda   # NVIDIA GPU (when available)

# Override the web dashboard port (default: 5001)
python -m src.main --port 5001
```

---

## Review Dashboard

After processing completes, the **Review UI** opens automatically (or via `--review-only`). It loads the annotated output videos and `tracking_log.json` and provides:

- **Video playback** — play/pause, frame stepping, and a seek slider.
- **Person logger** — expandable per-person appearance timelines.
- **Auto-tracking mode** — click any Person ID to full-screen their current camera and automatically follow them across cameras as they move.
- **Auto-detection wait** — if a person is not currently live, the UI waits for them to appear and then auto full-screens.
- **Manual camera switching** — Grid View, Cam 1, Cam 2, Cam 3 buttons + double-click to full-screen.

---

## Output

Annotated videos and reports are saved to `output/`:

```
output/
  cam1_20260507_160822_tracked.mp4
  cam2_20260507_160823_tracked.mp4
  id_summary_report.html
  tracking_log.json
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
               ┌─────────┴─────────────────────┐
         [TrackingLog]                   [output/*.mp4]
         per-person timelines            annotated video files
               │
         [Review UI]
         interactive tkinter dashboard
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
├── output/                     # Annotated output videos, HTML reports & tracking log
├── scripts/
│   └── configure.py            # Local web server for configuration UI
├── src/
│   ├── config.py               # Hardcoded defaults & settings loader
│   ├── detector.py             # YOLOv8 + ByteTrack person detector
│   ├── dispatcher.py           # Syncs frames across multiple cameras
│   ├── gallery.py              # Cross-camera global ID gallery & exemplar banks
│   ├── main.py                 # Entry point: two-phase process → review workflow
│   ├── pipeline.py             # Per-camera worker thread logic
│   ├── reid_model.py           # PersonViT feature extractor
│   ├── review_ui.py            # Post-processing review dashboard (tkinter)
│   ├── tk_ui.py                # Native tkinter live dashboard (zero-overhead)
│   ├── tracking_log.py         # Thread-safe per-person appearance timeline recorder
│   ├── video_loader.py         # Video discovery & CLI argument parsing
│   ├── web_server.py           # Flask web UI for live tracking dashboard (MJPEG)
│   └── templates/
│       └── index.html          # Web dashboard HTML template
├── pyrightconfig.json          # IDE configuration for type-checking
├── requirements.txt            # Python dependencies
└── README.md                   # You are here
```
