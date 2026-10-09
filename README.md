# Multi-Camera Person Tracking Pipeline

Real-time multi-camera person detection and re-identification using **YOLOv8** (bounding boxes + local tracking) and **PersonViT-ReID** (cross-camera global identity matching with DINOv2 fallback).

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

> **Note:** 
> - **YOLOv8** weights (`yolov8n.pt`, ~6 MB) auto-download on the first run.
> - **PersonViT-ReID** weights are pre-bundled in `models/personvit_reid/` (falls back to DINOv2 ViT-Small via `timm` if missing).
> - *(Optional)* For the AI Assistant features, copy `.env.example` to `.env` and set your `OPENAI_API_KEY`.

### 2. Configure Settings (UI)

Instead of editing code, you can use the built-in configuration UI to adjust similarity thresholds, especially for cameras with extreme angle differences (such as portrait vs landscape).

```bash
python scripts/configure.py
```
This opens your browser at `http://localhost:7700`.
- Adjust **Per-Camera-Pair Thresholds** (lower thresholds like `0.55` are recommended for cameras with very different viewpoints to prevent false negatives).
- Click **Save Settings** (saved to `config/pipeline_settings.json`).
- Stop the server with `Ctrl+C` in your terminal.

### 3. Run Pipeline

```bash
# Default: process all videos in data/, then open the interactive review dashboard
python -m src.main
```

The pipeline runs in **two phases**:
1. **Processing** — All camera feeds are processed headlessly across worker threads at maximum speed. Annotated output videos and `tracking_log.json` are saved to `output/`.
2. **Review** — An interactive review dashboard opens automatically for post-processing review.

---

## Review Dashboards

Two review interfaces are available:

### 1. Web Review Dashboard & AI Assistant (Recommended)
```bash
python scripts/run_web_ui.py
```
Opens in your browser at `http://localhost:5001`:
- **Synchronized Video Playback** with seek slider and camera switching.
- **Natural Language AI Queries** (e.g. *"Who entered the store after 16:08?"*, *"Find the person wearing a red jacket"*).
- **Automated Clothing / Attribute Descriptions** powered by OpenAI Vision.
- **Action Event Timeline** showing detected movements and spatiotemporal intervals.

### 2. Desktop Review UI (Tkinter)
```bash
python -m src.main --review-only
```
- Expandable per-person timelines and crop thumbnails.
- Interactive **ID Merging** to correct split tracks post-run.
- Auto-tracking mode that follows a selected person across camera feeds.

---

## Advanced Usage & CLI Flags

```bash
# Manually select specific videos instead of auto-loading all in data/
python -m src.main --videos data/cam1.mp4 data/cam2.mp4

# Skip the review UI (headless processing only)
python -m src.main --no-preview

# Skip processing and launch the desktop review UI directly
python -m src.main --review-only

# Performance tweak: extract Re-ID features less often (e.g., every 10 frames)
python -m src.main --reid-interval 10

# Force a specific compute device (auto-detects by default)
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
  tracking_log.json
```

Each video shows:
- **Bounding boxes** around every detected person with confidence score.
- **Global ID** (e.g. `ID:003`) — consistent across all cameras.
- **HUD overlay** — camera ID, frame progress, FPS, person count.

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
                  [Occlusion Filter]    suppresses updates for partially occluded crops
                         │
                  [Person crops]
                         │
                  [PersonViT (MSMT17)]  feature extraction (384-dim normalized)
                         │
                  [GlobalGallery]       cosine similarity cross-camera matching
                                        (supports per-pair thresholds & exemplar banks)
                         │
               ┌─────────┴─────────────────────┐
         [TrackingLog]                   [output/*.mp4]
         per-person timelines            annotated video files
               │
      ┌────────┴────────────────────────┐
      ▼                                 ▼
[Web Review UI + AI Assistant]    [Tkinter Desktop UI]
(scripts/run_web_ui.py)           (python -m src.main --review-only)
```

---

## Directory Structure

```text
Test-FYP-Sensen/
├── .env.example                # Example environment variables (OPENAI_API_KEY)
├── .venv/                      # Python virtual environment
├── config/
│   └── pipeline_settings.json  # Saved from scripts/configure.py
├── data/                       # Input video files (*.mp4)
├── docs/
│   └── partial-occlusion.md    # Documentation for occlusion handling logic
├── models/
│   └── personvit_reid/         # Bundled PersonViT-ReID model weights & config
├── output/                     # Annotated output videos, HTML reports & tracking logs
├── scripts/
│   ├── configure.py            # Local web server for threshold configuration UI
│   ├── download_weights.py     # TransReID weights downloader utility
│   ├── draw_zones.py           # Interactive zone drawing utility
│   ├── generate_test_videos.py # Synthetic video generation tool
│   ├── playback.py             # Multi-video synced playback tool
│   └── run_web_ui.py           # Web review dashboard & OpenAI assistant server
├── src/
│   ├── action_analysis.py      # Offline action & movement interval extraction
│   ├── action_events.py        # Spatiotemporal event data structures
│   ├── action_labels.py        # Action label definitions and formatters
│   ├── ai_assistant.py         # OpenAI vision & LLM tracking assistant
│   ├── config.py               # Settings, defaults & runtime loader
│   ├── detector.py             # YOLOv8 + ByteTrack detector
│   ├── dispatcher.py           # Syncs detections across cameras
│   ├── gallery.py              # Cross-camera global ID gallery & exemplar banks
│   ├── main.py                 # Pipeline entry point
│   ├── occlusion.py            # Partial occlusion detection & size history tracking
│   ├── pipeline.py             # Per-camera worker thread logic
│   ├── reid_model.py           # PersonViT feature extractor with DINOv2 fallback
│   ├── review_ui.py            # Post-processing desktop review UI (Tkinter)
│   ├── tk_ui.py                # Native Tkinter live preview helper
│   ├── tracking_log.py         # Thread-safe timeline recorder
│   ├── video_loader.py         # Video discovery & CLI argument parser
│   ├── web_review.py           # Web review API backend
│   └── templates/              # HTML templates for web UI
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```
