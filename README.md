# Multi-Camera Person Tracking & Video Intelligence Pipeline

An end-to-end multi-camera person detection, tracking, re-identification (Re-ID), and video intelligence system built with **YOLOv8**, **ByteTrack**, **PersonViT-ReID**, and **OpenAI GPT-4o-mini Vision**.

The pipeline detects and tracks individuals across multiple synchronized camera streams, resolves cross-camera global identities despite extreme viewpoint and lighting shifts, and provides a desktop review dashboard featuring interactive ID merging, multi-angle people galleries, PDF reporting, and an AI intelligence assistant with clickable timestamp citations.

---

## Table of Contents

- [System Architecture](#system-architecture)
  - [Architecture Flow Diagram](#architecture-flow-diagram)
  - [Core Architectural Components](#core-architectural-components)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
  - [1. Prerequisites & Environment](#1-prerequisites--environment)
  - [2. Environment Variables (.env)](#2-environment-variables-env)
  - [3. Pretrained Weights](#3-pretrained-weights)
- [Pipeline Execution](#pipeline-execution)
  - [Phase 1: Headless Batch Processing](#phase-1-headless-batch-processing)
  - [Phase 2: Interactive Review Dashboard](#phase-2-interactive-review-dashboard)
  - [CLI Flags & Options](#cli-flags--options)
- [Configuration & Helper Tools](#configuration--helper-tools)
  - [Web Configuration UI (`scripts/configure.py`)](#web-configuration-ui-scriptsconfigurepy)
  - [Exclusion Zone Polygon Tool (`scripts/draw_zones.py`)](#exclusion-zone-polygon-tool-scriptsdraw_zonespy)
  - [Synchronized Playback Viewer (`scripts/playback.py`)](#synchronized-playback-viewer-scriptsplaybackpy)
  - [Weight Downloader (`scripts/download_weights.py`)](#weight-downloader-scriptsdownload_weightspy)
- [Review Dashboard & AI Assistant](#review-dashboard--ai-assistant)
  - [Multi-Camera Playback & Auto-Tracking](#multi-camera-playback--auto-tracking)
  - [People Gallery & Live ID Merging](#people-gallery--live-id-merging)
  - [AI Video Intelligence (GPT-4o-mini Vision)](#ai-video-intelligence-gpt-4o-mini-vision)
  - [PDF Summary Report Export](#pdf-summary-report-export)
- [Output Artifacts](#output-artifacts)
- [Directory Structure](#directory-structure)

---

## System Architecture

### Architecture Flow Diagram

```text
  data/cam1.mp4 ──┐
  data/cam2.mp4 ──┼──► [VideoLoader] ──► Discovers and validates video streams
  data/cam3.mp4 ──┘
                             │
                             ▼
                  [DetectionDispatcher]  (Strict round-robin scheduler)
                  ├── Dedicated PersonDetector per camera (YOLOv8 + ByteTrack)
                  └── Serialized inference (prevents Apple MPS Metal race conditions)
                             │
                             ▼
                     [CameraWorkers]  (Concurrent background worker threads)
                     ├── 1. Gray World Color Normalization (corrects sensor tints)
                     ├── 2. Exclusion Zone Polygon Masking (filters reflections/doors)
                     ├── 3. Aspect Ratio & Quality Gating (filters partial occlusions)
                     └── 4. Intra-Frame Duplicate Suppression (cosine sim > 0.85)
                             │
                             ▼
                   [TransReIDEncoder]  (Shared thread-safe feature encoder)
                   ├── Default: PersonViT-ReID ViT-Small (MSMT17, 384-dim normalized)
                   └── Fallback: DINOv2 ViT-Small (contrastive self-supervised)
                             │
                             ▼
                     [GlobalGallery]  (Thread-safe identity bank across all streams)
                     ├── Phase 1: Local ByteTrack Continuity Verification
                     ├── Phase 2: Mutual Exclusivity Bipartite Matching
                     ├── Per-Camera-Pair Threshold Matrix (e.g. portrait vs landscape)
                     ├── Multi-Exemplar Feature Bank (8 diverse pose vectors per person)
                     ├── Redundancy-Based Eviction & Contamination Guard
                     └── Multi-Angle Thumbnail Selection (up to 3 best sharpness crops)
                             │
            ┌────────────────┴────────────────────────┐
            ▼                                         ▼
   [Annotated MP4 Videos]                     [TrackingLog]
   - Bounding boxes & Global IDs              - Camera timeline metadata
   - Color-coded per person                   - Appearance continuous intervals
   - Semi-transparent HUD overlay             - Sampled frame visibility & detections
            │                                         │
            └────────────────┬────────────────────────┘
                             ▼
                    [ReviewUI Dashboard]  (Tkinter desktop application)
                    ├── Multi-Camera Synchronized Playback (0.25x - 4x, Seek Slider)
                    ├── Camera View Switcher (Grid view / Single camera full-screen)
                    ├── Auto-Tracking Follow Mode (camera switches as target moves)
                    ├── People Gallery & Interactive Manual ID Merging
                    ├── Automated PDF Person Summary Report Generation
                    └── [AIAssistant]  (OpenAI GPT-4o-mini & Vision query engine)
                        ├── Multi-thumbnail visual profiling (SHA-256 cached)
                        ├── Spatiotemporal timeline query answering
                        └── Interactive timestamp links: [Cam X @ M:SS]
```

### Core Architectural Components

1. **Synchronized Ingestion & Detection Dispatcher (`src/dispatcher.py`)**:
   - Running multiple YOLO instances concurrently on Apple Silicon GPUs (MPS) causes memory corruption due to non-thread-safe Metal allocations.
   - `DetectionDispatcher` manages $N$ distinct `PersonDetector` instances (one per camera) running in a single dedicated thread using a round-robin queue.
   - Each detector maintains independent ByteTrack tracker state (`persist=True`), ensuring tracking IDs are never corrupted while all cameras remain within $\pm 1$ frame of each other.

2. **Per-Camera Processing Worker (`src/pipeline.py`)**:
   - Each video stream is handled by a `CameraWorker` thread that fetches detections from the dispatcher.
   - **Gray World Color Normalization**: Dynamically rescales RGB channels based on gray-world assumptions to neutralize color temperature differences across varied camera sensors.
   - **Exclusion Zone Filtering**: Masks out predefined polygon areas (e.g., reflective glass, mirrors, doorway exits) to suppress static false positives.
   - **Intra-Frame Duplicate Suppression**: Identifies multiple person crops in the same frame with embedding cosine similarity $> 0.85$ (e.g., reflections or double detections) and drops the lower-confidence candidate.
   - **Quality Gating**: Person crops are filtered for minimum resolution ($28 \times 60$ px) and full-body aspect ratio ($H/W \ge 1.5$) before passing to the Re-ID model.

3. **Re-ID Feature Encoder (`src/reid_model.py`)**:
   - Encodes cropped person images into unit-normalized 384-dimensional feature vectors.
   - **Primary Model**: `PersonViT-ReID` (ViT-Small fine-tuned on MSMT17 with BN-Neck). Input resolution: $256 \times 128$.
   - **Fallback Model**: `DINOv2-ViT-Small` via `timm` (self-supervised contrastive model, auto-downloaded if local weights are not found).

4. **Global Identity Gallery (`src/gallery.py`)**:
   - Thread-safe repository linking local per-camera tracks to persistent cross-camera global IDs.
   - **Multi-Exemplar Feature Bank**: Stores up to 8 diverse appearance embeddings per identity (capturing front, back, side, and lighting variations) rather than a single drifting average vector.
   - **Mutual Exclusivity Constraint**: Per-frame bipartite matching guarantees that two people appearing simultaneously in the same camera frame can never be assigned the same global ID.
   - **Contamination Guard**: Rejects updates that deviate significantly from a person's known feature cluster to prevent occlusions from corrupting identity representations.
   - **Per-Camera-Pair Threshold Matrix**: Allows setting different similarity thresholds between specific camera pairs (e.g., lowering threshold to 0.55 for portrait vs landscape cameras with severe viewpoint divergence).
   - **Thumbnail Selection**: Retains the 3 sharpest, highest-resolution crops for each person, prioritizing viewpoint diversity across different cameras.

5. **Spatiotemporal Tracking Log (`src/tracking_log.py`)**:
   - Aggregates frame-by-frame detection coordinates, camera parameters, and continuous appearance intervals.
   - Grouping sightings into continuous appearances with a configurable gap threshold (30 frames).
   - Serialized into `output/tracking_log.json` for post-processing and AI analysis.

---

## Key Features

- **Robust Cross-Camera Re-ID**: Handles occlusion, turn-arounds, and cross-camera transitions using multi-exemplar banks and per-pair similarity matrices.
- **Headless Fast Batch Processing**: High-throughput processing pipeline runs headlessly without GUI overhead.
- **Interactive Review UI**: Native desktop GUI for post-processing review, frame scrubbing, camera switching, and auto-tracking targets across feeds.
- **People Gallery & Live ID Merging**: Visual gallery of all tracked subjects with one-click manual merging that updates tracking logs, detections, and UI state persistently.
- **AI Video Intelligence Assistant**: Chat panel powered by GPT-4o-mini and Vision that understands camera timelines, describes clothing/appearance, and provides clickable timestamp citation links (`[Cam X @ M:SS]`).
- **PDF & HTML Reporting**: One-click generation of multi-page PDF summary reports and standalone HTML tracking summaries with embedded base64 thumbnails.
- **Exclusion Zone Editor**: Interactive tool for clicking polygon points on camera frames to filter out reflections, mirrors, and unwanted regions.
- **Web Configuration Server**: Browser-based slider interface (`http://localhost:7700`) for tuning Re-ID thresholds, exemplar bank sizes, and camera-pair matrices.

---

## Quick Start

### 1. Prerequisites & Environment

> **⚠️ Requires Python 3.10.** Python 3.9 or earlier is not supported.

```bash
# Clone repository
git clone https://github.com/RuiMeng1/Test-FYP-Sensen.git
cd Test-FYP-Sensen

# Create virtual environment with Python 3.10
python3.10 -m venv .venv

# Activate environment
# macOS / Linux:
source .venv/bin/activate
# Windows (cmd/powershell):
.venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Environment Variables (.env)

The pipeline integrates with OpenAI models for the natural language video intelligence assistant. Copy `.env.example` to `.env` and set your OpenAI API key:

```bash
cp .env.example .env
```

Edit `.env`:
```ini
OPENAI_API_KEY=sk-...your_openai_api_key_here...
OPENAI_MODEL=gpt-4o-mini
```

*(If you do not intend to use the AI Assistant panel in the review UI, you can skip setting this key; all other tracking, review, and merging features work completely offline.)*

### 3. Pretrained Weights

- **YOLOv8-nano**: Downloaded automatically by Ultralytics (`yolov8n.pt`) on first run.
- **PersonViT-ReID**: Pre-trained weights can be downloaded or will fall back to DINOv2 automatically. To download official MSMT17 weights:
  ```bash
  python scripts/download_weights.py
  ```

---

## Pipeline Execution

The pipeline operates in two phases: **Headless Batch Processing** followed by the **Interactive Review Dashboard**.

### Phase 1: Headless Batch Processing

Process all videos in `data/` at maximum speed and save outputs to `output/`:

```bash
# Run full pipeline: process videos then open review dashboard
python -m src.main
```

During processing:
- All video streams are synchronized frame-by-frame.
- Annotated videos are saved to `output/<video_stem>_tracked.mp4`.
- Per-person appearance logs and coordinates are written to `output/tracking_log.json`.
- A standalone HTML summary is exported to `output/id_summary_report.html`.

### Phase 2: Interactive Review Dashboard

Once processing completes, the Review UI opens automatically. If you have already processed videos and want to inspect previous results without reprocessing:

```bash
# Open review dashboard using existing output/ files
python -m src.main --review-only
```

### CLI Flags & Options

```bash
# Select specific video files instead of scanning data/
python -m src.main --videos data/cam1.mp4 data/cam2.mp4

# Specify a custom data directory
python -m src.main --data-dir /path/to/videos

# Headless processing only (do not launch Review UI afterwards)
python -m src.main --no-preview

# Skip video processing and launch Review UI directly
python -m src.main --review-only

# Adjust Re-ID extraction frequency (extract features every N frames)
python -m src.main --reid-interval 10

# Force compute device (auto-detects CUDA -> MPS -> CPU by default)
python -m src.main --device cpu
python -m src.main --device mps    # Apple Silicon GPU
python -m src.main --device cuda   # NVIDIA GPU
```

---

## Configuration & Helper Tools

### Web Configuration UI (`scripts/configure.py`)

Launch the local web server to adjust pipeline thresholds, exemplar banks, and per-camera-pair matrices:

```bash
python scripts/configure.py
```
- Opens automatically in your browser at `http://localhost:7700`.
- Set global cosine similarity thresholds (default: `0.65`).
- Adjust **Per-Camera-Pair Thresholds** (e.g. set `0.55` between portrait and landscape cameras to account for extreme viewpoint drift).
- Click **Save Settings** to persist configurations to `config/pipeline_settings.json`.

### Exclusion Zone Polygon Tool (`scripts/draw_zones.py`)

If a camera feed contains mirrors, glass walls, posters, or doorway reflections:

```bash
python scripts/draw_zones.py data/cam1_20260507_160822.mp4
```
- **Left click**: Add polygon vertex.
- **Right click**: Remove last vertex.
- **`c`**: Clear all points.
- **`Enter` or `q`**: Save and print coordinates.
- Copy the generated coordinates into `EXCLUSION_ZONES` in `src/config.py`.

### Synchronized Playback Viewer (`scripts/playback.py`)

A lightweight, standalone OpenCV-based synchronized multi-grid player for tracked videos:

```bash
# Play all tracked videos in output/
python scripts/playback.py

# Playback controls:
#   SPACE        — Play / Pause
#   ← / →        — Seek ±5 seconds
#   [ / ]        — Seek ±30 seconds
#   0 - 9        — Jump to 0% - 90%
#   + / -        — Adjust playback speed (0.25x to 4.0x)
#   q / ESC      — Quit
```

### Weight Downloader (`scripts/download_weights.py`)

Downloads the official TransReID MSMT17 model checkpoint using `gdown`:

```bash
python scripts/download_weights.py
```

---

## Review Dashboard & AI Assistant

The Review Dashboard (`src/review_ui.py`) is a native Tkinter application designed for investigating multi-camera surveillance footage.

### Multi-Camera Playback & Auto-Tracking
- **Synchronized Scrubbing**: Seek bar and playback controls keep all camera streams in exact frame synchrony.
- **View Switcher**: Instant switching between tiled 2x2 Grid View and individual full-screen camera feeds.
- **Auto-Tracking Mode**: Click any Person ID card in the timeline logger to lock focus. If the person moves between cameras, the interface automatically switches to full-screen their current camera. If they temporarily leave the scene, the UI displays a waiting badge and automatically jumps back when they reappear.

### People Gallery & Live ID Merging
- **People Gallery Tab**: Displays gallery cards for every tracked identity with appearance counts, visited camera badges, and up to 3 multi-angle thumbnails.
- **Interactive Manual Merge**: If extreme angle shifts or lighting changes caused the Re-ID algorithm to split one individual into two separate IDs:
  1. Click **Merge With...** on the source person's card.
  2. Click **Select as Target** on the target identity.
  3. Confirm the merge prompt.
  4. The UI immediately consolidates appearance timelines, updates detection labels across all video frames, re-saves `output/tracking_log.json`, and refreshes the display.

### AI Video Intelligence (GPT-4o-mini Vision)
- **Natural Language Chat**: Ask questions directly about the footage:
  - *"Which person visited both Camera 1 and Camera 3?"*
  - *"Who was wearing a red jacket and black backpack?"*
  - *"At what time did ID 2 leave Camera 2?"*
- **Visual Profiling with GPT-4o-mini Vision**: Analyzes person crops to identify clothing color, accessories, hats, and backpacks. Results are fingerprinted via SHA-256 and cached in `output/ai_person_profiles.json` to prevent hallucinations and eliminate redundant API calls.
- **Clickable Timestamp Links**: AI responses cite specific moments formatted as `[Cam X @ M:SS]`. Clicking any citation instantly seeks the video player to that camera and timestamp.

### PDF Summary Report Export
- Click **Export PDF Report** in the People Gallery to generate a formatted PDF report.
- Includes header metadata, total persons detected, per-person thumbnail photos, visited cameras, total detection frames, and complete appearance timeline tables.

---

## Output Artifacts

After running the pipeline, files are saved in `output/`:

| Artifact | Description |
| :--- | :--- |
| `cam*_tracked.mp4` | High-definition annotated video feeds with bounding boxes, global IDs, and telemetry HUD overlays. |
| `tracking_log.json` | JSON database containing camera metadata, appearance timelines, and detection bounding boxes. |
| `id_summary_report.html` | Self-contained HTML report with stats and embedded base64 person thumbnails. |
| `ai_person_profiles.json` | Cached GPT-4o-mini visual attribute descriptions and SHA-256 thumbnail fingerprints. |
| `person_summary_report.pdf` | (Optional) Exported multi-page PDF summary dossier generated from the Review UI. |

---

## Directory Structure

```text
Test-FYP-Sensen/
├── .env.example                # Template for OpenAI API credentials
├── .gitignore                  # Git ignore patterns for venv, weights, and caches
├── pyproject.toml              # Project configuration metadata
├── pyrightconfig.json          # Type checker configuration
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── config/
│   └── pipeline_settings.json  # Runtime configuration saved from scripts/configure.py
│
├── data/                       # Input video files (e.g. cam1_*.mp4, cam2_*.mp4)
│
├── models/                     # Deep learning weights directory
│   └── personvit_reid/         # PersonViT-ReID HuggingFace model files
│
├── output/                     # Output directory for processed videos and logs
│   ├── *_tracked.mp4           # Output annotated MP4 videos
│   ├── tracking_log.json       # Spatiotemporal tracking log & detection data
│   ├── id_summary_report.html  # Standalone HTML summary report
│   └── ai_person_profiles.json # Cached visual profile attributes from GPT-4o-mini
│
├── scripts/
│   ├── configure.py            # Web UI server for editing thresholds & settings
│   ├── download_weights.py     # TransReID MSMT17 weights downloader
│   ├── draw_zones.py           # Interactive exclusion polygon drawing utility
│   └── playback.py             # Lightweight synchronized OpenCV replay viewer
│
└── src/
    ├── __init__.py
    ├── ai_assistant.py         # OpenAI GPT-4o-mini Vision intelligence & query engine
    ├── assets/                 # Application branding and UI icons
    ├── config.py               # System defaults, path definitions, and exclusion zones
    ├── detector.py             # YOLOv8 + ByteTrack detector with aspect-ratio gating
    ├── dispatcher.py           # Round-robin frame dispatcher & MPS GPU sync scheduler
    ├── gallery.py              # Thread-safe global Re-ID gallery & multi-exemplar bank
    ├── main.py                 # Pipeline entry point (Phase 1 batch -> Phase 2 review)
    ├── pipeline.py             # CameraWorker thread, color normalization & video writer
    ├── reid_model.py           # TransReID / PersonViT / DINOv2 embedding extractor
    ├── review_ui.py            # Tkinter post-processing review dashboard & AI panel
    ├── tk_ui.py                # Optional native live tracking dashboard
    ├── tracking_log.py         # Thread-safe appearance logger & JSON exporter
    ├── video_loader.py         # Video discovery and CLI argument parser
    ├── web_server.py           # Optional Flask MJPEG streaming server
    └── templates/              # Web dashboard HTML templates
```
