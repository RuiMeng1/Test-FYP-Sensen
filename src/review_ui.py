"""
review_ui.py — Post-processing review dashboard for multi-camera person tracking.

Opens after processing is complete (or via --review-only / python -m src.review_ui).
Loads the annotated output videos and tracking_log.json, providing:
  - Video playback with play/pause, frame stepping, and seek slider
  - Person logger with expandable per-person appearance timelines
  - Auto-tracking mode: click any Person ID to follow across cameras
  - View switching: Grid View, individual camera full-screen buttons
  - NEW: "People Gallery" section with photos, statistics, and interactive
    manual merging of two IDs together with live persistence.

Runs on the main thread (macOS Cocoa/Tk requirement).
"""

from __future__ import annotations

import base64
import io
import json
import time
import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk


# ── Colour constants ─────────────────────────────────────────────────────────
BG_PRIMARY     = "#0a0e1a"
BG_SECONDARY   = "#111827"
BG_CARD        = "#151d2e"
BG_ACTIVE      = "#0d2818"
BG_EXPANDED    = "#0f1729"
BG_TRACKED     = "#1e1b4b"
BG_MERGE_SRC   = "#451a03"
BORDER         = "#1e293b"
BORDER_ACTIVE  = "#10b981"
BORDER_TRACKED = "#38bdf8"
BORDER_MERGE   = "#f59e0b"

TEXT_PRIMARY   = "#f1f5f9"
TEXT_SECONDARY = "#94a3b8"
TEXT_MUTED     = "#64748b"
ACCENT_GREEN   = "#10b981"
ACCENT_BLUE    = "#3b82f6"
ACCENT_RED     = "#ef4444"
ACCENT_YELLOW  = "#f59e0b"


class ReviewUI:
    """
    Post-processing review dashboard with interactive People Gallery & ID Merging.

    Parameters
    ----------
    tracking_data : Parsed JSON from tracking_log.json
    output_dir    : Directory containing the tracked output videos
    """

    def __init__(self, tracking_data: dict, output_dir: str):
        self.tracking_data = tracking_data
        self.output_dir = Path(output_dir)

        # Camera metadata
        self.cameras: Dict[int, dict] = {
            int(k): v for k, v in tracking_data.get("cameras", {}).items()
        }
        self.persons: Dict[int, dict] = {
            int(k): v for k, v in tracking_data.get("persons", {}).items()
        }

        # Video captures
        self._caps: Dict[int, cv2.VideoCapture] = {}
        self._n_cameras = len(self.cameras)

        # Playback state
        self._playing = False
        self._current_frame = 0
        self._max_frame = 0
        self._playback_fps = 25.0

        # View state
        self._main_section = "video"              # "video" or "gallery"
        self._view_mode = "grid"                  # "grid" or "single"
        self._selected_cam: Optional[int] = None  # Camera shown in single mode
        self._tracked_gid: Optional[int] = None   # Person ID currently being followed

        # Merge state
        self._merge_source_gid: Optional[int] = None

        # PhotoImage references (prevent GC)
        self._photo_refs: Dict[int, ImageTk.PhotoImage] = {}
        self._thumb_refs: Dict[int, ImageTk.PhotoImage] = {}
        self._gallery_thumb_refs: Dict[int, ImageTk.PhotoImage] = {}

        # Expanded person cards in sidebar
        self._expanded: Set[int] = set()

        # UI references
        self._cam_cards: Dict[int, tk.Frame] = {}
        self._cam_labels: Dict[int, tk.Label] = {}
        self._cam_overlays: Dict[int, tk.Label] = {}
        self._cam_view_btns: Dict[str, tk.Button] = {}

        # Raw video captures & per-frame detections
        self.detections: Dict[str, Dict[str, List[dict]]] = tracking_data.get("detections", {})
        self._raw_caps: Dict[int, cv2.VideoCapture] = {}

        # Open video files
        self._open_videos()
        self._build_ui()

    # ── Video management ──────────────────────────────────────────────────────

    def _open_videos(self) -> None:
        """Open all output video files and raw input videos if available."""
        from src.config import DATA_DIR

        for cam_id, meta in self.cameras.items():
            # 1. Try to open clean raw video from data/
            raw_name = meta.get("raw_video_file")
            if not raw_name:
                raw_name = meta["video_file"].replace("_tracked.mp4", ".mp4")
            raw_path = DATA_DIR / raw_name
            if raw_path.exists():
                raw_cap = cv2.VideoCapture(str(raw_path))
                if raw_cap.isOpened():
                    self._raw_caps[cam_id] = raw_cap

            # 2. Open tracked output video as fallback
            video_path = self.output_dir / meta["video_file"]
            if video_path.exists():
                cap = cv2.VideoCapture(str(video_path))
                if cap.isOpened():
                    self._caps[cam_id] = cap
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    self._max_frame = max(self._max_frame, total - 1)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    self._playback_fps = fps
                else:
                    print(f"[ReviewUI] Warning: Could not open {video_path}")
            elif cam_id in self._raw_caps:
                cap = self._raw_caps[cam_id]
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self._max_frame = max(self._max_frame, total - 1)
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                self._playback_fps = fps
            else:
                print(f"[ReviewUI] Warning: Video not found: {video_path}")

    def _read_frame(self, cam_id: int, frame_idx: int) -> Optional[np.ndarray]:
        """Read a specific frame from video and dynamically draw filtered bounding boxes."""
        raw_cap = self._raw_caps.get(cam_id)
        cam_dets_map = self.detections.get(str(cam_id), {})
        frame_dets = cam_dets_map.get(str(frame_idx), [])

        # If raw clean video and per-frame detections are available:
        if raw_cap is not None and self.detections:
            raw_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = raw_cap.read()
            if ret and frame is not None:
                frame_copy = frame.copy()
                if self._tracked_gid is not None:
                    to_draw = [d for d in frame_dets if d.get("gid") == self._tracked_gid]
                else:
                    to_draw = frame_dets

                for d in to_draw:
                    self._draw_detection_box(frame_copy, d["gid"], d["bbox"])
                return frame_copy

        # Fallback to pre-rendered output video
        cap = self._caps.get(cam_id)
        if cap is None:
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        return frame if ret else None

    def _draw_detection_box(self, frame: np.ndarray, gid: int, bbox: List[int]) -> None:
        """Draw bounding box and ID label on a clean frame."""
        palette = [
            (60, 180, 255), (100, 220, 0), (230, 80, 80),
            (50, 215, 255), (200, 130, 255), (255, 190, 30),
            (0, 140, 255), (180, 220, 50), (220, 20, 60),
        ]
        color = palette[gid % len(palette)] if gid >= 0 else (140, 140, 140)
        x1, y1, x2, y2 = bbox

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        id_text = f"ID:{gid:03d}"
        id_font = cv2.FONT_HERSHEY_DUPLEX
        id_scale = 1.4
        id_thick = 2
        (tw, th), bl = cv2.getTextSize(id_text, id_font, id_scale, id_thick)
        pad = 6
        lbl_x1 = x1
        lbl_y2 = y1
        lbl_y1 = max(0, y1 - th - bl - pad * 2)
        lbl_x2 = x1 + tw + pad * 2

        cv2.rectangle(frame, (lbl_x1, lbl_y1), (lbl_x2, lbl_y2), color, -1)
        cv2.putText(
            frame, id_text, (lbl_x1 + pad, lbl_y2 - pad - bl),
            id_font, id_scale, (0, 0, 0), id_thick, cv2.LINE_AA
        )

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.root = tk.Tk()
        self.root.title("Sensen — Multi-Camera Review & People Gallery")
        self.root.configure(bg=BG_PRIMARY)
        self.root.minsize(1200, 750)

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        win_w = min(1520, screen_w - 60)
        win_h = min(960, screen_h - 60)
        x = (screen_w - win_w) // 2
        y = (screen_h - win_h) // 2
        self.root.geometry(f"{win_w}x{win_h}+{x}+{y}")

        # ── Header ────────────────────────────────────────────────────────────
        header = tk.Frame(self.root, bg="#0f172a", height=50)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)

        logo = tk.Label(
            header, text=" S ", bg=ACCENT_BLUE, fg="white",
            font=("Helvetica", 12, "bold"), padx=5, pady=2
        )
        logo.pack(side=tk.LEFT, padx=(16, 8), pady=8)

        tk.Label(
            header, text="Sensen Dashboard",
            bg="#0f172a", fg=TEXT_PRIMARY,
            font=("Helvetica", 14, "bold")
        ).pack(side=tk.LEFT, pady=8, padx=(0, 20))

        # ── Section Navigation Tabs ───────────────────────────────────────────
        nav_frame = tk.Frame(header, bg="#0f172a")
        nav_frame.pack(side=tk.LEFT, pady=6)

        tab_style = {
            "font": ("Helvetica", 11, "bold"),
            "bd": 0, "cursor": "hand2", "padx": 14, "pady": 4
        }

        self._tab_video_btn = tk.Button(
            nav_frame, text="🎬 Cameras & Playback",
            command=self._show_video_section,
            bg=ACCENT_BLUE, fg="white",
            activebackground="#2563eb", activeforeground="white",
            **tab_style
        )
        self._tab_video_btn.pack(side=tk.LEFT, padx=3)

        self._tab_gallery_btn = tk.Button(
            nav_frame, text=f"👥 People Gallery ({len(self.persons)})",
            command=self._show_gallery_section,
            bg="#1e293b", fg=TEXT_SECONDARY,
            activebackground="#334155", activeforeground="white",
            **tab_style
        )
        self._tab_gallery_btn.pack(side=tk.LEFT, padx=3)

        self._time_label = tk.Label(
            header, text="0:00 / 0:00",
            bg="#0f172a", fg=TEXT_SECONDARY,
            font=("Menlo", 11)
        )
        self._time_label.pack(side=tk.RIGHT, padx=16, pady=8)

        # ── Main container ────────────────────────────────────────────────────
        self._content_root = tk.Frame(self.root, bg=BG_PRIMARY)
        self._content_root.pack(fill=tk.BOTH, expand=True)

        # 1. Video Playback Section View
        self._video_section_frame = tk.Frame(self._content_root, bg=BG_PRIMARY)
        self._build_video_section(self._video_section_frame)

        # 2. People Gallery Section View
        self._gallery_section_frame = tk.Frame(self._content_root, bg=BG_PRIMARY)
        self._build_gallery_section(self._gallery_section_frame)

        # Initially show video section
        self._show_video_section()

        # ── Status bar ────────────────────────────────────────────────────────
        status_bar = tk.Frame(self.root, bg="#0f172a", height=32)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        status_bar.pack_propagate(False)

        self._status_label = tk.Label(
            status_bar,
            text=f"  {len(self.persons)} unique persons  •  {self._n_cameras} cameras  •  {self._max_frame + 1} frames",
            bg="#0f172a", fg=TEXT_MUTED, font=("Menlo", 10),
            anchor="w"
        )
        self._status_label.pack(side=tk.LEFT, padx=12, pady=4)

        tk.Label(
            status_bar, text="People Gallery: Click 'Merge into...' to combine duplicate IDs",
            bg="#0f172a", fg="#64748b", font=("Helvetica", 10)
        ).pack(side=tk.RIGHT, padx=16, pady=4)

    # ── Section Switching ─────────────────────────────────────────────────────

    def _show_video_section(self) -> None:
        """Switch view to the multi-camera video playback section."""
        self._main_section = "video"
        self._gallery_section_frame.pack_forget()
        self._video_section_frame.pack(fill=tk.BOTH, expand=True)

        self._tab_video_btn.configure(bg=ACCENT_BLUE, fg="white")
        self._tab_gallery_btn.configure(bg="#1e293b", fg=TEXT_SECONDARY)
        self._update_display()

    def _show_gallery_section(self) -> None:
        """Switch view to the People Gallery grid section."""
        self._main_section = "gallery"
        self._video_section_frame.pack_forget()
        self._gallery_section_frame.pack(fill=tk.BOTH, expand=True)

        self._tab_video_btn.configure(bg="#1e293b", fg=TEXT_SECONDARY)
        self._tab_gallery_btn.configure(bg=ACCENT_BLUE, fg="white")
        self._rebuild_gallery_grid()

    # ── Video Section Construction ────────────────────────────────────────────

    def _build_video_section(self, parent: tk.Frame) -> None:
        """Build the video playback area with camera tiles and logger panel."""
        logger_panel = tk.Frame(parent, bg=BG_SECONDARY, width=400)
        logger_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=8)
        logger_panel.pack_propagate(False)
        self._build_logger(logger_panel)

        left_panel = tk.Frame(parent, bg=BG_PRIMARY)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_view_switcher(left_panel)

        self._camera_frame = tk.Frame(left_panel, bg=BG_PRIMARY)
        self._camera_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 4))

        controls = tk.Frame(left_panel, bg="#0f172a", height=56)
        controls.pack(fill=tk.X, padx=8, pady=(0, 8))
        controls.pack_propagate(False)
        self._build_controls(controls)

        self._build_camera_tiles()
        self._apply_view_layout()

    def _build_view_switcher(self, parent: tk.Frame) -> None:
        bar = tk.Frame(parent, bg="#0f172a", height=42)
        bar.pack(fill=tk.X, padx=8, pady=(8, 0))
        bar.pack_propagate(False)

        tk.Label(
            bar, text="VIEW:", bg="#0f172a", fg=TEXT_MUTED,
            font=("Menlo", 9, "bold")
        ).pack(side=tk.LEFT, padx=(12, 6), pady=8)

        btn_style = {
            "bg": "#1e293b", "fg": TEXT_PRIMARY, "font": ("Helvetica", 10, "bold"),
            "bd": 0, "activebackground": "#334155", "activeforeground": "white",
            "cursor": "hand2", "padx": 10, "pady": 3
        }

        self._btn_grid = tk.Button(
            bar, text="▦ All Cameras (Grid)",
            command=self._set_view_grid, **btn_style
        )
        self._btn_grid.pack(side=tk.LEFT, padx=3, pady=6)
        self._cam_view_btns["grid"] = self._btn_grid

        for cam_id in sorted(self._caps.keys()):
            b = tk.Button(
                bar, text=f"📷 Cam {cam_id}",
                command=lambda c=cam_id: self._set_view_camera(c), **btn_style
            )
            b.pack(side=tk.LEFT, padx=3, pady=6)
            self._cam_view_btns[f"cam_{cam_id}"] = b

        self._tracking_status_frame = tk.Frame(bar, bg="#0f172a")
        self._tracking_status_frame.pack(side=tk.RIGHT, padx=12, pady=6)

        self._tracking_badge = tk.Label(
            self._tracking_status_frame, text="",
            bg=BG_TRACKED, fg=BORDER_TRACKED,
            font=("Menlo", 10, "bold"), padx=8, pady=2
        )
        self._tracking_stop_btn = tk.Button(
            self._tracking_status_frame, text=" ✕ Stop Tracking ",
            command=self._stop_tracking,
            bg="#ef4444", fg="white", font=("Helvetica", 9, "bold"),
            bd=0, activebackground="#dc2626", activeforeground="white",
            cursor="hand2", padx=6, pady=2
        )

    def _build_controls(self, parent: tk.Frame) -> None:
        btn_frame = tk.Frame(parent, bg="#0f172a")
        btn_frame.pack(side=tk.LEFT, padx=12, pady=8)

        btn_style = {"bg": "#1e293b", "fg": TEXT_PRIMARY, "font": ("Helvetica", 14),
                     "bd": 0, "activebackground": "#334155", "activeforeground": "white",
                     "cursor": "hand2", "padx": 8, "pady": 2}

        self._btn_back = tk.Button(btn_frame, text="◀◀", command=self._step_back, **btn_style)
        self._btn_back.pack(side=tk.LEFT, padx=2)

        self._btn_play = tk.Button(btn_frame, text=" ▶ ", command=self._toggle_play, **btn_style)
        self._btn_play.pack(side=tk.LEFT, padx=2)

        self._btn_fwd = tk.Button(btn_frame, text="▶▶", command=self._step_forward, **btn_style)
        self._btn_fwd.pack(side=tk.LEFT, padx=2)

        self._frame_label = tk.Label(
            parent, text="Frame: 0", bg="#0f172a", fg=TEXT_MUTED,
            font=("Menlo", 10)
        )
        self._frame_label.pack(side=tk.LEFT, padx=(12, 8))

        self._seek_var = tk.IntVar(value=0)
        self._seek_slider = tk.Scale(
            parent, from_=0, to=self._max_frame,
            orient=tk.HORIZONTAL, variable=self._seek_var,
            command=self._on_seek,
            bg="#0f172a", fg=TEXT_SECONDARY, troughcolor="#1e293b",
            activebackground=ACCENT_BLUE, highlightthickness=0,
            bd=0, sliderrelief=tk.FLAT, sliderlength=16,
            showvalue=False, font=("Menlo", 9),
        )
        self._seek_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8, pady=8)

    def _build_logger(self, parent: tk.Frame) -> None:
        header = tk.Frame(parent, bg=BG_SECONDARY)
        header.pack(fill=tk.X, padx=12, pady=(12, 8))

        tk.Label(
            header, text="👁  Person Logger",
            bg=BG_SECONDARY, fg=TEXT_PRIMARY,
            font=("Helvetica", 13, "bold")
        ).pack(side=tk.LEFT)

        self._count_label = tk.Label(
            header, text=f"{len(self.persons)} IDs",
            bg=ACCENT_BLUE, fg="white",
            font=("Menlo", 10, "bold"), padx=8, pady=2
        )
        self._count_label.pack(side=tk.RIGHT)

        hint = tk.Label(
            parent, text="Click any person to auto-follow across cameras",
            bg=BG_SECONDARY, fg=TEXT_MUTED, font=("Helvetica", 9, "italic")
        )
        hint.pack(fill=tk.X, padx=12, pady=(0, 6))

        tk.Frame(parent, bg=BORDER, height=1).pack(fill=tk.X, padx=8)

        list_container = tk.Frame(parent, bg=BG_SECONDARY)
        list_container.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._canvas = tk.Canvas(
            list_container, bg=BG_SECONDARY,
            highlightthickness=0, bd=0
        )
        scrollbar = tk.Scrollbar(
            list_container, orient=tk.VERTICAL,
            command=self._canvas.yview,
            bg=BG_SECONDARY, troughcolor=BG_SECONDARY,
            activebackground=TEXT_MUTED, width=8
        )
        self._canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._person_list_frame = tk.Frame(self._canvas, bg=BG_SECONDARY)
        self._canvas_window = self._canvas.create_window(
            (0, 0), window=self._person_list_frame, anchor="nw"
        )

        self._person_list_frame.bind(
            "<Configure>",
            lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        )
        self._canvas.bind(
            "<Configure>",
            lambda e: self._canvas.itemconfig(self._canvas_window, width=e.width)
        )
        self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        self._build_person_cards()

    def _build_camera_tiles(self) -> None:
        for cam_id in sorted(self._caps.keys()):
            card = tk.Frame(self._camera_frame, bg=BG_SECONDARY, bd=0)
            card.grid_rowconfigure(0, weight=1)
            card.grid_columnconfigure(0, weight=1)

            cam_label = tk.Label(card, bg="black", bd=0, anchor="center", cursor="hand2")
            cam_label.grid(row=0, column=0, sticky="nsew")
            self._cam_labels[cam_id] = cam_label

            overlay = tk.Label(
                cam_label, text=f"  ● CAM {cam_id}  ",
                bg="#000000", fg="white",
                font=("Menlo", 10, "bold"),
            )
            overlay.place(x=8, y=8)
            self._cam_overlays[cam_id] = overlay
            self._cam_cards[cam_id] = card

            for w in [cam_label, overlay]:
                w.bind("<Double-Button-1>", lambda e, c=cam_id: self._toggle_camera_fullscreen(c))

    def _apply_view_layout(self) -> None:
        for r in range(10):
            self._camera_frame.grid_rowconfigure(r, weight=0, uniform="")
            self._camera_frame.grid_columnconfigure(r, weight=0, uniform="")

        cam_ids = sorted(self._caps.keys())

        if self._view_mode == "single" and self._selected_cam is not None:
            self._camera_frame.grid_rowconfigure(0, weight=1)
            self._camera_frame.grid_columnconfigure(0, weight=1)

            for cid, card in self._cam_cards.items():
                if cid == self._selected_cam:
                    card.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
                else:
                    card.grid_remove()

            self._btn_grid.configure(bg="#1e293b", fg=TEXT_SECONDARY)
            for cid in cam_ids:
                btn = self._cam_view_btns.get(f"cam_{cid}")
                if btn:
                    if cid == self._selected_cam:
                        btn.configure(bg=ACCENT_BLUE, fg="white")
                    else:
                        btn.configure(bg="#1e293b", fg=TEXT_SECONDARY)
        else:
            cols = 2
            n_rows = (len(cam_ids) + cols - 1) // cols

            for r in range(n_rows):
                self._camera_frame.grid_rowconfigure(r, weight=1, uniform="cam_row")
            for c in range(cols):
                self._camera_frame.grid_columnconfigure(c, weight=1, uniform="cam_col")

            for idx, cid in enumerate(cam_ids):
                row, col = divmod(idx, cols)
                card = self._cam_cards[cid]
                card.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")

            self._btn_grid.configure(bg=ACCENT_BLUE, fg="white")
            for cid in cam_ids:
                btn = self._cam_view_btns.get(f"cam_{cid}")
                if btn:
                    btn.configure(bg="#1e293b", fg=TEXT_SECONDARY)

        self._update_tracking_status_ui()
        self._update_display()

    # ── People Gallery Section ─────────────────────────────────────────────────

    def _build_gallery_section(self, parent: tk.Frame) -> None:
        """Build the People Gallery section for reviewing person profiles and merging IDs."""
        toolbar = tk.Frame(parent, bg="#0f172a", height=60)
        toolbar.pack(fill=tk.X, padx=12, pady=(10, 6))
        toolbar.pack_propagate(False)

        title_frame = tk.Frame(toolbar, bg="#0f172a")
        title_frame.pack(side=tk.LEFT, padx=12, pady=8)

        tk.Label(
            title_frame, text="👥 People Gallery",
            bg="#0f172a", fg=TEXT_PRIMARY,
            font=("Helvetica", 15, "bold")
        ).pack(anchor="w")

        self._gallery_subtitle = tk.Label(
            title_frame, text=f"{len(self.persons)} distinct person identities detected across cameras.",
            bg="#0f172a", fg=TEXT_MUTED, font=("Helvetica", 10)
        )
        self._gallery_subtitle.pack(anchor="w")

        self._merge_alert_frame = tk.Frame(parent, bg="#78350f", height=42)
        self._merge_alert_label = tk.Label(
            self._merge_alert_frame, text="",
            bg="#78350f", fg="#fef3c7", font=("Helvetica", 10, "bold")
        )
        self._merge_alert_label.pack(side=tk.LEFT, padx=16, pady=8)

        self._merge_cancel_btn = tk.Button(
            self._merge_alert_frame, text=" ✕ Cancel Merge ",
            command=self._cancel_merge,
            bg="#dc2626", fg="white", font=("Helvetica", 9, "bold"),
            bd=0, activebackground="#b91c1c", activeforeground="white",
            cursor="hand2", padx=8, pady=3
        )
        self._merge_cancel_btn.pack(side=tk.RIGHT, padx=12, pady=6)

        grid_container = tk.Frame(parent, bg=BG_PRIMARY)
        grid_container.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

        self._gallery_canvas = tk.Canvas(
            grid_container, bg=BG_PRIMARY, highlightthickness=0, bd=0
        )
        gallery_scrollbar = tk.Scrollbar(
            grid_container, orient=tk.VERTICAL,
            command=self._gallery_canvas.yview,
            bg=BG_PRIMARY, troughcolor=BG_PRIMARY,
            activebackground=TEXT_MUTED, width=10
        )
        self._gallery_canvas.configure(yscrollcommand=gallery_scrollbar.set)

        gallery_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._gallery_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._gallery_grid_frame = tk.Frame(self._gallery_canvas, bg=BG_PRIMARY)
        self._gallery_canvas_window = self._gallery_canvas.create_window(
            (0, 0), window=self._gallery_grid_frame, anchor="nw"
        )

        self._gallery_grid_frame.bind(
            "<Configure>",
            lambda e: self._gallery_canvas.configure(scrollregion=self._gallery_canvas.bbox("all"))
        )
        self._gallery_canvas.bind(
            "<Configure>",
            lambda e: self._gallery_canvas.itemconfig(self._gallery_canvas_window, width=e.width)
        )

    def _rebuild_gallery_grid(self) -> None:
        """Populate the People Gallery grid with cards for each person."""
        for widget in self._gallery_grid_frame.winfo_children():
            widget.destroy()
        self._gallery_thumb_refs.clear()

        count = len(self.persons)
        self._gallery_subtitle.configure(
            text=f"{count} distinct person identities detected across cameras. Click 'Merge into...' to merge duplicate IDs."
        )
        self._tab_gallery_btn.configure(text=f"👥 People Gallery ({count})")
        self._count_label.configure(text=f"{count} IDs")

        if self._merge_source_gid is not None:
            src = self._merge_source_gid
            self._merge_alert_label.configure(
                text=f"⚠️ MERGE MODE: Selected ID:{src:03d} — Now click 'Merge into ID...' on the target card below."
            )
            self._merge_alert_frame.pack(fill=tk.X, padx=12, pady=(0, 6), before=self._gallery_grid_frame.master)
        else:
            self._merge_alert_frame.pack_forget()

        if not self.persons:
            tk.Label(
                self._gallery_grid_frame, text="No person records found in tracking log.",
                bg=BG_PRIMARY, fg=TEXT_MUTED, font=("Helvetica", 14), pady=40
            ).pack()
            return

        cols = 3
        for idx, gid in enumerate(sorted(self.persons.keys())):
            row, col = divmod(idx, cols)
            self._build_one_gallery_card(self._gallery_grid_frame, gid, row, col)

    def _build_one_gallery_card(self, parent: tk.Frame, gid: int, row: int, col: int) -> None:
        """Build an individual profile card in the People Gallery."""
        person = self.persons[gid]
        color_hex = person.get("color_hex", "#64748b")
        is_merge_source = (self._merge_source_gid == gid)
        is_merge_active = (self._merge_source_gid is not None)

        border_color = BORDER_MERGE if is_merge_source else BORDER
        bg_card = BG_MERGE_SRC if is_merge_source else BG_CARD

        card_outer = tk.Frame(parent, bg=border_color, bd=1)
        card_outer.grid(row=row, column=col, padx=8, pady=8, sticky="nsew")
        parent.grid_columnconfigure(col, weight=1, uniform="gallery_col")

        card = tk.Frame(card_outer, bg=bg_card, padx=12, pady=12)
        card.pack(fill=tk.BOTH, expand=True)

        # Top Row
        top_row = tk.Frame(card, bg=bg_card)
        top_row.pack(fill=tk.X, pady=(0, 10))

        swatch = tk.Frame(top_row, bg=color_hex, width=14, height=14)
        swatch.pack(side=tk.LEFT, padx=(0, 8), pady=2)
        swatch.pack_propagate(False)

        tk.Label(
            top_row, text=f"Person ID: {gid:03d}",
            bg=bg_card, fg=TEXT_PRIMARY,
            font=("Menlo", 14, "bold")
        ).pack(side=tk.LEFT)

        if is_merge_source:
            tk.Label(
                top_row, text=" [MERGE SOURCE] ",
                bg="#f59e0b", fg="#000000",
                font=("Menlo", 9, "bold"), padx=4, pady=1
            ).pack(side=tk.RIGHT)
        else:
            tk.Label(
                top_row, text=f"×{person.get('total_detections', 0)} frames",
                bg=bg_card, fg=TEXT_MUTED, font=("Menlo", 10)
            ).pack(side=tk.RIGHT)

        # Middle Row
        mid_row = tk.Frame(card, bg=bg_card)
        mid_row.pack(fill=tk.X, pady=(0, 10))

        thumb_w, thumb_h = 95, 125
        thumb_frame = tk.Frame(mid_row, bg=BORDER, width=thumb_w, height=thumb_h)
        thumb_frame.pack_propagate(False)
        thumb_frame.pack(side=tk.LEFT, padx=(0, 14))

        if person.get("thumbnail_b64"):
            try:
                raw = base64.b64decode(person["thumbnail_b64"])
                pil_img = Image.open(io.BytesIO(raw))
                pil_img = pil_img.resize((thumb_w - 4, thumb_h - 4), Image.LANCZOS)
                photo = ImageTk.PhotoImage(pil_img)
                thumb_label = tk.Label(thumb_frame, image=photo, bg=BORDER, bd=0)
                thumb_label.pack(fill=tk.BOTH, expand=True)
                self._gallery_thumb_refs[gid] = photo
            except Exception:
                tk.Label(thumb_frame, text="👤", bg=BG_SECONDARY, fg=TEXT_MUTED,
                         font=("Helvetica", 28)).pack(fill=tk.BOTH, expand=True)
        else:
            tk.Label(thumb_frame, text="👤", bg=BG_SECONDARY, fg=TEXT_MUTED,
                     font=("Helvetica", 28)).pack(fill=tk.BOTH, expand=True)

        stats_col = tk.Frame(mid_row, bg=bg_card)
        stats_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        apps = person.get("appearances", [])
        total_sec = sum(a.get("duration_seconds", 0) for a in apps)
        tk.Label(
            stats_col, text=f"Appearances: {len(apps)} segment(s)",
            bg=bg_card, fg=TEXT_SECONDARY, font=("Helvetica", 10, "bold")
        ).pack(anchor="w", pady=(0, 2))

        tk.Label(
            stats_col, text=f"Total Duration: {total_sec:.1f}s",
            bg=bg_card, fg=TEXT_MUTED, font=("Helvetica", 10)
        ).pack(anchor="w", pady=(0, 6))

        tk.Label(
            stats_col, text="Cameras Visited:",
            bg=bg_card, fg=TEXT_SECONDARY, font=("Helvetica", 9, "bold")
        ).pack(anchor="w", pady=(0, 2))

        cam_tag_row = tk.Frame(stats_col, bg=bg_card)
        cam_tag_row.pack(anchor="w")

        cams = person.get("cameras_seen", [])
        if cams:
            for c in cams:
                tk.Label(
                    cam_tag_row, text=f" Cam {c} ", bg="#1e293b", fg=ACCENT_BLUE,
                    font=("Menlo", 9, "bold"), padx=4, pady=2
                ).pack(side=tk.LEFT, padx=(0, 4))
        else:
            tk.Label(cam_tag_row, text="None", bg=bg_card, fg=TEXT_MUTED, font=("Helvetica", 9)).pack()

        # Bottom Row
        btn_row = tk.Frame(card, bg=bg_card)
        btn_row.pack(fill=tk.X, pady=(6, 0))

        view_btn = tk.Button(
            btn_row, text="▶ View in Video",
            command=lambda g=gid: self._jump_to_person_video(g),
            bg="#1e293b", fg=TEXT_PRIMARY, font=("Helvetica", 9, "bold"),
            bd=0, activebackground="#334155", activeforeground="white",
            cursor="hand2", padx=8, pady=4
        )
        view_btn.pack(side=tk.LEFT, padx=(0, 6))

        if not is_merge_active:
            merge_btn = tk.Button(
                btn_row, text="🔗 Merge with...",
                command=lambda g=gid: self._start_merge(g),
                bg="#334155", fg=ACCENT_YELLOW, font=("Helvetica", 9, "bold"),
                bd=0, activebackground="#475569", activeforeground="white",
                cursor="hand2", padx=8, pady=4
            )
            merge_btn.pack(side=tk.RIGHT)
        elif is_merge_source:
            cancel_btn = tk.Button(
                btn_row, text="Cancel Selection",
                command=self._cancel_merge,
                bg="#dc2626", fg="white", font=("Helvetica", 9, "bold"),
                bd=0, activebackground="#b91c1c", activeforeground="white",
                cursor="hand2", padx=8, pady=4
            )
            cancel_btn.pack(side=tk.RIGHT)
        else:
            confirm_btn = tk.Button(
                btn_row, text=f"✓ Merge into ID:{gid:03d}",
                command=lambda g=gid: self._confirm_merge(self._merge_source_gid, g),
                bg=ACCENT_GREEN, fg="#000000", font=("Helvetica", 9, "bold"),
                bd=0, activebackground="#059669", activeforeground="white",
                cursor="hand2", padx=8, pady=4
            )
            confirm_btn.pack(side=tk.RIGHT)

    # ── Merge ID Functionality ────────────────────────────────────────────────

    def _start_merge(self, source_gid: int) -> None:
        self._merge_source_gid = source_gid
        self._rebuild_gallery_grid()

    def _cancel_merge(self) -> None:
        self._merge_source_gid = None
        self._rebuild_gallery_grid()

    def _confirm_merge(self, source_gid: int, target_gid: int) -> None:
        if source_gid == target_gid:
            return

        src_dets = self.persons.get(source_gid, {}).get("total_detections", 0)
        tgt_dets = self.persons.get(target_gid, {}).get("total_detections", 0)

        msg = (
            f"Are you sure you want to merge ID:{source_gid:03d} into ID:{target_gid:03d}?\n\n"
            f"• ID:{source_gid:03d} ({src_dets} detections) will be combined into ID:{target_gid:03d} ({tgt_dets} detections).\n"
            f"• All video bounding boxes, appearances, and timelines will be reassigned.\n"
            f"• Changes will be saved permanently to tracking_log.json."
        )

        confirm = messagebox.askyesno(
            title="Confirm Person ID Merge",
            message=msg,
            parent=self.root
        )

        if confirm:
            self._execute_merge(source_gid, target_gid)

    def _execute_merge(self, source_gid: int, target_gid: int) -> None:
        print(f"[ReviewUI] 🔄 Merging Person ID:{source_gid:03d} → ID:{target_gid:03d} …")

        source_person = self.persons.get(source_gid)
        target_person = self.persons.get(target_gid)

        if not source_person or not target_person:
            messagebox.showerror("Merge Error", "One of the selected person records was not found.", parent=self.root)
            self._cancel_merge()
            return

        # 1. Merge appearances
        target_apps = target_person.get("appearances", [])
        source_apps = source_person.get("appearances", [])
        combined_apps = target_apps + source_apps
        combined_apps.sort(key=lambda a: (a.get("camera_id", 0), a.get("first_frame", 0)))
        target_person["appearances"] = combined_apps

        # 2. Merge cameras seen
        cams_seen = set(target_person.get("cameras_seen", [])) | set(source_person.get("cameras_seen", []))
        target_person["cameras_seen"] = sorted(list(cams_seen))

        # 3. Sum total detections
        target_person["total_detections"] = (
            target_person.get("total_detections", 0) + source_person.get("total_detections", 0)
        )

        # 4. Use best thumbnail
        if not target_person.get("thumbnail_b64") and source_person.get("thumbnail_b64"):
            target_person["thumbnail_b64"] = source_person["thumbnail_b64"]

        # 5. Update per-frame detection boxes in memory
        for cam_id_str, frames_map in self.detections.items():
            for frame_str, det_list in frames_map.items():
                for det in det_list:
                    if det.get("gid") == source_gid or det.get("global_id") == source_gid:
                        det["gid"] = target_gid
                        det["global_id"] = target_gid

        # 6. Remove source person
        del self.persons[source_gid]
        if str(source_gid) in self.tracking_data.get("persons", {}):
            del self.tracking_data["persons"][str(source_gid)]

        # 7. Update tracking_data dictionary
        self.tracking_data["persons"][str(target_gid)] = target_person
        self.tracking_data["detections"] = self.detections

        # 8. Save updated log to JSON file
        log_path = self.output_dir / "tracking_log.json"
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(self.tracking_data, f, indent=2)
            print(f"[ReviewUI] ✓ Saved updated tracking log to {log_path}")
        except Exception as e:
            print(f"[ReviewUI] ⚠️ Error saving tracking log: {e}")

        # Reset merge state
        self._merge_source_gid = None

        # Rebuild UI elements
        self._rebuild_gallery_grid()
        self._build_person_cards()
        self._status_label.configure(
            text=f"  {len(self.persons)} unique persons  •  {self._n_cameras} cameras  •  {self._max_frame + 1} frames"
        )
        self._update_display()

        # Feedback Toast
        messagebox.showinfo(
            "Merge Successful",
            f"✓ Successfully merged Person ID:{source_gid:03d} into ID:{target_gid:03d}!\nUpdated data saved to tracking_log.json.",
            parent=self.root
        )

    def _jump_to_person_video(self, gid: int) -> None:
        person = self.persons.get(gid, {})
        apps = person.get("appearances", [])
        if apps:
            first_app = apps[0]
            first_frame = first_app.get("first_frame", 0)
            cam_id = first_app.get("camera_id")
            self._seek_to_frame(first_frame)
            if cam_id is not None:
                self._view_mode = "single"
                self._selected_cam = cam_id
                self._tracked_gid = gid
        self._show_video_section()
        self._apply_view_layout()

    # ── View Layout & Playback Controls ────────────────────────────────────────

    def _set_view_grid(self) -> None:
        self._view_mode = "grid"
        self._selected_cam = None
        self._tracked_gid = None
        self._apply_view_layout()
        self._rebuild_cards()

    def _set_view_camera(self, cam_id: int) -> None:
        self._view_mode = "single"
        self._selected_cam = cam_id
        self._tracked_gid = None
        self._apply_view_layout()
        self._rebuild_cards()

    def _toggle_camera_fullscreen(self, cam_id: int) -> None:
        if self._view_mode == "single" and self._selected_cam == cam_id:
            self._set_view_grid()
        else:
            self._set_view_camera(cam_id)

    def _get_person_current_camera(self, gid: int, frame_idx: int) -> Optional[int]:
        person = self.persons.get(gid, {})
        for app in person.get("appearances", []):
            if app["first_frame"] <= frame_idx <= app["last_frame"]:
                return app["camera_id"]
        return None

    def _on_person_clicked(self, gid: int) -> None:
        if self._tracked_gid == gid:
            self._stop_tracking()
            return

        self._tracked_gid = gid
        cam_id = self._get_person_current_camera(gid, self._current_frame)
        if cam_id is not None:
            self._view_mode = "single"
            self._selected_cam = cam_id

        self._apply_view_layout()
        self._rebuild_cards()

    def _stop_tracking(self) -> None:
        self._tracked_gid = None
        self._update_tracking_status_ui()
        self._rebuild_cards()

    def _update_tracking_status_ui(self) -> None:
        if self._tracked_gid is not None:
            gid = self._tracked_gid
            cam_id = self._get_person_current_camera(gid, self._current_frame)
            if cam_id is not None:
                text = f"🎯 Auto-Following ID:{gid:03d} on CAM {cam_id}"
                bg = BG_TRACKED
                fg = BORDER_TRACKED
            else:
                text = f"🎯 Auto-Following ID:{gid:03d} (Waiting for detection...)"
                bg = "#334155"
                fg = TEXT_SECONDARY

            self._tracking_badge.configure(text=text, bg=bg, fg=fg)
            self._tracking_badge.pack(side=tk.LEFT, padx=(0, 6))
            self._tracking_stop_btn.pack(side=tk.LEFT)
        else:
            self._tracking_badge.pack_forget()
            self._tracking_stop_btn.pack_forget()

    # ── Sidebar Person Cards ──────────────────────────────────────────────────

    def _build_person_cards(self) -> None:
        for widget in self._person_list_frame.winfo_children():
            widget.destroy()
        self._thumb_refs.clear()

        for gid in sorted(self.persons.keys()):
            person = self.persons[gid]
            self._build_one_person_card(person)

    def _build_one_person_card(self, person: dict) -> None:
        gid = person["global_id"]
        color_hex = person.get("color_hex", "#64748b")
        is_expanded = gid in self._expanded
        is_tracked = (self._tracked_gid == gid)

        active_cam = self._get_person_current_camera(gid, self._current_frame)
        is_active = (active_cam is not None)

        if is_tracked:
            border_color = BORDER_TRACKED
            bg = BG_TRACKED
        elif is_active:
            border_color = ACCENT_GREEN
            bg = BG_ACTIVE
        else:
            border_color = BORDER
            bg = BG_CARD

        outer = tk.Frame(self._person_list_frame, bg=border_color, bd=0)
        outer.pack(fill=tk.X, padx=6, pady=3)

        card = tk.Frame(outer, bg=bg, bd=0, cursor="hand2")
        card.pack(fill=tk.X, padx=(3, 1), pady=1)

        main_row = tk.Frame(card, bg=bg, cursor="hand2")
        main_row.pack(fill=tk.X, padx=8, pady=8)

        expand_text = "▼" if is_expanded else "▶"
        expand_label = tk.Label(
            main_row, text=expand_text, bg=bg, fg=TEXT_MUTED,
            font=("Helvetica", 10), cursor="hand2"
        )
        expand_label.pack(side=tk.LEFT, padx=(0, 6))
        expand_label.bind("<Button-1>", lambda e, g=gid: self._toggle_expand(g))

        thumb_frame = tk.Frame(main_row, bg=border_color, width=40, height=52)
        thumb_frame.pack_propagate(False)
        thumb_frame.pack(side=tk.LEFT, padx=(0, 8))

        if person.get("thumbnail_b64"):
            try:
                raw = base64.b64decode(person["thumbnail_b64"])
                pil_img = Image.open(io.BytesIO(raw))
                pil_img = pil_img.resize((38, 50), Image.LANCZOS)
                photo = ImageTk.PhotoImage(pil_img)
                thumb_label = tk.Label(thumb_frame, image=photo, bg=BORDER, bd=0)
                thumb_label.pack(fill=tk.BOTH, expand=True)
                self._thumb_refs[gid] = photo
            except Exception:
                tk.Label(thumb_frame, text="👤", bg=BG_CARD, fg=TEXT_MUTED,
                         font=("Helvetica", 16)).pack(fill=tk.BOTH, expand=True)
        else:
            tk.Label(thumb_frame, text="👤", bg=BG_CARD, fg=TEXT_MUTED,
                     font=("Helvetica", 16)).pack(fill=tk.BOTH, expand=True)

        info = tk.Frame(main_row, bg=bg, cursor="hand2")
        info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        row1 = tk.Frame(info, bg=bg, cursor="hand2")
        row1.pack(fill=tk.X)

        swatch = tk.Frame(row1, bg=color_hex, width=12, height=12)
        swatch.pack(side=tk.LEFT, padx=(0, 6), pady=2)
        swatch.pack_propagate(False)

        id_text = f"ID:{gid:03d}"
        id_label = tk.Label(
            row1, text=id_text, bg=bg, fg=TEXT_PRIMARY, font=("Menlo", 12, "bold")
        )
        id_label.pack(side=tk.LEFT)

        tk.Label(
            row1, text=f"×{person.get('total_detections', 0)}",
            bg=bg, fg=TEXT_MUTED, font=("Menlo", 10)
        ).pack(side=tk.RIGHT)

        if is_active or is_tracked:
            row2 = tk.Frame(info, bg=bg, cursor="hand2")
            row2.pack(fill=tk.X, pady=(2, 0))

            if is_active:
                live_frame = tk.Frame(row2, bg="#134e2e", bd=0)
                live_frame.pack(side=tk.LEFT, padx=(0, 4))
                tk.Label(
                    live_frame, text=f" ● LIVE (Cam {active_cam}) ",
                    bg="#134e2e", fg=ACCENT_GREEN,
                    font=("Menlo", 8, "bold"), padx=2, pady=1
                ).pack()

            if is_tracked:
                track_badge = tk.Label(
                    row2, text=" 🎯 FOLLOWING ",
                    bg="#312e81", fg=BORDER_TRACKED,
                    font=("Menlo", 8, "bold"), padx=4, pady=1
                )
                track_badge.pack(side=tk.LEFT)

        cam_row = tk.Frame(info, bg=bg, cursor="hand2")
        cam_row.pack(fill=tk.X, pady=(3, 0))
        for c in person.get("cameras_seen", []):
            tk.Label(
                cam_row, text=f" Cam {c} ", bg="#1e293b", fg=TEXT_MUTED,
                font=("Menlo", 8, "bold"), padx=3, pady=1
            ).pack(side=tk.LEFT, padx=(0, 4))

        for widget in [card, main_row, thumb_frame, info, row1, id_label, cam_row]:
            widget.bind("<Button-1>", lambda e, g=gid: self._on_person_clicked(g))

        if is_expanded:
            timeline = tk.Frame(outer, bg=BG_EXPANDED, bd=0)
            timeline.pack(fill=tk.X, padx=(3, 1), pady=(0, 1))

            appearances = person.get("appearances", [])
            if not appearances:
                tk.Label(timeline, text="  No appearances recorded",
                         bg=BG_EXPANDED, fg=TEXT_MUTED,
                         font=("Helvetica", 10), pady=8).pack(anchor="w")
            else:
                for i, app in enumerate(appearances):
                    self._build_appearance_row(timeline, app, gid, i)

    def _build_appearance_row(self, parent: tk.Frame, app: dict, gid: int, idx: int) -> None:
        row = tk.Frame(parent, bg=BG_EXPANDED, cursor="hand2")
        row.pack(fill=tk.X, padx=12, pady=2)

        cam_id = app["camera_id"]
        first_ts = app.get("first_timestamp", "?")
        last_ts = app.get("last_timestamp", "?")
        duration = app.get("duration_seconds", 0)
        first_frame = app.get("first_frame", 0)

        tk.Label(row, text=f"📷 Cam {cam_id}", bg=BG_EXPANDED, fg=ACCENT_BLUE,
                 font=("Menlo", 10, "bold")).pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(row, text=f"{first_ts} – {last_ts}", bg=BG_EXPANDED, fg=TEXT_SECONDARY,
                 font=("Menlo", 10)).pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(row, text=f"({duration}s)", bg=BG_EXPANDED, fg=TEXT_MUTED,
                 font=("Menlo", 9)).pack(side=tk.LEFT)

        go_btn = tk.Label(row, text=" ▶ GO ", bg="#1e293b", fg=ACCENT_YELLOW,
                          font=("Menlo", 9, "bold"), cursor="hand2", padx=4)
        go_btn.pack(side=tk.RIGHT, padx=4)

        def _seek_and_follow(e=None, f=first_frame, c=cam_id, g=gid):
            self._tracked_gid = g
            self._view_mode = "single"
            self._selected_cam = c
            self._seek_to_frame(f)
            self._apply_view_layout()

        for widget in [row, go_btn]:
            widget.bind("<Button-1>", _seek_and_follow)

    def _toggle_expand(self, gid: int) -> None:
        if gid in self._expanded:
            self._expanded.remove(gid)
        else:
            self._expanded.add(gid)
        self._build_person_cards()

    def _rebuild_cards(self) -> None:
        self._build_person_cards()

    # ── Playback Controls ─────────────────────────────────────────────────────

    def _toggle_play(self) -> None:
        self._playing = not self._playing
        self._btn_play.configure(text=" ⏸ " if self._playing else " ▶ ")

    def _step_back(self) -> None:
        self._playing = False
        self._btn_play.configure(text=" ▶ ")
        self._seek_to_frame(max(0, self._current_frame - 1))

    def _step_forward(self) -> None:
        self._playing = False
        self._btn_play.configure(text=" ▶ ")
        self._seek_to_frame(min(self._max_frame, self._current_frame + 1))

    def _on_seek(self, value: str) -> None:
        self._seek_to_frame(int(value))

    def _seek_to_frame(self, frame_idx: int) -> None:
        self._current_frame = frame_idx
        self._seek_var.set(frame_idx)
        self._handle_auto_tracking()
        self._update_display()

    def _handle_auto_tracking(self) -> None:
        if self._tracked_gid is not None:
            cam_id = self._get_person_current_camera(self._tracked_gid, self._current_frame)
            if cam_id is not None:
                if self._selected_cam != cam_id or self._view_mode != "single":
                    self._view_mode = "single"
                    self._selected_cam = cam_id
                    self._apply_view_layout()

    def _update_display(self) -> None:
        """Render current video frames to the camera widgets."""
        if self._main_section != "video":
            return

        cams_to_render = (
            [self._selected_cam]
            if self._view_mode == "single" and self._selected_cam is not None
            else list(self._caps.keys())
        )

        for cam_id in cams_to_render:
            label = self._cam_labels.get(cam_id)
            if label is None:
                continue

            frame = self._read_frame(cam_id, self._current_frame)
            if frame is None:
                continue

            w = max(label.winfo_width(), 100)
            h = max(label.winfo_height(), 100)

            fh, fw = frame.shape[:2]
            scale = min(w / fw, h / fh)
            nw = max(1, int(fw * scale))
            nh = max(1, int(fh * scale))

            resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            photo = ImageTk.PhotoImage(image=img)

            label.configure(image=photo)
            label.image = photo
            self._photo_refs[cam_id] = photo

        self._frame_label.configure(text=f"Frame: {self._current_frame}")
        current_ts = self._frame_to_timestamp(self._current_frame)
        total_ts = self._frame_to_timestamp(self._max_frame)
        self._time_label.configure(text=f"{current_ts} / {total_ts}")
        self._update_tracking_status_ui()

    def _frame_to_timestamp(self, frame: int) -> str:
        if self._playback_fps <= 0:
            return "0:00"
        total = int(frame / self._playback_fps)
        m, s = divmod(total, 60)
        return f"{m}:{s:02d}"

    def _on_mousewheel(self, event):
        self._canvas.yview_scroll(-1 * (event.delta // 120 or event.delta), "units")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        """Main update loop."""
        if self._playing:
            next_frame = self._current_frame + 1
            if next_frame > self._max_frame:
                self._playing = False
                self._btn_play.configure(text=" ▶ ")
            else:
                self._current_frame = next_frame
                self._seek_var.set(self._current_frame)
                self._handle_auto_tracking()
                self._update_display()

                if self._current_frame % 15 == 0:
                    self._rebuild_cards()

        delay = max(10, int(1000 / self._playback_fps))
        self.root.after(delay, self._tick)

    def run(self) -> None:
        """Start the review UI."""
        print(f"[ReviewUI] ✦ Review dashboard opened  ({len(self.persons)} persons, {self._n_cameras} cameras)")
        self._update_display()
        self.root.after(100, self._tick)
        self.root.mainloop()

        for cap in self._caps.values():
            cap.release()
        for cap in self._raw_caps.values():
            cap.release()

    def destroy(self) -> None:
        try:
            self.root.destroy()
        except tk.TclError:
            pass


if __name__ == "__main__":
    from src.config import OUTPUT_DIR
    from src.tracking_log import TrackingLog

    log_path = OUTPUT_DIR / "tracking_log.json"
    if not log_path.exists():
        print(f"[ReviewUI] No tracking log found at {log_path}")
        print(f"           Run pipeline first: python -m src.main")
        exit(1)

    print(f"[ReviewUI] Loading tracking log from {log_path} …")
    tracking_data = TrackingLog.load(str(log_path))
    ui = ReviewUI(tracking_data=tracking_data, output_dir=str(OUTPUT_DIR))
    ui.run()
