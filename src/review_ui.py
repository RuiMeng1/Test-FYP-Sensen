"""
review_ui.py — Post-processing review dashboard for multi-camera person tracking.

Opens after processing is complete. Loads the annotated output videos and
tracking_log.json, providing:
  - Video playback with play/pause, frame stepping, and seek slider
  - Person logger with expandable per-person appearance timelines
  - Click-to-seek: clicking a person or appearance row seeks all cameras
    to the frame where that person appears

Runs on the main thread (macOS Cocoa/Tk requirement).
"""

from __future__ import annotations

import base64
import io
import json
import threading
import time
import tkinter as tk
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk


# ── Colour constants ─────────────────────────────────────────────────────────
BG_PRIMARY   = "#0a0e1a"
BG_SECONDARY = "#111827"
BG_CARD      = "#151d2e"
BG_ACTIVE    = "#0d2818"
BG_EXPANDED  = "#0f1729"
BORDER       = "#1e293b"
BORDER_ACTIVE = "#10b981"
TEXT_PRIMARY  = "#f1f5f9"
TEXT_SECONDARY = "#94a3b8"
TEXT_MUTED    = "#64748b"
ACCENT_GREEN  = "#10b981"
ACCENT_BLUE   = "#3b82f6"
ACCENT_RED    = "#ef4444"
ACCENT_YELLOW = "#f59e0b"


class ReviewUI:
    """
    Post-processing review dashboard.

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
            int(k): v for k, v in tracking_data["cameras"].items()
        }
        self.persons: Dict[int, dict] = {
            int(k): v for k, v in tracking_data["persons"].items()
        }

        # Video captures
        self._caps: Dict[int, cv2.VideoCapture] = {}
        self._n_cameras = len(self.cameras)

        # Playback state
        self._playing = False
        self._current_frame = 0
        self._max_frame = 0
        self._playback_fps = 25.0

        # PhotoImage references (prevent GC)
        self._photo_refs: Dict[int, ImageTk.PhotoImage] = {}
        self._thumb_refs: Dict[int, ImageTk.PhotoImage] = {}

        # Expanded person cards
        self._expanded: Set[int] = set()

        # Open video files
        self._open_videos()
        self._build_ui()

    # ── Video management ──────────────────────────────────────────────────────

    def _open_videos(self) -> None:
        """Open all output video files."""
        for cam_id, meta in self.cameras.items():
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
            else:
                print(f"[ReviewUI] Warning: Video not found: {video_path}")

    def _read_frame(self, cam_id: int, frame_idx: int) -> Optional[np.ndarray]:
        """Read a specific frame from a video."""
        cap = self._caps.get(cam_id)
        if cap is None:
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        return frame if ret else None

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.root = tk.Tk()
        self.root.title("Sensen — Review Dashboard")
        self.root.configure(bg=BG_PRIMARY)
        self.root.minsize(1200, 700)

        # Default window size
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        win_w = min(1500, screen_w - 80)
        win_h = min(950, screen_h - 80)
        x = (screen_w - win_w) // 2
        y = (screen_h - win_h) // 2
        self.root.geometry(f"{win_w}x{win_h}+{x}+{y}")

        # ── Header ────────────────────────────────────────────────────────────
        header = tk.Frame(self.root, bg="#0f172a", height=48)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)

        logo = tk.Label(
            header, text=" S ", bg=ACCENT_BLUE, fg="white",
            font=("Helvetica", 12, "bold"), padx=4, pady=2
        )
        logo.pack(side=tk.LEFT, padx=(16, 8), pady=8)

        tk.Label(
            header, text="Sensen — Review Dashboard",
            bg="#0f172a", fg=TEXT_PRIMARY,
            font=("Helvetica", 15, "bold")
        ).pack(side=tk.LEFT, pady=8)

        self._time_label = tk.Label(
            header, text="0:00 / 0:00",
            bg="#0f172a", fg=TEXT_SECONDARY,
            font=("Menlo", 11)
        )
        self._time_label.pack(side=tk.RIGHT, padx=16, pady=8)

        # ── Main content ──────────────────────────────────────────────────────
        content = tk.Frame(self.root, bg=BG_PRIMARY)
        content.pack(fill=tk.BOTH, expand=True)

        # Left: Camera feeds + playback controls
        left_panel = tk.Frame(content, bg=BG_PRIMARY)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Camera grid
        self._camera_frame = tk.Frame(left_panel, bg=BG_PRIMARY)
        self._camera_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 4))

        # Playback controls
        controls = tk.Frame(left_panel, bg="#0f172a", height=56)
        controls.pack(fill=tk.X, padx=8, pady=(0, 8))
        controls.pack_propagate(False)
        self._build_controls(controls)

        # Right: Logger panel
        logger_panel = tk.Frame(content, bg=BG_SECONDARY, width=400)
        logger_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=8)
        logger_panel.pack_propagate(False)
        self._build_logger(logger_panel)

        # ── Camera tiles ──────────────────────────────────────────────────────
        self._cam_labels: Dict[int, tk.Label] = {}
        self._cam_overlays: Dict[int, tk.Label] = {}
        self._build_camera_grid()

        # ── Status bar ────────────────────────────────────────────────────────
        status_bar = tk.Frame(self.root, bg="#0f172a", height=32)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        status_bar.pack_propagate(False)

        self._status_label = tk.Label(
            status_bar,
            text=f"  {len(self.persons)} persons tracked  •  {self._n_cameras} cameras  •  {self._max_frame + 1} frames",
            bg="#0f172a", fg=TEXT_MUTED, font=("Menlo", 10),
            anchor="w"
        )
        self._status_label.pack(side=tk.LEFT, padx=12, pady=4)

        tk.Label(
            status_bar, text="Sensen Multi-Camera Tracking System",
            bg="#0f172a", fg="#334155", font=("Helvetica", 10)
        ).pack(side=tk.RIGHT, padx=16, pady=4)

    def _build_controls(self, parent: tk.Frame) -> None:
        """Build playback controls: step back, play/pause, step forward, seek slider."""
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

        # Frame counter
        self._frame_label = tk.Label(
            parent, text="Frame: 0", bg="#0f172a", fg=TEXT_MUTED,
            font=("Menlo", 10)
        )
        self._frame_label.pack(side=tk.LEFT, padx=(12, 8))

        # Seek slider
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
        """Build the person logger panel."""
        # Header
        header = tk.Frame(parent, bg=BG_SECONDARY)
        header.pack(fill=tk.X, padx=12, pady=(12, 8))

        tk.Label(
            header, text="👁  Person Logger",
            bg=BG_SECONDARY, fg=TEXT_PRIMARY,
            font=("Helvetica", 13, "bold")
        ).pack(side=tk.LEFT)

        self._count_label = tk.Label(
            header, text=str(len(self.persons)),
            bg=ACCENT_BLUE, fg="white",
            font=("Menlo", 11, "bold"), padx=8, pady=1
        )
        self._count_label.pack(side=tk.RIGHT)

        # Separator
        tk.Frame(parent, bg=BORDER, height=1).pack(fill=tk.X, padx=8)

        # Scrollable person list
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

        # Build initial person cards
        self._build_person_cards()

    def _build_camera_grid(self) -> None:
        """Create camera tiles in a 2-column grid."""
        cols = 2
        cam_ids = sorted(self._caps.keys())
        n_rows = (len(cam_ids) + cols - 1) // cols

        for r in range(n_rows):
            self._camera_frame.grid_rowconfigure(r, weight=1, uniform="cam_row", minsize=200)
        for c in range(cols):
            self._camera_frame.grid_columnconfigure(c, weight=1, uniform="cam_col", minsize=300)

        for idx, cam_id in enumerate(cam_ids):
            row, col = divmod(idx, cols)

            card = tk.Frame(self._camera_frame, bg=BG_SECONDARY, bd=0)
            card.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
            card.grid_rowconfigure(0, weight=1)
            card.grid_columnconfigure(0, weight=1)

            cam_label = tk.Label(card, bg="black", bd=0, anchor="center")
            cam_label.grid(row=0, column=0, sticky="nsew")
            self._cam_labels[cam_id] = cam_label

            overlay = tk.Label(
                cam_label, text=f"  ● CAM {cam_id}  ",
                bg="#000000", fg="white",
                font=("Menlo", 10, "bold"),
            )
            overlay.place(x=8, y=8)
            self._cam_overlays[cam_id] = overlay

    # ── Person cards ──────────────────────────────────────────────────────────

    def _build_person_cards(self) -> None:
        """Build all person cards in the logger."""
        for widget in self._person_list_frame.winfo_children():
            widget.destroy()
        self._thumb_refs.clear()

        for gid in sorted(self.persons.keys()):
            person = self.persons[gid]
            self._build_one_person_card(person)

    def _build_one_person_card(self, person: dict) -> None:
        """Build a single person card with expandable timeline."""
        gid = person["global_id"]
        color_hex = person.get("color_hex", "#64748b")
        is_expanded = gid in self._expanded

        # Check if person is currently visible
        is_active = self._is_person_visible(gid, self._current_frame)
        bg = BG_ACTIVE if is_active else BG_CARD
        border_color = ACCENT_GREEN if is_active else BORDER

        # Outer frame
        outer = tk.Frame(self._person_list_frame, bg=border_color, bd=0)
        outer.pack(fill=tk.X, padx=6, pady=3)

        # Inner card (clickable)
        card = tk.Frame(outer, bg=bg, bd=0, cursor="hand2")
        card.pack(fill=tk.X, padx=(3, 1), pady=1)

        # ── Main row ──────────────────────────────────────────────────────────
        main_row = tk.Frame(card, bg=bg, cursor="hand2")
        main_row.pack(fill=tk.X, padx=8, pady=8)

        # Expand indicator
        expand_text = "▼" if is_expanded else "▶"
        expand_label = tk.Label(
            main_row, text=expand_text, bg=bg, fg=TEXT_MUTED,
            font=("Helvetica", 10), cursor="hand2"
        )
        expand_label.pack(side=tk.LEFT, padx=(0, 6))

        # Thumbnail
        thumb_frame = tk.Frame(main_row, bg=BORDER, width=40, height=52)
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

        # Info column
        info = tk.Frame(main_row, bg=bg)
        info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Top: swatch + ID + LIVE badge + hit count
        top = tk.Frame(info, bg=bg)
        top.pack(fill=tk.X)

        swatch = tk.Frame(top, bg=color_hex, width=12, height=12)
        swatch.pack(side=tk.LEFT, padx=(0, 6))
        swatch.pack_propagate(False)

        tk.Label(top, text=f"ID:{gid:03d}", bg=bg, fg=TEXT_PRIMARY,
                 font=("Menlo", 13, "bold")).pack(side=tk.LEFT)

        if is_active:
            live_frame = tk.Frame(top, bg="#134e2e", bd=0)
            live_frame.pack(side=tk.LEFT, padx=(8, 0))
            tk.Label(live_frame, text=" ● LIVE ", bg="#134e2e", fg=ACCENT_GREEN,
                     font=("Menlo", 8, "bold")).pack()

        tk.Label(top, text=f"×{person.get('total_detections', 0)}",
                 bg=bg, fg=TEXT_MUTED, font=("Menlo", 11)).pack(side=tk.RIGHT)

        # Camera tags
        cam_row = tk.Frame(info, bg=bg)
        cam_row.pack(fill=tk.X, pady=(3, 0))
        for c in person.get("cameras_seen", []):
            tk.Label(cam_row, text=f" Cam {c} ", bg="#1e293b", fg=TEXT_MUTED,
                     font=("Menlo", 9, "bold"), padx=2).pack(side=tk.LEFT, padx=(0, 4))

        # Make the whole card clickable (toggle expand)
        for widget in [card, main_row, expand_label, info, top, cam_row]:
            widget.bind("<Button-1>", lambda e, g=gid: self._toggle_expand(g))

        # ── Expanded timeline ─────────────────────────────────────────────────
        if is_expanded:
            timeline = tk.Frame(outer, bg=BG_EXPANDED, bd=0)
            timeline.pack(fill=tk.X, padx=(3, 1), pady=(0, 1))

            appearances = person.get("appearances", [])
            if not appearances:
                tk.Label(timeline, text="  No appearances recorded",
                         bg=BG_EXPANDED, fg=TEXT_MUTED,
                         font=("Helvetica", 10), pady=8).pack(anchor="w")
            else:
                # Group appearances by camera
                for i, app in enumerate(appearances):
                    self._build_appearance_row(timeline, app, gid, i)

    def _build_appearance_row(self, parent: tk.Frame, app: dict, gid: int, idx: int) -> None:
        """Build a single clickable appearance row in the expanded timeline."""
        row = tk.Frame(parent, bg=BG_EXPANDED, cursor="hand2")
        row.pack(fill=tk.X, padx=12, pady=2)

        cam_id = app["camera_id"]
        first_ts = app.get("first_timestamp", "?")
        last_ts = app.get("last_timestamp", "?")
        duration = app.get("duration_seconds", 0)
        first_frame = app.get("first_frame", 0)

        # Camera indicator
        tk.Label(row, text=f"📷 Cam {cam_id}", bg=BG_EXPANDED, fg=ACCENT_BLUE,
                 font=("Menlo", 10, "bold")).pack(side=tk.LEFT, padx=(0, 8))

        # Time range
        tk.Label(row, text=f"{first_ts} – {last_ts}", bg=BG_EXPANDED, fg=TEXT_SECONDARY,
                 font=("Menlo", 10)).pack(side=tk.LEFT, padx=(0, 8))

        # Duration
        tk.Label(row, text=f"({duration}s)", bg=BG_EXPANDED, fg=TEXT_MUTED,
                 font=("Menlo", 9)).pack(side=tk.LEFT)

        # "Go to" button
        go_btn = tk.Label(row, text=" ▶ GO ", bg="#1e293b", fg=ACCENT_YELLOW,
                          font=("Menlo", 9, "bold"), cursor="hand2", padx=4)
        go_btn.pack(side=tk.RIGHT, padx=4)

        # Click to seek
        seek_frame = first_frame
        for widget in [row, go_btn]:
            widget.bind("<Button-1>", lambda e, f=seek_frame: self._seek_to_frame(f))

    def _is_person_visible(self, gid: int, frame_idx: int) -> bool:
        """Check if a person is visible at the current frame based on their appearances."""
        person = self.persons.get(gid, {})
        for app in person.get("appearances", []):
            if app["first_frame"] <= frame_idx <= app["last_frame"]:
                return True
        return False

    def _toggle_expand(self, gid: int) -> None:
        """Toggle expand/collapse for a person card."""
        if gid in self._expanded:
            self._expanded.discard(gid)
        else:
            self._expanded.add(gid)
        self._rebuild_cards()

    def _rebuild_cards(self) -> None:
        """Rebuild all person cards (preserving expand state)."""
        scroll_pos = self._canvas.yview()
        self._build_person_cards()
        self.root.update_idletasks()
        self._canvas.yview_moveto(scroll_pos[0])

    # ── Playback controls ─────────────────────────────────────────────────────

    def _toggle_play(self) -> None:
        if self._playing:
            self._playing = False
            self._btn_play.configure(text=" ▶ ")
        else:
            self._playing = True
            self._btn_play.configure(text=" ⏸ ")

    def _step_back(self) -> None:
        self._playing = False
        self._btn_play.configure(text=" ▶ ")
        self._seek_to_frame(max(0, self._current_frame - 1))

    def _step_forward(self) -> None:
        self._playing = False
        self._btn_play.configure(text=" ▶ ")
        self._seek_to_frame(min(self._max_frame, self._current_frame + 1))

    def _on_seek(self, value) -> None:
        frame = int(value)
        if frame != self._current_frame:
            self._seek_to_frame(frame)

    def _seek_to_frame(self, frame_idx: int) -> None:
        """Seek all cameras to a specific frame."""
        self._current_frame = max(0, min(frame_idx, self._max_frame))
        self._seek_var.set(self._current_frame)
        self._update_display()
        self._rebuild_cards()

    # ── Display update ────────────────────────────────────────────────────────

    def _update_display(self) -> None:
        """Update camera feeds and UI for the current frame."""
        for cam_id, cap in self._caps.items():
            frame = self._read_frame(cam_id, self._current_frame)
            if frame is None:
                continue

            label = self._cam_labels.get(cam_id)
            if label is None:
                continue

            lw = label.winfo_width()
            lh = label.winfo_height()
            if lw < 10 or lh < 10:
                lw, lh = 540, 304

            fh, fw = frame.shape[:2]
            scale = min(lw / fw, lh / fh)
            new_w = max(1, int(fw * scale))
            new_h = max(1, int(fh * scale))

            resized = cv2.resize(frame, (new_w, new_h))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            photo = ImageTk.PhotoImage(img)

            label.configure(image=photo)
            self._photo_refs[cam_id] = photo

        # Update frame counter and time
        self._frame_label.configure(text=f"Frame: {self._current_frame}")
        current_ts = self._frame_to_timestamp(self._current_frame)
        total_ts = self._frame_to_timestamp(self._max_frame)
        self._time_label.configure(text=f"{current_ts} / {total_ts}")

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
        """Main update loop, called from tkinter's after()."""
        if self._playing:
            next_frame = self._current_frame + 1
            if next_frame > self._max_frame:
                self._playing = False
                self._btn_play.configure(text=" ▶ ")
            else:
                self._current_frame = next_frame
                self._seek_var.set(self._current_frame)
                self._update_display()

                # Rebuild person cards every 15 frames (for active highlighting)
                if self._current_frame % 15 == 0:
                    self._rebuild_cards()

        # Schedule at playback FPS
        delay = max(10, int(1000 / self._playback_fps))
        self.root.after(delay, self._tick)

    def run(self) -> None:
        """Start the review UI. MUST be called from the main thread."""
        print(f"[ReviewUI] ✦ Review dashboard opened  ({len(self.persons)} persons, {self._n_cameras} cameras)")

        # Show initial frame
        self._update_display()

        # Start tick loop
        self.root.after(100, self._tick)
        self.root.mainloop()

        # Cleanup
        for cap in self._caps.values():
            cap.release()

    def destroy(self) -> None:
        try:
            self.root.destroy()
        except tk.TclError:
            pass
