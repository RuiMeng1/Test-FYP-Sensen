"""
tk_ui.py — Native tkinter dashboard for live multi-camera person tracking.

Replaces the Flask web server with a zero-overhead native GUI:
  - Camera feeds rendered directly from numpy arrays (no JPEG encoding)
  - Person logger panel with colour swatches, thumbnails, LIVE badges
  - Dark theme matching the web UI design

The UI runs on the main thread (macOS requirement for Cocoa/Tk).
CameraWorkers push frames via a thread-safe queue.
"""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

from src.pipeline import active_ids_by_camera, get_color_hex


# ── Colour constants ─────────────────────────────────────────────────────────
BG_PRIMARY   = "#0a0e1a"
BG_SECONDARY = "#111827"
BG_CARD      = "#151d2e"
BG_ACTIVE    = "#0d2818"
BORDER       = "#1e293b"
BORDER_ACTIVE = "#10b981"
TEXT_PRIMARY  = "#f1f5f9"
TEXT_SECONDARY = "#94a3b8"
TEXT_MUTED    = "#64748b"
ACCENT_GREEN  = "#10b981"
ACCENT_BLUE   = "#3b82f6"
ACCENT_RED    = "#ef4444"


class TrackingUI:
    """
    Native tkinter dashboard for the multi-camera tracker.

    Must be created and run on the main thread (macOS Cocoa requirement).

    Parameters
    ----------
    gallery     : The shared GlobalGallery instance.
    n_cameras   : Total number of camera streams.
    tile_w      : Width of each camera tile.
    tile_h      : Height of each camera tile.
    """

    def __init__(self, gallery, n_cameras: int, tile_w: int = 540, tile_h: int = 304):
        self.gallery = gallery
        self.n_cameras = n_cameras
        self.tile_w = tile_w
        self.tile_h = tile_h

        # Thread-safe frame buffer: {cam_id: np.ndarray (BGR)}
        self._latest_frames: Dict[int, np.ndarray] = {}
        self._frame_lock = threading.Lock()

        # PhotoImage references (prevent GC)
        self._photo_refs: Dict[int, ImageTk.PhotoImage] = {}
        self._thumb_refs: Dict[int, ImageTk.PhotoImage] = {}

        # Tracking state for efficient updates
        self._last_person_data: Optional[str] = None

        self._build_ui()

    # ── Frame ingestion (called from CameraWorker threads) ────────────────────

    def push_frame(self, cam_id: int, frame: np.ndarray) -> None:
        """Store the latest annotated frame for a camera (thread-safe)."""
        with self._frame_lock:
            self._latest_frames[cam_id] = frame

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.root = tk.Tk()
        self.root.title("Sensen — Multi-Camera Tracker")
        self.root.configure(bg=BG_PRIMARY)
        self.root.minsize(1100, 600)

        # Try to set a reasonable default size
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        win_w = min(1440, screen_w - 100)
        win_h = min(900, screen_h - 100)
        x = (screen_w - win_w) // 2
        y = (screen_h - win_h) // 2
        self.root.geometry(f"{win_w}x{win_h}+{x}+{y}")

        # ── Configure ttk styles ─────────────────────────────────────────────
        style = ttk.Style()
        style.theme_use("clam")

        # ── Header ────────────────────────────────────────────────────────────
        header = tk.Frame(self.root, bg="#0f172a", height=48)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)

        # Logo
        logo = tk.Label(
            header, text=" S ", bg=ACCENT_GREEN, fg="white",
            font=("Helvetica", 12, "bold"), padx=4, pady=2
        )
        logo.pack(side=tk.LEFT, padx=(16, 8), pady=8)

        title = tk.Label(
            header, text="Sensen — Multi-Camera Tracker",
            bg="#0f172a", fg=TEXT_PRIMARY,
            font=("Helvetica", 15, "bold")
        )
        title.pack(side=tk.LEFT, pady=8)

        # Status pill
        self._status_label = tk.Label(
            header, text="● Live", bg="#0f172a", fg=ACCENT_GREEN,
            font=("Helvetica", 11)
        )
        self._status_label.pack(side=tk.RIGHT, padx=16, pady=8)

        # ── Main content area ─────────────────────────────────────────────────
        content = tk.Frame(self.root, bg=BG_PRIMARY)
        content.pack(fill=tk.BOTH, expand=True)

        # Left: Camera feeds
        self._camera_frame = tk.Frame(content, bg=BG_PRIMARY)
        self._camera_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Right: Logger panel
        logger_panel = tk.Frame(content, bg=BG_SECONDARY, width=360)
        logger_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=8)
        logger_panel.pack_propagate(False)

        # Logger header
        logger_header = tk.Frame(logger_panel, bg=BG_SECONDARY)
        logger_header.pack(fill=tk.X, padx=12, pady=(12, 8))

        tk.Label(
            logger_header, text="👁  Person Logger",
            bg=BG_SECONDARY, fg=TEXT_PRIMARY,
            font=("Helvetica", 13, "bold")
        ).pack(side=tk.LEFT)

        self._count_label = tk.Label(
            logger_header, text="0", bg=ACCENT_BLUE, fg="white",
            font=("Menlo", 11, "bold"), padx=8, pady=1
        )
        self._count_label.pack(side=tk.RIGHT)

        # Separator
        tk.Frame(logger_panel, bg=BORDER, height=1).pack(fill=tk.X, padx=8)

        # Scrollable person list
        list_container = tk.Frame(logger_panel, bg=BG_SECONDARY)
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

        # Mouse wheel scrolling
        self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # ── Status bar ────────────────────────────────────────────────────────
        status_bar = tk.Frame(self.root, bg="#0f172a", height=32)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        status_bar.pack_propagate(False)

        self._status_tracked = tk.Label(
            status_bar, text="Tracked: 0", bg="#0f172a", fg=TEXT_MUTED,
            font=("Menlo", 10)
        )
        self._status_tracked.pack(side=tk.LEFT, padx=16, pady=4)

        self._status_active = tk.Label(
            status_bar, text="Active: 0", bg="#0f172a", fg=TEXT_MUTED,
            font=("Menlo", 10)
        )
        self._status_active.pack(side=tk.LEFT, padx=8, pady=4)

        tk.Label(
            status_bar, text=f"Cameras: {self.n_cameras}",
            bg="#0f172a", fg=TEXT_MUTED, font=("Menlo", 10)
        ).pack(side=tk.LEFT, padx=8, pady=4)

        tk.Label(
            status_bar, text="Sensen Multi-Camera Tracking System",
            bg="#0f172a", fg="#334155", font=("Helvetica", 10)
        ).pack(side=tk.RIGHT, padx=16, pady=4)

        # ── Camera canvases ───────────────────────────────────────────────────
        self._cam_labels: Dict[int, tk.Label] = {}
        self._cam_overlay_labels: Dict[int, tk.Label] = {}
        self._build_camera_grid()

    def _build_camera_grid(self) -> None:
        """Create camera tile grid (2 columns) with evenly-sized tiles."""
        cols = 2
        n_rows = (self.n_cameras + cols - 1) // cols

        # Configure grid: uniform groups force all rows/columns to the same size
        for r in range(n_rows):
            self._camera_frame.grid_rowconfigure(
                r, weight=1, uniform="cam_row", minsize=200
            )
        for c in range(cols):
            self._camera_frame.grid_columnconfigure(
                c, weight=1, uniform="cam_col", minsize=300
            )

        for idx in range(self.n_cameras):
            cam_id = idx + 1
            row, col = divmod(idx, cols)

            # Card frame — propagate=False so it fills the grid cell
            card = tk.Frame(self._camera_frame, bg=BG_SECONDARY, bd=0)
            card.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
            card.grid_rowconfigure(0, weight=1)
            card.grid_columnconfigure(0, weight=1)

            # Camera image label — fills the entire card
            cam_label = tk.Label(card, bg="black", bd=0, anchor="center")
            cam_label.grid(row=0, column=0, sticky="nsew")
            self._cam_labels[cam_id] = cam_label

            # Overlay label (camera name)
            overlay = tk.Label(
                cam_label, text=f"  ● CAM {cam_id}  ",
                bg="#000000", fg="white",
                font=("Menlo", 10, "bold"),
            )
            overlay.place(x=8, y=8)
            self._cam_overlay_labels[cam_id] = overlay

    def _on_mousewheel(self, event):
        self._canvas.yview_scroll(-1 * (event.delta // 120 or event.delta), "units")

    # ── Update loop ───────────────────────────────────────────────────────────

    def _update(self) -> None:
        """Called every ~33ms from the tkinter main loop."""
        self._update_camera_feeds()
        self._update_logger()
        self.root.after(33, self._update)  # ~30 Hz

    def _update_camera_feeds(self) -> None:
        """Render latest camera frames into tkinter labels."""
        with self._frame_lock:
            frames = dict(self._latest_frames)

        for cam_id, frame in frames.items():
            if cam_id not in self._cam_labels:
                continue

            label = self._cam_labels[cam_id]
            # Get the label's current display size
            lw = label.winfo_width()
            lh = label.winfo_height()
            if lw < 10 or lh < 10:
                lw, lh = self.tile_w, self.tile_h

            # Resize frame to fit the label while maintaining aspect ratio
            fh, fw = frame.shape[:2]
            scale = min(lw / fw, lh / fh)
            new_w = int(fw * scale)
            new_h = int(fh * scale)

            resized = cv2.resize(frame, (new_w, new_h))
            # Convert BGR → RGB
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            photo = ImageTk.PhotoImage(img)

            label.configure(image=photo)
            self._photo_refs[cam_id] = photo  # Prevent GC

    def _update_logger(self) -> None:
        """Update the person logger panel with current gallery state."""
        entries = self.gallery.get_all_entries_snapshot()

        # Build active-person mapping
        all_active: Dict[int, list] = {}
        for cam_id, gids in dict(active_ids_by_camera).items():
            for gid in gids:
                all_active.setdefault(gid, []).append(cam_id)

        # Build a fingerprint to avoid unnecessary rebuilds
        fingerprint = str([
            (e["global_id"], e["hit_count"], sorted(all_active.get(e["global_id"], [])))
            for e in entries
        ])

        if fingerprint == self._last_person_data:
            return
        self._last_person_data = fingerprint

        # Update counters
        total = len(entries)
        active_count = sum(1 for e in entries if e["global_id"] in all_active)
        self._count_label.configure(text=str(total))
        self._status_tracked.configure(text=f"Tracked: {total}")
        self._status_active.configure(text=f"Active: {active_count}")

        # Sort by global ID only (stable order — no jumping)
        entries.sort(key=lambda e: e["global_id"])

        # Clear old person cards
        for widget in self._person_list_frame.winfo_children():
            widget.destroy()
        self._thumb_refs.clear()

        if not entries:
            tk.Label(
                self._person_list_frame, text="🔍  Waiting for detections…",
                bg=BG_SECONDARY, fg=TEXT_MUTED,
                font=("Helvetica", 12), pady=60
            ).pack()
            return

        # Build person cards
        for entry in entries:
            gid = entry["global_id"]
            is_active = gid in all_active
            active_cams = sorted(all_active.get(gid, []))
            color_hex = get_color_hex(gid)

            self._build_person_card(
                entry, is_active, active_cams, color_hex
            )

    def _build_person_card(
        self, entry: dict, is_active: bool,
        active_cams: list, color_hex: str
    ) -> None:
        """Build a single person card in the logger."""
        gid = entry["global_id"]
        bg = BG_ACTIVE if is_active else BG_CARD
        border_color = ACCENT_GREEN if is_active else BORDER

        # Outer border frame
        outer = tk.Frame(self._person_list_frame, bg=border_color, bd=0)
        outer.pack(fill=tk.X, padx=6, pady=3)

        # Inner card
        card = tk.Frame(outer, bg=bg, bd=0)
        card.pack(fill=tk.X, padx=(3, 1), pady=1)

        # ── Left: Colour bar + thumbnail ──────────────────────────────────────
        left = tk.Frame(card, bg=bg)
        left.pack(side=tk.LEFT, padx=(8, 4), pady=8)

        # Thumbnail
        thumb_frame = tk.Frame(left, bg=BORDER, width=40, height=52)
        thumb_frame.pack_propagate(False)
        thumb_frame.pack()

        if entry.get("thumbnail_b64"):
            import base64
            import io
            raw = base64.b64decode(entry["thumbnail_b64"])
            pil_img = Image.open(io.BytesIO(raw))
            pil_img = pil_img.resize((38, 50), Image.LANCZOS)
            photo = ImageTk.PhotoImage(pil_img)
            thumb_label = tk.Label(thumb_frame, image=photo, bg=BORDER, bd=0)
            thumb_label.pack(fill=tk.BOTH, expand=True)
            self._thumb_refs[gid] = photo
        else:
            tk.Label(
                thumb_frame, text="👤", bg=BG_CARD, fg=TEXT_MUTED,
                font=("Helvetica", 16)
            ).pack(fill=tk.BOTH, expand=True)

        # ── Right: Info ───────────────────────────────────────────────────────
        right = tk.Frame(card, bg=bg)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 8), pady=8)

        # Top row: swatch + ID + LIVE badge + hit count
        top_row = tk.Frame(right, bg=bg)
        top_row.pack(fill=tk.X)

        # Colour swatch
        swatch = tk.Frame(top_row, bg=color_hex, width=12, height=12)
        swatch.pack(side=tk.LEFT, padx=(0, 6))
        swatch.pack_propagate(False)

        # ID
        id_text = f"ID:{gid:03d}"
        tk.Label(
            top_row, text=id_text, bg=bg, fg=TEXT_PRIMARY,
            font=("Menlo", 13, "bold")
        ).pack(side=tk.LEFT)

        # LIVE badge
        if is_active:
            live_frame = tk.Frame(top_row, bg="#134e2e", bd=0)
            live_frame.pack(side=tk.LEFT, padx=(8, 0))
            tk.Label(
                live_frame, text=" ● LIVE ", bg="#134e2e", fg=ACCENT_GREEN,
                font=("Menlo", 8, "bold")
            ).pack()

        # Hit count
        tk.Label(
            top_row, text=f"×{entry['hit_count']}",
            bg=bg, fg=TEXT_MUTED, font=("Menlo", 11)
        ).pack(side=tk.RIGHT)

        # Camera tags row
        cam_row = tk.Frame(right, bg=bg)
        cam_row.pack(fill=tk.X, pady=(4, 0))

        for c in entry.get("cameras_seen", []):
            is_active_cam = c in active_cams
            tag_bg = "#134e2e" if is_active_cam else "#1e293b"
            tag_fg = ACCENT_GREEN if is_active_cam else TEXT_MUTED
            tag = tk.Label(
                cam_row, text=f" Cam {c} ",
                bg=tag_bg, fg=tag_fg,
                font=("Menlo", 9, "bold"), padx=2
            )
            tag.pack(side=tk.LEFT, padx=(0, 4))

        # Active cameras row (only shown when active)
        if is_active and active_cams:
            active_row = tk.Frame(right, bg=bg)
            active_row.pack(fill=tk.X, pady=(3, 0))
            tk.Label(
                active_row, text="📷",
                bg=bg, fg=ACCENT_GREEN, font=("Helvetica", 9)
            ).pack(side=tk.LEFT, padx=(0, 4))
            for c in active_cams:
                tk.Label(
                    active_row, text=f" Cam {c} ",
                    bg="#134e2e", fg=ACCENT_GREEN,
                    font=("Menlo", 9, "bold"), padx=2
                ).pack(side=tk.LEFT, padx=(0, 3))

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Start the UI update loop and enter tkinter mainloop.
        MUST be called from the main thread.
        """
        print("[UI] ✦ Tracking dashboard opened")
        self.root.after(100, self._update)
        self.root.mainloop()

    def is_alive(self) -> bool:
        """Check if the window is still open."""
        try:
            self.root.winfo_exists()
            return True
        except tk.TclError:
            return False

    def destroy(self) -> None:
        """Close the window."""
        try:
            self.root.destroy()
        except tk.TclError:
            pass
