"""
generate_test_videos.py — Generates synthetic multi-camera MP4 videos for testing.

Renders 4 simulated camera feeds (Front Desk, Main Aisle, Lounge Area, Checkout)
with animated human figures walking along realistic trajectories across cameras.
"""

from pathlib import Path
import cv2
import numpy as np


def draw_human_figure(
    frame: np.ndarray,
    x: int,
    y: int,
    height: int = 140,
    top_color=(0, 0, 220),       # BGR (e.g. Red Jacket)
    bottom_color=(200, 100, 0),  # BGR (e.g. Blue Jeans)
    skin_color=(180, 200, 230),
    head_hair=(30, 30, 30),
    label: str = "",
) -> None:
    """Draw a realistic human silhouette on a frame."""
    width = int(height * 0.45)
    head_r = int(height * 0.12)
    head_cy = y - height + head_r
    torso_top = head_cy + head_r
    torso_h = int(height * 0.42)
    leg_h = height - (head_r * 2) - torso_h

    # Legs (Bottom)
    leg_w = width // 3
    cv2.rectangle(
        frame,
        (x - width // 2 + 2, torso_top + torso_h),
        (x - 2, y),
        bottom_color,
        -1,
    )
    cv2.rectangle(
        frame,
        (x + 2, torso_top + torso_h),
        (x + width // 2 - 2, y),
        bottom_color,
        -1,
    )

    # Torso / Upper body
    cv2.rectangle(
        frame,
        (x - width // 2, torso_top),
        (x + width // 2, torso_top + torso_h),
        top_color,
        -1,
    )

    # Head
    cv2.circle(frame, (x, head_cy), head_r, skin_color, -1)
    # Hair / Cap
    cv2.ellipse(frame, (x, head_cy - 2), (head_r, head_r - 3), 0, 180, 360, head_hair, -1)

    # Shoes
    cv2.ellipse(frame, (x - leg_w, y - 2), (leg_w // 2, 4), 0, 0, 360, (20, 20, 20), -1)
    cv2.ellipse(frame, (x + leg_w, y - 2), (leg_w // 2, 4), 0, 0, 360, (20, 20, 20), -1)

    if label:
        cv2.putText(
            frame,
            label,
            (x - width // 2, head_cy - head_r - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def generate_camera_background(title: str, w=640, h=360) -> np.ndarray:
    """Create a realistic indoor room background with floor tiles and walls."""
    bg = np.zeros((h, w, 3), dtype=np.uint8)

    
    # Wall (top half)
    wall = np.full((h // 2, w, 3), (45, 35, 25), dtype=np.uint8)
    # Floor (bottom half)
    floor = np.full((h // 2, w, 3), (80, 85, 90), dtype=np.uint8)
    bg[: h // 2, :] = wall
    bg[h // 2 :, :] = floor

    # Grid tiles on floor
    for x in range(0, w, 40):
        cv2.line(bg, (x, h // 2), (int(w / 2 + (x - w / 2) * 1.5), h), (60, 65, 70), 1)
    for y in range(h // 2, h, 30):
        cv2.line(bg, (0, y), (w, y), (60, 65, 70), 1)

    # Title Banner
    cv2.putText(
        bg,
        title,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 220, 255),
        2,
        cv2.LINE_AA,
    )
    return bg


def main():
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)

    w, h = 640, 360
    fps = 25
    duration_sec = 12
    total_frames = fps * duration_sec

    cams_info = [
        ("cam1_test.mp4", "Camera 1 — Front Entry"),
        ("cam2_test.mp4", "Camera 2 — Main Aisle"),
        ("cam3_test.mp4", "Camera 3 — Lounge Area"),
        ("cam4_test.mp4", "Camera 4 — Exit Checkout"),
    ]

    writers = []
    for fname, _ in cams_info:
        path = output_dir / fname
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        writers.append(writer)

    print(f"[TestVideoGen] Generating {len(cams_info)} synthetic test videos in: {output_dir}")

    # Person 0 (Red Jacket, Blue Jeans) — walks Cam 1 -> Cam 2 -> Cam 4
    # Person 1 (Green Sweater, Dark Pants) — walks Cam 2 -> Cam 3
    # Person 2 (Yellow Hoodie, Blue Shorts) — walks Cam 3 -> Cam 4

    for frame_idx in range(total_frames):
        t = frame_idx / total_frames  # normalized 0.0 -> 1.0

        frames = [generate_camera_background(title, w, h) for _, title in cams_info]

        # ── Person 0 ─────────────────────────────────────────────────────────
        # Cam 1 (0.0 -> 0.35)
        if 0.0 <= t <= 0.35:
            progress = t / 0.35
            px = int(80 + progress * 480)
            py = int(270 + np.sin(progress * np.pi) * 20)
            draw_human_figure(
                frames[0],
                px,
                py,
                height=150,
                top_color=(30, 30, 220),      # Red top
                bottom_color=(180, 80, 10),   # Blue jeans
                label="Person 0",
            )
        # Cam 2 (0.30 -> 0.70)
        if 0.30 <= t <= 0.70:
            progress = (t - 0.30) / 0.40
            px = int(550 - progress * 460)
            py = int(280 + np.cos(progress * np.pi) * 15)
            draw_human_figure(
                frames[1],
                px,
                py,
                height=140,
                top_color=(30, 30, 220),
                bottom_color=(180, 80, 10),
                label="Person 0",
            )
        # Cam 4 (0.65 -> 1.0)
        if 0.65 <= t <= 1.0:
            progress = (t - 0.65) / 0.35
            px = int(100 + progress * 440)
            py = int(290 - progress * 30)
            draw_human_figure(
                frames[3],
                px,
                py,
                height=135,
                top_color=(30, 30, 220),
                bottom_color=(180, 80, 10),
                label="Person 0",
            )

        # ── Person 1 ─────────────────────────────────────────────────────────
        # Cam 2 (0.10 -> 0.50)
        if 0.10 <= t <= 0.50:
            progress = (t - 0.10) / 0.40
            px = int(100 + progress * 400)
            py = int(260 + np.sin(progress * 2 * np.pi) * 15)
            draw_human_figure(
                frames[1],
                px,
                py,
                height=145,
                top_color=(30, 180, 30),      # Green sweater
                bottom_color=(40, 40, 40),    # Dark pants
                label="Person 1",
            )
        # Cam 3 (0.45 -> 0.90)
        if 0.45 <= t <= 0.90:
            progress = (t - 0.45) / 0.45
            px = int(500 - progress * 380)
            py = int(275 + np.cos(progress * np.pi) * 20)
            draw_human_figure(
                frames[2],
                px,
                py,
                height=150,
                top_color=(30, 180, 30),
                bottom_color=(40, 40, 40),
                label="Person 1",
            )

        # ── Person 2 ─────────────────────────────────────────────────────────
        # Cam 3 (0.0 -> 0.40)
        if 0.0 <= t <= 0.40:
            progress = t / 0.40
            px = int(120 + progress * 420)
            py = int(285 - progress * 20)
            draw_human_figure(
                frames[2],
                px,
                py,
                height=130,
                top_color=(20, 220, 240),     # Yellow hoodie
                bottom_color=(160, 60, 20),   # Blue shorts
                label="Person 2",
            )
        # Cam 4 (0.35 -> 0.80)
        if 0.35 <= t <= 0.80:
            progress = (t - 0.35) / 0.45
            px = int(520 - progress * 380)
            py = int(270 + progress * 15)
            draw_human_figure(
                frames[3],
                px,
                py,
                height=135,
                top_color=(20, 220, 240),
                bottom_color=(160, 60, 20),
                label="Person 2",
            )

        # Write frame to each camera writer
        for idx, writer in enumerate(writers):
            # Frame counter HUD
            cv2.putText(
                frames[idx],
                f"Frame: {frame_idx}/{total_frames}",
                (w - 180, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )
            writer.write(frames[idx])

    for writer in writers:
        writer.release()

    print("[TestVideoGen] Successfully created 4 test video files:")
    for fname, _ in cams_info:
        p = output_dir / fname
        print(f"  - {p} ({p.stat().st_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
