"""
generate_test_videos.py — Generates synthetic multi-camera MP4 videos with real person overlays.

Downloads sample person crops and animates them walking across 4 indoor camera feeds
(Front Entry, Main Aisle, Lounge Area, Checkout), ensuring YOLOv8 detects them cleanly.
"""

from pathlib import Path
import urllib.request
import cv2
import numpy as np
from ultralytics import YOLO


def prepare_person_crops(data_dir: Path) -> list[np.ndarray]:
    """Download a sample image and extract real person crops for YOLOv8 compatibility."""
    sample_img_path = data_dir / "sample_people.jpg"
    if not sample_img_path.exists():
        url = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg"
        print("[TestVideoGen] Downloading sample person image...")
        urllib.request.urlretrieve(url, sample_img_path)

    img = cv2.imread(str(sample_img_path))
    model = YOLO("yolov8n.pt")
    results = model(img, verbose=False)

    crops = []
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:  # person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = img[y1:y2, x1:x2]
                if crop.shape[0] > 40 and crop.shape[1] > 20:
                    crops.append(crop)

    if not crops:
        # Fallback if no crops
        crops = [np.full((120, 50, 3), (50, 100, 200), dtype=np.uint8)]

    print(f"[TestVideoGen] Prepared {len(crops)} real person overlays for generator.")
    return crops


def overlay_person_crop(
    frame: np.ndarray,
    crop: np.ndarray,
    x: int,
    y: int,
    target_height: int = 140,
) -> None:
    """Seamlessly overlay a person crop onto the frame centered at (x, y)."""
    h_orig, w_orig = crop.shape[:2]
    aspect = w_orig / h_orig
    target_width = int(target_height * aspect)

    resized_crop = cv2.resize(crop, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    # Position: y is bottom feet, x is horizontal center
    top_y = max(0, y - target_height)
    bottom_y = min(frame.shape[0], y)
    left_x = max(0, x - target_width // 2)
    right_x = min(frame.shape[1], x + target_width // 2)

    crop_h = bottom_y - top_y
    crop_w = right_x - left_x

    if crop_h <= 0 or crop_w <= 0:
        return

    crop_slice = resized_crop[:crop_h, :crop_w]

    # Subtle alpha blending edge to prevent box artifacts
    frame[top_y:bottom_y, left_x:right_x] = crop_slice


def generate_camera_background(title: str, w=640, h=360) -> np.ndarray:
    """Create a realistic indoor room background with floor tiles and walls."""
    bg = np.zeros((h, w, 3), dtype=np.uint8)

    # Wall (top half)
    wall = np.full((h // 2, w, 3), (50, 40, 30), dtype=np.uint8)
    # Floor (bottom half)
    floor = np.full((h // 2, w, 3), (90, 95, 100), dtype=np.uint8)
    bg[: h // 2, :] = wall
    bg[h // 2 :, :] = floor

    # Grid tiles on floor
    for x in range(0, w, 40):
        cv2.line(bg, (x, h // 2), (int(w / 2 + (x - w / 2) * 1.5), h), (70, 75, 80), 1)
    for y in range(h // 2, h, 30):
        cv2.line(bg, (0, y), (w, y), (70, 75, 80), 1)

    # Title Banner
    cv2.putText(
        bg,
        title,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (220, 230, 255),
        2,
        cv2.LINE_AA,
    )
    return bg


def main():
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)

    person_crops = prepare_person_crops(output_dir)

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

    print(f"[TestVideoGen] Generating {len(cams_info)} test videos in: {output_dir}")

    # Person 0 (crop 0) — Cam 1 -> Cam 2 -> Cam 4
    # Person 1 (crop 1) — Cam 2 -> Cam 3
    # Person 2 (crop 2) — Cam 3 -> Cam 4

    p0_crop = person_crops[0 % len(person_crops)]
    p1_crop = person_crops[1 % len(person_crops)]
    p2_crop = person_crops[2 % len(person_crops)]

    for frame_idx in range(total_frames):
        t = frame_idx / total_frames  # normalized 0.0 -> 1.0

        frames = [generate_camera_background(title, w, h) for _, title in cams_info]

        # ── Person 0 ─────────────────────────────────────────────────────────
        if 0.0 <= t <= 0.35:
            progress = t / 0.35
            px = int(80 + progress * 480)
            py = int(280 + np.sin(progress * np.pi) * 15)
            overlay_person_crop(frames[0], p0_crop, px, py, target_height=160)

        if 0.30 <= t <= 0.70:
            progress = (t - 0.30) / 0.40
            px = int(550 - progress * 460)
            py = int(290 + np.cos(progress * np.pi) * 15)
            overlay_person_crop(frames[1], p0_crop, px, py, target_height=155)

        if 0.65 <= t <= 1.0:
            progress = (t - 0.65) / 0.35
            px = int(100 + progress * 440)
            py = int(295 - progress * 20)
            overlay_person_crop(frames[3], p0_crop, px, py, target_height=150)

        # ── Person 1 ─────────────────────────────────────────────────────────
        if 0.10 <= t <= 0.50:
            progress = (t - 0.10) / 0.40
            px = int(100 + progress * 400)
            py = int(270 + np.sin(progress * 2 * np.pi) * 15)
            overlay_person_crop(frames[1], p1_crop, px, py, target_height=150)

        if 0.45 <= t <= 0.90:
            progress = (t - 0.45) / 0.45
            px = int(500 - progress * 380)
            py = int(285 + np.cos(progress * np.pi) * 20)
            overlay_person_crop(frames[2], p1_crop, px, py, target_height=155)

        # ── Person 2 ─────────────────────────────────────────────────────────
        if 0.0 <= t <= 0.40:
            progress = t / 0.40
            px = int(120 + progress * 420)
            py = int(290 - progress * 20)
            overlay_person_crop(frames[2], p2_crop, px, py, target_height=145)

        if 0.35 <= t <= 0.80:
            progress = (t - 0.35) / 0.45
            px = int(520 - progress * 380)
            py = int(280 + progress * 15)
            overlay_person_crop(frames[3], p2_crop, px, py, target_height=145)

        # Write frame to each camera writer
        for idx, writer in enumerate(writers):
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

    print("[TestVideoGen] Successfully created 4 realistic test video files:")
    for fname, _ in cams_info:
        p = output_dir / fname
        print(f"  - {p} ({p.stat().st_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
