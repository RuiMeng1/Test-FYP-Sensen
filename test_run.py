import cv2
import time
from pathlib import Path
from src.dispatcher import DetectionDispatcher
from src.gallery import GlobalGallery
from src.reid_model import TransReIDEncoder
from src.pipeline import CameraWorker
import threading

videos = [str(p) for p in sorted(Path("data").glob("*.mp4"))]
dispatcher = DetectionDispatcher()
reid_model = TransReIDEncoder(device="mps")
gallery = GlobalGallery()

workers = []
for i, vpath in enumerate(videos):
    w = CameraWorker(i, vpath, dispatcher, reid_model, gallery, reid_interval=30)
    workers.append(w)

start_time = time.time()
frame_idx = 0
while frame_idx < 300:
    for w in workers:
        w.step()
    frame_idx += 1
    if frame_idx % 30 == 0:
        print(f"Processed {frame_idx} frames. Gallery IDs: {len(gallery._entries)}")

for w in workers:
    w.release()

print(f"Done in {time.time()-start_time:.2f}s")
print(f"Gallery distinct persons: {len(gallery._entries)}")
