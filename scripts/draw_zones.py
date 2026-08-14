import cv2
import argparse
from pathlib import Path

def draw_polygon(event, x, y, flags, param):
    global points, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        drawing = True
    elif event == cv2.EVENT_RBUTTONDOWN:
        if points:
            points.pop()

parser = argparse.ArgumentParser(description="Helper script to define exclusion polygons for a video.")
parser.add_argument("video", help="Path to the video file")
args = parser.parse_args()

cap = cv2.VideoCapture(args.video)
ret, frame = cap.read()
cap.release()

if not ret:
    print(f"Error: Could not read {args.video}")
    exit(1)

points = []
drawing = False

clone = frame.copy()
window_name = f"Draw Exclusion Zone - {Path(args.video).name}"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, draw_polygon)

print("Instructions:")
print(" - Left click: Add a point to the polygon")
print(" - Right click: Remove the last point")
print(" - Press 'c' to clear all points")
print(" - Press 'q' or 'Enter' when finished")

while True:
    img = clone.copy()
    if len(points) > 0:
        cv2.polylines(img, [__import__('numpy').array(points)], isClosed=False, color=(0, 0, 255), thickness=2)
        if len(points) > 2:
            cv2.polylines(img, [__import__('numpy').array(points)], isClosed=True, color=(0, 255, 0), thickness=2)
            # Fill the polygon with a slight green tint
            overlay = img.copy()
            cv2.fillPoly(overlay, [__import__('numpy').array(points)], (0, 255, 0))
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
            
    for p in points:
        cv2.circle(img, p, 4, (0, 0, 255), -1)
        
    cv2.imshow(window_name, img)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("c"):
        points = []
    elif key == ord("q") or key == 13: # 13 is Enter
        break

cv2.destroyAllWindows()

print("\n--- Polygon Coordinates ---")
if len(points) > 2:
    print(f"    # {Path(args.video).name}:")
    print(f"    {points},")
    print("\nCopy the coordinates above and paste them into EXCLUSION_ZONES in src/config.py")
else:
    print("No valid polygon drawn (needs at least 3 points).")
