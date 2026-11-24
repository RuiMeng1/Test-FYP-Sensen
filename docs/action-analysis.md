# Office activity analysis

Run this optional pass after normal tracking. It reads original footage and the
saved tracking log, then writes `output/action_events.json` for both review UIs
and the assistant. Restart the review server or desktop UI after a run.

```sh
python -m src.action_analysis --output-dir output
python -m src.action_analysis --output-dir output --camera 6 --start 5 --end 30
```

The default samples one frame per second. `--step 0.5` samples twice as often;
`--walk-speed 0.25` sets the minimum ground-point travel, measured in person
heights per second. Use `--data-dir` if the original videos are elsewhere.
`--device cpu` is the portable default; `mps` and `cuda` are optional.
On CPU, two cameras are analyzed concurrently by default, with separate models
per worker. Use `--workers 1` to compare serial processing or `--workers 4` to
benchmark more parallel work; GPU runs default to one worker.
Object detection uses `--object-size 1920` by default to retain small phones in
wide CCTV frames; lower it for faster analysis if phone recall is less important.

The pass uses the existing COCO `yolov8n.pt` object model and the official
`yolov8n-pose.pt` checkpoint. Ultralytics downloads the pose checkpoint on first
use if it is absent. You can supply local checkpoint paths with `--pose-model`
and `--object-model`. No OpenAI call or extra video copy is made.

Events are **unreviewed suggestions**:

- **Sitting / standing:** visible shoulder, hip, knee and ankle geometry in a
  matched pose. Sitting also requires a detected chair or couch under the hip
  to avoid logging crouching on the floor as office seating. If the joints or
  seat are hidden or ambiguous, no posture is reported.
- **Walking:** a person classified as standing whose tracked ground point moves
  at least the configured speed across adjacent samples in one fixed camera.
- **Possible phone use:** the COCO object model detects a phone near a visible
  wrist inside the tracked person's box. It does not establish that the person
  is interacting with the phone.

An interval needs at least two consecutive positive samples. A missed frame or
uncertain pose creates a gap; the system does not fill it. The first movement
sample has no prior point, so walking starts only after a comparison. Sitting
and standing are postures; they do not establish the exact moment of a
sit-down or stand-up transition. Scores are model and landmark confidence
indicators, not calibrated probabilities of behavior.

Small or hidden phones, desk occlusion, steep camera angles, camera movement,
and identity swaps can cause missed or incorrect events. Tune and validate on
labelled footage from the actual cameras before relying on counts or duration.
The existing person tracker can exclude seated people behind desks. This pass
cannot attribute an event to a person who is absent from the tracking log; a
seated worker hidden by a chair in the sample workspace footage was missed by
the stock YOLO detectors tested here.
Absent events do not prove inactivity. The saved sidecar is invalidated if the
tracking log or source footage changes; a limited run replaces its earlier
coverage. ID merges also invalidate it. Existing tracking videos are untouched.

Sources: [Ultralytics YOLOv8 pose models](https://docs.ultralytics.com/models/yolov8/),
[pose keypoint format](https://docs.ultralytics.com/tasks/pose/), and
[COCO phone class](https://docs.ultralytics.com/datasets/detect/coco/).
