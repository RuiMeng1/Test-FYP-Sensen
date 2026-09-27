# Optional SlowFast action analysis

Normal tracking and review commands remain unchanged. Run action analysis after
processing completes and close/reopen review to load its results.

```sh
pip install -r requirements-actions.txt
python -m src.action_analysis --output-dir output --camera 1 --start 5 --end 10
python -m src.main --review-only
```

Omit camera/start/end for all footage. `--data-dir` selects the directory containing
the original input videos. The filenames must match the tracking run. A limited run
replaces the previous action sidecar with only its new coverage.

The first run downloads the official PyTorchVideo SlowFast R50 AVA checkpoint.
CPU is the portable default (including Macs); CUDA is optional. MPS is not assumed
compatible. Expect extra offline processing time; inference never runs in review.

The pass uses raw RGB clips and saved boxes at the reference frame, 32 temporal
samples, 256-pixel short side, normalization mean 0.45/std 0.225, and slow/fast
ratio 4. The clip spans approximately 2.13 seconds. Scores are sampled once per
second by default; tune `--step` and `--threshold` (default 0.6) on labelled footage.

All 60 evaluated AVA v2.2 labels are enabled (see `src/action_labels.py`), including
running, phone use, and fighting/hitting. These are unreviewed predictions. Actions require two
adjacent positive samples; possible falls preserve single samples. Intervals report
sample times, not exact movement boundaries. Sitting does not mean sitting down;
standing does not establish a stand-up transition. Scores are not probabilities
of correctness. Falls need human verification; automatic confirmation is absent.

Expand a person's logger card to see clickable suggestions. The assistant receives
the events and analyzed camera/time coverage. No extra vision API calls are used.
Use **Action display settings…** above the camera views to toggle multiple overlay
labels, or select All/None. Apply saves the selection in
`output/action_display_settings.json`. Filtering affects only footage overlays;
all retained events stay in the logger and assistant context. Overlays use current
saved person boxes during each event interval; ambiguous or missing boxes suppress
the label. Existing five-action results must be regenerated to include new classes.
Data is stored atomically in `output/action_events.json`; the tracking log and
rendered videos are untouched. Old results are rejected if the tracking JSON or
source video size/modification time/path changes. ID merges invalidate the results.
The sidecar keeps the checkpoint identity, threshold, sampling interval, supporting
frame and box, and skipped-window counts. Duplicate global IDs at the reference
frame are skipped rather than assigned an action.

Limitations: identity swaps across different frames can still misattribute actions.
The current detector's aspect-ratio and minimum-height filters can remove seated
or fallen people, so absent suggestions are not evidence that no event occurred.
Only saved boxes are analyzed. Heavy occlusion and missing detections reduce coverage.
No fine-tuning or accuracy claims are included in this prototype.

Sources:
- https://pytorchvideo.org/docs/tutorial_torchhub_detection_inference
- https://dl.fbaipublicfiles.com/pytorchvideo/data/class_names/ava_action_list.pbtxt
- https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/models/hub/slowfast.py

Checks: `python -m unittest discover -s tests -p 'test_action_events.py'`.
