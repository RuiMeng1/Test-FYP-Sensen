# Occlusion diagnostics

Run `python -m src.main --no-preview` normally. Each camera writes a uniquely
named JSONL file under `output/occlusion_logs/`; previous runs are retained.
Each line is a JSON event with source filename, camera, frame, seconds, local ID,
global ID (or -1 if unknown), and available bounding-box evidence.
No extra model, API calls, image storage, or identity changes are introduced.
Existing review-only commands do not generate these diagnostics retroactively.

Events:
- `suspected_occlusion`: overlapping person boxes (intersection / smaller area
  >= 0.35), height below 65% of the track's smoothed baseline, or the existing
  detector partial-occlusion flag. Logged on changes of the evidence reasons.
- `occlusion_cues_cleared`: those visible-box cues no longer apply.
- `track_missing`: absent for at least 0.2 seconds (minimum two frames).
- `track_reappeared`: the same local tracker ID returns within three seconds;
  includes previous/current global IDs and prior occlusion cues.
- `possible_reappearance_new_track`: a new local ID overlaps a recently missing
  track's last box; includes candidate old IDs, not a confirmed identity link.
- `session_start` / `session_end`: thresholds and completion markers. Missing
  end marker means the log may be incomplete.

These are unreviewed heuristics, not confirmed occlusions or calibrated confidence
scores. Person-box overlap cannot tell which person is in front. Shrinkage can
also mean sitting, bending, perspective change, or detection noise; tables are
not explicitly detected. Missing tracks may have exited the scene. New-track
matching is spatial only and misses returns elsewhere or after three seconds.
Only detections surviving the pipeline's existing filters are observable.
The logger does not merge IDs or infer hidden people's positions.

Use the source, camera and seconds fields to inspect the footage. No web UI or
AI assistant integration is added; this is a separate diagnostic log.
