# Partial occlusion

Each camera tracks reliable box area and height for each ByteTrack local ID.
An area drop greater than 35%, a height drop greater than 25%, or failure of
the existing full-body filter marks an established track as possibly occluded.
The reliable baseline follows gradual changes with an exponential moving
average (30% current size), and freezes while occlusion is suspected.

The partial box remains visible and keeps its existing global identity, but
is excluded from appearance extraction, gallery exemplar updates, and thumbnail
selection. A track without a global identity waits for a reliable crop before
receiving one. New local tracks still have to pass the full-body filter.

Appearance updates resume after three consecutive detected frames reach at
least 85% of baseline area and 90% of baseline height and pass the full-body
filter. Size history expires after more than 30 frames without that detection.

This is a conservative size heuristic, not a visibility estimate. Crouching or
abrupt scale changes can pause learning until size recovers. Occlusion that
does not shrink a box may go undetected. YOLO's existing confidence cutoff
still applies; full disappearance and new local IDs rely on ByteTrack and
the existing gallery matching. Thresholds need validation on real videos.
