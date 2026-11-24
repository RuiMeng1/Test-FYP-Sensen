# Web review (independent of the Python dashboard)

From the repository root, with the virtual environment active:

```sh
pip install -r requirements-web.txt
python -m src.web_review
```

Open http://127.0.0.1:5002 after the terminal prints “Web review ready”.
The existing `python -m src.main --review-only` remains the Python backup.
No existing UI, tracking, or action-analysis code is replaced.

First launch creates browser-compatible H.264 720p proxies in
`output/web_video_cache/`. This can take several minutes and extra disk space;
subsequent launches reuse the cache unless source files change. Original files
remain intact. Processing and action analysis still run with their existing commands.
The proxy is silent: this dashboard, like the Python review, reviews video rather than audio.

Options: `--output-dir output`, `--data-dir data`, `--port 5002`. Supply the directory
containing the original videos used in that tracking run. If originals were moved,
pass the correct directory. Without them, annotated outputs can play, but baked-in
boxes cannot be hidden or reassigned; merges are therefore disabled in that case.
Action sidecars retain their existing source validation and can require regeneration
after files move. Restart the server after new tracking or action analysis.

## Features

- Synchronized grid / individual cameras, seek, frame stepping, speed, double-click focus.
- Person timelines and auto-follow camera handoffs, persistent logger cards, search.
- Gallery corrections. Use **Delete ID…** for a false track such as a reflection,
  and **Merge with…** for duplicate IDs of the same person. Confirming either
  removes the gap in numbering across the gallery, map, and footage overlays.
  Review saves a `tracking_log.before-delete-<revision>.json` or
  `tracking_log.before-merge-<revision>.json` backup. Original videos are used to
  draw corrected IDs. Activity for unaffected people follows the new numbers.
  Rerun activity analysis for a merged person. Chat is cleared so previous
  references cannot point to a different person after renumbering.
- Existing assistant, conversation clearing, camera/time citations, clickable visual evidence.
- All saved action events, configurable footage labels, All/None controls, persisted selections.
- Person-summary PDF download using the same existing report renderer without opening Tk.

## 3D office map

Click **Show everyone** in the workspace to see people recorded at the current
playback time. Markers use the same colors as the identity boxes. Hover over a
marker or focus it with Tab to see the ID, then select it to follow that person.
People who share a room get separate markers. An ID seen by cameras assigned
to different rooms can appear in both. Sightings from cameras without a mapped
coverage area are counted separately.

Click **Follow** beside a person to open the map. Its play button and scrubber
share the footage timeline. Expand **Movement history** to jump to any recorded
sighting, or turn on **Show full recorded route** to see all observed zones and
calibrated path segments. The timeline marks recorded coverage. The current
location clears during gaps, and the map leaves separate segments across camera
handoffs. Simultaneous sightings can show more than one camera observation.

The built-in view arranges labeled camera zones as a schematic. To draw the
actual building, use one sketch per floor showing the rooms, corridors, stairs,
entrances, camera locations, and viewing directions. Include a known distance
for scale and label the cameras with their original video filenames. For an
estimated position within a view, supply four corresponding points on the same
flat floor that appear both in a camera frame and on the plan. Doorway corners
on the ground, tile intersections, and floor markings work well. A camera
without calibration can still show a room or corridor zone.

### Place cameras in the browser

From **Workspace**, click **Edit cameras**. Pick a camera by its filename and
check its footage preview. Click the top-down plan to place the selected camera,
or drag an existing pin. Use the facing direction slider to turn its arrow.
**Coverage area** sets where that camera's sightings appear in the route and
can be a different room from the camera pin. Click **Save camera layout** to
apply the edits to the 3D map. Pins and coverage changes are kept in
`output/office_camera_placements.json` by layout and camera filename.
They are reused across questions, browser reloads, and server restarts with the
same floor plan, output folder, and camera filenames. A saved edit applies on
the next assistant question, including in an existing chat.

The web assistant uses the saved coverage areas and pins, the followed person,
and the playback time when you send a question. Its movement summaries refer
to recorded sightings and link to camera times. Markdown is supported in
questions and replies, including lists, headings, bold, tables, and code.

Camera pins help identify the physical camera position. Room-level location
uses the selected coverage area. Movement within a room still needs the four
floor landmarks below. Editing a pin or heading keeps an existing calibration
when its coverage area stays the same. Reassigning coverage uses the new room
until that view is calibrated there.

### Layout file and floor calibration

Save the layout as `config/office_map.json`, or pass `--office-map /path/map.json`.
Floor and zone bounds are `[x, y, width, depth]` in consistent plan units. Floor
elevations use the same units. Camera keys are original video filename stems.
Each calibration uses corresponding `image` points (fractions of frame width
and height, origin at the upper left) and `world` points (plan coordinates).
List the four points in perimeter order. The projection uses the bottom center
of the person's box and only projects points inside the calibrated area.

```json
{
  "name": "Office",
  "floors": [
    {"id": "ground", "label": "Ground floor", "elevation": 0, "bounds": [0, 0, 18, 12]},
    {"id": "upstairs", "label": "Level 1", "elevation": 3.4, "bounds": [0, 0, 18, 12]}
  ],
  "zones": [
    {"id": "entry", "label": "Entrance", "floor": "ground", "bounds": [0, 0, 5, 4]},
    {"id": "workspace", "label": "Workspace 1", "floor": "upstairs", "bounds": [4, 2, 10, 8]}
  ],
  "cameras": {
    "1-entrance-stairs_000": {
      "zone": "entry",
      "calibration": {
        "image": [[0.27, 0.45], [0.72, 0.45], [0.92, 0.92], [0.08, 0.91]],
        "world": [[0.5, 0.5], [4.5, 0.5], [4.5, 3.5], [0.5, 3.5]]
      }
    },
    "6-workspace-1_000": {"zone": "workspace"}
  }
}
```

The example coordinates illustrate the file format. Whole-day replay requires a
tracking log and camera recordings that cover that day, with a shared start time.
Playback shows where the selected ID was observed during the recording. Camera
blind spots and times when the tracker loses the person show as gaps. The current
review pipeline reads saved recordings. Live CCTV feeds would need a separate input
and tracking setup.

## Playback design

Videos use native HTML5 decoding and HTTP range requests. The server does not stream
per-frame JPEGs. Canvas overlays use cached ten-second chunks of detection metadata.
The overlay interpolates boxes across gaps of at most eight frames (about 0.32 seconds
at 25 fps) when the same unique ID is present on both sides; longer gaps remain blank.
Hidden cameras pause. Logger nodes persist while visibility badges update at 5 Hz.
Grid videos correct drift only above 450 ms rather than seeking every frame.
Smoothness still depends on camera count, hardware decoding, and browser support;
this is not a guarantee of zero buffering on every machine. Source timestamps and
camera FPS are used; absent recording-start offsets are assumed aligned at time zero.

The server binds to localhost and checks a request token for writes. It is intended
for a single local reviewer, not public hosting. API keys stay on the Python backend.
AI calls use the existing configured provider; no extra inference runs during playback.

Validation: `python -m unittest discover -s tests -p 'test_web_review.py'`.
