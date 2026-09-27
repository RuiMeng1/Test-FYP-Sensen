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
- Gallery and confirmed ID merges. Merges save to the shared tracking JSON and keep
  a `tracking_log.before-merge-<revision>.json` backup. Concurrent external edits are
  rejected. Actions and chat history are invalidated after merges.
- Existing assistant, conversation clearing, camera/time citations, clickable visual evidence.
- All saved action events, configurable footage labels, All/None controls, persisted selections.
- Person-summary PDF download using the same existing report renderer without opening Tk.

## Playback design

Videos use native HTML5 decoding and HTTP range requests. The server does not stream
per-frame JPEGs. Canvas overlays use cached ten-second chunks of detection metadata.
Hidden cameras pause. Logger nodes persist while visibility badges update at 5 Hz.
Grid videos correct drift only above 450 ms rather than seeking every frame.
Smoothness still depends on camera count, hardware decoding, and browser support;
this is not a guarantee of zero buffering on every machine. Source timestamps and
camera FPS are used; absent recording-start offsets are assumed aligned at time zero.

The server binds to localhost and checks a request token for writes. It is intended
for a single local reviewer, not public hosting. API keys stay on the Python backend.
AI calls use the existing configured provider; no extra inference runs during playback.

Validation: `python -m unittest discover -s tests -p 'test_web_review.py'`.
