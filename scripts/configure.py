"""
configure.py — Launch the pipeline configuration UI.

Starts a local HTTP server on port 7700, opens your default browser,
and lets you configure all camera-pair thresholds and pipeline settings
through a web interface. Settings are saved to:

    config/pipeline_settings.json

The pipeline (src/main.py) picks this file up automatically on next run.

Usage:
    python scripts/configure.py
    python scripts/configure.py --port 7700 --no-browser
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data"
CONFIG_DIR = ROOT / "config"
SETTINGS_FILE = CONFIG_DIR / "pipeline_settings.json"

# ── Default settings (mirrors src/config.py) ─────────────────────────────────
DEFAULTS = {
    "global_threshold":    0.65,
    "reid_every_n_frames": 5,
    "gallery_ttl_frames":  50000,
    "max_exemplars":       8,
    "ema_alpha":           0.15,
    "pair_thresholds":     {},   # "stemA|stemB" -> float  (stems sorted alphabetically)
}


def _pair_key(a: str, b: str) -> str:
    """Canonical order-independent key for a camera pair."""
    return "|".join(sorted([a, b]))


def _load_settings() -> dict:
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r") as f:
                data = json.load(f)
            # Back-fill missing keys with defaults
            for k, v in DEFAULTS.items():
                data.setdefault(k, v)
            return data
        except Exception as e:
            print(f"[Config] Warning: could not load {SETTINGS_FILE}: {e}")
    return dict(DEFAULTS)


def _save_settings(data: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[Config] Saved → {SETTINGS_FILE}")


def _scan_cameras() -> list[dict]:
    """Return list of {stem, filename} for every .mp4 in data/."""
    cameras = []
    if DATA_DIR.exists():
        for p in sorted(DATA_DIR.glob("*.mp4")):
            cameras.append({"stem": p.stem, "filename": p.name})
    return cameras


# ── HTML UI (self-contained) ─────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Pipeline Configuration · Multi-Camera Tracker</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:        #0d0f14;
    --surface:   #161a23;
    --surface2:  #1e2332;
    --border:    #2a3045;
    --accent:    #4f8ef7;
    --accent2:   #7c5ef7;
    --green:     #34d399;
    --yellow:    #fbbf24;
    --red:       #f87171;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --radius:    12px;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Inter', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    padding: 0 0 80px;
  }

  /* ── Header ── */
  header {
    background: linear-gradient(135deg, #0d0f14 0%, #1a1f35 100%);
    border-bottom: 1px solid var(--border);
    padding: 28px 40px;
    display: flex;
    align-items: center;
    gap: 16px;
    position: sticky;
    top: 0;
    z-index: 100;
    backdrop-filter: blur(12px);
  }

  .logo {
    width: 40px; height: 40px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
  }

  header h1 { font-size: 1.25rem; font-weight: 600; }
  header p  { font-size: 0.8rem; color: var(--muted); margin-top: 2px; }

  .header-actions { margin-left: auto; display: flex; gap: 12px; }

  /* ── Buttons ── */
  .btn {
    padding: 10px 22px;
    border-radius: 8px;
    border: none;
    font-family: inherit;
    font-size: 0.875rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.18s ease;
    display: flex; align-items: center; gap: 8px;
  }
  .btn-primary {
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: #fff;
  }
  .btn-primary:hover { opacity: 0.88; transform: translateY(-1px); box-shadow: 0 4px 20px rgba(79,142,247,0.35); }
  .btn-ghost { background: var(--surface2); color: var(--text); border: 1px solid var(--border); }
  .btn-ghost:hover { background: var(--border); }

  /* ── Layout ── */
  .container { max-width: 1100px; margin: 0 auto; padding: 40px 24px; }

  .section {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    margin-bottom: 28px;
    overflow: hidden;
  }

  .section-header {
    padding: 20px 24px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .section-icon {
    width: 32px; height: 32px;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 15px;
  }
  .section-header h2 { font-size: 1rem; font-weight: 600; }
  .section-header p  { font-size: 0.78rem; color: var(--muted); margin-top: 2px; }

  .section-body { padding: 24px; }

  /* ── Global settings grid ── */
  .settings-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 20px;
  }

  .field { display: flex; flex-direction: column; gap: 8px; }
  .field label {
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .field input[type=number] {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    font-family: inherit;
    font-size: 0.9rem;
    padding: 10px 14px;
    width: 100%;
    transition: border-color 0.15s;
  }
  .field input[type=number]:focus {
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(79,142,247,0.15);
  }
  .field .hint { font-size: 0.72rem; color: var(--muted); }

  /* ── Camera pair matrix ── */
  .camera-list {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .pair-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px;
    display: grid;
    grid-template-columns: 1fr auto 1fr auto;
    gap: 16px;
    align-items: center;
    transition: border-color 0.2s;
  }
  .pair-card:hover { border-color: var(--accent); }

  .cam-pill {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 0.82rem;
    font-weight: 500;
  }
  .cam-pill .cam-id {
    font-size: 0.65rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 2px;
  }
  .cam-pill .cam-stem {
    color: var(--text);
    word-break: break-all;
  }

  .pair-arrow { color: var(--muted); font-size: 1.1rem; }

  .threshold-control {
    display: flex;
    flex-direction: column;
    gap: 10px;
    min-width: 280px;
  }

  .threshold-row {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .threshold-row input[type=range] {
    flex: 1;
    -webkit-appearance: none;
    height: 6px;
    border-radius: 3px;
    outline: none;
    cursor: pointer;
    background: var(--border);
  }
  .threshold-row input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 18px; height: 18px;
    border-radius: 50%;
    cursor: pointer;
    border: 2px solid var(--bg);
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
  }

  .threshold-value {
    min-width: 46px;
    text-align: right;
    font-size: 0.9rem;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
  }

  .threshold-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 3px 8px;
    border-radius: 5px;
  }
  .label-strict   { background: rgba(248,113,113,0.15); color: var(--red); }
  .label-moderate { background: rgba(251,191,36,0.15);  color: var(--yellow); }
  .label-lenient  { background: rgba(52,211,153,0.15);  color: var(--green); }

  .pair-use-global {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.78rem;
    color: var(--muted);
  }
  .pair-use-global input[type=checkbox] { accent-color: var(--accent); width: 15px; height: 15px; }

  /* ── Empty state ── */
  .empty {
    text-align: center;
    padding: 48px 24px;
    color: var(--muted);
  }
  .empty .icon { font-size: 2.5rem; margin-bottom: 12px; }
  .empty p { font-size: 0.875rem; }

  /* ── Toast ── */
  #toast {
    position: fixed;
    bottom: 28px;
    right: 28px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 22px;
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 0.875rem;
    font-weight: 500;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    transform: translateY(80px);
    opacity: 0;
    transition: all 0.3s ease;
    z-index: 999;
  }
  #toast.show { transform: translateY(0); opacity: 1; }
  #toast.success { border-color: var(--green); }
  #toast.error   { border-color: var(--red); }

  /* ── Info callout ── */
  .callout {
    display: flex;
    gap: 12px;
    background: rgba(79,142,247,0.08);
    border: 1px solid rgba(79,142,247,0.25);
    border-radius: 8px;
    padding: 14px 16px;
    font-size: 0.8rem;
    color: #94b4f7;
    margin-bottom: 20px;
  }

  .divider { border: none; border-top: 1px solid var(--border); margin: 20px 0; }

  /* ── Threshold guide ── */
  .guide-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 12px;
  }
  .guide-item {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 16px;
  }
  .guide-item .g-value {
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 4px;
  }
  .guide-item .g-label { font-size: 0.78rem; color: var(--muted); }

  @media (max-width: 680px) {
    .pair-card { grid-template-columns: 1fr; }
    header { padding: 16px 20px; }
    .container { padding: 24px 16px; }
  }
</style>
</head>
<body>

<header>
  <div class="logo">⚙</div>
  <div>
    <h1>Pipeline Configuration</h1>
    <p>Multi-Camera Person Tracking · YOLOv8 + TransReID</p>
  </div>
  <div class="header-actions">
    <button class="btn btn-ghost" onclick="resetToDefaults()">↺ Reset</button>
    <button class="btn btn-primary" onclick="saveSettings()">💾 Save Settings</button>
  </div>
</header>

<div class="container">

  <!-- ── Global Settings ── -->
  <div class="section">
    <div class="section-header">
      <div class="section-icon" style="background:rgba(79,142,247,0.15)">🌐</div>
      <div>
        <h2>Global Settings</h2>
        <p>Applied to all cameras unless overridden by a per-pair threshold below</p>
      </div>
    </div>
    <div class="section-body">
      <div class="settings-grid">

        <div class="field">
          <label>Global Threshold</label>
          <input type="number" id="global_threshold" min="0.30" max="0.95" step="0.01" value="0.65">
          <span class="hint">Default cosine similarity cutoff for matching across cameras (0.40–0.95)</span>
        </div>

        <div class="field">
          <label>Re-ID Interval (frames)</label>
          <input type="number" id="reid_every_n_frames" min="1" max="30" step="1" value="5">
          <span class="hint">Extract embeddings every N frames. Higher = faster but less accurate</span>
        </div>

        <div class="field">
          <label>Gallery TTL (frames)</label>
          <input type="number" id="gallery_ttl_frames" min="100" max="500000" step="100" value="50000">
          <span class="hint">Frames before an unseen person's ID is released</span>
        </div>

        <div class="field">
          <label>Max Exemplars per Person</label>
          <input type="number" id="max_exemplars" min="1" max="20" step="1" value="8">
          <span class="hint">Number of appearance vectors stored per person (front/back/side views)</span>
        </div>

        <div class="field">
          <label>EMA Alpha</label>
          <input type="number" id="ema_alpha" min="0.01" max="0.50" step="0.01" value="0.15">
          <span class="hint">Blend factor when refining duplicate exemplars (higher = faster adapt)</span>
        </div>

      </div>
    </div>
  </div>

  <!-- ── Per-Pair Thresholds ── -->
  <div class="section">
    <div class="section-header">
      <div class="section-icon" style="background:rgba(124,94,247,0.15)">🎯</div>
      <div>
        <h2>Per-Camera-Pair Thresholds</h2>
        <p>Override the global threshold for specific camera pairs with hard angle differences</p>
      </div>
    </div>
    <div class="section-body">

      <div class="callout">
        ℹ️&nbsp; <span>Lower threshold = more likely to link the same person across cameras (fewer missed matches). Use lower values (0.50–0.58) for cameras with very different angles like portrait vs landscape, overhead vs eye-level.</span>
      </div>

      <div class="guide-grid" style="margin-bottom:24px">
        <div class="guide-item">
          <div class="g-value" style="color:var(--green)">0.50–0.58</div>
          <div class="g-label">Extreme angle difference<br>(portrait vs landscape, overhead)</div>
        </div>
        <div class="guide-item">
          <div class="g-value" style="color:var(--yellow)">0.58–0.65</div>
          <div class="g-label">Moderate angle difference<br>(same height, ~45° apart)</div>
        </div>
        <div class="guide-item">
          <div class="g-value" style="color:var(--red)">0.65–0.80</div>
          <div class="g-label">Similar angles<br>(same view, overlapping FOV)</div>
        </div>
        <div class="guide-item">
          <div class="g-value" style="color:var(--accent)">Use Global</div>
          <div class="g-label">Pair not listed → falls back to Global Threshold above</div>
        </div>
      </div>

      <div class="camera-list" id="pair-list">
        <div class="empty">
          <div class="icon">📂</div>
          <p>No cameras detected in <code>data/</code></p>
        </div>
      </div>

    </div>
  </div>

  <!-- ── Output file location ── -->
  <div class="section">
    <div class="section-header">
      <div class="section-icon" style="background:rgba(52,211,153,0.15)">📁</div>
      <div>
        <h2>Settings File</h2>
        <p>Settings are saved here and loaded automatically by the pipeline</p>
      </div>
    </div>
    <div class="section-body">
      <code style="background:var(--surface2);padding:10px 16px;border-radius:8px;display:block;font-size:0.85rem;color:var(--green)">config/pipeline_settings.json</code>
    </div>
  </div>

</div>

<div id="toast"></div>

<script>
let cameras = [];
let savedSettings = {};

// ── Fetch cameras + settings on load ──────────────────────────────────────────
async function init() {
  try {
    const [camRes, settRes] = await Promise.all([
      fetch('/api/cameras'),
      fetch('/api/settings'),
    ]);
    cameras = await camRes.json();
    savedSettings = await settRes.json();
    applySettings(savedSettings);
    renderPairs();
  } catch(e) {
    showToast('⚠ Could not load data: ' + e.message, 'error');
  }
}

function applySettings(s) {
  document.getElementById('global_threshold').value    = s.global_threshold    ?? 0.65;
  document.getElementById('reid_every_n_frames').value = s.reid_every_n_frames ?? 5;
  document.getElementById('gallery_ttl_frames').value  = s.gallery_ttl_frames  ?? 50000;
  document.getElementById('max_exemplars').value       = s.max_exemplars       ?? 8;
  document.getElementById('ema_alpha').value           = s.ema_alpha           ?? 0.15;
}

// ── Render camera pair rows ───────────────────────────────────────────────────
function pairKey(a, b) {
  return [a, b].sort().join('|');
}

function thresholdLabel(v) {
  if (v <= 0.58) return `<span class="threshold-label label-lenient">Lenient</span>`;
  if (v <= 0.65) return `<span class="threshold-label label-moderate">Moderate</span>`;
  return `<span class="threshold-label label-strict">Strict</span>`;
}

function sliderColor(v) {
  // green (0.50) → yellow (0.65) → red (0.80)
  const t = (v - 0.50) / 0.30;
  const r = Math.round(52  + (248 - 52)  * t);
  const g = Math.round(211 + (113 - 211) * t);
  const b = Math.round(153 + (113 - 153) * t);
  return `rgb(${r},${g},${b})`;
}

function renderPairs() {
  const list = document.getElementById('pair-list');
  if (cameras.length < 2) {
    list.innerHTML = `<div class="empty"><div class="icon">📂</div><p>No cameras detected in <code>data/</code></p></div>`;
    return;
  }

  const pairs = [];
  for (let i = 0; i < cameras.length; i++)
    for (let j = i+1; j < cameras.length; j++)
      pairs.push([cameras[i], cameras[j]]);

  list.innerHTML = pairs.map(([a, b]) => {
    const key = pairKey(a.stem, b.stem);
    const useGlobal = !(savedSettings.pair_thresholds && key in savedSettings.pair_thresholds);
    const val = useGlobal
      ? (savedSettings.global_threshold ?? 0.65)
      : savedSettings.pair_thresholds[key];

    return `
      <div class="pair-card" id="card-${key}">
        <div class="cam-pill">
          <div class="cam-id">Camera ${cameras.indexOf(a)+1}</div>
          <div class="cam-stem">${a.stem}</div>
        </div>
        <div class="pair-arrow">↔</div>
        <div class="cam-pill">
          <div class="cam-id">Camera ${cameras.indexOf(b)+1}</div>
          <div class="cam-stem">${b.stem}</div>
        </div>
        <div class="threshold-control">
          <div class="pair-use-global">
            <input type="checkbox" id="chk-${key}" ${useGlobal ? 'checked' : ''}
              onchange="toggleGlobal('${key}', this.checked, '${a.stem}', '${b.stem}')">
            <label for="chk-${key}">Use global threshold</label>
          </div>
          <div class="threshold-row" id="row-${key}" style="${useGlobal ? 'opacity:0.35;pointer-events:none' : ''}">
            <input type="range" id="slider-${key}"
              min="0.40" max="0.90" step="0.01"
              value="${val}"
              oninput="updateSlider('${key}', this.value)"
              style="--track: ${sliderColor(val)}">
            <span class="threshold-value" id="val-${key}">${parseFloat(val).toFixed(2)}</span>
            <span id="lbl-${key}">${thresholdLabel(val)}</span>
          </div>
        </div>
      </div>`;
  }).join('');

  // Style slider tracks
  pairs.forEach(([a,b]) => {
    const key = pairKey(a.stem, b.stem);
    styleSlider(key);
  });
}

function styleSlider(key) {
  const slider = document.getElementById('slider-' + key);
  if (!slider) return;
  const v = parseFloat(slider.value);
  const pct = ((v - 0.40) / 0.50) * 100;
  const color = sliderColor(v);
  slider.style.background = `linear-gradient(to right, ${color} ${pct}%, var(--border) ${pct}%)`;
  slider.style.setProperty('--thumb', color);
  slider.style.setProperty('--webkit-slider-thumb', `background:${color}`);
}

function updateSlider(key, rawVal) {
  const v = parseFloat(rawVal);
  document.getElementById('val-' + key).textContent = v.toFixed(2);
  document.getElementById('lbl-' + key).innerHTML = thresholdLabel(v);
  styleSlider(key);
}

function toggleGlobal(key, useGlobal, stemA, stemB) {
  const row = document.getElementById('row-' + key);
  if (useGlobal) {
    row.style.opacity = '0.35';
    row.style.pointerEvents = 'none';
    // Snap slider to current global threshold for visual clarity
    const g = parseFloat(document.getElementById('global_threshold').value);
    const slider = document.getElementById('slider-' + key);
    if (slider) { slider.value = g; updateSlider(key, g); }
  } else {
    row.style.opacity = '1';
    row.style.pointerEvents = '';
  }
}

// ── Collect settings from UI ──────────────────────────────────────────────────
function collectSettings() {
  const globalThresh = parseFloat(document.getElementById('global_threshold').value);
  const pairThresholds = {};

  document.querySelectorAll('.pair-card').forEach(card => {
    const key = card.id.replace('card-', '');
    const chk = document.getElementById('chk-' + key);
    if (!chk || chk.checked) return; // using global → don't store
    const val = parseFloat(document.getElementById('slider-' + key).value);
    pairThresholds[key] = val;
  });

  return {
    global_threshold:    globalThresh,
    reid_every_n_frames: parseInt(document.getElementById('reid_every_n_frames').value),
    gallery_ttl_frames:  parseInt(document.getElementById('gallery_ttl_frames').value),
    max_exemplars:       parseInt(document.getElementById('max_exemplars').value),
    ema_alpha:           parseFloat(document.getElementById('ema_alpha').value),
    pair_thresholds:     pairThresholds,
  };
}

// ── Save ──────────────────────────────────────────────────────────────────────
async function saveSettings() {
  const data = collectSettings();
  try {
    const res = await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (res.ok) {
      savedSettings = data;
      showToast('✓ Settings saved to config/pipeline_settings.json', 'success');
    } else {
      showToast('✗ Server error: ' + res.status, 'error');
    }
  } catch(e) {
    showToast('✗ ' + e.message, 'error');
  }
}

// ── Reset ─────────────────────────────────────────────────────────────────────
function resetToDefaults() {
  const defaults = {
    global_threshold: 0.65, reid_every_n_frames: 5,
    gallery_ttl_frames: 50000, max_exemplars: 8, ema_alpha: 0.15,
    pair_thresholds: {}
  };
  applySettings(defaults);
  savedSettings = defaults;
  renderPairs();
  showToast('↺ Reset to defaults', 'success');
}

// ── Toast notification ────────────────────────────────────────────────────────
function showToast(msg, type='success') {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = `show ${type}`;
  clearTimeout(t._timer);
  t._timer = setTimeout(() => { t.className = ''; }, 3500);
}

// Also restyle slider thumbs via CSS variable
document.addEventListener('DOMContentLoaded', () => {
  const style = document.createElement('style');
  style.textContent = `
    input[type=range]::-webkit-slider-thumb { background: var(--accent); }
  `;
  document.head.appendChild(style);
});

init();
</script>
</body>
</html>
"""


# ── HTTP Handler ─────────────────────────────────────────────────────────────

class ConfigHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Suppress default access log spam; only show errors
        if int(args[1]) >= 400:
            super().log_message(fmt, *args)

    def _send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path

        if path == "/" or path == "/index.html":
            body = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif path == "/api/cameras":
            self._send_json(_scan_cameras())

        elif path == "/api/settings":
            self._send_json(_load_settings())

        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/settings":
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            try:
                data = json.loads(body)
                _save_settings(data)
                self._send_json({"status": "ok"})
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
        else:
            self.send_error(404)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Launch pipeline configuration UI")
    parser.add_argument("--port",       type=int, default=7700, help="Local port (default 7700)")
    parser.add_argument("--no-browser", action="store_true",    help="Don't open browser automatically")
    args = parser.parse_args()

    url = f"http://localhost:{args.port}"
    server = HTTPServer(("localhost", args.port), ConfigHandler)

    print(f"\n{'='*55}")
    print(f"  Pipeline Configuration UI")
    print(f"{'='*55}")
    print(f"  URL     : {url}")
    print(f"  Cameras : {DATA_DIR}")
    print(f"  Config  : {SETTINGS_FILE}")
    print(f"{'='*55}")
    print(f"  Press  Ctrl+C  to stop\n")

    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Config] Server stopped.")


if __name__ == "__main__":
    main()
