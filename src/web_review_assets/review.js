"use strict";
const $ = (id) => document.getElementById(id),
  el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  };
const fmt = (t) =>
    `${Math.floor(Math.max(0, t) / 60)}:${String(Math.floor(Math.max(0, t) % 60)).padStart(2, "0")}`,
  name = (a) => a.replaceAll("_", " ");
let state,
  time = 0,
  playing = false,
  focus = null,
  followed = null,
  duration = 0,
  speed = 1,
  showBoxes = true,
  selected = new Set(),
  tiles = new Map(),
  people = new Map(),
  chunks = new Map(),
  requests = new Map(),
  editSource = null,
  busy = false;
const officeMap = new OfficeMap({
  requestHistory: (person) => api(`/api/office-map/${person}`),
  requestOverview: (index) => api(`/api/office-map/overview/${index}`),
  follow: (gid, camera) => follow(gid, camera),
  seek: (t) => { setPlaying(false); seek(t); },
  jump: (camera, t) => { setPlaying(false); jump(camera, t, followed); },
  togglePlayback: () => $("play").click(),
});
const officeEditor = new OfficeCameraEditor({
  save: (changes) => api("/api/office-map/placements", changes),
  onSaved: ({office_map}) => {
    state.office_map = office_map;
    if (officeMap.open) showOfficeMap();
    notice("Camera layout saved.");
  },
});
function notice(text) {
  $("notice").textContent = text;
}
async function api(path, body) {
  const r = await fetch(
    path,
    body === undefined
      ? {}
      : {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-Review-Token": state.token,
          },
          body: JSON.stringify(body),
        },
  );
  if (!r.ok) {
    let e;
    try {
      e = await r.json();
    } catch {}
    throw Error(e?.error || `Request failed (${r.status})`);
  }
  return r.json();
}
function button(text, fn, cls) {
  const b = el("button", cls, text);
  b.onclick = fn;
  return b;
}
function photo(person, cls = "thumb") {
  const n = el("img", cls);
  n.alt = `Person ${person.global_id ?? ""}`;
  if (person.thumbnail_b64)
    n.src = "data:image/jpeg;base64," + person.thumbnail_b64;
  return n;
}
function visibleCameras() {
  return [...tiles.keys()].filter((c) => focus === null || c === focus);
}
function activeCameras(person, t = time) {
  return (person.appearances || [])
    .filter((a) => {
      const fps = state.cameras[a.camera_id]?.fps || 25;
      return t >= a.first_frame / fps && t < (a.last_frame + 1) / fps;
    })
    .map((a) => String(a.camera_id));
}
async function chunk(cid, t) {
  const index = Math.max(0, Math.floor(t / 10)),
    key = `${state.revision}:${cid}:${index}`;
  if (chunks.has(key)) return chunks.get(key);
  if (requests.has(key)) return requests.get(key);
  const promise = api(`/api/boxes/${cid}/${index}`)
    .then((data) => {
      chunks.set(key, data);
      if (chunks.size > 30) chunks.delete(chunks.keys().next().value);
      return data;
    })
    .catch((e) => {
      notice(e.message);
      return {};
    })
    .finally(() => requests.delete(key));
  requests.set(key, promise);
  return promise;
}
function currentBoxes(cid, t) {
  const key = `${state.revision}:${cid}:${Math.floor(t / 10)}`;
  return (
    chunks.get(key)?.[String(Math.floor(t * state.cameras[cid].fps + 0.001))] ||
    []
  );
}
function view(cid) {
  focus = cid;
  $("cameras").classList.toggle("single", cid !== null);
  for (const [id, tile] of tiles) {
    tile.card.hidden = cid !== null && cid !== id;
    if (tile.card.hidden) tile.video.pause();
    else {
      if (Math.abs(tile.video.currentTime - time) > 0.2)
        tile.video.currentTime = Math.min(time, Number.isFinite(tile.video.duration) ? tile.video.duration : time);
      if (playing) playVideo(tile);
    }
  }
  for (const b of $("cameraButtons").children)
    b.classList.toggle("selected", b.dataset.camera === (cid ?? "all"));
}
function playVideo(tile) {
  if (tile.video.ended) return;
  tile.video.playbackRate = speed;
  tile.video.play().catch(() => {
    tile.status.textContent = "Press Play to resume";
  });
}
function setPlaying(value) {
  playing = value;
  $("play").textContent = value ? "Ⅱ Pause" : "▶ Play";
  for (const [cid, tile] of tiles) {
    if (value && visibleCameras().includes(cid)) playVideo(tile);
    else tile.video.pause();
  }
  officeMap.update(time, playing);
}
function seek(t) {
  time = Math.max(0, Math.min(duration, t));
  for (const [cid, tile] of tiles) {
    if (!tile.card.hidden) {
      tile.video.currentTime = Math.min(
        time,
        Number.isFinite(tile.video.duration) ? tile.video.duration : time,
      );
      chunk(cid, time);
    }
  }
  updateUI();
}
function follow(gid, camera) {
  followed = String(gid);
  $("followBar").hidden = false;
  $("followText").textContent = `Following Person ${gid}`;
  const p = state.persons[gid],
    cams = activeCameras(p);
  if (camera && state.cameras[camera]) view(camera);
  else if (cams.length) view(cams[0]);
  else if (p.appearances?.length) {
    const a = p.appearances[0];
    view(String(a.camera_id));
    seek(a.first_frame / state.cameras[a.camera_id].fps);
  }
  showOfficeMap();
}
function stopFollow() {
  const wasFollowing = followed !== null;
  followed = null;
  $("followBar").hidden = true;
  if (wasFollowing) officeMap.setOpen(false);
}
function showOfficeMap() {
  officeMap.show(followed, state, duration);
  officeMap.update(time, playing);
}
function jump(cid, t, gid) {
  $("workspaceTab").click();
  if (gid !== undefined) {
    followed = String(gid);
    $("followBar").hidden = false;
    $("followText").textContent = `Following Person ${gid}`;
  }
  view(String(cid));
  seek(t);
  if (officeMap.open) showOfficeMap();
}
function buildCameras() {
  for (const t of tiles.values()) t.video.pause();
  tiles.clear();
  $("cameras").replaceChildren();
  $("cameraButtons").replaceChildren();
  const all = button("All cameras", () => {
    stopFollow();
    view(null);
  });
  all.dataset.camera = "all";
  $("cameraButtons").append(all);
  for (const [cid, c] of Object.entries(state.cameras)) {
    const b = button(`Camera ${cid}`, () => {
      stopFollow();
      view(cid);
    });
    b.dataset.camera = cid;
    $("cameraButtons").append(b);
    const card = el("article", "cameraTile"),
      caption = el("div", "cameraCaption"),
      status = el("span", "", "Loading…"),
      viewport = el("div", "viewport"),
      video = el("video"),
      canvas = el("canvas");
    caption.append(el("strong", "", `CAM ${cid}`), status);
    card.append(caption, viewport);
    viewport.append(video, canvas);
    video.muted = true;
    video.playsInline = true;
    video.preload = "auto";
    video.loop = false;
    if (c.available) video.src = `/video/${cid}`;
    else status.textContent = c.requires_original ? "Original video needed for corrected IDs" : "Recording unavailable";
    const tile = { card, video, canvas, status };
    tiles.set(cid, tile);
    video.addEventListener("loadedmetadata", () => {
      status.textContent = c.raw
        ? "Recorded · original source"
        : "Recorded · baked-in IDs";
      video.currentTime = Math.min(time, video.duration);
    });
    video.addEventListener(
      "waiting",
      () => (status.textContent = "Buffering…"),
    );
    video.addEventListener("playing", () => (status.textContent = "Playing"));
    video.addEventListener("pause", () => {
      if (!video.ended) status.textContent = "Paused";
    });
    video.addEventListener(
      "error",
      () => (status.textContent = "Video unavailable — check server console"),
    );
    viewport.ondblclick = () => view(focus === cid ? null : cid);
    $("cameras").append(card);
    chunk(cid, 0);
  }
  view(focus);
}
function buildPeople() {
  people.clear();
  $("people").replaceChildren();
  $("galleryCards").replaceChildren();
  for (const [gid, p] of Object.entries(state.persons)) {
    const card = el("article", "person"),
      top = el("div", "personTop"),
      info = el("div", "info"),
      badge = el("span", "badge", "Not visible");
    info.append(el("strong", "", `Person ${gid.padStart(3, "0")}`), badge);
    top.append(
      photo(p),
      info,
      button("Follow", () => follow(gid)),
    );
    card.append(top);
    const details = el("details");
    details.append(
      el(
        "summary",
        "",
        `${p.appearances?.length || 0} appearances · cameras ${(p.cameras_seen || []).join(", ")}`,
      ),
    );
    for (const a of p.appearances || [])
      details.append(
        button(
          `Cam ${a.camera_id} · ${a.first_timestamp}–${a.last_timestamp} · ${a.duration_seconds}s`,
          () =>
            jump(
              a.camera_id,
              a.first_frame / state.cameras[a.camera_id].fps,
              gid,
            ),
          "sighting",
        ),
      );
    const events = (state.actions.events || []).filter(
      (e) => String(e.person_id) === gid,
    );
    details.append(
      el("p", "subtle", `Suggested actions · ${events.length} unreviewed`),
    );
    for (const e of events)
      details.append(
        button(
          `${name(e.action)} · Cam ${e.camera_id} · ${fmt(e.start_seconds)}–${fmt(e.end_seconds)}`,
          () =>
            jump(
              e.camera_id,
              e.evidence_frame / state.cameras[e.camera_id].fps,
              gid,
            ),
          "event",
        ),
      );
    card.append(details);
    $("people").append(card);
    people.set(gid, { card, badge, last: "" });
    const g = el("article", "galleryCard");
    g.append(
      el("h2", "", `Person ${gid.padStart(3, "0")}`),
      photo(p),
      el("p", "", `${p.appearances?.length || 0} appearances`),
      el(
        "p",
        "",
        `${(p.appearances || []).reduce((s, a) => s + a.duration_seconds, 0).toFixed(1)} seconds visible`,
      ),
      el("p", "", `Cameras ${(p.cameras_seen || []).join(", ")}`),
      el("p", "subtle", `${state.person_counts?.[gid] ?? p.total_detections ?? 0} detections`),
    );
    const controls = el("div", "toolbar");
    controls.append(
      button("View in footage", () => {
        $("workspaceTab").click();
        follow(gid);
      }),
      button("Merge with…", () => openMerge(gid)),
      button("Delete ID…", () => openDeleteId(gid), "danger"),
    );
    g.append(controls);
    $("galleryCards").append(g);
  }
  updateUI();
}
function updateUI() {
  officeMap.update(time, playing);
  for (const [gid, p] of people) {
    const cams = activeCameras(state.persons[gid]),
      signature = cams.join(",");
    if (signature !== p.last) {
      p.badge.textContent = cams.length
        ? `Visible · Cam ${signature}`
        : "Not visible";
      p.badge.classList.toggle("live", !!cams.length);
      p.card.classList.toggle("active", !!cams.length);
      p.last = signature;
    }
  }
  $("clock").textContent = `${fmt(time)} / ${fmt(duration)}`;
  if (document.activeElement !== $("seek")) $("seek").value = time;
}
function draw(cid, tile) {
  const { video, canvas } = tile,
    c = state.cameras[cid];
  const w = canvas.clientWidth,
    h = canvas.clientHeight;
  if (!w || !h) return;
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, w, h);
  const vw = video.videoWidth || c.width,
    vh = video.videoHeight || c.height;
  if (!vw || !vh) return;
  const scale = Math.min(w / vw, h / vh),
    ox = (w - vw * scale) / 2,
    oy = (h - vh * scale) / 2,
    sx = (vw * scale) / c.width,
    sy = (vh * scale) / c.height;
  const t = video.currentTime,
    boxes = currentBoxes(cid, t),
    counts = {};
  for (const d of boxes) counts[d.gid] = (counts[d.gid] || 0) + 1;
  for (const d of boxes) {
    if (followed !== null && String(d.gid) !== followed) continue;
    const [x1, y1, x2, y2] = d.bbox,
      x = ox + x1 * sx,
      y = oy + y1 * sy;
    const unknown = d.gid < 0;
    const color = unknown ? "#b1bac7" : state.persons[d.gid]?.color_hex || "#67dccc";
    ctx.font = "12px system-ui";
    if (showBoxes && c.raw) {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, (x2 - x1) * sx, (y2 - y1) * sy);
      ctx.fillStyle = color;
      ctx.fillRect(x, Math.max(oy, y - 20), 70, 20);
      ctx.fillStyle = "#061018";
      ctx.fillText(
        unknown ? "Unknown" : `ID:${String(d.gid).padStart(3, "0")}`,
        x + 5,
        Math.max(oy, y - 20) + 14,
      );
    }
    if (unknown || counts[d.gid] !== 1) continue;
    const labels = (state.actions.events || [])
      .filter(
        (e) =>
          e.camera_id === Number(cid) &&
          e.person_id === d.gid &&
          selected.has(e.action) &&
          e.start_seconds <= t &&
          e.end_seconds >= t,
      )
      .map((e) => name(e.action));
    labels.forEach((label, i) => {
      const text = `Suggested: ${label}`,
        tw = Math.min(w - 10, ctx.measureText(text).width + 10),
        tx = Math.max(0, Math.min(w - tw, x)),
        ty = Math.max(0, Math.min(h - 20, y + 24 + i * 20));
      ctx.fillStyle = "#102332ee";
      ctx.fillRect(tx, ty, tw, 20);
      ctx.fillStyle = "#93efd5";
      ctx.fillText(text, tx + 5, ty + 14, tw - 10);
    });
  }
}
let lastTick = 0,
  lastUI = 0;
function tick(now) {
  if (state && now - lastTick > 40) {
    const elapsed = Math.min(.25, (now - lastTick) / 1000);
    lastTick = now;
    const ids = visibleCameras(),
      master = ids
        .map((id) => tiles.get(id))
        .find((t) => !t.video.ended && t.video.readyState >= 2);
    if (playing && officeMap.open) {
      time = Math.min(duration, time + elapsed * speed);
      if (time >= duration) setPlaying(false);
    } else if (playing && master) {
      time = master.video.currentTime;
    }
    if (playing && (officeMap.open || master)) {
      for (const cid of ids) {
        const t = tiles.get(cid);
        if (
          (officeMap.open || t !== master) &&
          t.video.readyState >= 2 &&
          !t.video.seeking &&
          Math.abs(t.video.currentTime - time) > 0.45 &&
          !t.video.ended
        )
          t.video.currentTime = Math.min(time, t.video.duration || time);
      }
    }
    if (playing && !officeMap.open && ids.length && ids.every((id) => tiles.get(id).video.ended))
      setPlaying(false);
    if (followed !== null) {
      const cams = activeCameras(state.persons[followed]);
      if (cams.length && !cams.includes(focus)) {
        view(cams[0]);
      }
    }
    for (const cid of visibleCameras()) {
      const tile = tiles.get(cid);
      chunk(cid, tile.video.currentTime);
      chunk(cid, tile.video.currentTime + 10);
      draw(cid, tile);
    }
    if (now - lastUI > 200) {
      lastUI = now;
      updateUI();
    }
  }
  requestAnimationFrame(tick);
}
function formatMessage(node, html) {
  if (typeof html !== "string") return;
  const content = node.querySelector(".messageContent");
  // The chat API renders Markdown with raw HTML and images disabled.
  content.innerHTML = html;
  content.classList.add("formatted");
  for (const link of content.querySelectorAll("a")) {
    link.target = "_blank";
    link.rel = "noopener noreferrer";
  }
}
function message(role, text, html) {
  const node = el("div", `message ${role}`);
  node.append(
    el("strong", "messageAuthor", role === "user" ? "You" : "Sensen AI"),
    el("div", "messageContent", text),
  );
  formatMessage(node, html);
  $("messages").append(node);
  $("messages").scrollTop = $("messages").scrollHeight;
  return node;
}
async function ask(event) {
  event.preventDefault();
  const prompt = $("prompt").value;
  if (!prompt.trim() || busy) return;
  busy = true;
  $("send").disabled = true;
  $("send").textContent = "Thinking…";
  const userMessage = message("user", prompt);
  $("prompt").value = "";
  try {
    const result = await api("/api/chat", { prompt, view: { camera: focus, followed, seconds: time } });
    formatMessage(userMessage, result.prompt_html);
    const node = message("ai", result.reply, result.reply_html);
    const links = el("div", "toolbar");
    const seen = new Set();
    for (const m of result.reply.matchAll(
      /\[?Cam\s*(\d+)\s*(?:@|at)\s*(\d+):(\d{2})\]?/gi,
    )) {
      const camera = String(Number(m[1]));
      const seconds = Number(m[2]) * 60 + Number(m[3]);
      const key = `${camera}:${seconds}`;
      if (!state.cameras[camera] || seen.has(key)) continue;
      seen.add(key);
      links.append(
        button(`Cam ${m[1]} · ${m[2]}:${m[3]}`, () =>
          jump(camera, seconds),
        ),
      );
    }
    const gids = new Set(
      [...result.reply.matchAll(/\[?Person\s*(\d+)\]?/gi)].map((m) => String(Number(m[1]))),
    );
    for (const gid of gids)
      if (state.persons[gid])
        links.append(button(`Follow ID ${gid}`, () => follow(gid)));
    if (links.childElementCount) node.append(links);
    for (const evidence of result.evidence || []) {
      node.append(
        el("p", "subtle", `Visual evidence · Person ${evidence.global_id}`),
      );
      const row = el("div", "evidence");
      (evidence.thumbnails_b64 || []).forEach((src, i) => {
        const b = button("", () => follow(String(evidence.global_id)));
        const image = el("img");
        image.src = "data:image/jpeg;base64," + src;
        image.alt = `Person ${evidence.global_id}, evidence view ${i + 1}`;
        b.append(image);
        row.append(b);
      });
      node.append(row);
    }
    $("messages").scrollTop = $("messages").scrollHeight;
  } catch (e) {
    message("ai", e.message);
  } finally {
    busy = false;
    $("send").disabled = false;
    $("send").textContent = "Ask ↗";
  }
}
function openMerge(gid) {
  editSource = gid;
  $("mergeDescription").textContent =
    `Move the sightings from Person ${personLabel(gid)} into the selected identity.`;
  $("mergeError").hidden = true;
  $("mergeTarget").replaceChildren();
  for (const id of Object.keys(state.persons))
    if (id !== gid) {
      const o = el("option", "", `Person ${personLabel(id)}`);
      o.value = id;
      $("mergeTarget").append(o);
    }
  $("confirmMerge").disabled = !$("mergeTarget").options.length;
  $("mergeDialog").showModal();
}
function openDeleteId(gid) {
  editSource = gid;
  $("deleteIdDescription").textContent = `Remove Person ${personLabel(gid)} and their sightings from the gallery, map, and footage overlays.`;
  $("deleteIdError").hidden = true;
  $("deleteIdDialog").showModal();
}
async function load() {
  state = await api("/api/state");
  $("editOfficeCameras").disabled = Object.keys(state.cameras).length === 0;
  $("showEveryone").disabled = $("editOfficeCameras").disabled;
  selected = new Set(state.selected);
  duration = Math.max(
    0,
    ...Object.values(state.cameras).map((c) => c.total_frames / c.fps),
  );
  $("seek").max = duration;
  $("summary").textContent =
    `${Object.keys(state.cameras).length} cameras · ${Object.keys(state.persons).length} people`;
  notice(state.actions.status);
  buildCameras();
  buildPeople();
  $("actionChoices").replaceChildren();
  for (const action of [...state.labels].sort()) {
    const label = el("label"),
      input = el("input");
    input.type = "checkbox";
    input.value = action;
    input.checked = selected.has(action);
    label.append(input, document.createTextNode(" " + name(action)));
    $("actionChoices").append(label);
  }
  if (officeMap.open) showOfficeMap();
}
$("play").onclick = () => {
  if (time >= duration) seek(0);
  setPlaying(!playing);
};
$("previous").onclick = () => {
  setPlaying(false);
  seek(time - 1 / state.cameras[focus || Object.keys(state.cameras)[0]].fps);
};
$("next").onclick = () => {
  setPlaying(false);
  seek(time + 1 / state.cameras[focus || Object.keys(state.cameras)[0]].fps);
};
$("seek").oninput = () => seek(Number($("seek").value));
$("speed").onchange = () => {
  speed = Number($("speed").value);
  for (const t of tiles.values()) t.video.playbackRate = speed;
};
$("stopFollow").onclick = stopFollow;
$("showEveryone").onclick = () => {
  if (officeMap.overview) officeMap.setOpen(false);
  else {
    stopFollow();
    showOfficeMap();
  }
};
$("editOfficeCameras").onclick = () => {
  setPlaying(false);
  officeEditor.open(state.office_map, state.cameras, focus);
};
$("officeMapToggle").onclick = () => {
  if (officeMap.open) officeMap.setOpen(false);
  else showOfficeMap();
};
$("workspaceTab").onclick = () => {
  $("workspace").hidden = false;
  $("gallery").hidden = true;
  $("workspaceTab").classList.add("selected");
  $("galleryTab").classList.remove("selected");
};
$("galleryTab").onclick = () => {
  setPlaying(false);
  $("workspace").hidden = true;
  $("gallery").hidden = false;
  $("galleryTab").classList.add("selected");
  $("workspaceTab").classList.remove("selected");
};
$("peopleTab").onclick = () => {
  $("peoplePanel").hidden = false;
  $("aiPanel").hidden = true;
  $("peopleTab").classList.add("selected");
  $("aiTab").classList.remove("selected");
};
$("aiTab").onclick = () => {
  $("peoplePanel").hidden = true;
  $("aiPanel").hidden = false;
  $("aiTab").classList.add("selected");
  $("peopleTab").classList.remove("selected");
};
$("search").oninput = () => {
  for (const [gid, p] of people)
    p.card.hidden = !gid.includes($("search").value.trim());
};
$("settingsButton").onclick = () => $("settings").showModal();
$("allActions").onclick = () =>
  document
    .querySelectorAll("#actionChoices input")
    .forEach((i) => (i.checked = true));
$("noActions").onclick = () =>
  document
    .querySelectorAll("#actionChoices input")
    .forEach((i) => (i.checked = false));
$("saveSettings").onclick = async () => {
  const actions = [
    ...document.querySelectorAll("#actionChoices input:checked"),
  ].map((i) => i.value);
  try {
    await api("/api/settings", { actions });
    selected = new Set(actions);
    showBoxes = $("boxesToggle").checked;
    $("settings").close();
  } catch (e) {
    notice(e.message);
  }
};
$("chatForm").onsubmit = ask;
$("clearChat").onclick = async () => {
  if (busy) return;
  try {
    await api("/api/chat/clear", {});
    $("messages").replaceChildren();
  } catch (e) {
    notice(e.message);
  }
};
$("cancelMerge").onclick = () => $("mergeDialog").close();
$("cancelDeleteId").onclick = () => $("deleteIdDialog").close();
$("confirmMerge").onclick = () => applyIdentityCorrection(true);
$("confirmDeleteId").onclick = () => applyIdentityCorrection(false);
for (const [dialog, cancel] of [["mergeDialog", "cancelMerge"], ["deleteIdDialog", "cancelDeleteId"]]) {
  $(dialog).addEventListener("cancel", (event) => {
    if ($(cancel).disabled) event.preventDefault();
  });
}
async function applyIdentityCorrection(merge) {
  const dialog = $(merge ? "mergeDialog" : "deleteIdDialog");
  const confirm = $(merge ? "confirmMerge" : "confirmDeleteId");
  const cancel = $(merge ? "cancelMerge" : "cancelDeleteId");
  const error = $(merge ? "mergeError" : "deleteIdError");
  setPlaying(false);
  confirm.disabled = cancel.disabled = true;
  error.hidden = true;
  let saved = false;
  try {
    const payload = {source: editSource, revision: state.revision};
    if (merge) payload.target = $("mergeTarget").value;
    const result = await api(merge ? "/api/merge" : "/api/delete-id", payload);
    saved = true;
    dialog.close();
    stopFollow();
    focus = null;
    chunks.clear();
    $("messages").replaceChildren();
    await load();
    const summary = merge ? `Merged into Person ${personLabel(result.kept_id)}.` : "False ID deleted.";
    notice([summary, "Remaining IDs renumbered.", result.activity_note].filter(Boolean).join(" "));
  } catch (e) {
    error.textContent = e.message;
    error.hidden = false;
    notice(saved ? "The edit was saved. Reload review to see the updated IDs." : e.message);
  } finally {
    confirm.disabled = cancel.disabled = false;
  }
}
document.addEventListener("keydown", (e) => {
  if (
    ["INPUT", "TEXTAREA", "SELECT", "BUTTON"].includes(e.target.tagName) ||
    document.querySelector("dialog[open]")
  )
    return;
  if (e.code === "Space") {
    e.preventDefault();
    $("play").click();
  }
  if (e.code === "ArrowLeft") $("previous").click();
  if (e.code === "ArrowRight") $("next").click();
});
load()
  .then(() => requestAnimationFrame(tick))
  .catch((e) => notice(e.message));
