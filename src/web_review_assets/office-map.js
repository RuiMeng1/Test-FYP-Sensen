"use strict";

class OfficeMap {
  constructor({requestHistory, requestOverview, follow, seek, jump, togglePlayback}) {
    this.get = (id) => document.getElementById(id);
    this.panel = this.get("officeMap");
    this.scene = this.get("officeScene");
    this.coverage = this.get("officeCoverage");
    this.floor = this.get("officeFloor");
    this.slider = this.get("officeSeek");
    this.requestHistory = requestHistory;
    this.requestOverview = requestOverview;
    this.follow = follow;
    this.markers = this.get("officeMarkers");
    this.markerButtons = new Map();
    this.jump = jump;
    this.generation = 0;
    this.pageSize = 20;
    this.yaw = -34;
    this.top = false;
    this.time = 0;

    this.get("officePlay").onclick = togglePlayback;
    this.slider.oninput = () => seek(Number(this.slider.value));
    this.floor.onchange = () => this.update(this.time, this.playing);
    this.get("officeRotate").oninput = (event) => {
      this.yaw = Number(event.target.value);
      this.drawScene();
    };
    this.get("officeTop").onclick = (event) => {
      this.top = !this.top;
      event.currentTarget.setAttribute("aria-pressed", String(this.top));
      this.drawScene();
    };
    this.get("officeFullRoute").onchange = () => this.drawScene();
    this.get("officeEarlier").onclick = () => { this.page--; this.renderHistory(); };
    this.get("officeLater").onclick = () => { this.page++; this.renderHistory(); };
    new ResizeObserver(() => {
      if (this.open) {
        this.drawScene();
        this.drawCoverage();
      }
    }).observe(this.panel);
  }

  get open() {
    return !this.panel.hidden;
  }

  get overview() {
    return this.open && this.person === null;
  }

  setOpen(open) {
    this.panel.hidden = !open;
    this.get("reviewStage").classList.toggle("withOfficeMap", open);
    const button = this.get("officeMapToggle");
    button.textContent = open ? "Close 3D map" : "Open 3D map";
    button.setAttribute("aria-expanded", String(open));
    const everyone = this.get("showEveryone");
    everyone.classList.toggle("selected", this.overview);
    everyone.setAttribute("aria-pressed", String(this.overview));
    this.markers.hidden = !this.overview;
    if (!this.overview) this.clearMarkers();
    if (!open) {
      this.generation++;
      this.person = null;
    }
  }

  async show(person, state, duration) {
    const {office_map: layout, revision, cameras, persons} = state;
    if (this.open && this.person === person && this.revision === revision && this.layout.revision === layout.revision) return;
    const request = ++this.generation;
    const sameLayout = this.layout === layout;
    this.person = person;
    this.revision = revision;
    this.layout = layout;
    this.cameras = cameras;
    this.people = persons;
    this.duration = duration;
    this.segments = [];
    this.page = 0;
    this.message = "Loading recorded sightings…";
    this.overviewChunks = new Map();
    this.observations = [];
    const overview = person === null;
    this.zoneById = new Map(layout.zones.map(zone => [zone.id, zone]));
    this.floorById = new Map(layout.floors.map(floor => [floor.id, floor]));
    if (!sameLayout) {
      const previousFloor = this.floor.value;
      this.floor.replaceChildren(new Option("All floors", ""));
      for (const floor of layout.floors) this.floor.append(new Option(floor.label, floor.id));
      if (layout.floors.some(floor => floor.id === previousFloor)) this.floor.value = previousFloor;
      this.floor.parentElement.hidden = layout.floors.length === 1;
    }
    this.slider.max = duration;
    this.get("officeMapTitle").textContent = overview ? "Everyone · 3D office map" : `Person ${personLabel(person)} · 3D office map`;
    this.scene.setAttribute("aria-label", overview ? "3D view of camera zones and visible people" : "3D view of camera zones and the followed person's route");
    for (const id of ["officeHistory", "officeCoverage", "officeSightingsLegend", "officeGapsLegend"]) this.get(id).hidden = overview;
    this.get("officeOverviewHelp").hidden = !overview;
    this.get("officeMapNote").textContent = layout.schematic
      ? "Schematic camera zones · layout and positions are illustrative"
      : layout.approximate ? `${layout.name} · approximate layout`
        : `${layout.name} · positions estimated from recorded camera views`;
    this.get("officePinLegend").hidden = !Object.values(layout.cameras).some(camera => camera.position);
    this.setOpen(true);
    this.update(this.time, this.playing || false);
    if (overview) return;
    this.renderHistory();
    try {
      const history = await this.requestHistory(person);
      if (request !== this.generation) return;
      if (history.revision !== revision || history.map_revision !== layout.revision)
        throw new Error("Tracking or camera layout changed. Reload the review to update the map.");
      this.segments = history.segments;
      this.message = this.segments.length ? "" : "No recorded sightings for this person.";
    } catch (error) {
      if (request !== this.generation) return;
      this.message = error.message;
    }
    this.renderHistory();
    this.drawCoverage();
    this.update(this.time, this.playing);
  }

  update(time, playing) {
    if (!this.open) return;
    this.time = time;
    this.playing = playing;
    this.slider.value = time;
    this.get("officePlay").textContent = playing ? "Pause" : "Play";
    this.get("officeClock").textContent = `${officeTime(time)} / ${officeTime(this.duration)}`;
    if (this.overview) {
      this.updateOverview(time);
      this.drawScene();
      return;
    }
    this.active = [];
    for (const segment of this.segments) {
      if (segment.start > time) break;
      if (segment.end > time) this.active.push(segment);
    }
    const places = [...new Set(this.active.map(segment => {
      const zone = this.zoneById.get(segment.zone);
      const camera = this.layout.cameras[segment.camera];
      return zone ? `${zone.label} · ${segment.points.length ? "estimated position" : "camera zone"}`
        : `${camera.label} · camera has no map placement`;
    }))];
    const text = this.message || places.join(" / ") || "No recorded sighting at this time.";
    const location = this.get("officeLocation");
    if (location.textContent !== text) location.textContent = text;
    this.drawScene();
  }

  renderHistory() {
    const list = this.get("officeSightings");
    list.replaceChildren();
    this.get("officeHistoryCount").textContent = `(${this.segments.length})`;
    const start = this.page * this.pageSize;
    this.get("officeEarlier").disabled = this.page === 0;
    this.get("officeLater").disabled = start + this.pageSize >= this.segments.length;
    list.start = start + 1;
    for (const segment of this.segments.slice(start, start + this.pageSize)) {
      const row = document.createElement("li");
      const button = document.createElement("button");
      const name = this.layout.cameras[segment.camera].label;
      button.textContent = `${officeTime(segment.start, true)} – ${officeTime(segment.end, true)} · ${name}`;
      button.onclick = () => this.jump(segment.camera, segment.start);
      row.append(button);
      list.append(row);
    }
  }

  drawCoverage() {
    const frame = canvasFrame(this.coverage);
    if (!frame || !this.duration) return;
    const {ctx, width, height} = frame;
    ctx.fillStyle = "#28384b";
    ctx.fillRect(0, 0, width, height);
    for (const segment of this.segments) {
      ctx.fillStyle = segment.zone === null ? "#d7af74" : "#62dac5";
      ctx.fillRect(width * segment.start / this.duration, 0,
        Math.max(1, width * (segment.end - segment.start) / this.duration), height);
    }
  }

  loadOverviewChunk(index) {
    if (this.overviewChunks.has(index)) return this.overviewChunks.get(index);
    const entry = {};
    const generation = this.generation;
    this.overviewChunks.set(index, entry);
    if (this.overviewChunks.size > 4) this.overviewChunks.delete(this.overviewChunks.keys().next().value);
    this.requestOverview(index).then(result => {
      if (generation !== this.generation) return;
      if (result.revision !== this.revision || result.map_revision !== this.layout.revision)
        throw new Error("Tracking or camera layout changed. Reload the review to update the map.");
      entry.frames = result.frames;
    }).catch(error => { entry.error = error.message; }).finally(() => {
      if (generation === this.generation && Math.floor(this.time / 10) === index)
        this.update(this.time, this.playing);
    });
    return entry;
  }

  updateOverview(time) {
    const index = Math.floor(time / 10);
    const {frames, error} = this.loadOverviewChunk(index);
    if (this.playing && time % 10 >= 8 && (index + 1) * 10 < this.duration) this.loadOverviewChunk(index + 1);

    const visible = new Set(), mapped = new Set(), onFloor = new Set(), observations = new Map();
    const floor = this.floor.value;
    for (const [camera, sightings] of Object.entries(frames || {})) {
      const frame = Math.floor(time * this.cameras[camera].fps + 1e-7);
      const zone = this.zoneById.get(this.layout.cameras[camera].zone);
      for (const [gid, x, y] of sightings[frame] || []) {
        visible.add(gid);
        if (!zone) continue;
        mapped.add(gid);
        if (floor && zone.floor !== floor) continue;
        onFloor.add(gid);
        const key = `${gid}:${zone.id}`;
        const position = x === undefined ? null : [x, y];
        if (!observations.has(key) || (position && !observations.get(key).position))
          observations.set(key, {key, gid, camera, zone, position});
      }
    }
    this.observations = [...observations.values()];
    const count = onFloor.size;
    const unmapped = visible.size - mapped.size;
    const text = error || (!frames ? "Loading visible people…" : !visible.size ? "No recorded sightings at this time."
      : `${count} ${count === 1 ? "person" : "people"} ${floor ? "on this floor" : "on the map"}` +
        (unmapped ? ` · ${unmapped} without mapped coverage` : ""));
    const location = this.get("officeLocation");
    if (location.textContent !== text) location.textContent = text;
  }

  clearMarkers() {
    this.markers.replaceChildren();
    this.markerButtons.clear();
  }

  drawMarkers({ctx, width, height}, project) {
    const occupied = [], shown = new Set();
    const observations = [...this.observations].sort((a, b) => Number(!!b.position) - Number(!!a.position) || a.gid - b.gid);
    for (const {key, gid, camera, zone, position} of observations) {
      const point = project(...(position || center(zone.bounds)), this.floorById.get(zone.floor).elevation);
      if (!position) point[1] -= 8;
      const offset = markerPosition(point, occupied, width, height);
      occupied.push(offset);
      if (position && Math.hypot(offset[0] - point[0], offset[1] - point[1]) > 4) {
        ctx.beginPath();
        ctx.moveTo(...point);
        ctx.lineTo(...offset);
        ctx.lineWidth = 1;
        ctx.strokeStyle = "#92a9bb";
        ctx.stroke();
      }

      let button = this.markerButtons.get(key);
      if (!button) {
        button = document.createElement("button");
        button.type = "button";
        button.className = "officePersonMarker";
        button.dataset.label = `ID ${personLabel(gid)}`;
        button.dataset.gid = gid;
        button.setAttribute("aria-label", `Follow Person ${personLabel(gid)} in ${zone.label}`);
        button.style.backgroundColor = this.people[gid].color_hex || "#67dccc";
        button.onclick = () => this.follow(button.dataset.gid, button.dataset.camera);
        this.markerButtons.set(key, button);
        this.markers.append(button);
      }
      button.dataset.camera = camera;
      button.style.left = `${offset[0]}px`;
      button.style.top = `${offset[1]}px`;
      shown.add(key);
    }
    for (const [key, button] of this.markerButtons) {
      if (shown.has(key)) continue;
      button.remove();
      this.markerButtons.delete(key);
    }
  }

  drawScene() {
    const frame = canvasFrame(this.scene);
    if (!frame || !this.layout) return;
    const {ctx, width, height} = frame;
    const shown = this.floor.value;
    const floors = this.layout.floors.filter(floor => !shown || floor.id === shown)
      .sort((a, b) => a.elevation - b.elevation);
    if (!floors.length) return;

    const theta = this.yaw * Math.PI / 180;
    const [sin, cos] = [Math.sin(theta), Math.cos(theta)];
    const tilt = this.top ? 1 : .61;
    const raw = (x, y, z) => [x * cos - y * sin, (x * sin + y * cos) * tilt - z * (this.top ? 0 : .8)];
    const footprint = floors.flatMap(floor => corners(floor.bounds).map(p => raw(...p, floor.elevation)));
    const x = footprint.map(p => p[0]), y = footprint.map(p => p[1]);
    const [left, right, top, bottom] = [Math.min(...x), Math.max(...x), Math.min(...y), Math.max(...y)];
    const scale = Math.min((width - 34) / (right - left), (height - 69) / (bottom - top));
    if (scale <= 0) return;
    const project = (x, y, z) => {
      const [sx, sy] = raw(x, y, z);
      return [width / 2 + (sx - (left + right) / 2) * scale,
        (height - 25) / 2 + (sy - (top + bottom) / 2) * scale];
    };
    const full = this.get("officeFullRoute").checked;
    const route = [];
    const visited = new Set();
    for (const segment of this.segments) {
      if (!full && segment.start > this.time) break;
      if (segment.zone !== null) visited.add(segment.zone);
      if (segment.points.length) route.push(segment);
    }

    for (const floor of floors) {
      const z = floor.elevation;
      const slab = corners(floor.bounds).map(p => project(...p, z));
      if (!this.top) {
        for (let i = 0; i < 4; i++) {
          const a = slab[i], b = slab[(i + 1) % 4];
          if (b[0] < a[0]) continue;
          polygon(ctx, [a, b, [b[0], b[1] + 8], [a[0], a[1] + 8]], "#172b3b", "#37536a");
        }
      }
      polygon(ctx, slab, "#122131e8", "#3a576c");
      for (const zone of this.layout.zones) {
        if (zone.floor !== floor.id) continue;
        const shape = corners(zone.bounds).map(p => project(...p, z));
        const fill = visited.has(zone.id) ? "#22524d" : "#1b3447";
        if (!this.top) {
          for (let i = 0; i < 4; i++) {
            const a = shape[i], b = shape[(i + 1) % 4];
            if (b[0] < a[0]) continue;
            polygon(ctx, [a, b, [b[0], b[1] + 4], [a[0], a[1] + 4]], "#142b37", "#3b5363");
          }
        }
        polygon(ctx, shape, fill, visited.has(zone.id) ? "#78d9c9" : "#4b687e");
        const [cx, cy] = project(...center(zone.bounds), z);
        const labelWidth = Math.max(...shape.map(p => p[0])) - Math.min(...shape.map(p => p[0]));
        if (labelWidth < 20) continue;
        ctx.font = "600 11px system-ui";
        ctx.textAlign = "center";
        ctx.fillStyle = "#e5f1fa";
        ctx.fillText(zone.label, cx, cy + 14, labelWidth - 5);
      }
      for (const segment of route) {
        if (this.zoneById.get(segment.zone)?.floor !== floor.id) continue;
        ctx.beginPath();
        let first = true;
        for (const [timestamp, px, py] of segment.points) {
          if (!full && timestamp > this.time) break;
          const point = project(px, py, z);
          if (first) ctx.moveTo(...point);
          else ctx.lineTo(...point);
          first = false;
        }
        ctx.lineWidth = 2;
        ctx.lineJoin = "round";
        ctx.strokeStyle = "#77e2ca";
        ctx.stroke();
      }
      for (const [id, camera] of Object.entries(this.layout.cameras)) {
        if (camera.floor !== floor.id || !camera.position) continue;
        const [x, y] = camera.position;
        const [dx, dy] = cameraDirection(camera.heading);
        const point = project(x, y, z), target = project(x + dx, y + dy, z);
        paintCameraPin(ctx, point, [target[0] - point[0], target[1] - point[1]], `C${id}`);
      }
    }

    if (this.overview) {
      this.drawMarkers(frame, project);
      return;
    }
    const color = this.people[this.person]?.color_hex || "#3ec9b4";
    ctx.lineWidth = 2;
    const zoneDots = new Set();
    for (const segment of this.active || []) {
      const zone = this.zoneById.get(segment.zone);
      if (!zone || (shown && zone.floor !== shown)) continue;
      if (!segment.points.length && zoneDots.has(zone.id)) continue;
      if (!segment.points.length) zoneDots.add(zone.id);
      const position = segment.points.length ? routePosition(segment.points, this.time) : center(zone.bounds);
      const [px, py] = project(...position, this.floorById.get(zone.floor).elevation);
      ctx.beginPath();
      ctx.arc(px, py, 9, 0, 2 * Math.PI);
      ctx.fillStyle = "#6df0d94d";
      ctx.fill();
      ctx.beginPath();
      ctx.arc(px, py, 5, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = "#e3fff8";
      ctx.stroke();
    }
  }
}

function personLabel(gid) {
  return String(gid).padStart(3, "0");
}

function markerPosition([x, y], occupied, width, height) {
  const clamp = ([px, py]) => [Math.max(12, Math.min(width - 12, px)), Math.max(15, Math.min(height - 55, py))];
  const first = clamp([x, y]);
  const free = ([px, py]) => occupied.every(([ox, oy]) => Math.hypot(px - ox, py - oy) >= 22);
  if (free(first)) return first;
  for (let ring = 1; ring <= 4; ring++) {
    for (let step = 0; step < 6 * ring; step++) {
      const angle = step * 2 * Math.PI / (6 * ring);
      const point = clamp([x + 23 * ring * Math.sin(angle), y - 23 * ring * Math.cos(angle)]);
      if (free(point)) return point;
    }
  }
  return first;
}

function officeTime(seconds, detailed = false) {
  const t = Math.max(0, seconds);
  const h = Math.floor(t / 3600), m = Math.floor(t % 3600 / 60), s = Math.floor(t % 60);
  const pad = (n) => String(n).padStart(2, "0");
  const clock = h ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
  return detailed ? `${clock}.${Math.floor(t * 10) % 10}` : clock;
}

function canvasFrame(canvas) {
  const {width, height} = canvas.getBoundingClientRect();
  if (width < 1 || height < 1) return null;
  const ratio = Math.min(2, window.devicePixelRatio || 1);
  const pixelWidth = Math.round(width * ratio), pixelHeight = Math.round(height * ratio);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return {ctx, width, height};
}

function corners([x, y, w, d]) {
  return [[x, y], [x + w, y], [x + w, y + d], [x, y + d]];
}

function center([x, y, w, d]) {
  return [x + w / 2, y + d / 2];
}

function polygon(ctx, points, fill, stroke) {
  ctx.beginPath();
  ctx.moveTo(...points[0]);
  for (const point of points.slice(1)) ctx.lineTo(...point);
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.lineWidth = 1;
  ctx.strokeStyle = stroke;
  ctx.stroke();
}

function cameraDirection(heading) {
  const angle = heading * Math.PI / 180;
  return [Math.sin(angle), -Math.cos(angle)];
}

function paintCameraPin(ctx, [x, y], [dx, dy], label, selected = false) {
  const length = Math.hypot(dx, dy);
  const ux = dx / length, uy = dy / length;
  const color = selected ? "#58dfc5" : "#8fbafa";
  polygon(ctx, [[x + 18 * ux, y + 18 * uy], [x + 7 * ux + 5 * uy, y + 7 * uy - 5 * ux],
    [x + 7 * ux - 5 * uy, y + 7 * uy + 5 * ux]], color, "#0d1826");
  ctx.beginPath();
  ctx.arc(x, y, 9, 0, Math.PI * 2);
  ctx.fillStyle = "#172b40";
  ctx.fill();
  ctx.strokeStyle = color;
  ctx.lineWidth = selected ? 2 : 1;
  ctx.stroke();
  ctx.fillStyle = "#edf6ff";
  ctx.font = "700 8px system-ui";
  ctx.textAlign = "center";
  ctx.fillText(label, x, y + 3, 15);
}

function routePosition(points, time) {
  let lo = 0, hi = points.length - 1;
  while (hi - lo > 1) {
    const mid = Math.floor((lo + hi) / 2);
    if (points[mid][0] <= time) lo = mid;
    else hi = mid;
  }
  const [start, x, y] = points[lo], [end, nx, ny] = points[hi];
  const amount = end > start ? Math.max(0, Math.min(1, (time - start) / (end - start))) : 0;
  return [x + amount * (nx - x), y + amount * (ny - y)];
}
