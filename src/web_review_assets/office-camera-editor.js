"use strict";

class OfficeCameraEditor {
  constructor({save, onSaved}) {
    this.get = (id) => document.getElementById(`officeEditor${id}`);
    this.dialog = document.getElementById("officeCameraEditor");
    this.plan = this.get("Plan");
    this.camera = this.get("Camera");
    this.floor = this.get("Floor");
    this.zone = this.get("Zone");
    this.video = this.get("Video");
    this.heading = this.get("Heading");
    this.save = save;
    this.onSaved = onSaved;

    this.camera.onchange = () => this.selectCamera(this.camera.value);
    this.floor.onchange = () => this.draw();
    this.zone.onchange = () => this.update({zone: this.zone.value || null});
    this.heading.oninput = () => this.update({heading: Number(this.heading.value)});
    this.get("Remove").onclick = () => this.update({position: null, floor: null});
    this.get("Close").onclick = () => this.dialog.close();
    this.get("Save").onclick = () => this.saveLayout();
    this.dialog.addEventListener("cancel", (event) => { if (this.saving) event.preventDefault(); });
    this.dialog.addEventListener("close", () => {
      this.video.pause();
      this.video.removeAttribute("src");
      this.video.load();
    });

    this.plan.addEventListener("pointerdown", (event) => this.startDrag(event));
    this.plan.addEventListener("pointermove", (event) => this.moveDrag(event));
    this.plan.addEventListener("pointerup", (event) => this.endDrag(event));
    this.plan.addEventListener("pointercancel", () => { this.drag = null; this.draw(); });
    this.plan.addEventListener("keydown", (event) => this.nudge(event));
    new ResizeObserver(() => { if (this.dialog.open) this.draw(); }).observe(this.plan);
  }

  open(layout, recordings, preferredCamera) {
    this.layout = layout;
    this.recordings = recordings;
    this.draft = Object.fromEntries(Object.entries(layout.cameras).map(([id, pin]) => [id, {...pin}]));
    this.edited = new Set();
    this.drag = null;
    this.get("PlanName").textContent = layout.approximate ? `${layout.name} · approximate layout` : layout.name;
    this.get("Status").textContent = "";
    this.camera.replaceChildren();
    for (const [id, pin] of Object.entries(this.draft)) {
      this.camera.append(new Option(`Camera ${id} · ${pin.filename}`, id));
    }
    this.floor.replaceChildren(...layout.floors.map(floor => new Option(floor.label, floor.id)));
    this.zone.replaceChildren(new Option("No coverage area assigned", ""));
    for (const floor of layout.floors) {
      const group = document.createElement("optgroup");
      group.label = floor.label;
      for (const zone of layout.zones.filter(zone => zone.floor === floor.id)) {
        group.append(new Option(zone.label, zone.id));
      }
      if (group.children.length) this.zone.append(group);
    }
    this.dialog.showModal();
    this.selectCamera(this.draft[preferredCamera] ? preferredCamera : this.camera.options[0].value);
  }

  selectCamera(id) {
    this.camera.value = id;
    const pin = this.draft[id];
    const floor = pin.floor || this.layout.zones.find(zone => zone.id === pin.zone)?.floor;
    if (floor) this.floor.value = floor;
    this.video.hidden = !this.recordings[id].available;
    this.get("NoVideo").hidden = !this.video.hidden;
    if (this.video.hidden) {
      this.video.removeAttribute("src");
      this.video.load();
    } else {
      this.video.src = `/video/${id}`;
    }
    this.render();
  }

  render() {
    const pin = this.draft[this.camera.value];
    this.zone.value = pin.zone || "";
    this.heading.value = pin.heading;
    this.heading.disabled = !pin.position || this.saving;
    this.get("Degrees").textContent = `${pin.heading}°`;
    this.get("Remove").disabled = !pin.position || this.saving;
    const placed = Object.values(this.draft).filter(camera => camera.position).length;
    this.get("PinCount").textContent = `${placed} of ${this.camera.options.length} cameras placed`;
    this.get("Save").disabled = !this.edited.size || this.saving;
    this.draw();
  }

  update(changes) {
    if (this.saving) return;
    const id = this.camera.value;
    Object.assign(this.draft[id], changes);
    this.edited.add(id);
    this.get("Status").textContent = "Unsaved changes";
    this.render();
  }

  zoneAt(floor, position) {
    if (!position) return null;
    const [x, y] = position;
    return this.layout.zones.find(zone => {
      const [left, top, w, h] = zone.bounds;
      return zone.floor === floor && x >= left && x <= left + w && y >= top && y <= top + h;
    });
  }

  place(point) {
    const pin = this.draft[this.camera.value], floor = this.floor.value;
    const previous = this.zoneAt(pin.floor, pin.position);
    const next = this.zoneAt(floor, point);
    const zone = next && next.id !== previous?.id ? next.id : pin.zone;
    this.update({floor, position: point, zone});
  }

  eventPosition(event) {
    const box = this.plan.getBoundingClientRect();
    return [event.clientX - box.left, event.clientY - box.top];
  }

  startDrag(event) {
    if (this.saving || this.drag || event.button !== 0 || !this.project) return;
    const point = this.eventPosition(event);
    const hit = Object.entries(this.draft).reverse().find(([, pin]) => {
      if (!pin.position || pin.floor !== this.floor.value) return false;
      const [x, y] = this.project(...pin.position);
      return Math.hypot(x - point[0], y - point[1]) <= 13;
    });
    const [left, top, right, bottom] = this.screenBounds;
    if (!hit && (point[0] < left || point[0] > right || point[1] < top || point[1] > bottom)) return;
    if (hit && hit[0] !== this.camera.value) this.selectCamera(hit[0]);
    this.plan.focus({preventScroll: true});
    this.plan.setPointerCapture(event.pointerId);
    this.drag = {id: event.pointerId, start: point, hit: !!hit, moved: false};
  }

  moveDrag(event) {
    if (this.drag?.id !== event.pointerId) return;
    const screen = this.eventPosition(event);
    if (!this.drag.moved && Math.hypot(screen[0] - this.drag.start[0], screen[1] - this.drag.start[1]) < 4) return;
    this.drag.moved = true;
    this.drag.position = this.worldPoint(screen);
    this.draw();
  }

  endDrag(event) {
    if (this.drag?.id !== event.pointerId) return;
    const {moved, hit} = this.drag;
    this.drag = null;
    if (moved || !hit) this.place(this.worldPoint(this.eventPosition(event)));
    else this.draw();
  }

  nudge(event) {
    const moves = {ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1]};
    if (!moves[event.key] || this.saving || !this.project) return;
    event.preventDefault();
    const floor = this.layout.floors.find(f => f.id === this.floor.value);
    const pin = this.draft[this.camera.value];
    const zone = this.layout.zones.find(zone => zone.id === pin.zone && zone.floor === floor.id);
    const start = pin.floor === floor.id && pin.position ? pin.position : center(zone?.bounds || floor.bounds);
    const step = Math.min(...floor.bounds.slice(2)) * (event.shiftKey ? .05 : .012);
    const next = start.map((value, i) => value + moves[event.key][i] * step);
    this.place(this.worldPoint(this.project(...next)));
  }

  draw() {
    const frame = canvasFrame(this.plan);
    if (!frame) return;
    const {ctx, width, height} = frame;
    const floor = this.layout.floors.find(item => item.id === this.floor.value);
    const [x, y, w, h] = floor.bounds;
    const scale = Math.min((width - 36) / w, (height - 36) / h);
    if (scale <= 0) return;
    const [dx, dy] = [(width - w * scale) / 2, (height - h * scale) / 2];
    this.screenBounds = [dx, dy, dx + w * scale, dy + h * scale];
    this.project = (px, py) => [dx + (px - x) * scale, dy + (py - y) * scale];
    this.worldPoint = ([px, py]) => [x + Math.min(w, Math.max(0, (px - dx) / scale)),
      y + Math.min(h, Math.max(0, (py - dy) / scale))];
    const outline = (rect, fill, stroke) => polygon(ctx, corners(rect).map(point => this.project(...point)), fill, stroke);
    const selected = this.camera.value;
    outline(floor.bounds, "#0f1c2a", "#35516b");
    for (const zone of this.layout.zones) {
      if (zone.floor !== floor.id) continue;
      const coverage = this.draft[selected].zone === zone.id;
      outline(zone.bounds, coverage ? "#17453f" : "#203244", coverage ? "#62d8bd" : "#597083");
      const [cx, cy] = this.project(...center(zone.bounds));
      ctx.font = "600 13px system-ui";
      ctx.textAlign = "center";
      ctx.fillStyle = "#f0f7ff";
      ctx.fillText(zone.label, cx, cy + 4, Math.max(1, zone.bounds[2] * scale - 8));
    }
    const pins = Object.entries(this.draft).sort(([a], [b]) => Number(a === selected) - Number(b === selected));
    for (const [id, pin] of pins) {
      const moving = id === selected && this.drag?.position;
      const point = moving || (pin.floor === floor.id && pin.position);
      if (point) paintCameraPin(ctx, this.project(...point), cameraDirection(pin.heading), `C${id}`, id === selected);
    }
  }

  async saveLayout() {
    this.saving = true;
    this.get("Close").disabled = true;
    this.get("Status").textContent = "Saving camera layout…";
    this.render();
    try {
      const cameras = {};
      for (const id of this.edited) {
        const {zone, floor, position, heading} = this.draft[id];
        cameras[id] = {zone, floor, position, heading};
      }
      this.onSaved(await this.save({revision: this.layout.revision, cameras}));
      this.dialog.close();
    } catch (error) {
      this.get("Status").textContent = error.message;
    } finally {
      this.saving = false;
      this.get("Close").disabled = false;
      if (this.dialog.open) this.render();
    }
  }
}
