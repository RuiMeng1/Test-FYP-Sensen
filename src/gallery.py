"""
gallery.py — Thread-safe global identity gallery for multi-camera Person Re-ID.

Core improvements:
  1. Multi-Exemplar Feature Bank:
     Stores multiple diverse appearance vectors per person (front, back, side angles)
     instead of a single EMA vector that degrades when a person turns around.
     Match score is the maximum similarity across the person's exemplar bank.

  2. Mutual Exclusivity Constraint (Per-Frame Bipartite Matching):
     No two active person detections in the same camera/frame can ever share
     the same Global ID.

  3. Smart Same-Camera Continuity:
     Tracks from the same camera can re-match previous identities if the track
     was lost (e.g. occlusion / turn-around), provided that identity is not
     currently claimed by another person visible in the same frame.

  4. Per-Camera-Pair Similarity Thresholds (improve/reid-robustness):
     Cameras with extreme angle differences (e.g. portrait vs landscape) use a
     lower matching threshold than well-aligned pairs to reduce false negatives
     caused by natural embedding drift across viewpoints.

  5. Larger Exemplar Bank + Cross-Camera Relaxed Contamination Guard:
     Bank size raised from 6 → 8 for better viewpoint diversity.
     Cross-camera exemplar updates use a looser contamination floor since
     different-camera appearances are inherently more dissimilar.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional, Set, Tuple

import numpy as np
import cv2


@dataclass
class GalleryEntry:
    MAX_THUMBNAILS: ClassVar[int] = 3

    global_id: int
    embeddings: List[np.ndarray] = field(default_factory=list)  # unit-norm feature bank
    last_seen_frame: int = 0                                    # for TTL expiry
    last_camera_id: int = 0
    hit_count: int = 1                                          # number of detections
    cameras_seen: Set[int] = field(default_factory=set)         # set of camera IDs
    thumbnails: List[np.ndarray] = field(default_factory=list)  # list of cropped images
    thumbnail_camera_ids: List[int] = field(default_factory=list)
    thumbnail_scores: List[float] = field(default_factory=list)

    def add_thumbnail(self, crop: np.ndarray, camera_id: int) -> None:
        """Keep the three best crops, preferring coverage across cameras."""
        if crop is None or crop.size == 0:
            return

        h, w = crop.shape[:2]
        if h <= 0 or w <= 0:
            return

        # Larger, sharper crops are generally more useful to the vision model.
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        quality = float(h * w) * (1.0 + min(sharpness, 1000.0) / 1000.0)

        # Keep enough source detail for clothing/accessory inspection while
        # avoiding large tracking-log payloads.
        target_h = min(256, h)
        target_w = int(w * (target_h / h))
        if target_w <= 0:
            return
        resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)

        if len(self.thumbnails) < self.MAX_THUMBNAILS:
            self.thumbnails.append(resized)
            self.thumbnail_camera_ids.append(camera_id)
            self.thumbnail_scores.append(quality)
            return

        camera_counts = {
            cid: self.thumbnail_camera_ids.count(cid)
            for cid in set(self.thumbnail_camera_ids)
        }

        duplicate_indices: List[int] = []
        if camera_id not in camera_counts:
            # Make room for a new viewpoint by replacing the weakest duplicate
            # camera view. If all views are already from different cameras,
            # replace the weakest crop only when the new one is better.
            duplicate_indices = [
                idx for idx, cid in enumerate(self.thumbnail_camera_ids)
                if camera_counts[cid] > 1
            ]
            candidate_indices = duplicate_indices or list(range(len(self.thumbnails)))
        else:
            candidate_indices = [
                idx for idx, cid in enumerate(self.thumbnail_camera_ids)
                if cid == camera_id
            ]

        replace_idx = min(candidate_indices, key=lambda idx: self.thumbnail_scores[idx])
        adds_new_camera = camera_id not in camera_counts and bool(duplicate_indices)
        if adds_new_camera or quality > self.thumbnail_scores[replace_idx]:
            self.thumbnails[replace_idx] = resized
            self.thumbnail_camera_ids[replace_idx] = camera_id
            self.thumbnail_scores[replace_idx] = quality

    def similarity(self, query_emb: np.ndarray) -> float:
        """Maximum cosine similarity across all stored exemplars for this person."""
        if not self.embeddings or query_emb is None:
            return -1.0
        # Dot product against all stored unit-norm exemplars
        sims = [float(np.dot(query_emb, e)) for e in self.embeddings]
        return max(sims)

    def add_or_update_exemplar(
        self,
        new_emb: np.ndarray,
        max_exemplars: int = 8,
        dup_thresh: float = 0.92,
        ema_alpha: float = 0.15,
        min_update_sim: float = 0.60,
        is_cross_camera: bool = False,
    ) -> bool:
        """
        Maintains a diverse set of appearance vectors (e.g. front/back/side) with Contamination Guard.

        - If new_emb has very low similarity (< min_update_sim) to existing exemplars of this person,
          reject the update to prevent gallery contamination from misidentified frames/occlusions.
          Cross-camera updates use a lower floor (is_cross_camera=True) because viewpoint shift
          naturally pushes embeddings further apart even for the same person.
        - If new_emb is almost identical (>= dup_thresh), EMA-refine the closest exemplar.
        - If new_emb captures a genuine new angle/pose, append to the bank.
        - If bank is full, replace the closest existing exemplar (least diverse w.r.t. new_emb).

        Returns True if the exemplar was accepted/updated, False if rejected by contamination guard.
        """
        if new_emb is None or new_emb.size == 0:
            return False

        if not self.embeddings:
            self.embeddings.append(new_emb.copy())
            return True

        # Find closest existing exemplar
        sims = [float(np.dot(new_emb, e)) for e in self.embeddings]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]

        # Contamination Guard: reject updates that are too dissimilar from known appearances.
        # Cross-camera updates get a relaxed floor since viewpoint drift is expected.
        effective_min_sim = min_update_sim * (0.85 if is_cross_camera else 1.0)
        if best_sim < effective_min_sim:
            return False

        if best_sim >= dup_thresh:
            # Refine existing exemplar with EMA.
            # Cap ema_alpha so that high-frequency same-camera updates don't cause
            # runaway drift over thousands of frames (e.g. lighting change accumulates).
            safe_alpha = min(ema_alpha, 0.10)
            updated = (1.0 - safe_alpha) * self.embeddings[best_idx] + safe_alpha * new_emb
            updated /= (np.linalg.norm(updated) + 1e-8)
            self.embeddings[best_idx] = updated.astype(np.float32)
        elif len(self.embeddings) < max_exemplars:
            # Add as new appearance variation (e.g. back view, different lighting, or new camera angle)
            self.embeddings.append(new_emb.copy())
        else:
            # Bank is full: replace the LEAST DIVERSE exemplar rather than the closest to
            # new_emb.  "Least diverse" = the one that is most similar to all OTHER exemplars
            # (i.e. most redundant), so we preserve the widest spread of viewpoints.
            # This prevents the bank from converging to a narrow pose cluster over time.
            redundancy = []
            for i, e in enumerate(self.embeddings):
                others = [self.embeddings[j] for j in range(len(self.embeddings)) if j != i]
                avg_sim_to_others = float(np.mean([np.dot(e, o) for o in others])) if others else 0.0
                redundancy.append(avg_sim_to_others)
            replace_idx = int(np.argmax(redundancy))  # most redundant slot
            self.embeddings[replace_idx] = new_emb.copy()

        return True


class GlobalGallery:
    """
    Maintains a shared embedding store for all active persons across cameras.
    Thread-safe: protected by a re-entrant threading.RLock.
    """

    MAX_EXEMPLARS_PER_PERSON = 8  # Raised from 6 for better viewpoint diversity

    def __init__(
        self,
        reid_threshold: float = 0.65,
        ttl_frames: int = 50000,
        threshold_matrix: Optional[Dict] = None,
        camera_name_map: Optional[Dict[int, str]] = None,
    ):
        """
        Args:
            reid_threshold:   Default cosine similarity cutoff for all camera pairs.
            ttl_frames:       Frames of continuous absence before an identity expires.
            threshold_matrix: Optional dict keyed by frozenset({stem_a, stem_b}) with
                              per-pair similarity thresholds.  When two cameras have very
                              different angles (e.g. portrait vs landscape) a lower value
                              (e.g. 0.55) reduces false negatives from viewpoint drift.
            camera_name_map:  Maps camera_id (int) -> video stem (str) so the gallery
                              can look up the per-pair threshold at match time.
        """
        self.reid_threshold    = reid_threshold
        self.ttl_frames        = ttl_frames
        self._threshold_matrix = threshold_matrix or {}
        self._camera_name_map  = camera_name_map  or {}  # {camera_id: video_stem}

        self._lock = threading.RLock()
        self._entries: Dict[int, GalleryEntry] = {}
        self._next_id: int = 0

        # (camera_id, local_id) -> global_id (fast continuity for ongoing ByteTrack tracks)
        self._local_map: Dict[Tuple[int, int], int] = {}

    # ── Threshold helpers ─────────────────────────────────────────────────────

    def _get_threshold(self, cam_a: int, cam_b: int) -> float:
        """Return per-pair threshold if configured, otherwise fall back to global default."""
        name_a = self._camera_name_map.get(cam_a)
        name_b = self._camera_name_map.get(cam_b)
        if name_a and name_b:
            key = frozenset({name_a, name_b})
            if key in self._threshold_matrix:
                return self._threshold_matrix[key]
        return self.reid_threshold

    # ── Public Batch API (Enforces Mutual Exclusivity) ─────────────────────────

    def match_batch(
        self,
        camera_id: int,
        frame_idx: int,
        detections: List[Tuple[int, Optional[np.ndarray]]],
    ) -> Dict[int, int]:
        """
        Match all detections in a single camera frame simultaneously with mutual exclusivity.

        Uses per-camera-pair thresholds when configured so that camera pairs with
        extreme angle differences (e.g. portrait cam3 vs landscape cam1/cam2) apply
        a lower similarity cutoff, reducing false negatives from viewpoint drift.

        Args:
            camera_id:   Integer camera identifier.
            frame_idx:   Current frame index on this camera.
            detections:  List of tuples: (local_id, embedding_or_none)

        Returns:
            Dict mapping local_id -> global_id for all detections in this frame.
        """
        if not detections:
            return {}

        results: Dict[int, int] = {}
        claimed_gids_in_frame: Set[int] = set()

        with self._lock:
            # ── Phase 1: Keep active ongoing tracks (already in local_map) ────
            unmapped_detections: List[Tuple[int, Optional[np.ndarray]]] = []

            for local_id, emb in detections:
                key = (camera_id, local_id)
                if key in self._local_map:
                    gid = self._local_map[key]
                    if gid in self._entries:
                        # Ongoing known track
                        entry = self._entries[gid]

                        # Verify identity stability if embedding is available
                        current_sim = entry.similarity(emb) if emb is not None else 1.0

                        # If similarity to the assigned person drops below threshold, don't blindly
                        # lock-in — re-evaluate in Phase 2 to avoid carrying a drifted/wrong ID
                        # forward for the rest of the video.  0.58 is tighter than the old 0.50
                        # floor so contamination is caught earlier.
                        if emb is not None and current_sim < 0.58:
                            del self._local_map[key]
                            unmapped_detections.append((local_id, emb))
                            continue

                        if emb is not None:
                            # Same-camera update: use standard contamination guard
                            entry.add_or_update_exemplar(
                                emb,
                                max_exemplars=self.MAX_EXEMPLARS_PER_PERSON,
                                min_update_sim=max(0.60, self.reid_threshold - 0.05),
                                is_cross_camera=False,
                            )
                        entry.last_seen_frame = frame_idx
                        entry.last_camera_id = camera_id
                        entry.hit_count += 1

                        results[local_id] = gid
                        claimed_gids_in_frame.add(gid)
                        continue
                    else:
                        # Expired entry
                        del self._local_map[key]

                unmapped_detections.append((local_id, emb))

            if not unmapped_detections:
                return results

            # ── Phase 2: Compute candidate match scores for unmapped tracks ──
            # Candidate gallery entries: any entry NOT already claimed by another person in this frame
            candidates = [
                gid for gid in self._entries.keys()
                if gid not in claimed_gids_in_frame
            ]

            # Build list of potential matches: (similarity, local_id, gid, emb, threshold)
            match_proposals: List[Tuple[float, int, int, np.ndarray, float]] = []

            for local_id, emb in unmapped_detections:
                if emb is None:
                    continue
                for gid in candidates:
                    entry = self._entries[gid]
                    # Determine threshold: use per-pair if the candidate was last seen on a
                    # different camera (cross-camera match); otherwise use global default.
                    last_cam = entry.last_camera_id
                    is_cross = (last_cam != camera_id)
                    threshold = (
                        self._get_threshold(camera_id, last_cam)
                        if is_cross
                        else self.reid_threshold
                    )
                    sim = entry.similarity(emb)
                    if sim >= threshold:
                        match_proposals.append((sim, local_id, gid, emb, threshold))

            # Sort proposals by similarity in descending order (highest confidence first)
            match_proposals.sort(key=lambda x: x[0], reverse=True)

            assigned_locals: Set[int] = set()

            for sim, local_id, gid, emb, threshold in match_proposals:
                if local_id in assigned_locals or gid in claimed_gids_in_frame:
                    continue  # already assigned or claimed by a higher-similarity match

                # Valid match — determine if this is a cross-camera update
                entry = self._entries[gid]
                is_cross = (entry.last_camera_id != camera_id)
                entry.add_or_update_exemplar(
                    emb,
                    max_exemplars=self.MAX_EXEMPLARS_PER_PERSON,
                    # Cross-camera updates use a relaxed contamination floor because
                    # appearance naturally shifts more across camera viewpoints.
                    min_update_sim=max(0.50, threshold - 0.05) if is_cross else max(0.60, threshold - 0.05),
                    is_cross_camera=is_cross,
                )
                entry.last_seen_frame = frame_idx
                entry.last_camera_id = camera_id
                entry.hit_count += 1

                results[local_id] = gid
                self._local_map[(camera_id, local_id)] = gid
                claimed_gids_in_frame.add(gid)
                assigned_locals.add(local_id)

            # ── Phase 3: Register any remaining unmapped tracks as new persons ─
            for local_id, emb in unmapped_detections:
                if local_id in results:
                    continue

                if emb is None:
                    continue  # Do not assign a new global ID until a valid crop is obtained

                gid = self._next_id
                self._next_id += 1

                entry = GalleryEntry(
                    global_id=gid,
                    embeddings=[emb.copy()],
                    last_seen_frame=frame_idx,
                    last_camera_id=camera_id,
                    hit_count=1,
                )
                self._entries[gid] = entry
                self._local_map[(camera_id, local_id)] = gid
                claimed_gids_in_frame.add(gid)
                results[local_id] = gid

            return results

    # ── Single match fallback (wrapper around match_batch) ───────────────────

    def match_or_register(
        self,
        embedding: np.ndarray,
        camera_id: int,
        local_id: int,
        frame_idx: int,
    ) -> int:
        """Compatibility wrapper for single track query."""
        res = self.match_batch(camera_id, frame_idx, [(local_id, embedding)])
        return res.get(local_id, self._next_id)

    # ── Housekeeping ─────────────────────────────────────────────────────────

    def cleanup(self, current_frame: int) -> int:
        """Expire gallery entries not seen for ttl_frames."""
        with self._lock:
            stale = [
                gid for gid, e in self._entries.items()
                if (current_frame - e.last_seen_frame) > self.ttl_frames
            ]
            for gid in stale:
                del self._entries[gid]
            self._local_map = {
                k: v for k, v in self._local_map.items()
                if v in self._entries
            }
            return len(stale)

    def add_thumbnail_to_entry(self, gid: int, crop: np.ndarray, camera_id: int) -> None:
        with self._lock:
            if gid in self._entries:
                entry = self._entries[gid]
                entry.cameras_seen.add(camera_id)
                entry.add_thumbnail(crop, camera_id)

    def get_all_entries_snapshot(self) -> List[dict]:
        """
        Return a thread-safe snapshot of all gallery entries for the web UI.
        Each entry is a plain dict with serialisable fields (thumbnails as base64 JPEG).
        """
        import base64

        snapshot: List[dict] = []
        with self._lock:
            for gid in sorted(self._entries.keys()):
                entry = self._entries[gid]
                # Encode up to three selected thumbnails. ``thumbnail_b64`` is
                # retained for compatibility with existing UI consumers.
                thumbnails_b64: List[str] = []
                ranked_indices = sorted(
                    range(len(entry.thumbnails)),
                    key=lambda idx: entry.thumbnail_scores[idx],
                    reverse=True,
                )[:GalleryEntry.MAX_THUMBNAILS]
                for idx in ranked_indices:
                    thumbnail = entry.thumbnails[idx]
                    ret, buf = cv2.imencode(".jpg", thumbnail)
                    if ret:
                        thumbnails_b64.append(base64.b64encode(buf).decode("utf-8"))
                thumb_b64: Optional[str] = thumbnails_b64[0] if thumbnails_b64 else None

                snapshot.append({
                    "global_id": entry.global_id,
                    "hit_count": entry.hit_count,
                    "cameras_seen": sorted(entry.cameras_seen),
                    "last_camera_id": entry.last_camera_id,
                    "last_seen_frame": entry.last_seen_frame,
                    "thumbnail_b64": thumb_b64,
                    "thumbnails_b64": thumbnails_b64,
                })
        return snapshot

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._entries)

    def generate_html_report(self, output_dir: str) -> None:
        """Generates an HTML report summarizing all tracked IDs."""
        import base64
        from pathlib import Path
        
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        report_path = out_path / "id_summary_report.html"
        
        html_content = [
            "<html><head><title>Tracking ID Summary</title>",
            "<style>",
            "body { font-family: sans-serif; background: #f4f4f9; padding: 20px; }",
            "h1 { text-align: center; color: #333; }",
            ".grid { display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; }",
            ".card { background: #fff; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); padding: 15px; width: 300px; }",
            ".card h2 { margin-top: 0; font-size: 1.2em; color: #444; border-bottom: 2px solid #eee; padding-bottom: 5px; }",
            ".stats { font-size: 0.9em; color: #666; margin-bottom: 10px; }",
            ".badges { margin-bottom: 15px; }",
            ".badge { background: #007bff; color: #fff; padding: 3px 8px; border-radius: 12px; font-size: 0.8em; margin-right: 5px; }",
            ".thumbs { display: flex; gap: 5px; overflow-x: auto; padding-bottom: 5px; }",
            ".thumbs img { height: 100px; border-radius: 4px; border: 1px solid #ddd; object-fit: cover; }",
            "</style></head><body>",
            "<h1>Multi-Camera Tracking Summary</h1>",
            "<div class='grid'>"
        ]

        with self._lock:
            entries = sorted(self._entries.values(), key=lambda e: e.global_id)
            for entry in entries:
                cam_badges = "".join([f"<span class='badge'>Cam {cid}</span>" for cid in sorted(entry.cameras_seen)])
                
                img_tags = []
                for idx, thumb in enumerate(entry.thumbnails):
                    ret, buffer = cv2.imencode('.jpg', thumb)
                    if ret:
                        b64 = base64.b64encode(buffer).decode('utf-8')
                        img_tags.append(f"<img src='data:image/jpeg;base64,{b64}' alt='ID {entry.global_id} img {idx}'>")
                
                imgs_html = "".join(img_tags) if img_tags else "<p style='color: #999; font-size: 0.8em;'>No image</p>"
                
                card = f"""
                <div class='card'>
                    <h2>Person ID: {entry.global_id}</h2>
                    <div class='stats'>Total Detections: {entry.hit_count}</div>
                    <div class='badges'>{cam_badges}</div>
                    <div class='thumbs'>{imgs_html}</div>
                </div>
                """
                html_content.append(card)

        html_content.append("</div></body></html>")
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html_content))
        print(f"\n[Gallery] ID Summary Report generated at: {report_path}")
