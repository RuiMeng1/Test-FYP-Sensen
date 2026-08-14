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
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class GalleryEntry:
    global_id: int
    embeddings: List[np.ndarray] = field(default_factory=list)  # unit-norm feature bank
    last_seen_frame: int = 0                                    # for TTL expiry
    last_camera_id: int = 0
    hit_count: int = 1                                          # number of detections

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
        max_exemplars: int = 6,
        dup_thresh: float = 0.92,
        ema_alpha: float = 0.15,
    ) -> None:
        """
        Maintains a diverse set of appearance vectors (e.g. front/back/side).
        - If new_emb is almost identical (>0.92) to an existing exemplar, EMA-refine it.
        - If new_emb captures a new pose/angle (sim < 0.92), append it to the bank.
        - If bank is full, replace the exemplar with the highest redundancy.
        """
        if new_emb is None or new_emb.size == 0:
            return

        if not self.embeddings:
            self.embeddings.append(new_emb.copy())
            return

        # Find closest existing exemplar
        sims = [float(np.dot(new_emb, e)) for e in self.embeddings]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]

        if best_sim >= dup_thresh:
            # Refine existing exemplar with EMA
            updated = (1.0 - ema_alpha) * self.embeddings[best_idx] + ema_alpha * new_emb
            updated /= (np.linalg.norm(updated) + 1e-8)
            self.embeddings[best_idx] = updated.astype(np.float32)
        elif len(self.embeddings) < max_exemplars:
            # Add as new appearance variation (e.g. back view or different lighting)
            self.embeddings.append(new_emb.copy())
        else:
            # Bank is full: find the pair of exemplars that are closest to each other
            # and replace the one closest to new_emb
            self.embeddings[best_idx] = new_emb.copy()


class GlobalGallery:
    """
    Maintains a shared embedding store for all active persons across cameras.
    Thread-safe: protected by a re-entrant threading.RLock.
    """

    MAX_EXEMPLARS_PER_PERSON = 6

    def __init__(self, reid_threshold: float = 0.65, ttl_frames: int = 50000):
        """
        Args:
            reid_threshold: Cosine similarity cutoff for matching to an existing person.
            ttl_frames:     Frames of continuous absence before an identity expires.
        """
        self.reid_threshold = reid_threshold
        self.ttl_frames = ttl_frames

        self._lock = threading.RLock()
        self._entries: Dict[int, GalleryEntry] = {}
        self._next_id: int = 0

        # (camera_id, local_id) -> global_id (fast continuity for ongoing ByteTrack tracks)
        self._local_map: Dict[Tuple[int, int], int] = {}

    # ── Public Batch API (Enforces Mutual Exclusivity) ─────────────────────────

    def match_batch(
        self,
        camera_id: int,
        frame_idx: int,
        detections: List[Tuple[int, Optional[np.ndarray]]],
    ) -> Dict[int, int]:
        """
        Match all detections in a single camera frame simultaneously with mutual exclusivity.

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
                        if emb is not None:
                            entry.add_or_update_exemplar(emb, max_exemplars=self.MAX_EXEMPLARS_PER_PERSON)
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

            # Build list of potential matches: (similarity, local_id, gid, emb)
            match_proposals: List[Tuple[float, int, int, np.ndarray]] = []

            for local_id, emb in unmapped_detections:
                if emb is None:
                    continue
                for gid in candidates:
                    sim = self._entries[gid].similarity(emb)
                    if sim >= self.reid_threshold:
                        match_proposals.append((sim, local_id, gid, emb))

            # Sort proposals by similarity in descending order (highest confidence first)
            match_proposals.sort(key=lambda x: x[0], reverse=True)

            assigned_locals: Set[int] = set()

            for sim, local_id, gid, emb in match_proposals:
                if local_id in assigned_locals or gid in claimed_gids_in_frame:
                    continue  # already assigned or claimed by a higher-similarity match

                # Valid match
                entry = self._entries[gid]
                entry.add_or_update_exemplar(emb, max_exemplars=self.MAX_EXEMPLARS_PER_PERSON)
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

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._entries)
