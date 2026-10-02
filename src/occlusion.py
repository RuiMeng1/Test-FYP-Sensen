"""Conservative per-track appearance gate for suspected partial occlusion."""

from dataclasses import dataclass


@dataclass
class _SizeHistory:
    area: float
    height: float
    last_seen: int
    occluded: bool = False
    recovery_frames: int = 0


class PartialOcclusionGuard:
    """Compare boxes with an EMA of reliable sizes; never learn from shrinkage.

    This is a suspicion signal, not an occlusion classifier: crouching and
    abrupt perspective changes can also pause appearance updates.
    """

    def __init__(self):
        self._history: dict[int, _SizeHistory] = {}
        self._frame = 0

    def advance(self):
        self._frame += 1
        # A returning/reused local ID must not inherit an indefinitely old size.
        self._history = {
            tid: state for tid, state in self._history.items()
            if self._frame - state.last_seen <= 30
        }

    def check(self, local_id, width, height, full_body):
        """Return (retain_detection, suspected_occlusion) for a valid box."""
        area = width * height
        state = self._history.get(local_id)
        if state is None:
            if not full_body:
                return False, False
            self._history[local_id] = _SizeHistory(area, height, self._frame)
            return True, False

        consecutive = state.last_seen == self._frame - 1
        state.last_seen = self._frame
        if not consecutive:
            state.recovery_frames = 0
        if not full_body or area < 0.65 * state.area or height < 0.75 * state.height:
            state.occluded = True
            state.recovery_frames = 0

        if state.occluded:
            recovered = full_body and area >= 0.85 * state.area and height >= 0.90 * state.height
            state.recovery_frames = state.recovery_frames + 1 if recovered else 0
            if state.recovery_frames < 3:
                return True, True
            state.occluded = False
            state.recovery_frames = 0

        # Follow gradual scale changes as the person walks toward/away from us.
        state.area = 0.7 * state.area + 0.3 * area
        state.height = 0.7 * state.height + 0.3 * height
        return True, False
