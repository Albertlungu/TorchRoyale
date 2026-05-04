"""
Persistence window tracker for on-field detections.

The YOLOv8 detectors are frame-by-frame and have no memory. A card that is
genuinely on the field can go undetected for several sampled frames due to:
  - Occlusion by other units or effects
  - Low confidence from model uncertainty (e.g. unfamiliar arena color)
  - Brief visual obstruction (spells, king tower animations)

Without any smoothing, these gaps create false "card disappeared then
reappeared" events that corrupt training episodes.

Solution — persistence window:
  Each unique card slot is tracked by its class, owner, and coarse tile
  position (rounded to the nearest 2 tiles so minor movement doesn't break
  identity). A detection stays "alive" for `window` sampled frames after
  the last time it was seen. Only after `window` consecutive misses is the
  card considered gone.

Public API:
  OnFieldTracker  -- call update() each sampled frame
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.types import DetectionDict


# Coarse tile grouping: round to the nearest N tiles.
# Cards move slowly relative to the sample rate so ±1 coarse cell covers drift.
_TILE_BUCKET = 2


def _key(det: DetectionDict) -> Tuple[str, int, int, bool]:
    """Stable identity key for a detection across frames."""
    return (
        det["class_name"],
        round(det["tile_x"] / _TILE_BUCKET),
        round(det["tile_y"] / _TILE_BUCKET),
        bool(det["is_opponent"]),
    )


@dataclass
class _Slot:
    det: DetectionDict   # most recent detection for this slot
    last_seen: int       # sampled frame index when last detected


class OnFieldTracker:
    """
    Applies a persistence window to raw per-frame on-field detections.

    Cards that disappear for fewer than `window` consecutive sampled frames
    are kept in the output at their last known position and confidence.

    Args:
        window: number of consecutive missed sampled frames before a card
                is removed from the tracked set. Default 4 (~0.4 s at the
                standard frame_skip=6 on a 57 fps video, i.e. ~10 fps
                effective sample rate).
    """

    def __init__(self, window: int = 4) -> None:
        self.window: int = window
        self._slots: Dict[Tuple, _Slot] = {}
        self._frame: int = 0

    def update(self, detections: List[DetectionDict]) -> List[DetectionDict]:
        """
        Merge new detections with persisted slots and return the smoothed list.

        Args:
            detections: raw on-field detections for the current sampled frame.

        Returns:
            Deduplicated list combining fresh detections with any slots still
            within their persistence window.
        """
        # Refresh slots for cards detected this frame
        seen_keys = set()
        for det in detections:
            k = _key(det)
            self._slots[k] = _Slot(det=det, last_seen=self._frame)
            seen_keys.add(k)

        # Collect all live slots (fresh + persisted within window)
        live: List[DetectionDict] = []
        dead_keys = []
        for k, slot in self._slots.items():
            age = self._frame - slot.last_seen
            if age > self.window:
                dead_keys.append(k)
            else:
                live.append(slot.det)

        for k in dead_keys:
            del self._slots[k]

        self._frame += 1
        return live

    def reset(self) -> None:
        """Clear all state between games."""
        self._slots.clear()
        self._frame = 0
