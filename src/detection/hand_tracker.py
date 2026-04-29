"""
Maintains a stable 4-card hand state across frames.

Roboflow frequently detects fewer than 4 in-hand cards per frame.
HandTracker forward-fills from the last known state and removes a card
when it appears as a new on-field deployment.

Play detection uses (card_name, tile_x, tile_y) tuples so replaying the
same card type at a different position is correctly treated as a new play.

Public API:
  HandTracker -- stateful tracker; call reset() between games, update() each frame
"""
from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

from src.types import DetectionDict

_SUFFIX_RE = re.compile(
    r"-(in-hand|next|on-field|on_field|evolution|ability)$",
    re.IGNORECASE,
)


def _base(name: str) -> str:
    """Strip known KataCR label suffixes to get the canonical card name."""
    return _SUFFIX_RE.sub("", name.lower()).strip()


class HandTracker:
    """
    Forward-filling hand state tracker.

    Accumulates in-hand card detections across frames, evicts cards that
    appear as new on-field deployments, and always returns exactly up to
    HAND_SIZE entries.
    """

    HAND_SIZE: int = 4

    def __init__(self) -> None:
        self._hand: List[str] = []                   # canonical base names
        self._prev_field: Set[Tuple[str, int, int]] = set()

    def reset(self) -> None:
        """Clear all state. Call between games."""
        self._hand = []
        self._prev_field = set()

    def update(self, detections: List[DetectionDict]) -> List[str]:
        """
        Update hand state from the current frame's detections.

        Args:
            detections: list of detection dicts with keys class_name,
                        is_opponent, is_on_field, tile_x, tile_y.

        Returns:
            Tracked hand as a list of "<name>-in-hand" strings (up to HAND_SIZE).
        """
        in_hand_raw: List[str] = []
        on_field_ids: Set[Tuple[str, int, int]] = set()

        for det in detections:
            if det.get("is_opponent"):
                continue
            name: str = det.get("class_name", "")
            if det.get("is_on_field") and "-next" not in name.lower():
                on_field_ids.add((
                    _base(name),
                    int(det.get("tile_x", 0)),
                    int(det.get("tile_y", 0)),
                ))
            elif "-in-hand" in name.lower():
                in_hand_raw.append(name)

        in_hand_bases: List[str] = [_base(nm) for nm in in_hand_raw]

        newly_on_field: Set[str] = {b for b, _, _ in (on_field_ids - self._prev_field)}
        played: Set[str] = {b for b in self._hand if b in newly_on_field}

        if not self._hand:
            if in_hand_bases:
                # Use dict.fromkeys to deduplicate while preserving order
                deduped: Dict[str, None] = dict.fromkeys(in_hand_bases)
                self._hand = list(deduped)[: self.HAND_SIZE]
            self._prev_field = on_field_ids
            return [f"{b}-in-hand" for b in self._hand]

        self._hand = [b for b in self._hand if b not in played]

        tracked_set: Set[str] = set(self._hand)
        for base in dict.fromkeys(in_hand_bases):
            if base not in tracked_set and len(self._hand) < self.HAND_SIZE:
                self._hand.append(base)
                tracked_set.add(base)

        self._prev_field = on_field_ids
        return [f"{b}-in-hand" for b in self._hand]
