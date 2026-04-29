"""
Maintains a stable 4-card hand state across frames.

Roboflow frequently detects fewer than 4 in-hand cards per frame.
HandTracker forward-fills from the last known state and removes a card
when it appears as a new on-field deployment.

Play detection uses (card_name, tile_x, tile_y) tuples so replaying the
same card type at a different position is correctly treated as a new play.
"""
from __future__ import annotations

import re
from typing import List, Optional, Set, Tuple

_SUFFIX_RE = re.compile(
    r"-(in-hand|next|on-field|on_field|evolution|ability)$",
    re.IGNORECASE,
)


def _base(name: str) -> str:
    return _SUFFIX_RE.sub("", name.lower()).strip()


class HandTracker:
    HAND_SIZE = 4

    def __init__(self):
        self._hand: List[str] = []                   # canonical base names
        self._prev_field: Set[Tuple] = set()          # (name, tile_x, tile_y) last frame

    def reset(self) -> None:
        self._hand = []
        self._prev_field = set()

    def update(self, detections: List[dict]) -> List[str]:
        """
        Args:
            detections: list of dicts with keys class_name, is_opponent,
                        is_on_field, tile_x, tile_y.
        Returns:
            Tracked hand as list of '<name>-in-hand' strings.
        """
        in_hand_raw: List[str] = []
        on_field_ids: Set[Tuple] = set()

        for det in detections:
            if det.get("is_opponent"):
                continue
            name = det.get("class_name", "")
            if det.get("is_on_field") and "-next" not in name.lower():
                on_field_ids.add((
                    _base(name),
                    int(det.get("tile_x", 0)),
                    int(det.get("tile_y", 0)),
                ))
            elif "-in-hand" in name.lower():
                in_hand_raw.append(name)

        in_hand_bases = [_base(n) for n in in_hand_raw]

        newly_on_field = {b for b, _, _ in (on_field_ids - self._prev_field)}
        played = {b for b in self._hand if b in newly_on_field}

        if not self._hand:
            if in_hand_bases:
                self._hand = list(dict.fromkeys(in_hand_bases))[: self.HAND_SIZE]
            self._prev_field = on_field_ids
            return [f"{b}-in-hand" for b in self._hand]

        self._hand = [b for b in self._hand if b not in played]

        tracked_set = set(self._hand)
        for b in dict.fromkeys(in_hand_bases):
            if b not in tracked_set and len(self._hand) < self.HAND_SIZE:
                self._hand.append(b)
                tracked_set.add(b)

        self._prev_field = on_field_ids
        return [f"{b}-in-hand" for b in self._hand]
