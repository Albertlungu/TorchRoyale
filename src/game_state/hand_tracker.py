"""
HandTracker maintains a stable 4-card hand state across video frames.

Roboflow frequently fails to detect all four hand cards in a given frame
(grayed-out cards during play animation, partial occlusion, etc.). Raw
per-frame detection therefore produces hands with 0-3 cards most of the
time, making the hand_cards field unreliable for training and inference.

Strategy
--------
- Maintain a tracked list of exactly 4 in-hand card base names.
- Each frame, accept the raw Roboflow detections and:
    1. Extract detected in-hand cards (class names containing "-in-hand").
    2. Extract newly on-field player cards to detect play events.
    3. When a card is played (was in tracked hand, now appears freshly on
       field), remove it from the tracked hand.
    4. When a new in-hand card is detected that was not in the tracked
       hand, add it to fill the open slot.
    5. If fewer than 4 cards are detected but no play events occurred,
       keep the tracked hand unchanged (forward-fill).
- Base names are compared without suffixes so "hog-rider-in-hand" and
  "hog-rider" match the same slot.
"""

from typing import List, Optional, Set
import re


_SUFFIXES = re.compile(
    r"-(in-hand|next|on-field|on_field|evolution|ability)$",
    re.IGNORECASE,
)


def _base(name: str) -> str:
    """Strip Roboflow class suffixes to get the canonical card name."""
    return _SUFFIXES.sub("", name.lower()).strip()


class HandTracker:
    """Tracks the player's 4-card hand across frames."""

    HAND_SIZE = 4

    def __init__(self):
        self._hand: List[str] = []          # tracked hand (base names)
        self._prev_on_field: Set[str] = set()  # base names on field last frame

    def reset(self) -> None:
        self._hand = []
        self._prev_on_field = set()

    def update(self, detections: List[dict]) -> List[str]:
        """
        Update the tracked hand from one frame's Roboflow detections.

        Args:
            detections: List of detection dicts with keys class_name,
                        is_opponent, is_on_field.

        Returns:
            The tracked hand as a list of class names (with -in-hand suffix),
            always aiming for HAND_SIZE entries.
        """
        # --- Separate detections by type ---
        in_hand_raw: List[str] = []     # raw class names with -in-hand
        on_field_bases: Set[str] = set()

        for det in detections:
            if det.get("is_opponent"):
                continue
            name = det.get("class_name", "")
            if det.get("is_on_field"):
                on_field_bases.add(_base(name))
            elif "-in-hand" in name.lower():
                in_hand_raw.append(name)

        in_hand_bases = [_base(n) for n in in_hand_raw]

        # --- Detect play events ---
        # A card was played if it was in our tracked hand and has just
        # appeared on-field for the first time (not in prev_on_field).
        newly_on_field = on_field_bases - self._prev_on_field
        played_bases = {b for b in self._hand if b in newly_on_field}

        # --- Initialise tracked hand on first non-empty detection ---
        if not self._hand:
            if in_hand_bases:
                self._hand = list(dict.fromkeys(in_hand_bases))[: self.HAND_SIZE]
            self._prev_on_field = on_field_bases
            return self._to_in_hand(self._hand)

        # --- Remove played cards ---
        self._hand = [b for b in self._hand if b not in played_bases]

        # --- Add newly detected in-hand cards not already tracked ---
        tracked_set = set(self._hand)
        for b in dict.fromkeys(in_hand_bases):  # preserve order, dedup
            if b not in tracked_set and len(self._hand) < self.HAND_SIZE:
                self._hand.append(b)
                tracked_set.add(b)

        # --- If still short, fill from in-hand detections we already track ---
        # (handles the case where we see fewer than 4 but no play happened)

        self._prev_on_field = on_field_bases
        return self._to_in_hand(self._hand)

    @staticmethod
    def _to_in_hand(bases: List[str]) -> List[str]:
        """Re-attach the -in-hand suffix for downstream consumers."""
        return [f"{b}-in-hand" for b in bases]
