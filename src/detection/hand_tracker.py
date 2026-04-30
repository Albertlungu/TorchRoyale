"""
Maintains a stable 4-card hand state across frames.

Roboflow frequently detects fewer than 4 in-hand cards per frame.
HandTracker forward-fills from the last known state and removes a card
when it appears as a new on-field deployment.

Play detection uses (card_name, tile_x, tile_y) tuples so replaying the
same card type at a different position is correctly treated as a new play.

Evolution logic: for ice spirit, skeletons, cannon, and musketeer, the
tracker counts plays. When a card returns to hand with a cycle count of 2,
the hand classifier is run once to check for an evolution. If confirmed,
every subsequent cycle-of-2 returns the card labelled as evolution. If not
confirmed, the card is treated as normal forever.

Public API:
  HandTracker -- stateful tracker; call reset() between games, update() each frame
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.types import DetectionDict

_SUFFIX_RE = re.compile(
    r"-(in-hand|next|on-field|on_field|evolution|ability)$",
    re.IGNORECASE,
)

# Cards eligible for evolution tracking.
# cannon is included for when evo cannon is added to the classifier dataset.
_EVO_CANDIDATES: frozenset = frozenset({"ice spirit", "skeletons", "cannon", "musketeer"})


def _base(name: str) -> str:
    """Strip known KataCR label suffixes to get the canonical card name."""
    return _SUFFIX_RE.sub("", name.lower()).strip()


def _is_evo_label(classified: str, base: str) -> bool:
    """Return True if the classifier label indicates an evolved version of base."""
    c = classified.lower()
    return base in c and ("evo" in c or "evolution" in c)


class HandTracker:
    """
    Forward-filling hand state tracker with evolution detection.

    Accumulates in-hand card detections across frames, evicts cards that
    appear as new on-field deployments, and always returns exactly up to
    HAND_SIZE entries.
    """

    HAND_SIZE: int = 4

    def __init__(self) -> None:
        self._hand: List[str] = []
        self._prev_field: Set[Tuple[str, int, int]] = set()

        # Evo tracking per candidate card
        # None = unknown, True = has evo, False = no evo (stop tracking)
        self._evo_status: Dict[str, Optional[bool]] = {}
        self._cycle_count: Dict[str, int] = {}

    def reset(self) -> None:
        """Clear all state. Call between games."""
        self._hand = []
        self._prev_field = set()
        self._evo_status = {}
        self._cycle_count = {}

    def update(
        self,
        detections: List[DetectionDict],
        frame: Optional[np.ndarray] = None,
        game_strip: Optional[Tuple[int, int]] = None,
    ) -> List[str]:
        """
        Update hand state from the current frame's detections.

        Args:
            detections:  list of detection dicts with keys class_name,
                         is_opponent, is_on_field, tile_x, tile_y.
            frame:       full BGR frame — required for evo classifier scans.
            game_strip:  (x_left, x_right) passed through to HandClassifier.

        Returns:
            Tracked hand as a list of card name strings (up to HAND_SIZE).
            Evolved cards are returned as "<name>-evolution-in-hand".
            Normal cards are returned as "<name>-in-hand".
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

        # Increment cycle counter for every evo candidate that was just played
        for card in played:
            if card in _EVO_CANDIDATES and self._evo_status.get(card) is not False:
                self._cycle_count[card] = self._cycle_count.get(card, 0) + 1

        if not self._hand:
            if in_hand_bases:
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

        # Check evo candidates that are back in hand with cycle == 2
        if frame is not None:
            for card in self._hand:
                if card not in _EVO_CANDIDATES:
                    continue
                if self._evo_status.get(card) is not None:
                    continue  # already confirmed evo or normal
                if self._cycle_count.get(card, 0) == 2:
                    self._run_evo_scan(frame, game_strip, card)

        self._prev_field = on_field_ids
        return self._build_output()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_evo_scan(
        self,
        frame: np.ndarray,
        game_strip: Optional[Tuple[int, int]],
        card: str,
    ) -> None:
        """Run the hand classifier once to confirm whether card is evolved."""
        try:
            from src.detection.hand_classifier import HandClassifier  # pylint: disable=import-outside-toplevel
            clf = HandClassifier()
            labels = clf.classify(frame, game_strip=game_strip)
            found_evo = any(
                lbl is not None and _is_evo_label(lbl, card)
                for lbl in labels
            )
            self._evo_status[card] = found_evo
            status = "EVO confirmed" if found_evo else "normal — no evo"
            print(f"[HandTracker] Evo scan for '{card}': {status}")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"[HandTracker] Evo scan failed for '{card}': {exc}")

    def _build_output(self) -> List[str]:
        """Build the hand list, labelling evo cards at cycle==2 and resetting."""
        result: List[str] = []
        for card in self._hand:
            if (
                self._evo_status.get(card) is True
                and self._cycle_count.get(card, 0) == 2
            ):
                result.append(f"{card}-evolution-in-hand")
                self._cycle_count[card] = 0
            else:
                result.append(f"{card}-in-hand")
        return result
