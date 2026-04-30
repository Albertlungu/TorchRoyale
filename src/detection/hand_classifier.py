"""
Hand card classifier using a local YOLOv8 classification model.

Crops the four hand card slots from a frame and classifies each one.
Slot positions were calibrated for a 1920x1080 landscape source with the
portrait game strip centred horizontally (~497px wide).

Public API:
  HandClassifier -- load weights once, call classify(frame) each frame
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_WEIGHTS_PATH = (
    Path(__file__).parents[2] / "data/models/hand_classifier/hand_classifier.pt"
)

# Cards whose dataset folder names differ from the pipeline's canonical names.
_NAME_ALIASES = {
    "skeletons": "skeleton",  # dataset uses plural; KataCR and CARD_NAMES use singular
}


def _normalise(name: str) -> str:
    """
    Normalise a classifier output label to the pipeline's canonical format.

    Rules applied in order:
      1. "evo <card>" → "<card>-evolution"  (spaces converted to dashes)
      2. Alias substitution (e.g. "skeletons" → "skeleton")
      3. Spaces → dashes  (to match KataCR class names and CARD_NAMES vocab)
    """
    n = name.lower().strip()
    if n.startswith("evo "):
        base = n[4:]
        base = _NAME_ALIASES.get(base, base).replace(" ", "-")
        return f"{base}-evolution"
    n = _NAME_ALIASES.get(n, n)
    return n.replace(" ", "-")


# Vertical crop of the hand area as fractions of frame height
_Y_TOP_FRAC: float = 0.824
_Y_BOT_FRAC: float = 0.945

# The "Next card" preview occupies this fraction of the game strip before slot 0
_NEXT_CARD_FRAC: float = 0.115

# Next-card bounding box edges (fractions). These allow explicit control of the
# left/right/top/bottom edges of the preview box instead of a single width frac.
# Defaults chosen to match legacy behavior.
_NEXT_LEFT_FRAC: float = 0.048
_NEXT_RIGHT_FRAC: float = 0.145
_NEXT_TOP_FRAC: float = 0.92
_NEXT_BOTTOM_FRAC: float = 0.975

# Horizontal inset applied symmetrically to each slot, as a fraction of slot_w
_INSET_FRAC: float = 0.091

# Per-slot horizontal offsets as fractions of slot_w (positive = right).
# Calibrated visually on 1920x1080; expressed as fractions so they scale to any res.
_SLOT_OFFSET_FRACS: Tuple[float, float, float, float] = (0.409, 0.273, 0.136, -0.095)

# Minimum confidence to return a label; below this returns None
_MIN_CONF: float = 0.40


def get_next_bbox(frame_h: int, frame_w: int, x_left: int, x_right: int):
    """Return pixel bbox (x1, y1, x2, y2) for the Next-card preview area.

    Args:
        frame_h, frame_w: frame dimensions
        x_left, x_right: detected game strip left/right columns
    """
    game_w = x_right - x_left
    next_left = x_left + int(game_w * _NEXT_LEFT_FRAC)
    next_right = x_left + int(game_w * _NEXT_RIGHT_FRAC)
    next_top = int(frame_h * _NEXT_TOP_FRAC)
    next_bottom = int(frame_h * _NEXT_BOTTOM_FRAC)
    return (
        max(0, next_left),
        max(0, next_top),
        min(frame_w, next_right),
        min(frame_h, next_bottom),
    )


class HandClassifier:
    """
    Classifies the four hand card slots in a Clash Royale frame.

    Weights are loaded lazily on the first call to classify().
    """

    def __init__(self, weights_path: Optional[str] = None) -> None:
        """
        Args:
            weights_path: path to best.pt; defaults to the trained local weights.
        """
        self._weights = Path(weights_path) if weights_path else _WEIGHTS_PATH
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self._weights.exists():
            raise FileNotFoundError(
                f"Hand classifier weights not found: {self._weights}\n"
                "Run: python scripts/train_hand_classifier.py"
            )
        from ultralytics import YOLO  # pylint: disable=import-outside-toplevel

        self._model = YOLO(str(self._weights))

    def classify(
        self, frame: np.ndarray, game_strip: Optional[Tuple[int, int]] = None
    ) -> List[Optional[str]]:
        """
        Classify all four hand card slots in the frame.

        Args:
            frame:      full BGR frame from OpenCV.
            game_strip: (x_left, x_right) pixel bounds of the game content strip.
                        If None, they are auto-detected from column brightness.

        Returns:
            List of four card name strings (or None if confidence is too low).
        """
        self._load()
        frame_h, frame_w = frame.shape[:2]

        if game_strip is not None:
            x_left, x_right = game_strip
        else:
            gray = np.mean(frame, axis=2) if frame.ndim == 3 else frame
            cols = np.where(np.mean(gray, axis=0) > 30)[0]
            if cols.size == 0:
                return [None, None, None, None]
            x_left, x_right = int(cols.min()), int(cols.max())

        game_w = x_right - x_left
        y_top = int(frame_h * _Y_TOP_FRAC)
        y_bot = int(frame_h * _Y_BOT_FRAC)
        # Next-card bounding box edges (pixel coords)
        next_left = x_left + int(game_w * _NEXT_LEFT_FRAC)
        next_right = x_left + int(game_w * _NEXT_RIGHT_FRAC)
        next_end = next_right
        cards_w = x_right - next_end
        slot_w = cards_w // 4

        inset = int(slot_w * _INSET_FRAC)
        results: List[Optional[str]] = []
        for i, offset_frac in enumerate(_SLOT_OFFSET_FRACS):
            offset = int(slot_w * offset_frac)
            x1 = next_end + i * slot_w + inset + offset
            x2 = x1 + slot_w - 2 * inset
            x1 = max(0, x1)
            x2 = min(frame_w, x2)
            crop = frame[y_top:y_bot, x1:x2]
            if crop.size == 0:
                results.append(None)
                continue
            pred = self._model(crop, verbose=False)[0]
            conf = float(pred.probs.top1conf)
            raw = pred.names[pred.probs.top1] if conf >= _MIN_CONF else None
            results.append(_normalise(raw) if raw is not None else None)

        return results
