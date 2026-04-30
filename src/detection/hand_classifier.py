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

_WEIGHTS_PATH = Path(__file__).parents[2] / "data/models/hand_classifier/hand_classifier.pt"

# Vertical crop of the hand area as fractions of frame height
_Y_TOP_FRAC: float = 0.845
_Y_BOT_FRAC: float = 0.965

# The "Next card" preview occupies this fraction of the game strip before slot 0
_NEXT_CARD_FRAC: float = 0.115

# Horizontal inset applied symmetrically to each slot, as a fraction of slot_w
_INSET_FRAC: float = 0.091

# Per-slot horizontal offsets as fractions of slot_w (positive = right).
# Calibrated visually on 1920x1080; expressed as fractions so they scale to any res.
_SLOT_OFFSET_FRACS: Tuple[float, float, float, float] = (0.409, 0.273, 0.136, 0.0)

# Minimum confidence to return a label; below this returns None
_MIN_CONF: float = 0.40


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

    def classify(self, frame: np.ndarray, game_strip: Optional[Tuple[int, int]] = None) -> List[Optional[str]]:
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
        next_end = x_left + int(game_w * _NEXT_CARD_FRAC)
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
            name = pred.names[pred.probs.top1] if conf >= _MIN_CONF else None
            results.append(name)

        return results
