"""
EasyOCR-based detector for the timer, elixir count, and multiplier icon.

All three detectors share a single EasyOCR reader instance that is
loaded lazily on first use (pass preload=True to warm it up at startup).

Public API:
  OCRResult     -- value/confidence/detected result from an OCR read
  DigitDetector -- detect_timer(), detect_elixir(), detect_multiplier()
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2  # type: ignore[import]
import numpy as np


@dataclass
class OCRResult:
    """Result from a single OCR detection attempt."""

    value: int
    confidence: float
    detected: bool
    raw_text: str = field(default="")


class DigitDetector:
    """
    Reads numeric UI elements from a video frame using EasyOCR.

    The EasyOCR reader is loaded lazily to avoid paying the startup cost
    unless detection is actually needed.
    """

    def __init__(self, preload: bool = False) -> None:
        """
        Args:
            preload: if True, initialise the EasyOCR reader immediately.
        """
        self._reader = None
        if preload:
            self._get_reader()

    def _get_reader(self):  # type: ignore[return]
        """Return the shared EasyOCR reader, initialising it on first call."""
        if self._reader is None:
            import easyocr  # type: ignore  # pylint: disable=import-outside-toplevel
            self._reader = easyocr.Reader(["en"], gpu=False)
        return self._reader

    def _preprocess(self, roi: np.ndarray) -> np.ndarray:
        """
        Convert an ROI to a high-contrast binary image suitable for digit OCR.

        Args:
            roi: BGR or greyscale image crop.

        Returns:
            Binary (0/255) greyscale image, upscaled to at least 128 px tall.
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        if gray.shape[0] < 128:
            scale = 128 / gray.shape[0]
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        return binary

    # ------------------------------------------------------------------
    # Timer
    # ------------------------------------------------------------------

    def detect_timer(
        self, frame: np.ndarray, region: Tuple[int, int, int, int]
    ) -> Optional[int]:
        """
        Read the countdown timer from a frame region.

        Args:
            frame:  full BGR frame.
            region: (x1, y1, x2, y2) crop rectangle.

        Returns:
            Time remaining in seconds, or None if unreadable.
        """
        x1, y1, x2, y2 = region
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        processed = self._preprocess(roi)
        try:
            results: List = self._get_reader().readtext(
                processed, allowlist="0123456789:", paragraph=False, min_size=5
            )
        except Exception:  # pylint: disable=broad-exception-caught
            return None
        if not results:
            return None
        full_text = "".join(r[1] for r in results).strip()
        return self._parse_timer(full_text)

    @staticmethod
    def _parse_timer(text: str) -> Optional[int]:
        """
        Parse a timer string into total seconds.

        Args:
            text: raw OCR text, e.g. "1:23" or "123".

        Returns:
            Seconds remaining, or None if unparseable.
        """
        match = re.match(r"(\d{1,2}):(\d{2})", text)
        if match:
            return int(match.group(1)) * 60 + int(match.group(2))
        if text.isdigit():
            if len(text) == 3:
                mins, secs = int(text[0]), int(text[1:])
                return mins * 60 + secs if secs < 60 else None
            if len(text) == 4:
                mins, secs = int(text[:2]), int(text[2:])
                return mins * 60 + secs if secs < 60 else None
            return int(text)
        return None

    # ------------------------------------------------------------------
    # Elixir
    # ------------------------------------------------------------------

    def detect_elixir(
        self, frame: np.ndarray, region: Tuple[int, int, int, int]
    ) -> OCRResult:
        """
        Read the player's current elixir count from a frame region.

        Args:
            frame:  full BGR frame.
            region: (x1, y1, x2, y2) crop rectangle.

        Returns:
            OCRResult with value in [0, 10]. Falls back to value=5 if undetected.
        """
        x1, y1, x2, y2 = region
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return OCRResult(5, 0.0, False)
        processed = self._preprocess(roi)
        try:
            results: List = self._get_reader().readtext(
                processed, allowlist="0123456789", paragraph=False, min_size=5
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return OCRResult(5, 0.0, False, str(exc))
        if not results:
            return OCRResult(5, 0.0, False)
        best = max(results, key=lambda item: item[2])
        text, conf = best[1].strip(), best[2]
        value = self._parse_elixir(text)
        if value is not None:
            return OCRResult(value, conf, True, text)
        return OCRResult(5, 0.0, False, text)

    @staticmethod
    def _parse_elixir(text: str) -> Optional[int]:
        """
        Parse an elixir string into an integer in [0, 10].

        Args:
            text: raw digit string from OCR.

        Returns:
            Elixir count, or None if unparseable.
        """
        if not text:
            return None
        if text == "10" or "10" in text:
            return 10
        if len(text) > 1 and len(set(text)) == 1:
            return min(int(text[0]), 10)
        if len(text) == 1 and text.isdigit():
            return int(text)
        if text[0].isdigit():
            digit = int(text[0])
            if digit == 1 and len(text) > 1 and text[1] == "0":
                return 10
            return min(digit, 10)
        return None

    # ------------------------------------------------------------------
    # Multiplier icon
    # ------------------------------------------------------------------

    def detect_multiplier(
        self, frame: np.ndarray, region: Tuple[int, int, int, int]
    ) -> int:
        """
        Detect the current elixir multiplier (1, 2, or 3) from the UI icon.

        Uses purple hue detection first; if the icon is absent returns 1.

        Args:
            frame:  full BGR frame.
            region: (x1, y1, x2, y2) crop rectangle.

        Returns:
            1, 2, or 3.
        """
        x1, y1, x2, y2 = region
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 1
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        purple = cv2.inRange(hsv, np.array([120, 50, 50]), np.array([170, 255, 255]))
        if cv2.countNonZero(purple) < roi.shape[0] * roi.shape[1] * 0.05:
            return 1
        processed = self._preprocess(roi)
        right_half = processed[:, processed.shape[1] // 2:]
        try:
            results: List = self._get_reader().readtext(
                right_half, allowlist="0123456789", paragraph=False, min_size=5
            )
        except Exception:  # pylint: disable=broad-exception-caught
            return 2
        if results:
            best = max(results, key=lambda item: item[2])
            for ch in reversed(best[1].strip()):
                if ch in "23":
                    return int(ch)
        return 2
