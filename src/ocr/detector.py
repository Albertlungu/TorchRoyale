"""
EasyOCR-based detector for timer, elixir count, and multiplier icon.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class OCRResult:
    value: int
    confidence: float
    detected: bool
    raw_text: str = ""


class DigitDetector:
    def __init__(self, preload: bool = False):
        self._reader = None
        if preload:
            self._get_reader()

    def _get_reader(self):
        if self._reader is None:
            import easyocr  # type: ignore
            self._reader = easyocr.Reader(["en"], gpu=False)
        return self._reader

    def _preprocess(self, roi: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        if gray.shape[0] < 128:
            scale = 128 / gray.shape[0]
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        return binary

    # ------------------------------------------------------------------
    # Timer
    # ------------------------------------------------------------------

    def detect_timer(self, frame: np.ndarray, region: Tuple[int, int, int, int]) -> Optional[int]:
        x1, y1, x2, y2 = region
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        processed = self._preprocess(roi)
        try:
            results = self._get_reader().readtext(
                processed, allowlist="0123456789:", paragraph=False, min_size=5
            )
        except Exception:
            return None
        if not results:
            return None
        full_text = "".join(r[1] for r in results).strip()
        return self._parse_timer(full_text)

    @staticmethod
    def _parse_timer(text: str) -> Optional[int]:
        m = re.match(r"(\d{1,2}):(\d{2})", text)
        if m:
            return int(m.group(1)) * 60 + int(m.group(2))
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

    def detect_elixir(self, frame: np.ndarray, region: Tuple[int, int, int, int]) -> OCRResult:
        x1, y1, x2, y2 = region
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return OCRResult(5, 0.0, False)
        processed = self._preprocess(roi)
        try:
            results = self._get_reader().readtext(
                processed, allowlist="0123456789", paragraph=False, min_size=5
            )
        except Exception as e:
            return OCRResult(5, 0.0, False, str(e))
        if not results:
            return OCRResult(5, 0.0, False)
        best = max(results, key=lambda x: x[2])
        text, conf = best[1].strip(), best[2]
        value = self._parse_elixir(text)
        if value is not None:
            return OCRResult(value, conf, True, text)
        return OCRResult(5, 0.0, False, text)

    @staticmethod
    def _parse_elixir(text: str) -> Optional[int]:
        if not text:
            return None
        if text == "10" or "10" in text:
            return 10
        if len(text) > 1 and len(set(text)) == 1:
            return min(int(text[0]), 10)
        if len(text) == 1 and text.isdigit():
            return int(text)
        if text[0].isdigit():
            d = int(text[0])
            if d == 1 and len(text) > 1 and text[1] == "0":
                return 10
            return min(d, 10)
        return None

    # ------------------------------------------------------------------
    # Multiplier icon
    # ------------------------------------------------------------------

    def detect_multiplier(self, frame: np.ndarray, region: Tuple[int, int, int, int]) -> int:
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
            results = self._get_reader().readtext(
                right_half, allowlist="0123456789", paragraph=False, min_size=5
            )
        except Exception:
            return 2
        if results:
            best = max(results, key=lambda x: x[2])
            for ch in reversed(best[1].strip()):
                if ch in "23":
                    return int(ch)
        return 2
