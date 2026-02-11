"""
Digit detector for reading numbers from Clash Royale UI.

Uses EasyOCR for reliable detection of:
- Elixir count (0-10)
- Timer (MM:SS format)
- Card costs
- Multiplier icons (x2/x3)
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
import re

# Lazy import of EasyOCR to avoid loading model until needed
_ocr_reader = None


def _get_ocr_reader():
    """Get or create the EasyOCR reader (singleton)."""
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        # Use GPU if available, only load English, disable verbose output
        _ocr_reader = easyocr.Reader(
            ['en'],
            gpu=True,
            verbose=False,
            quantize=True,  # Use quantized model for faster inference
        )
    return _ocr_reader


@dataclass
class DetectionResult:
    """Result of a detection operation."""
    value: int
    confidence: float
    detected: bool
    raw_text: str = ""


class DigitDetector:
    """
    Detects digits and icons in Clash Royale UI regions using EasyOCR.

    EasyOCR provides accurate text recognition for game UI elements
    without requiring manual template creation.

    Usage:
        detector = DigitDetector()
        result = detector.detect_elixir(frame, (x1, y1, x2, y2))
        print(f"Elixir: {result.value}")
    """

    def __init__(self, preload_ocr: bool = False):
        """
        Initialize digit detector.

        Args:
            preload_ocr: If True, load OCR model immediately.
                        If False (default), load on first use.
        """
        self._reader = None
        if preload_ocr:
            self._reader = _get_ocr_reader()

    @property
    def reader(self):
        """Lazy-load the OCR reader."""
        if self._reader is None:
            self._reader = _get_ocr_reader()
        return self._reader

    def detect_elixir(
        self,
        image: np.ndarray,
        region: Tuple[int, int, int, int]
    ) -> DetectionResult:
        """
        Detect elixir value (0-10) from a UI region.

        Args:
            image: Full frame (BGR format)
            region: (x_min, y_min, x_max, y_max) of elixir number area

        Returns:
            DetectionResult with detected elixir value
        """
        x1, y1, x2, y2 = region
        roi = image[y1:y2, x1:x2]

        if roi.size == 0:
            return DetectionResult(value=5, confidence=0.0, detected=False)

        # Preprocess for better OCR
        processed = self._preprocess_for_ocr(roi)

        # Run OCR - only look for digits
        try:
            results = self.reader.readtext(
                processed,
                allowlist='0123456789',
                paragraph=False,
                min_size=5,
            )
        except Exception as e:
            return DetectionResult(value=5, confidence=0.0, detected=False, raw_text=str(e))

        if not results:
            return DetectionResult(value=5, confidence=0.0, detected=False)

        # Get the result with highest confidence
        best_result = max(results, key=lambda x: x[2])
        text = best_result[1].strip()
        confidence = best_result[2]

        # Parse the number - handle OCR quirks
        value = self._parse_elixir_text(text)

        if value is not None:
            return DetectionResult(
                value=value,
                confidence=confidence,
                detected=True,
                raw_text=text
            )
        else:
            return DetectionResult(
                value=5,
                confidence=0.0,
                detected=False,
                raw_text=text
            )

    def _parse_elixir_text(self, text: str) -> Optional[int]:
        """
        Parse OCR text to elixir value, handling common OCR errors.

        Handles cases like:
        - "8" -> 8
        - "10" -> 10
        - "88" -> 8 (OCR saw the digit twice)
        - "000" -> 0
        - "110" -> 10 (OCR doubled the "1")

        Args:
            text: Raw OCR text

        Returns:
            Elixir value (0-10) or None if parsing failed
        """
        if not text or not text.strip():
            return None

        text = text.strip()

        # Handle "10" explicitly
        if text == "10":
            return 10

        # Handle repeated "10" like "1010" or "110"
        if "10" in text:
            return 10

        # If all characters are the same digit (e.g., "88", "999", "00")
        # This happens when OCR sees the same digit multiple times
        if len(text) > 1 and len(set(text)) == 1:
            digit = int(text[0])
            return min(digit, 10)

        # Single digit
        if len(text) == 1 and text.isdigit():
            return int(text)

        # Two different digits - take the first one (likely the real digit)
        # This handles cases like "81" when the actual value is 8
        if len(text) >= 1 and text[0].isdigit():
            first_digit = int(text[0])
            # But check if it might be "10" misread
            if first_digit == 1 and len(text) > 1 and text[1] == '0':
                return 10
            return min(first_digit, 10)

        return None

    def detect_timer(
        self,
        image: np.ndarray,
        region: Tuple[int, int, int, int]
    ) -> Optional[int]:
        """
        Detect game timer and return seconds remaining.

        Timer format is M:SS or MM:SS.

        Args:
            image: Full frame (BGR format)
            region: (x_min, y_min, x_max, y_max) of timer area

        Returns:
            Seconds remaining, or None if detection failed
        """
        x1, y1, x2, y2 = region
        roi = image[y1:y2, x1:x2]

        if roi.size == 0:
            return None

        # Preprocess for better OCR
        processed = self._preprocess_for_ocr(roi)

        # Run OCR - allow digits and colon
        try:
            results = self.reader.readtext(
                processed,
                allowlist='0123456789:',
                paragraph=False,
                min_size=5,
            )
        except Exception:
            return None

        if not results:
            return None

        # Combine all detected text
        full_text = ''.join(r[1] for r in results).strip()

        # Try to parse as time format (M:SS or MM:SS)
        time_match = re.match(r'(\d{1,2}):(\d{2})', full_text)
        if time_match:
            minutes = int(time_match.group(1))
            seconds = int(time_match.group(2))
            return minutes * 60 + seconds

        # Try parsing as just seconds if no colon
        if full_text.isdigit():
            return int(full_text)

        return None

    def detect_multiplier_icon(
        self,
        image: np.ndarray,
        region: Tuple[int, int, int, int]
    ) -> int:
        """
        Detect x2 or x3 elixir multiplier icon.

        Both x2 and x3 icons have a purple background with a white digit,
        so we use OCR to read the digit. If no purple background is present,
        there is no multiplier icon (single elixir).

        Args:
            image: Full frame (BGR format)
            region: (x_min, y_min, x_max, y_max) of multiplier area

        Returns:
            1 (no multiplier), 2 (x2), or 3 (x3)
        """
        x1, y1, x2, y2 = region
        roi = image[y1:y2, x1:x2]

        if roi.size == 0:
            return 1

        # Check for purple background to see if icon is present
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        purple_lower = np.array([120, 50, 50])
        purple_upper = np.array([170, 255, 255])
        purple_mask = cv2.inRange(hsv, purple_lower, purple_upper)
        purple_pixels = cv2.countNonZero(purple_mask)
        threshold = roi.shape[0] * roi.shape[1] * 0.05

        if purple_pixels < threshold:
            return 1  # No multiplier icon present

        # Icon is present -- use OCR to read the digit (2 or 3)
        # Crop to right half to skip the "x" prefix which confuses OCR
        processed = self._preprocess_for_ocr(roi)
        right_half = processed[:, processed.shape[1] // 2:]
        try:
            results = self.reader.readtext(
                right_half,
                allowlist='0123456789',
                paragraph=False,
                min_size=5,
            )
        except Exception:
            return 2  # Default to x2 if OCR fails but icon is present

        if results:
            best = max(results, key=lambda x: x[2])
            text = best[1].strip()
            # Take last digit in case of OCR artifacts
            for ch in reversed(text):
                if ch in '23':
                    return int(ch)

        return 2  # Purple icon present but digit unclear, default x2

    def detect_card_cost(
        self,
        image: np.ndarray,
        region: Tuple[int, int, int, int]
    ) -> DetectionResult:
        """
        Detect card elixir cost from a card region.

        Args:
            image: Full frame (BGR format)
            region: (x_min, y_min, x_max, y_max) of card cost area

        Returns:
            DetectionResult with detected cost
        """
        # Card costs are single digits 1-10, same as elixir detection
        return self.detect_elixir(image, region)

    def detect_digits_in_region(
        self,
        image: np.ndarray,
        region: Tuple[int, int, int, int]
    ) -> List[int]:
        """
        Detect all digits in a region.

        Args:
            image: Full frame (BGR format)
            region: (x_min, y_min, x_max, y_max)

        Returns:
            List of detected digits
        """
        x1, y1, x2, y2 = region
        roi = image[y1:y2, x1:x2]

        if roi.size == 0:
            return []

        processed = self._preprocess_for_ocr(roi)

        try:
            results = self.reader.readtext(
                processed,
                allowlist='0123456789',
                paragraph=False,
            )
        except Exception:
            return []

        digits = []
        for _, text, _ in results:
            for char in text:
                if char.isdigit():
                    digits.append(int(char))

        return digits

    def _preprocess_for_ocr(self, roi: np.ndarray) -> np.ndarray:
        """
        Preprocess image region for better OCR results.

        Text in Clash Royale UI is near-white on darker backgrounds,
        so we isolate bright pixels with a simple threshold. EasyOCR
        handles white-on-dark text well, so no inversion is needed.

        Args:
            roi: Region of interest (BGR format)

        Returns:
            Preprocessed image (white text on black background)
        """
        # Convert to grayscale
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi

        # Scale up for reliable OCR
        min_height = 128
        if gray.shape[0] < min_height:
            scale = min_height / gray.shape[0]
            gray = cv2.resize(
                gray,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

        # Threshold to isolate bright (near-white) text
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        return binary

    def __repr__(self) -> str:
        return f"DigitDetector(ocr_loaded={self._reader is not None})"
