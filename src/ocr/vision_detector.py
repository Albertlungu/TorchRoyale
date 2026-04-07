"""
Digit/text detector using Moondream2 vision-language model.

Drop-in replacement for DigitDetector (EasyOCR-based) that uses
Moondream2 for reading numbers and text from Clash Royale UI.

More robust across different screen sizes and resolutions since
the VLM understands visual context rather than relying on pixel
thresholds and hardcoded preprocessing.

Requires: pip install transformers einops
Model: vikhyatk/moondream2 (~1.8B params, ~3.5GB VRAM FP16)
"""

import cv2
import numpy as np
import re
from typing import Optional, Tuple, List
from dataclasses import dataclass

from .digit_detector import DetectionResult


class VisionDetector:
    """
    Detects digits and text in Clash Royale UI regions using Moondream2.

    Same public API as DigitDetector for drop-in replacement.

    Usage:
        detector = VisionDetector()
        result = detector.detect_elixir(frame, (x1, y1, x2, y2))
        print(f"Elixir: {result.value}")
    """

    def __init__(self, device: str = "auto", preload: bool = False):
        """
        Args:
            device: "auto", "cuda", "cpu", or "mps".
            preload: If True, load model immediately instead of on first use.
        """
        self._model = None
        self._tokenizer = None
        self._device_str = device
        self._device = None

        if preload:
            self._load_model()

    def _load_model(self):
        """Load Moondream2 model and tokenizer."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self._device_str == "auto":
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")
        else:
            self._device = torch.device(self._device_str)

        model_id = "vikhyatk/moondream2"
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self._device.type != "cpu" else torch.float32,
            device_map={"": self._device},
        )
        self._model.eval()

    def _ensure_loaded(self):
        if self._model is None:
            self._load_model()

    def _ask(self, image: np.ndarray, question: str) -> str:
        """
        Ask Moondream2 a question about an image region.

        Args:
            image: BGR or RGB numpy array (the ROI).
            question: Question to ask about the image.

        Returns:
            Model's text response.
        """
        from PIL import Image

        self._ensure_loaded()

        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif len(image.shape) == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = image

        pil_image = Image.fromarray(rgb)

        enc_image = self._model.encode_image(pil_image)
        answer = self._model.answer_question(enc_image, question, self._tokenizer)

        return answer.strip()

    def _extract_number(self, text: str) -> Optional[int]:
        """Extract the first integer from model response text."""
        numbers = re.findall(r"\d+", text)
        if numbers:
            return int(numbers[0])
        return None

    def detect_elixir(
        self,
        image: np.ndarray,
        region: Tuple[int, int, int, int],
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

        try:
            answer = self._ask(roi, "What number is shown in this image? Reply with just the number.")
            value = self._extract_number(answer)

            if value is not None and 0 <= value <= 10:
                return DetectionResult(
                    value=value, confidence=0.9, detected=True, raw_text=answer
                )
        except Exception as e:
            return DetectionResult(value=5, confidence=0.0, detected=False, raw_text=str(e))

        return DetectionResult(value=5, confidence=0.0, detected=False, raw_text=answer)

    def detect_timer(
        self,
        image: np.ndarray,
        region: Tuple[int, int, int, int],
    ) -> Optional[int]:
        """
        Detect game timer and return seconds remaining.

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

        try:
            answer = self._ask(roi, "What time is shown on this game timer? Reply in M:SS format.")

            # Try M:SS or MM:SS
            time_match = re.search(r"(\d{1,2}):(\d{2})", answer)
            if time_match:
                minutes = int(time_match.group(1))
                seconds = int(time_match.group(2))
                return minutes * 60 + seconds

            # Try just a number
            value = self._extract_number(answer)
            if value is not None:
                return value

        except Exception:
            pass

        return None

    def detect_multiplier_icon(
        self,
        image: np.ndarray,
        region: Tuple[int, int, int, int],
    ) -> int:
        """
        Detect x2 or x3 elixir multiplier icon.

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

        # Quick check: is there a purple icon at all?
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        purple_lower = np.array([120, 50, 50])
        purple_upper = np.array([170, 255, 255])
        purple_mask = cv2.inRange(hsv, purple_lower, purple_upper)
        purple_pixels = cv2.countNonZero(purple_mask)
        threshold = roi.shape[0] * roi.shape[1] * 0.05

        if purple_pixels < threshold:
            return 1

        try:
            answer = self._ask(roi, "What multiplier number is shown? Reply with just 2 or 3.")
            value = self._extract_number(answer)
            if value in (2, 3):
                return value
        except Exception:
            pass

        return 2  # Purple present but unclear, default x2

    def detect_card_cost(
        self,
        image: np.ndarray,
        region: Tuple[int, int, int, int],
    ) -> DetectionResult:
        """Detect card elixir cost from a card region."""
        return self.detect_elixir(image, region)

    def detect_digits_in_region(
        self,
        image: np.ndarray,
        region: Tuple[int, int, int, int],
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

        try:
            answer = self._ask(roi, "What numbers are shown in this image? List all digits.")
            return [int(d) for d in re.findall(r"\d", answer)]
        except Exception:
            return []

    def detect_hp(
        self,
        image: np.ndarray,
        region: Tuple[int, int, int, int],
        hp_max: int,
    ) -> Optional[int]:
        """
        Detect tower HP value from a region.

        Specialized method for tower health that gives the VLM
        context about what it's reading.

        Args:
            image: Full frame (BGR format)
            region: (x_min, y_min, x_max, y_max) of HP text area
            hp_max: Maximum possible HP for validation

        Returns:
            HP value, or None if detection failed
        """
        x1, y1, x2, y2 = region
        roi = image[y1:y2, x1:x2]

        if roi.size == 0:
            return None

        try:
            answer = self._ask(
                roi,
                "What is the health points number shown in this game image? Reply with just the number."
            )
            value = self._extract_number(answer)
            if value is not None and 1 <= value <= hp_max:
                return value
        except Exception:
            pass

        return None

    def detect_game_outcome(
        self,
        image: np.ndarray,
    ) -> Optional[str]:
        """
        Detect Victory/Defeat from a full game end screen.

        Args:
            image: Full frame (BGR format)

        Returns:
            "win", "loss", or None
        """
        h, w = image.shape[:2]
        x1 = int(0.15 * w)
        y1 = int(0.35 * h)
        x2 = int(0.85 * w)
        y2 = int(0.55 * h)

        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        try:
            answer = self._ask(
                roi,
                "Does this image show 'Victory' or 'Defeat'? Reply with just one word."
            )
            lower = answer.lower()
            if "victory" in lower or "win" in lower:
                return "win"
            if "defeat" in lower or "loss" in lower or "lose" in lower:
                return "loss"
        except Exception:
            pass

        return None

    def __repr__(self) -> str:
        loaded = self._model is not None
        return f"VisionDetector(moondream2, loaded={loaded})"
