"""
Game outcome detection from Clash Royale replay videos.

Uses EasyOCR to detect "Victory" or "Defeat" text from the
end-of-game screen, following the same OCR pattern as DigitDetector.
"""

import cv2
import numpy as np
from enum import Enum
from typing import Optional


_ocr_reader = None


class GameOutcome(Enum):
    WIN = "win"
    LOSS = "loss"
    UNKNOWN = "unknown"


class OutcomeDetector:
    """
    Detects game outcome (Victory/Defeat) from replay video end screens.

    Scans the last N frames of a video for the outcome text that appears
    at the center of the screen after the match ends.
    """

    def __init__(self):
        self._reader = None

    def _initialize_reader(self):
        import easyocr
        self._reader = easyocr.Reader(
            ["en"],
            gpu=True,
            verbose=False,
            quantize=True,
        )

    @property
    def reader(self):
        if self._reader is None:
            self._initialize_reader()
        return self._reader

    def detect_from_video(
        self,
        video_path: str,
        check_last_n_frames: int = 30,
    ) -> GameOutcome:
        """
        Detect game outcome from the last frames of a video.

        Args:
            video_path: Path to the replay video file.
            check_last_n_frames: Number of final frames to scan.

        Returns:
            GameOutcome enum value.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return GameOutcome.UNKNOWN

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = max(0, total_frames - check_last_n_frames)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for _ in range(check_last_n_frames):
            ret, frame = cap.read()
            if not ret:
                break

            result = self.detect_from_frame(frame)
            if result != GameOutcome.UNKNOWN:
                cap.release()
                return result

        cap.release()
        return GameOutcome.UNKNOWN

    def detect_from_frame(self, frame: np.ndarray) -> GameOutcome:
        """
        Detect game outcome from a single frame.

        Looks for "Victory" or "Defeat" text in the center band
        of the screen where the end-game text appears.

        Args:
            frame: BGR image (numpy array).

        Returns:
            GameOutcome enum value.
        """
        h, w = frame.shape[:2]

        # Center region where Victory/Defeat text appears
        x1 = int(0.15 * w)
        y1 = int(0.35 * h)
        x2 = int(0.85 * w)
        y2 = int(0.55 * h)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return GameOutcome.UNKNOWN

        # Preprocess: grayscale, scale up, threshold
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        min_height = 128
        if gray.shape[0] < min_height:
            scale = min_height / gray.shape[0]
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        try:
            results = self.reader.readtext(binary, paragraph=False, min_size=10)
        except Exception:
            return GameOutcome.UNKNOWN

        for _, text, confidence in results:
            text_lower = text.lower().strip()
            if confidence < 0.3:
                continue
            if "victory" in text_lower or "victoire" in text_lower:
                return GameOutcome.WIN
            if "defeat" in text_lower or "defaite" in text_lower:
                return GameOutcome.LOSS

        return GameOutcome.UNKNOWN

    def detect_from_analysis(
        self,
        video_path: str,
        check_last_n_seconds: float = 5.0,
    ) -> GameOutcome:
        """
        Detect outcome by scanning the last few seconds of video.
        More reliable than frame count since it accounts for variable FPS.

        Args:
            video_path: Path to the replay video file.
            check_last_n_seconds: Seconds from end to scan.

        Returns:
            GameOutcome enum value.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return GameOutcome.UNKNOWN

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_check = int(fps * check_last_n_seconds)
        start_frame = max(0, total_frames - frames_to_check)

        cap.release()
        return self.detect_from_video(video_path, check_last_n_frames=frames_to_check)
