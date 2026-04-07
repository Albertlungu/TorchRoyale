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
        check_last_n_seconds: float = 10.0,
    ) -> GameOutcome:
        """
        Detect game outcome from the last seconds of a video.

        Reads through entire video and checks the last N seconds worth of frames.

        Args:
            video_path: Path to the replay video file.
            check_last_n_seconds: Number of seconds from end to scan (default: 10).

        Returns:
            GameOutcome enum value.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return GameOutcome.UNKNOWN

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames_to_keep = int(fps * check_last_n_seconds)

        # Read all frames and keep only the last N seconds in a buffer
        frame_buffer = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame_buffer.append(frame)

            # Keep buffer size manageable
            if len(frame_buffer) > frames_to_keep:
                frame_buffer.pop(0)

        cap.release()

        # Now check the buffered frames for outcome
        for frame in frame_buffer:
            result = self.detect_from_frame(frame)
            if result != GameOutcome.UNKNOWN:
                return result

        return GameOutcome.UNKNOWN

    def _is_cyan_blue(self, color_bgr: np.ndarray) -> bool:
        """
        Check if a BGR color is close to cyan-blue (#64FFFE).

        Args:
            color_bgr: BGR color as numpy array [B, G, R].

        Returns:
            True if color is close to cyan-blue.
        """
        target_rgb = np.array([100, 255, 254])  # #64FFFE
        target_bgr = target_rgb[::-1]  # Convert to BGR

        # Allow some tolerance for color matching
        tolerance = 50
        diff = np.abs(color_bgr.astype(int) - target_bgr.astype(int))
        return np.all(diff < tolerance)

    def _check_winner_text_color(self, frame: np.ndarray, bbox: list) -> bool:
        """
        Check if the bounding box region contains cyan-blue color.

        Args:
            frame: Original BGR frame.
            bbox: Bounding box from OCR [top_left, top_right, bottom_right, bottom_left].

        Returns:
            True if cyan-blue color is detected in the bbox region.
        """
        try:
            # Extract bounding box coordinates
            x_coords = [int(pt[0]) for pt in bbox]
            y_coords = [int(pt[1]) for pt in bbox]
            x1, x2 = min(x_coords), max(x_coords)
            y1, y2 = min(y_coords), max(y_coords)

            # Extract region
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return False

            # Sample pixels in the region and check for cyan-blue
            for y in range(0, roi.shape[0], max(1, roi.shape[0] // 10)):
                for x in range(0, roi.shape[1], max(1, roi.shape[1] // 10)):
                    if self._is_cyan_blue(roi[y, x]):
                        return True

            return False
        except Exception:
            return False

    def detect_from_frame(self, frame: np.ndarray) -> GameOutcome:
        """
        Detect game outcome from a single frame.

        Looks for "Winner" text with cyan-blue color (#64FFFE) for victory,
        or "Victory"/"Defeat" text in the center band of the screen.

        Args:
            frame: BGR image (numpy array).

        Returns:
            GameOutcome enum value.
        """
        h, w = frame.shape[:2]

        # Center region where Victory/Defeat/Winner text appears
        x1 = int(0.15 * w)
        y1 = int(0.35 * h)
        x2 = int(0.85 * w)
        y2 = int(0.55 * h)

        roi = frame[y1:y2, x1:x2]
        roi_full_frame = frame.copy()  # Keep full frame for color checking
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

        # Also check on the ROI from original frame for "Winner" with color
        try:
            roi_results = self.reader.readtext(roi, paragraph=False, min_size=10)
        except Exception:
            roi_results = []

        # Check for "Winner" with cyan-blue color first
        for bbox, text, confidence in roi_results:
            text_lower = text.lower().strip()
            if confidence < 0.3:
                continue
            if "winner" in text_lower:
                # Adjust bbox coordinates to ROI space
                adjusted_bbox = [[pt[0] + x1, pt[1] + y1] for pt in bbox]
                if self._check_winner_text_color(roi_full_frame, adjusted_bbox):
                    return GameOutcome.WIN
                else:
                    return GameOutcome.LOSS

        # Fall back to Victory/Defeat detection
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
        check_last_n_seconds: float = 10.0,
    ) -> GameOutcome:
        """
        Detect outcome by scanning the last few seconds of video.
        Alias for detect_from_video() for backward compatibility.

        Args:
            video_path: Path to the replay video file.
            check_last_n_seconds: Seconds from end to scan.

        Returns:
            GameOutcome enum value.
        """
        return self.detect_from_video(video_path, check_last_n_seconds=check_last_n_seconds)
