"""
Video processor for extracting frames from game recordings.

Provides frame extraction with configurable skip intervals,
timestamp tracking, and resolution information.
"""

import cv2
from pathlib import Path
from typing import Generator, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class VideoInfo:
    """Metadata about a video file."""
    path: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float

    def __repr__(self) -> str:
        return (
            f"VideoInfo(path='{Path(self.path).name}', "
            f"{self.width}x{self.height}, "
            f"{self.fps:.1f}fps, "
            f"{self.duration_seconds:.1f}s, "
            f"{self.total_frames} frames)"
        )


class VideoProcessor:
    """
    Processes video files and extracts frames at specified intervals.

    Usage:
        processor = VideoProcessor(frame_skip=6)
        info = processor.open("game_recording.mp4")

        for frame, frame_num, timestamp_ms in processor.frames():
            # Process frame
            pass

        processor.close()

    Or using context manager:
        with VideoProcessor(frame_skip=6) as processor:
            processor.open("game_recording.mp4")
            for frame, frame_num, timestamp_ms in processor.frames():
                pass
    """

    def __init__(self, frame_skip: int = 6):
        """
        Initialize video processor.

        Args:
            frame_skip: Process every Nth frame.
                       Default 6 means ~5 FPS from 30 FPS video.
                       Set to 1 to process every frame.
        """
        if frame_skip < 1:
            raise ValueError("frame_skip must be >= 1")

        self.frame_skip = frame_skip
        self._cap: Optional[cv2.VideoCapture] = None
        self._video_info: Optional[VideoInfo] = None

    def open(self, video_path: str) -> VideoInfo:
        """
        Open a video file for processing.

        Args:
            video_path: Path to MP4 or other video file

        Returns:
            VideoInfo with metadata about the video

        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video cannot be opened
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        self._cap = cv2.VideoCapture(str(path))

        if not self._cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = self._cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self._video_info = VideoInfo(
            path=str(path),
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=fps,
            total_frames=total_frames,
            duration_seconds=total_frames / fps if fps > 0 else 0
        )

        return self._video_info

    def frames(self) -> Generator[Tuple[np.ndarray, int, int], None, None]:
        """
        Generator that yields frames at the configured skip interval.

        Yields:
            Tuple of (frame, frame_number, timestamp_ms)
            - frame: BGR image as numpy array
            - frame_number: Original frame number in video
            - timestamp_ms: Timestamp in milliseconds

        Raises:
            RuntimeError: If no video is open
        """
        if self._cap is None or self._video_info is None:
            raise RuntimeError("No video opened. Call open() first.")

        frame_number = 0

        while True:
            ret, frame = self._cap.read()

            if not ret:
                break

            if frame_number % self.frame_skip == 0:
                timestamp_ms = int((frame_number / self._video_info.fps) * 1000)
                yield frame, frame_number, timestamp_ms

            frame_number += 1

    def get_frame_at(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Get a specific frame by number.

        Args:
            frame_number: Frame index (0-based)

        Returns:
            Frame as numpy array, or None if frame not found

        Raises:
            RuntimeError: If no video is open
        """
        if self._cap is None:
            raise RuntimeError("No video opened. Call open() first.")

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self._cap.read()
        return frame if ret else None

    def get_frame_at_time(self, time_ms: int) -> Optional[np.ndarray]:
        """
        Get the frame at a specific timestamp.

        Args:
            time_ms: Timestamp in milliseconds

        Returns:
            Frame as numpy array, or None if not found

        Raises:
            RuntimeError: If no video is open
        """
        if self._cap is None or self._video_info is None:
            raise RuntimeError("No video opened. Call open() first.")

        frame_number = int((time_ms / 1000) * self._video_info.fps)
        return self.get_frame_at(frame_number)

    def reset(self):
        """Reset video to the beginning."""
        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def close(self):
        """Release video resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def video_info(self) -> Optional[VideoInfo]:
        """Get info about the currently open video."""
        return self._video_info

    @property
    def effective_fps(self) -> float:
        """
        Get effective FPS after applying frame skip.

        Returns:
            Effective frames per second being processed
        """
        if self._video_info is None:
            return 0
        return self._video_info.fps / self.frame_skip

    @property
    def frames_to_process(self) -> int:
        """
        Get estimated number of frames that will be processed.

        Returns:
            Approximate number of frames after applying skip
        """
        if self._video_info is None:
            return 0
        return self._video_info.total_frames // self.frame_skip

    def __enter__(self) -> "VideoProcessor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self) -> str:
        status = "open" if self._cap is not None else "closed"
        return f"VideoProcessor(frame_skip={self.frame_skip}, status={status})"
