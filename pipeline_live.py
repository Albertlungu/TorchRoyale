#!/usr/bin/env python3
"""
Live TorchRoyale pipeline optimized for real-time processing.

This pipeline prioritizes speed over debug features:
- Minimal console output
- No debug image generation
- Cached OCR results
- Reduced processing frequency for expensive operations

Usage:
    from pipeline_live import LivePipeline

    pipeline = LivePipeline()
    pipeline.process_frame(frame)  # Returns game state
"""

import os
import time
from pathlib import Path
from typing import Any, Dict

import cv2

# Suppress optional model dependency warnings
os.environ.setdefault("QWEN_2_5_ENABLED", "False")
os.environ.setdefault("QWEN_3_ENABLED", "False")
os.environ.setdefault("CORE_MODEL_SAM_ENABLED", "False")
os.environ.setdefault("CORE_MODEL_SAM3_ENABLED", "False")
os.environ.setdefault("CORE_MODEL_GAZE_ENABLED", "False")
os.environ.setdefault("CORE_MODEL_YOLO_WORLD_ENABLED", "False")

# Add project root to path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from detection_test import DetectionPipeline
from src.constants import UIRegions
from src.game_state import GamePhaseTracker, TowerHealthDetector
from src.ocr import DigitDetector
from src.recommendation.elixir_manager import OpponentElixirTracker


class LivePipeline:
    """Real-time optimized game state processing pipeline."""

    def __init__(self, target_fps: float = 30.0):
        """Initialize the live pipeline.

        Args:
            target_fps: Target processing rate (frames per second)
        """
        self.target_fps = target_fps
        self.frame_time_budget = 1.0 / target_fps  # Max time per frame

        # Initialize components (lazy loading for faster startup)
        self._detector = None
        self._regions = None
        self._phase_tracker = None
        self._opponent_tracker = None
        self._health_detector = None
        self._detection_pipeline = None

        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()

        # Caching for expensive operations
        self._last_ocr_frame = 0
        self._ocr_cache = {"elixir": None, "timer": None, "multiplier": None}

        self._last_detection_frame = 0
        self._detection_cache = []

        self._last_tower_frame = 0
        self._tower_cache = {}

        # Processing intervals (frames between operations)
        self.ocr_interval = max(1, int(target_fps // 10))  # OCR 10x per second
        self.detection_interval = 2  # Detection every 2nd frame
        self.tower_interval = max(1, int(target_fps // 2))  # Tower health 2x per second

        # Game state
        self.player_level = 15
        self.opponent_level = 15
        self.levels_detected = False
        self.last_tower_hp = {}

        print(f"Live pipeline initialized (target: {target_fps} FPS)")
        print(
            f"Processing intervals - OCR: every {self.ocr_interval} frames, "
            f"Detection: every {self.detection_interval} frames, "
            f"Towers: every {self.tower_interval} frames"
        )

    def _lazy_init_components(self, frame_width: int, frame_height: int):
        """Initialize components on first frame (lazy loading)."""
        if self._detector is None:
            self._detector = DigitDetector()
            self._regions = UIRegions(frame_width, frame_height)
            self._phase_tracker = GamePhaseTracker()
            self._opponent_tracker = OpponentElixirTracker()
            self._health_detector = TowerHealthDetector()
            self._detection_pipeline = DetectionPipeline()

    def process_frame(self, frame: cv2.Mat) -> Dict[str, Any]:
        """Process a single frame and return game state.

        Args:
            frame: Input frame from video capture

        Returns:
            Dictionary containing current game state:
            - elixir: Current player elixir count
            - opponent_elixir: Estimated opponent elixir
            - timer: Game timer (seconds remaining)
            - game_phase: Current game phase
            - tower_health: Tower health status
            - detections: Card/tower detections (if updated this frame)
            - processing_time: Time taken to process this frame
        """
        process_start = time.time()

        h, w = frame.shape[:2]
        self._lazy_init_components(w, h)

        timestamp_ms = int(self.frame_count * 1000 / self.target_fps)

        # OCR operations (cached)
        elixir_result = self._ocr_cache["elixir"]
        timer = self._ocr_cache["timer"]
        mult = self._ocr_cache["multiplier"]

        if self.frame_count % self.ocr_interval == 0:
            elixir_result = self._detector.detect_elixir(
                frame, self._regions.elixir_number.to_tuple()
            )
            timer = self._detector.detect_timer(frame, self._regions.timer.to_tuple())
            mult = self._detector.detect_multiplier_icon(
                frame, self._regions.multiplier_icon.to_tuple()
            )

            self._ocr_cache["elixir"] = elixir_result
            self._ocr_cache["timer"] = timer
            self._ocr_cache["multiplier"] = mult
            self._last_ocr_frame = self.frame_count

        # Update game phase
        phase = self._phase_tracker.update(multiplier_detected=mult)

        # Detection pipeline (cached)
        all_dets = self._detection_cache
        detections_updated = False

        if self.frame_count % self.detection_interval == 0:
            try:
                rf_results = self._detection_pipeline.process_image_array(frame)
                all_dets = rf_results.get("detections", [])
                self._detection_cache = all_dets
                self._last_detection_frame = self.frame_count
                detections_updated = True
            except Exception:
                all_dets = []

        # Extract relevant detections
        tower_dets = []
        opponent_on_field = []
        for det in all_dets:
            if "tower" in det.class_name:
                tower_dets.append(
                    {
                        "class_name": det.class_name,
                        "pixel_x": det.pixel_x,
                        "pixel_y": det.pixel_y,
                        "pixel_width": det.pixel_width,
                        "pixel_height": det.pixel_height,
                        "is_opponent": det.is_opponent,
                    }
                )
            if det.is_opponent and det.is_on_field:
                opponent_on_field.append(det)

        # Detect tower levels (once)
        if not self.levels_detected and tower_dets:
            for td in tower_dets:
                if "king" in td["class_name"]:
                    level = self._health_detector.detect_tower_level(
                        frame,
                        td["pixel_x"],
                        td["pixel_y"],
                        td["pixel_width"],
                        td["pixel_height"],
                        td["is_opponent"],
                    )
                    if td["is_opponent"]:
                        self.opponent_level = level
                    else:
                        self.player_level = level
            self.levels_detected = True

        # Tower health detection (cached)
        tower_health = self._tower_cache

        if self.frame_count % self.tower_interval == 0 and tower_dets:
            tower_health = self._health_detector.detect_all_towers(
                frame,
                tower_dets,
                player_level=self.player_level,
                opponent_level=self.opponent_level,
            )

            # Use last known HP for OCR failures
            for name, res in tower_health.items():
                if res.health_percent is not None:
                    self.last_tower_hp[name] = res
                elif name in self.last_tower_hp:
                    tower_health[name] = self.last_tower_hp[name]

            self._tower_cache = tower_health
            self._last_tower_frame = self.frame_count

        # Update opponent tracker
        opp_elixir = self._opponent_tracker.update(
            timestamp_ms=timestamp_ms,
            game_phase=phase,
            opponent_detections=opponent_on_field,
        )

        self.frame_count += 1
        processing_time = time.time() - process_start

        # Return game state
        return {
            "elixir": elixir_result.value if elixir_result else 0,
            "opponent_elixir": opp_elixir,
            "timer": timer,
            "game_phase": phase,
            "tower_health": tower_health,
            "detections": all_dets if detections_updated else None,
            "processing_time": processing_time,
            "frame_count": self.frame_count,
        }

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        elapsed = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0

        return {
            "frames_processed": self.frame_count,
            "elapsed_time": elapsed,
            "average_fps": avg_fps,
            "target_fps": self.target_fps,
        }


if __name__ == "__main__":
    """Example usage - process video in real-time simulation."""
    import sys

    if len(sys.argv) != 2:
        print("Usage: python pipeline_live.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    pipeline = LivePipeline(target_fps=fps)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        state = pipeline.process_frame(frame)

        # Simple console output every second
        if frame_count % int(fps) == 0:
            print(
                f"Frame {frame_count}: Elixir={state['elixir']}, "
                f"Opp={state['opponent_elixir']:.1f}, "
                f"Time={state['processing_time'] * 1000:.1f}ms"
            )

        frame_count += 1

        # Limit to 10 seconds for demo
        if frame_count > fps * 10:
            break

    cap.release()

    # Print performance stats
    stats = pipeline.get_performance_stats()
    print("\nPerformance Summary:")
    print(
        f"Processed {stats['frames_processed']} frames in {stats['elapsed_time']:.2f}s"
    )
    print(
        f"Average FPS: {stats['average_fps']:.2f} (target: {stats['target_fps']:.1f})"
    )
