"""
Main video analysis pipeline for Clash Royale game recordings.

Orchestrates all components to process videos and extract:
- Frame-by-frame detections (via Roboflow)
- Player elixir (visual detection)
- Opponent elixir (calculated)
- Game phase (single/double/triple elixir)
- Tower health

Outputs JSON with timestamps and optionally saves annotated frames.
"""

import json
import cv2
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.video.video_processor import VideoProcessor, VideoInfo
from src.constants.game_constants import GamePhase, ElixirConstants
from src.constants.ui_regions import UIRegions
from src.ocr.digit_detector import DigitDetector
from src.game_state.game_phase import GamePhaseTracker
from src.game_state.health_detector import TowerHealthDetector, TowerHealthResult
from src.recommendation.elixir_manager import OpponentElixirTracker, PlayerElixirTracker


@dataclass
class TowerHealth:
    """Health status for a single tower."""
    hp_current: Optional[int]
    hp_max: int
    health_percent: Optional[float]  # None = unknown (OCR failure)
    is_destroyed: bool = False


@dataclass
class FrameState:
    """Complete game state for a single frame."""
    # Timing
    timestamp_ms: int
    frame_number: int

    # Game phase
    game_phase: str
    elixir_multiplier: int

    # Elixir
    player_elixir: int
    opponent_elixir_estimated: float

    # Timer (None if not detected)
    game_time_remaining: Optional[int] = None

    # Detections
    detections: List[Dict[str, Any]] = field(default_factory=list)

    # Tower health
    player_towers: Dict[str, Dict] = field(default_factory=dict)
    opponent_towers: Dict[str, Dict] = field(default_factory=dict)

    # Cards in hand
    hand_cards: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "timestamp_ms": self.timestamp_ms,
            "frame_number": self.frame_number,
            "game_time_remaining": self.game_time_remaining,
            "game_phase": self.game_phase,
            "elixir_multiplier": self.elixir_multiplier,
            "player_elixir": self.player_elixir,
            "opponent_elixir_estimated": round(self.opponent_elixir_estimated, 2),
            "detections": self.detections,
            "player_towers": self.player_towers,
            "opponent_towers": self.opponent_towers,
            "hand_cards": self.hand_cards,
        }


class VideoAnalyzer:
    """
    Main pipeline that combines all components to analyze game videos.

    Processes video files and outputs:
    - JSON file with frame-by-frame game state
    - Optionally, annotated frames saved as images

    Usage:
        analyzer = VideoAnalyzer()
        result = analyzer.analyze_video("game_recording.mp4")

    Or with custom settings:
        analyzer = VideoAnalyzer(
            frame_skip=6,
            save_annotated_frames=True,
            output_dir="output"
        )
    """

    def __init__(
        self,
        frame_skip: int = 6,
        save_annotated_frames: bool = False,
        annotated_frame_interval: int = 30,
        output_dir: str = "output",
        verbose: bool = True,
    ):
        """
        Initialize video analyzer.

        Args:
            frame_skip: Process every Nth frame (default 6 = ~5 FPS from 30 FPS)
            save_annotated_frames: Whether to save annotated images
            annotated_frame_interval: Save every Nth processed frame as image
            output_dir: Directory for output files
            verbose: Print progress messages
        """
        self.frame_skip = frame_skip
        self.save_frames = save_annotated_frames
        self.frame_save_interval = annotated_frame_interval
        self.output_dir = Path(output_dir)
        self.verbose = verbose

        # Components (initialized on first use)
        self._video_processor: Optional[VideoProcessor] = None
        self._detection_pipeline = None
        self._digit_detector: Optional[DigitDetector] = None
        self._phase_tracker: Optional[GamePhaseTracker] = None
        self._opponent_tracker: Optional[OpponentElixirTracker] = None
        self._player_tracker: Optional[PlayerElixirTracker] = None
        self._health_detector: Optional[TowerHealthDetector] = None
        self._ui_regions: Optional[UIRegions] = None

        # Tower levels (detected once and reused)
        self._player_level: int = 15
        self._opponent_level: int = 15
        self._levels_detected: bool = False

        # Last known tower health (for fallback when OCR fails on princess)
        self._last_tower_health: Dict[str, TowerHealthResult] = {}

    def _initialize_components(self):
        """Initialize all analysis components."""
        self._video_processor = VideoProcessor(frame_skip=self.frame_skip)
        self._digit_detector = DigitDetector()
        self._phase_tracker = GamePhaseTracker()
        self._opponent_tracker = OpponentElixirTracker()
        self._player_tracker = PlayerElixirTracker()
        self._health_detector = TowerHealthDetector()

        # Detection pipeline is imported lazily to avoid circular imports
        # and to not require Roboflow API key until needed
        try:
            from detection_test import DetectionPipeline
            self._detection_pipeline = DetectionPipeline()
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not load DetectionPipeline: {e}")
                print("Detection features will be disabled.")
            self._detection_pipeline = None

    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """
        Analyze a complete video file.

        Args:
            video_path: Path to video file (MP4, etc.)

        Returns:
            Dictionary containing:
            - video_info: Metadata about the video
            - frames: List of FrameState dictionaries
            - summary: Game summary statistics
        """
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self._initialize_components()

        # Open video
        video_info = self._video_processor.open(video_path)

        # Initialize UI regions for video resolution
        self._ui_regions = UIRegions(video_info.width, video_info.height)

        # Reset trackers for new game
        self._phase_tracker.reset()
        self._opponent_tracker.reset()
        self._player_tracker.reset()
        self._levels_detected = False
        self._player_level = 15
        self._opponent_level = 15
        self._last_tower_health = {}

        if self.verbose:
            print(f"Processing video: {video_path}")
            print(f"Resolution: {video_info.width}x{video_info.height}")
            print(f"Duration: {video_info.duration_seconds:.1f}s")
            print(f"Source FPS: {video_info.fps:.1f}")
            print(f"Effective FPS: {self._video_processor.effective_fps:.1f}")
            print(f"Frames to process: ~{self._video_processor.frames_to_process}")
            print("-" * 50)

        # Process frames
        frame_states: List[FrameState] = []
        processed_count = 0

        for frame, frame_num, timestamp_ms in self._video_processor.frames():
            # Process single frame
            state = self._process_frame(frame, frame_num, timestamp_ms)
            frame_states.append(state)
            processed_count += 1

            # Save annotated frame if enabled
            if self.save_frames and processed_count % self.frame_save_interval == 0:
                self._save_annotated_frame(frame, state, frame_num)

            # Progress update
            if self.verbose and processed_count % 50 == 0:
                progress = (frame_num / video_info.total_frames) * 100
                print(
                    f"Progress: {progress:.1f}% | "
                    f"Frame {frame_num} | "
                    f"Time {timestamp_ms/1000:.1f}s | "
                    f"Phase: {state.game_phase} | "
                    f"Player Elixir: {state.player_elixir} | "
                    f"Opp Elixir: {state.opponent_elixir_estimated:.1f}"
                )

        # Close video
        self._video_processor.close()

        if self.verbose:
            print("-" * 50)
            print(f"Processed {processed_count} frames")

        # Build result
        result = {
            "video_info": {
                "path": video_path,
                "width": video_info.width,
                "height": video_info.height,
                "fps": video_info.fps,
                "duration_seconds": video_info.duration_seconds,
                "frame_skip": self.frame_skip,
                "total_frames_processed": len(frame_states),
            },
            "frames": [s.to_dict() for s in frame_states],
            "summary": self._generate_summary(frame_states),
        }

        # Save JSON output
        video_name = Path(video_path).stem
        output_path = self.output_dir / f"{video_name}_analysis.json"

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        if self.verbose:
            print(f"Analysis saved to: {output_path}")

        return result

    def _process_frame(
        self,
        frame,
        frame_num: int,
        timestamp_ms: int
    ) -> FrameState:
        """
        Process a single frame and extract game state.

        Args:
            frame: BGR image as numpy array
            frame_num: Frame number in video
            timestamp_ms: Timestamp in milliseconds

        Returns:
            FrameState with all detected information
        """
        # Default values
        detections = []
        hand_cards = []
        raw_detections = []

        # Run Roboflow detection if available
        if self._detection_pipeline is not None:
            try:
                results = self._detection_pipeline.process_image_array(frame)
                raw_detections = results.get("detections", [])

                # Convert detections to serializable format
                for det in raw_detections:
                    detections.append({
                        "class_name": det.class_name,
                        "tile_x": det.tile_x,
                        "tile_y": det.tile_y,
                        "is_opponent": det.is_opponent,
                        "is_on_field": det.is_on_field,
                        "confidence": round(det.confidence, 3),
                    })

                # Extract hand cards
                hand_cards = [
                    det.class_name
                    for det in raw_detections
                    if not det.is_opponent and not det.is_on_field
                ]
            except Exception as e:
                if self.verbose:
                    print(f"Detection error at frame {frame_num}: {e}")

        # Detect player elixir
        elixir_result = self._digit_detector.detect_elixir(
            frame,
            self._ui_regions.elixir_number.to_tuple()
        )
        player_elixir = self._player_tracker.update(
            elixir_result.value if elixir_result.detected else -1,
            elixir_result.confidence
        )

        # Detect timer (optional)
        game_time = self._digit_detector.detect_timer(
            frame,
            self._ui_regions.timer.to_tuple()
        )

        # Detect multiplier icon
        multiplier = self._digit_detector.detect_multiplier_icon(
            frame,
            self._ui_regions.multiplier_icon.to_tuple()
        )

        # Update game phase
        game_phase = self._phase_tracker.update(
            multiplier_detected=multiplier,
            timer_seconds=game_time,
            timestamp_ms=timestamp_ms
        )

        # Get opponent detections for elixir tracking
        opponent_detections = [d for d in detections if d.get("is_opponent", False)]

        # Create detection-like objects for opponent tracker
        class DetectionProxy:
            def __init__(self, d):
                self.class_name = d.get("class_name", "")
                self.is_opponent = d.get("is_opponent", False)
                self.is_on_field = d.get("is_on_field", False)
                self.tile_x = d.get("tile_x", 0)
                self.tile_y = d.get("tile_y", 0)

        opponent_det_proxies = [DetectionProxy(d) for d in opponent_detections]

        # Update opponent elixir estimate
        opponent_elixir = self._opponent_tracker.update(
            timestamp_ms=timestamp_ms,
            game_phase=game_phase,
            opponent_detections=opponent_det_proxies
        )

        # Extract tower detections from Roboflow results
        tower_dets = []
        for det in raw_detections:
            if "tower" in det.class_name:
                tower_dets.append({
                    "class_name": det.class_name,
                    "pixel_x": det.pixel_x,
                    "pixel_y": det.pixel_y,
                    "pixel_width": det.pixel_width,
                    "pixel_height": det.pixel_height,
                    "is_opponent": det.is_opponent,
                })

        # Detect tower levels from king towers (once)
        if not self._levels_detected and tower_dets:
            for td in tower_dets:
                if "king" in td["class_name"]:
                    level = self._health_detector.detect_tower_level(
                        frame,
                        td["pixel_x"], td["pixel_y"],
                        td["pixel_width"], td["pixel_height"],
                        td["is_opponent"],
                    )
                    if td["is_opponent"]:
                        self._opponent_level = level
                    else:
                        self._player_level = level
            self._levels_detected = True

        # Detect tower health via OCR
        tower_health = {}
        if tower_dets:
            tower_health = self._health_detector.detect_all_towers(
                frame, tower_dets,
                player_level=self._player_level,
                opponent_level=self._opponent_level,
            )

        # Use last known health for princess towers where OCR failed
        for name, result in tower_health.items():
            if result.health_percent is not None:
                self._last_tower_health[name] = result
            elif name in self._last_tower_health:
                tower_health[name] = self._last_tower_health[name]

        # Separate player and opponent towers
        player_towers = {}
        opponent_towers = {}
        for name, th in tower_health.items():
            tower_data = {
                "hp_current": th.hp_current,
                "hp_max": th.hp_max,
                "health_percent": round(th.health_percent, 1) if th.health_percent is not None else None,
                "is_destroyed": th.is_destroyed,
            }
            if name.startswith("player_"):
                player_towers[name] = tower_data
            else:
                opponent_towers[name] = tower_data

        # Build frame state
        return FrameState(
            timestamp_ms=timestamp_ms,
            frame_number=frame_num,
            game_time_remaining=game_time,
            game_phase=game_phase.value,
            elixir_multiplier=self._phase_tracker.elixir_multiplier,
            player_elixir=player_elixir,
            opponent_elixir_estimated=opponent_elixir,
            detections=detections,
            player_towers=player_towers,
            opponent_towers=opponent_towers,
            hand_cards=hand_cards,
        )

    def _save_annotated_frame(self, frame, state: FrameState, frame_num: int):
        """Save an annotated frame with overlays."""
        annotated = frame.copy()

        # Add text overlays
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        color = (255, 255, 255)
        bg_color = (0, 0, 0)

        # Helper to draw text with background
        def draw_text(img, text, pos, color=color):
            (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            x, y = pos
            cv2.rectangle(img, (x-2, y-h-2), (x+w+2, y+2), bg_color, -1)
            cv2.putText(img, text, pos, font, font_scale, color, thickness)

        # Frame info
        draw_text(annotated, f"Frame: {frame_num}", (10, 30))
        draw_text(annotated, f"Time: {state.timestamp_ms/1000:.1f}s", (10, 55))

        # Elixir info
        draw_text(annotated, f"Player Elixir: {state.player_elixir}", (10, 85), (255, 100, 255))
        draw_text(annotated, f"Opp Elixir: {state.opponent_elixir_estimated:.1f}", (10, 110), (100, 100, 255))

        # Game phase
        draw_text(annotated, f"Phase: {state.game_phase} (x{state.elixir_multiplier})", (10, 140), (100, 255, 100))

        # Detection count
        draw_text(annotated, f"Detections: {len(state.detections)}", (10, 170))

        # Save
        frames_dir = self.output_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        output_path = frames_dir / f"frame_{frame_num:06d}.png"
        cv2.imwrite(str(output_path), annotated)

    def _generate_summary(self, frame_states: List[FrameState]) -> Dict[str, Any]:
        """Generate summary statistics from processed frames."""
        if not frame_states:
            return {}

        # Collect statistics
        phases_seen = set(s.game_phase for s in frame_states)
        max_player_elixir = max(s.player_elixir for s in frame_states)
        max_opponent_elixir = max(s.opponent_elixir_estimated for s in frame_states)

        # Count detections
        total_detections = sum(len(s.detections) for s in frame_states)

        return {
            "total_frames": len(frame_states),
            "duration_ms": frame_states[-1].timestamp_ms - frame_states[0].timestamp_ms,
            "phases_detected": list(phases_seen),
            "max_player_elixir": max_player_elixir,
            "max_opponent_elixir_estimated": round(max_opponent_elixir, 2),
            "total_detections": total_detections,
            "opponent_cards_played": self._opponent_tracker.total_cards_played if self._opponent_tracker else 0,
            "avg_opponent_card_cost": round(
                self._opponent_tracker.average_elixir_per_card if self._opponent_tracker else 0,
                2
            ),
        }


def main():
    """CLI entry point for video analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze Clash Royale game recordings"
    )
    parser.add_argument(
        "video_path",
        help="Path to video file to analyze"
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=6,
        help="Process every Nth frame (default: 6)"
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save annotated frames as images"
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory (default: output)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    analyzer = VideoAnalyzer(
        frame_skip=args.frame_skip,
        save_annotated_frames=args.save_frames,
        output_dir=args.output_dir,
        verbose=not args.quiet
    )

    analyzer.analyze_video(args.video_path)


if __name__ == "__main__":
    main()
