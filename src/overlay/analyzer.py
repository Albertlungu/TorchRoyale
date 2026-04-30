"""
VideoAnalyzer: processes a replay video frame by frame, producing a JSON
analysis file with game state per frame.

Detection pipeline:
  - KataCRDetector for battlefield (troops, buildings, spells, towers)
  - HandClassifier for hand card identification (local YOLOv8 classifier)
  - HandTracker to stabilise hand state and track evo cycles across frames
  - OCR for timer, elixir, and multiplier icon

Post-processing rules applied per frame:
  - If hero musketeer is in hand, any on-field "musketeer" is relabelled as
    "hero musketeer" (KataCR cannot distinguish them).
  - If an evo card was just played (HandTracker.last_played_evo), newly
    appearing on-field units matching that card are labelled as evo.

Public API:
  VideoAnalyzer -- construct once, call analyze_video() per replay file
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional

import cv2

from src.detection.hand_classifier import HandClassifier
from src.detection.hand_tracker import HandTracker
from src.detection.dual_model_detector import DualModelDetector
from src.ocr.detector import DigitDetector
from src.ocr.regions import UIRegions
from src.types import DetectionDict, FrameDict, VideoInfoDict


class VideoAnalyzer:
    """
    Frame-by-frame video processor that produces a structured analysis JSON.

    Each processed frame yields a FrameDict containing OCR fields, on-field
    detections, tracked hand cards, and tower state placeholders.
    """

    def __init__(
        self,
        output_dir: str = "output/analysis",
        frame_skip: int = 6,
        device: str = "auto",
        preload_ocr: bool = False,
        verbose: bool = True,
        cicadas_weights: str = "data/models/onfield/cicadas_best.pt",
        visionbot_weights: str = "data/models/onfield/visionbot_best.pt",
    ) -> None:
        """
        Args:
            output_dir:  directory where <stem>_analysis.json files are written.
            frame_skip:  process every Nth frame (1 = every frame).
            device:      PyTorch device for detection models.
            preload_ocr: if True, initialise the EasyOCR reader at startup.
            verbose:     print progress to stdout.
            cicadas_weights: path to Cicadas model weights (player cards).
            visionbot_weights: path to Vision Bot model weights (opponent cards).
        """
        self.output_dir = Path(output_dir)
        self.frame_skip: int = frame_skip
        self.verbose: bool = verbose

        self._dual_detector = DualModelDetector(
            cicadas_weights=cicadas_weights,
            visionbot_weights=visionbot_weights,
            device=device,
        )
        self._ocr = DigitDetector(preload=preload_ocr)
        self._hand_tracker = HandTracker()
        self._hand_clf = HandClassifier()

    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """
        Process a replay video and write a per-frame analysis JSON.

        Args:
            video_path: path to the video file (any format OpenCV can read).

        Returns:
            Dict with keys "video_info" (VideoInfoDict) and "frames" (list of FrameDict).
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(video_path).stem
        out_path = self.output_dir / f"{stem}_analysis.json"

        cap = cv2.VideoCapture(video_path)
        fps: float = cap.get(cv2.CAP_PROP_FPS)
        total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_w: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration: float = total_frames / fps if fps > 0 else 0.0

        if self.verbose:
            print(f"Analyzing: {video_path}")
            print(f"Resolution: {vid_w}x{vid_h}  FPS: {fps:.1f}  Duration: {duration:.1f}s")
            effective_fps = fps / self.frame_skip
            frames_to_process = total_frames // self.frame_skip
            print(f"Effective FPS: {effective_fps:.1f}  Frames: ~{frames_to_process}")
            print("-" * 50)

        # Calibrate detector from a mid-game frame (10% into the video)
        calib_idx = max(0, int(total_frames * 0.10))
        cap.set(cv2.CAP_PROP_POS_FRAMES, calib_idx)
        ret, calib_frame = cap.read()
        if ret:
            self._dual_detector.calibrate(calib_frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self._hand_tracker.reset()

        frames: List[FrameDict] = []
        frame_num: int = 0

        while True:
            for _ in range(self.frame_skip - 1):
                cap.grab()
            ret, frame = cap.read()
            if not ret:
                break

            ts_ms: int = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            # Rebuild per-frame UI regions for alignment drift
            frame_ui = UIRegions(vid_w, vid_h)
            frame_ui.align_timer(frame)
            frame_ui.align_elixir(frame)
            frame_ui.align_multiplier(frame)

            timer_secs = self._ocr.detect_timer(frame, frame_ui.timer.to_tuple())
            elixir_result = self._ocr.detect_elixir(frame, frame_ui.elixir_number.to_tuple())
            mult = self._ocr.detect_multiplier(frame, frame_ui.multiplier_icon.to_tuple())
            player_elixir: Optional[int] = (
                elixir_result.value if elixir_result.detected else None
            )

            # Dual-model battlefield detection
            field_result = self._dual_detector.detect(frame)
            on_field_dets: List[DetectionDict] = [
                DetectionDict(
                    class_name=det.class_name,
                    tile_x=det.tile_x,
                    tile_y=det.tile_y,
                    is_opponent=det.is_opponent,
                    is_on_field=True,
                    confidence=det.confidence,
                )
                for det in field_result.on_field
            ]

            # Local hand classifier — identify the 4 hand slots
            x_left, x_right = (
                self._dual_detector._game_strip[:2]
                if self._dual_detector._game_strip
                else (None, None)
            )
            game_strip = (x_left, x_right) if x_left is not None else None
            hand_dets: List[DetectionDict] = []
            try:
                classified = self._hand_clf.classify(frame, game_strip=game_strip)
                for card_name in classified:
                    if card_name:
                        hand_dets.append(DetectionDict(
                            class_name=f"{card_name}-in-hand",
                            tile_x=0,
                            tile_y=31,
                            is_opponent=False,
                            is_on_field=False,
                            confidence=1.0,
                        ))
            except Exception:  # pylint: disable=broad-exception-caught
                pass

            all_dets: List[DetectionDict] = on_field_dets + hand_dets
            tracked_hand: List[str] = self._hand_tracker.update(all_dets, frame=frame, game_strip=game_strip)

            # Hero musketeer: KataCR cannot distinguish it from regular musketeer.
            # If it is in the hand, every on-field musketeer this match is hero musketeer.
            has_hero_musketeer = any("hero musketeer" in h for h in tracked_hand)
            last_played_evo = self._hand_tracker.last_played_evo

            final_field_dets: List[DetectionDict] = []
            for det in on_field_dets:
                name = det["class_name"]
                if not det.get("is_opponent"):
                    if name == "musketeer" and has_hero_musketeer:
                        name = "hero musketeer"
                    elif name in last_played_evo:
                        name = f"{name}-evolution"
                final_field_dets.append(DetectionDict(
                    class_name=name,
                    tile_x=det["tile_x"],
                    tile_y=det["tile_y"],
                    is_opponent=det["is_opponent"],
                    is_on_field=det["is_on_field"],
                    confidence=det["confidence"],
                ))
            all_dets = final_field_dets + hand_dets

            frames.append(FrameDict(
                timestamp_ms=ts_ms,
                frame_number=frame_num,
                game_time_remaining=timer_secs,
                elixir_multiplier=mult,
                game_phase=None,
                player_elixir=player_elixir,
                opponent_elixir_estimated=None,
                detections=all_dets,
                hand_cards=tracked_hand,
                player_towers={},
                opponent_towers={},
            ))
            frame_num += 1

            if self.verbose and frame_num % 100 == 0:
                print(f"  {frame_num} frames processed ({ts_ms/1000:.1f}s)")

        cap.release()

        video_info: VideoInfoDict = VideoInfoDict(
            path=video_path,
            width=vid_w,
            height=vid_h,
            fps=fps,
            duration_seconds=round(duration, 2),
            frame_skip=self.frame_skip,
            total_frames_processed=len(frames),
        )
        result: Dict[str, Any] = {
            "video_info": dict(video_info),
            "frames": [dict(fr) for fr in frames],
        }

        self._write_json(out_path, result)
        if self.verbose:
            print(f"Saved: {out_path} ({len(frames)} frames)")
        return result

    @staticmethod
    def _write_json(path: Path, data: Dict[str, Any]) -> None:
        """
        Atomically write a JSON file via a temporary file.

        Args:
            path: destination path.
            data: JSON-serialisable dict.
        """
        with NamedTemporaryFile(
            "w", delete=False, dir=str(path.parent), suffix=".tmp", encoding="utf-8"
        ) as tmp:
            json.dump(data, tmp, indent=2, ensure_ascii=False)
            tmp_name = tmp.name
        try:
            shutil.move(tmp_name, str(path))
        finally:
            try:
                Path(tmp_name).unlink()
            except Exception:  # pylint: disable=broad-exception-caught
                pass
