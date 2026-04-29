"""
VideoAnalyzer: processes a replay video frame by frame, producing a JSON
analysis file with game state per frame.

Detection pipeline:
  - KataCRDetector for battlefield (troops, buildings, spells, towers)
  - Roboflow torchroyale/4 for hand cards (has -in-hand labels)
  - HandTracker to stabilise hand state across frames
  - OCR for timer, elixir, multiplier icon
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from src.detection.hand_tracker import HandTracker
from src.detection.katacr import KataCRDetector
from src.ocr.detector import DigitDetector
from src.ocr.regions import UIRegions


class VideoAnalyzer:
    def __init__(
        self,
        output_dir: str = "output/analysis",
        frame_skip: int = 6,
        device: str = "auto",
        preload_ocr: bool = False,
        verbose: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.frame_skip = frame_skip
        self.verbose = verbose

        self._katacr = KataCRDetector(device=device)
        self._ocr = DigitDetector(preload=preload_ocr)
        self._hand_tracker = HandTracker()
        self._roboflow = None  # loaded lazily

    def _load_roboflow(self):
        if self._roboflow is not None:
            return
        try:
            from inference import get_model  # type: ignore
            self._roboflow = get_model("torchroyale/4")
        except Exception as e:
            if self.verbose:
                print(f"[Roboflow] Not available: {e} — hand detection via HandTracker only")
            self._roboflow = False

    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(video_path).stem
        out_path = self.output_dir / f"{stem}_analysis.json"

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total / fps if fps > 0 else 0

        if self.verbose:
            print(f"Analyzing: {video_path}")
            print(f"Resolution: {vid_w}x{vid_h}  FPS: {fps:.1f}  Duration: {duration:.1f}s")
            effective_fps = fps / self.frame_skip
            frames_to_process = total // self.frame_skip
            print(f"Effective FPS: {effective_fps:.1f}  Frames: ~{frames_to_process}")
            print("-" * 50)

        # Calibrate KataCR from a mid-game frame
        calib_frame_idx = max(0, int(total * 0.10))
        cap.set(cv2.CAP_PROP_POS_FRAMES, calib_frame_idx)
        ret, calib_frame = cap.read()
        if ret:
            self._katacr.calibrate(calib_frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        ui = UIRegions(vid_w, vid_h)
        self._load_roboflow()
        self._hand_tracker.reset()

        frames: List[Dict[str, Any]] = []
        frame_num = 0

        while True:
            for _ in range(self.frame_skip - 1):
                cap.grab()
            ret, frame = cap.read()
            if not ret:
                break

            ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            # OCR — rebuild regions each frame for alignment
            frame_ui = UIRegions(vid_w, vid_h)
            frame_ui.align_timer(frame)
            frame_ui.align_elixir(frame)
            frame_ui.align_multiplier(frame)

            timer_secs = self._ocr.detect_timer(frame, frame_ui.timer.to_tuple())
            elixir_r   = self._ocr.detect_elixir(frame, frame_ui.elixir_number.to_tuple())
            mult       = self._ocr.detect_multiplier(frame, frame_ui.multiplier_icon.to_tuple())
            player_elixir = elixir_r.value if elixir_r.detected else None

            # KataCR battlefield detection
            field_result = self._katacr.detect(frame)
            on_field_dets = [
                {
                    "class_name": d.class_name,
                    "tile_x": d.tile_x,
                    "tile_y": d.tile_y,
                    "is_opponent": d.is_opponent,
                    "is_on_field": True,
                    "confidence": d.confidence,
                }
                for d in field_result.on_field
            ]

            # Roboflow hand detection (if available)
            hand_dets: List[Dict] = []
            if self._roboflow:
                try:
                    rb_results = self._roboflow.infer(frame)
                    preds = rb_results[0].predictions if hasattr(rb_results[0], "predictions") else []
                    for pred in preds:
                        name = getattr(pred, "class_name", "")
                        if "-in-hand" in name.lower() or "-next" in name.lower():
                            hand_dets.append({
                                "class_name": name,
                                "tile_x": 0,
                                "tile_y": 31,
                                "is_opponent": False,
                                "is_on_field": False,
                                "confidence": float(getattr(pred, "confidence", 1.0)),
                            })
                except Exception:
                    pass

            all_dets = on_field_dets + hand_dets
            tracked_hand = self._hand_tracker.update(all_dets)

            frames.append({
                "timestamp_ms": ts_ms,
                "frame_number": frame_num,
                "game_time_remaining": timer_secs,
                "elixir_multiplier": mult,
                "game_phase": None,      # filled by patch_ocr_fields after analysis
                "player_elixir": player_elixir,
                "opponent_elixir_estimated": None,
                "detections": all_dets,
                "hand_cards": tracked_hand,
                "player_towers": {},
                "opponent_towers": {},
            })
            frame_num += 1

            if self.verbose and frame_num % 100 == 0:
                print(f"  {frame_num} frames processed ({ts_ms/1000:.1f}s)")

        cap.release()

        result = {
            "video_info": {
                "path": video_path,
                "width": vid_w,
                "height": vid_h,
                "fps": fps,
                "duration_seconds": round(duration, 2),
                "frame_skip": self.frame_skip,
                "total_frames_processed": len(frames),
            },
            "frames": frames,
        }

        self._write_json(out_path, result)
        if self.verbose:
            print(f"Saved: {out_path} ({len(frames)} frames)")
        return result

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        tmp = NamedTemporaryFile("w", delete=False, dir=str(path.parent), suffix=".tmp")
        try:
            json.dump(data, tmp, indent=2, ensure_ascii=False)
            tmp.flush()
            tmp.close()
            shutil.move(tmp.name, str(path))
        finally:
            try:
                Path(tmp.name).unlink()
            except Exception:
                pass
