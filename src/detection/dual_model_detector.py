"""
Dual-model on-field detector using two separate YOLOv8 models.

Replaces KataCRDetector with a dedicated dual-model approach:
  - Cicadas model: detects PLAYER'S cards (Hog 2.6 deck)
  - Vision Bot model: detects OPPONENT'S cards (all cards in the game)

Model assignment determines ownership directly (is_opponent flag).
No motion-based ownership correction is applied.

Public API:
  DualModelDetector -- load both models once, calibrate from a frame, detect per frame
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from src.detection.opponent_onnx_detector import OnnxOpponentDetector
from src.detection.result import Detection, FrameDetections
from src.grid.coordinate_mapper import CoordinateMapper
from src.ocr.regions import Region, UIRegions

# Default weights paths
_DEFAULT_CICADAS = "data/models/onfield/cicadas_best.pt"
_DEFAULT_VISIONBOT = "data/models/units_M_480x352.onnx"

# part2 crop params per aspect-ratio bucket (x, y, w, h as fractions of game strip)
# Mirrors the values from katacr.py for pixel-perfect compatibility
_PART2_PARAMS: dict[tuple[float, float], tuple[float, float, float, float]] = {
    (2.13, 2.14): (0.026, 0.048, 0.960, 0.710),
    (2.16, 2.18): (0.021, 0.073, 0.960, 0.700),
    (2.22, 2.24): (0.020, 0.070, 0.960, 0.690),
}
_PART2_TARGET_W, _PART2_TARGET_H = 576, 896
_ONNX_ARENA_TOP_MARGIN = 35
_ONNX_ARENA_BOTTOM_MARGIN = 40
_KATACR_SRC_CANDIDATES = [
    Path(__file__).parents[2].parent / "KataCR",
    Path("/tmp/katacr_repo"),
]


def _ensure_katacr_src_on_path() -> None:
    """Add KataCR source tree to sys.path so custom checkpoints can unpickle."""
    for candidate in _KATACR_SRC_CANDIDATES:
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            break


class DualModelDetector:
    """
    Battlefield detector using two separate YOLOv8 models.

    Cicadas model detects player cards, Vision Bot model detects opponent cards.
    Ownership is determined by model assignment, not by tile position.

    Call calibrate(frame) once per video to set up the coordinate mapper,
    then call detect(frame) for each frame.
    """

    def __init__(
        self,
        cicadas_weights: str = _DEFAULT_CICADAS,
        visionbot_weights: str = _DEFAULT_VISIONBOT,
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.5,
        device: str = "auto",
    ) -> None:
        """
        Args:
            cicadas_weights: path to Cicadas YOLOv8 weights file.
            visionbot_weights: path to Vision Bot YOLOv8 weights file.
            conf_threshold: confidence threshold for detections.
            iou_threshold: IoU threshold for cross-model NMS.
            device: PyTorch device string ("auto", "cpu", "cuda", "mps").
                    "auto" selects MPS > CUDA > CPU in that order.
        """
        self.conf_threshold: float = conf_threshold
        self.iou_threshold: float = iou_threshold

        # Resolve device
        if device == "auto":
            import torch

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"
            elif torch.cuda.is_available():
                self._device = "cuda"
            else:
                self._device = "cpu"
        else:
            self._device = device

        # Verify weights exist
        cicadas_path = Path(cicadas_weights)
        visionbot_path = Path(visionbot_weights)
        if not cicadas_path.exists():
            raise FileNotFoundError(
                f"Cicadas weights not found: {cicadas_path}\n"
                f"Run: python scripts/train_cicadas.py"
            )
        if not visionbot_path.exists():
            raise FileNotFoundError(
                f"Vision Bot weights not found: {visionbot_path}\n"
                f"Run: python scripts/train_visionbot.py"
            )

        # Load models
        from ultralytics import YOLO

        _ensure_katacr_src_on_path()
        self._cicadas = YOLO(str(cicadas_path)).to(self._device)
        self._visionbot_uses_onnx = visionbot_path.suffix.lower() == ".onnx"
        if self._visionbot_uses_onnx:
            side_model_path = visionbot_path.with_name("side.onnx")
            if not side_model_path.exists():
                raise FileNotFoundError(
                    f"Side classifier not found: {side_model_path}"
                )
            self._visionbot = OnnxOpponentDetector(
                str(visionbot_path),
                str(side_model_path),
            )
        else:
            self._visionbot = YOLO(str(visionbot_path)).to(self._device)
        print(f"[DualModel] Cicadas model loaded on {self._device}")
        backend = "ONNX" if self._visionbot_uses_onnx else "YOLO"
        print(f"[DualModel] Vision Bot {backend} model loaded on {self._device}")

        # State (initialized in calibrate)
        self._mapper: Optional[CoordinateMapper] = None
        self._game_strip: Optional[tuple[int, int, int, int]] = None
        self._part2_params: Optional[tuple[float, float, float, float]] = None
        self._ui_regions: Optional[UIRegions] = None

    def calibrate(self, frame: np.ndarray) -> None:
        """
        Detect game content bounds and set up the coordinate mapper.

        Call once with a mid-game frame (not a loading screen) before
        running detect().

        Args:
            frame: BGR image array from OpenCV.
        """
        frame_h, frame_w = frame.shape[:2]
        gray = np.mean(frame, axis=2) if frame.ndim == 3 else frame
        cols = np.where(np.mean(gray, axis=0) > 30)[0]
        rows = np.where(np.mean(gray, axis=1) > 30)[0]

        if cols.size > 0 and (cols.max() - cols.min()) < frame_w * 0.80:
            x_left, x_right = int(cols.min()), int(cols.max())
            y_top, y_bot = int(rows.min()), int(rows.max())
        else:
            x_left, y_top = 0, 0
            x_right, y_bot = frame_w, frame_h

        self._game_strip = (x_left, x_right, y_top, y_bot)
        game_h = max(1, y_bot - y_top)
        game_w = max(1, x_right - x_left)
        ratio = game_h / game_w

        # Pick the closest part2 params bucket
        best_params = list(_PART2_PARAMS.values())[0]
        best_dist = float("inf")
        for (lo, hi), params in _PART2_PARAMS.items():
            mid = (lo + hi) / 2
            if abs(ratio - mid) < best_dist:
                best_dist = abs(ratio - mid)
                best_params = params
        self._part2_params = best_params

        # Mapper calibrated from the game strip dimensions
        self._mapper = CoordinateMapper()
        self._mapper.calibrate_from_image(game_w, game_h)
        self._ui_regions = UIRegions(frame_w, frame_h)

    def detect(self, frame: np.ndarray) -> FrameDetections:
        """
        Run battlefield detection on one full video frame.

        Args:
            frame: BGR image array from OpenCV.

        Returns:
            FrameDetections with on_field populated.
        """
        if self._mapper is None or self._game_strip is None:
            self.calibrate(frame)

        x_left, x_right, y_top, y_bot = self._game_strip  # type: ignore[misc]
        part2, crop_x, crop_y = self._crop_part2(frame)

        # Run both models
        cicadas_dets = self._run_model(self._cicadas, part2, crop_x, crop_y, is_opponent=False)
        if self._visionbot_uses_onnx:
            visionbot_dets = self._run_onnx_model(frame, x_left, y_top)
        else:
            visionbot_dets = self._run_model(
                self._visionbot,
                part2,
                crop_x,
                crop_y,
                is_opponent=True,
            )

        # Merge and apply cross-model NMS
        all_dets = cicadas_dets + visionbot_dets
        merged = self._apply_cross_model_nms(all_dets)

        return FrameDetections(on_field=merged)

    def _crop_part2(self, frame: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """
        Crop and resize the arena region from the full frame.

        Uses the exact same logic as KataCR's _crop_part2 method.

        Returns:
            Tuple of (resized crop, crop x-offset in frame, crop y-offset in frame).
        """
        x_left, x_right, y_top, y_bot = self._game_strip  # type: ignore[misc]
        game_w = x_right - x_left
        game_h = y_bot - y_top
        cx, cy, cw, ch = self._part2_params  # type: ignore[misc]
        x1 = x_left + int(game_w * cx)
        y1 = y_top + int(game_h * cy)
        x2 = x_left + int(game_w * (cx + cw))
        y2 = y_top + int(game_h * (cy + ch))
        crop = frame[y1:y2, x1:x2]
        resized = cv2.resize(crop, (_PART2_TARGET_W, _PART2_TARGET_H))
        return resized, x1, y1

    def _run_model(
        self,
        model: "YOLO",
        part2: np.ndarray,
        crop_x: int,
        crop_y: int,
        is_opponent: bool,
    ) -> list[Detection]:
        """
        Run a single YOLO model on the arena crop and convert to Detection objects.

        Args:
            model: loaded YOLO model.
            part2: resized arena crop (576x896 BGR).
            crop_x: x-offset of the crop within the full frame.
            crop_y: y-offset of the crop within the full frame.
            is_opponent: ownership flag for all detections from this model.

        Returns:
            List of Detection objects with tile coordinates.
        """
        results = model.predict(
            part2,
            conf=self.conf_threshold,
            verbose=False,
            device=self._device,
        )

        if not results or results[0].boxes is None:
            return []

        boxes = results[0].boxes.data.cpu().numpy()  # (N, 6): xyxy, conf, cls
        return self._convert_to_detections(
            boxes, model.names, crop_x, crop_y, is_opponent, part2.shape
        )

    def _run_onnx_model(
        self,
        frame: np.ndarray,
        game_x: int,
        game_y: int,
    ) -> list[Detection]:
        raw_detections = self._visionbot.detect(frame)
        return self._convert_onnx_to_detections(
            raw_detections,
            game_x,
            game_y,
        )

    def _convert_to_detections(
        self,
        raw: np.ndarray,
        names: dict[int, str],
        crop_x: int,
        crop_y: int,
        is_opponent: bool,
        part2_shape: tuple[int, ...],
    ) -> list[Detection]:
        """
        Convert raw YOLO output to Detection objects with tile coordinates.

        Args:
            raw: Nx6 array from YOLO model.
            names: class name mapping from the model.
            crop_x: x-offset of the crop within the full frame.
            crop_y: y-offset of the crop within the full frame.
            is_opponent: ownership flag for these detections.
            part2_shape: shape of the resized arena crop.

        Returns:
            List of Detection objects.
        """
        detections: list[Detection] = []
        if len(raw) == 0:
            return detections

        x_left, x_right, y_top, y_bot = self._game_strip  # type: ignore[misc]
        crop_w, crop_h = self._part2_params[2], self._part2_params[3]  # type: ignore[index]
        part2_h, part2_w = part2_shape[:2]
        scale_x = (x_right - x_left) * crop_w / part2_w
        scale_y = (y_bot - y_top) * crop_h / part2_h

        for row in raw:
            bx1, by1, bx2, by2, conf, cls_idx = (
                float(row[0]),
                float(row[1]),
                float(row[2]),
                float(row[3]),
                float(row[4]),
                int(row[5]),
            )

            # Convert part2 pixels to full-frame pixels
            fx1 = crop_x + bx1 * scale_x
            fx2 = crop_x + bx2 * scale_x
            fy1 = crop_y + by1 * scale_y
            fy2 = crop_y + by2 * scale_y

            # Centre pixel to tile (game-strip-relative coords)
            cx = int((fx1 + fx2) / 2) - x_left
            cy = int((fy1 + fy2) / 2) - y_top
            tile_col, tile_row = self._mapper.pixel_to_tile(cx, cy)  # type: ignore[union-attr]

            card_name = names.get(cls_idx, f"unit_{cls_idx}").lower()

            detections.append(
                Detection(
                    class_name=card_name,
                    tile_x=tile_col,
                    tile_y=tile_row,
                    is_opponent=is_opponent,
                    is_on_field=True,
                    confidence=round(conf, 3),
                    bbox_px=(int(fx1), int(fy1), int(fx2), int(fy2)),
                )
            )

        return detections

    def _convert_onnx_to_detections(
        self,
        raw_detections,
        game_x: int,
        game_y: int,
    ) -> list[Detection]:
        detections: list[Detection] = []
        if not raw_detections:
            return detections

        bounds = self._mapper.bounds  # type: ignore[union-attr]
        y_min = bounds.y_min + _ONNX_ARENA_TOP_MARGIN
        y_max = bounds.y_max - _ONNX_ARENA_BOTTOM_MARGIN
        for raw_det in raw_detections:
            bx1, by1, bx2, by2 = raw_det.bbox_px
            fx1 = game_x + bx1
            fx2 = game_x + bx2
            fy1 = game_y + by1
            fy2 = game_y + by2

            center_x = int((fx1 + fx2) / 2)
            center_y = int((fy1 + fy2) / 2)
            cx = center_x - game_x
            cy = center_y - game_y
            if not (
                bounds.x_min <= cx <= bounds.x_max
                and y_min <= cy <= y_max
            ):
                continue
            if self._is_in_tower_region(center_x, center_y):
                continue
            tile_col, tile_row = self._mapper.pixel_to_tile(cx, cy)  # type: ignore[union-attr]

            detections.append(
                Detection(
                    class_name=raw_det.class_name,
                    tile_x=tile_col,
                    tile_y=tile_row,
                    is_opponent=True,
                    is_on_field=True,
                    confidence=round(raw_det.confidence, 3),
                    bbox_px=(int(fx1), int(fy1), int(fx2), int(fy2)),
                )
            )
        return detections

    def _is_in_tower_region(self, center_x: int, center_y: int) -> bool:
        if self._ui_regions is None:
            return False
        tower_regions = (
            self._ui_regions.player_tower_left,
            self._ui_regions.player_tower_king,
            self._ui_regions.player_tower_right,
            self._ui_regions.opponent_tower_left,
            self._ui_regions.opponent_tower_king,
            self._ui_regions.opponent_tower_right,
        )
        return any(self._contains_point(region, center_x, center_y) for region in tower_regions)

    @staticmethod
    def _contains_point(region: Region, x: int, y: int) -> bool:
        return region.x_min <= x <= region.x_max and region.y_min <= y <= region.y_max

    def _apply_cross_model_nms(
        self, detections: list[Detection]
    ) -> list[Detection]:
        """
        Apply NMS only where both models detect overlapping boxes.

        If two detections from different models have IoU > threshold,
        keep the one with higher confidence.

        Args:
            detections: merged list of detections from both models.

        Returns:
            List with duplicate cross-model detections removed.
        """
        if len(detections) <= 1:
            return detections

        # Group by rough location to find potential cross-model duplicates
        keep: list[bool] = [True] * len(detections)

        for i in range(len(detections)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(detections)):
                if not keep[j]:
                    continue

                det_i = detections[i]
                det_j = detections[j]

                # Only apply NMS for detections at overlapping pixel locations
                iou = self._bbox_iou(det_i.bbox_px, det_j.bbox_px)
                if iou > self.iou_threshold:
                    # Keep higher confidence detection
                    if det_i.confidence >= det_j.confidence:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break  # det_i is removed, continue with det_j

        return [detections[i] for i in range(len(detections)) if keep[i]]

    @staticmethod
    def _bbox_iou(
        box1: tuple[int, int, int, int],
        box2: tuple[int, int, int, int],
    ) -> float:
        """
        Calculate IoU between two bounding boxes.

        Args:
            box1: (x1, y1, x2, y2) for first box.
            box2: (x1, y1, x2, y2) for second box.

        Returns:
            IoU value between 0 and 1.
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        inter_w = max(0, xi2 - xi1)
        inter_h = max(0, yi2 - yi1)
        inter_area = inter_w * inter_h

        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0.0
        return inter_area / union_area
