"""
KataCR battlefield detector.

Wraps the two YOLOv8 models from KataCR (v0.7.13) to detect troops,
buildings, spells, and towers on the arena. Returns FrameDetections
with on_field populated; hand detection is handled separately.

Model weights are stored at data/models/katacr/ and downloaded on first
use from Google Drive if absent.

KataCR output format per detection:
  [x1, y1, x2, y2, conf, cls_idx]
  Coordinates are in the part2 crop (576×896), not the full frame.
  is_opponent is derived from tile_y: units spawning above PLAYER_SIDE_MIN_ROW
  belong to the opponent and retain that label even after crossing the river.

Screen partitioning for portrait-in-landscape (ratio ~2.16):
  part2 crop: x=2.1%, y=7.3%, w=96%, h=70% of the game content strip

Public API:
  KataCRDetector -- load weights once, calibrate from a frame, detect per frame
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# KataCR weights were saved with a custom YOLOv8 subclass. The katacr source
# tree must be on sys.path so torch can unpickle the checkpoint. We look for
# it next to this repo (cloned as "KataCR") and fall back to /tmp/katacr_repo.
_KATACR_SRC_CANDIDATES = [
    Path(__file__).parents[2].parent / "KataCR",
    Path("/tmp/katacr_repo"),
]
for _candidate in _KATACR_SRC_CANDIDATES:
    if _candidate.is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))
        break

from src.detection.result import Detection, FrameDetections
from src.grid.coordinate_mapper import CoordinateMapper

_WEIGHTS_DIR = Path(__file__).parents[2] / "data/models/katacr"

# part2 crop params per aspect-ratio bucket (x, y, w, h as fractions of game strip)
_PART2_PARAMS: Dict[Tuple[float, float], Tuple[float, float, float, float]] = {
    (2.13, 2.14): (0.026, 0.048, 0.960, 0.710),
    (2.16, 2.18): (0.021, 0.073, 0.960, 0.700),
    (2.22, 2.24): (0.020, 0.070, 0.960, 0.690),
}
_PART2_TARGET_W, _PART2_TARGET_H = 576, 896

_SUFFIX_RE = re.compile(
    r"-(in-hand|next|on-field|on_field|evolution-symbol|evolution|ability|bar|level)$",
    re.IGNORECASE,
)


def _base(name: str) -> str:
    """Strip known suffixes from a KataCR class name to get the base card name."""
    return _SUFFIX_RE.sub("", name.lower()).strip()


class KataCRDetector:
    """
    Battlefield detector using KataCR's dual YOLOv8 combo.

    Call calibrate(frame) once per video to set up the coordinate mapper,
    then call detect(frame) for each frame.
    """

    CONF_THRESHOLD: float = 0.5
    IOU_THRESHOLD: float = 0.6

    def __init__(self, device: str = "auto") -> None:
        """
        Args:
            device: PyTorch device string ("auto", "cpu", "cuda", "mps").
                    "auto" selects MPS > CUDA > CPU in that order.
        """
        self._models: Optional[List] = None
        self._mapper: Optional[CoordinateMapper] = None
        self._game_strip: Optional[Tuple[int, int, int, int]] = None
        self._part2_params: Optional[Tuple[float, float, float, float]] = None

        if device == "auto":
            import torch  # pylint: disable=import-outside-toplevel
            if torch.backends.mps.is_available():
                self._device = "mps"
            elif torch.cuda.is_available():
                self._device = "cuda"
            else:
                self._device = "cpu"
        else:
            self._device = device

    def _load(self) -> None:
        """Lazily load both YOLO model weights from disk."""
        if self._models is not None:
            return
        weight1 = _WEIGHTS_DIR / "detector1_v0.7.13.pt"
        weight2 = _WEIGHTS_DIR / "detector2_v0.7.13.pt"
        for weight in (weight1, weight2):
            if not weight.exists():
                raise FileNotFoundError(
                    f"KataCR weight not found: {weight}\n"
                    f"Place detector1_v0.7.13.pt and detector2_v0.7.13.pt in {_WEIGHTS_DIR}"
                )
        from ultralytics import YOLO  # type: ignore  # pylint: disable=import-outside-toplevel
        self._models = [
            YOLO(str(weight1)).to(self._device),
            YOLO(str(weight2)).to(self._device),
        ]
        print(f"[KataCR] Models loaded on {self._device}")

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
        best_params: Tuple[float, float, float, float] = list(_PART2_PARAMS.values())[0]
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

    def detect(self, frame: np.ndarray) -> FrameDetections:
        """
        Run battlefield detection on one full video frame.

        Args:
            frame: BGR image array from OpenCV.

        Returns:
            FrameDetections with on_field populated.
        """
        self._load()
        if self._mapper is None or self._game_strip is None:
            self.calibrate(frame)

        part2, crop_x, crop_y = self._crop_part2(frame)
        raw = self._run_combo(part2)
        detections = self._parse(raw, part2.shape, crop_x, crop_y)
        return FrameDetections(on_field=detections)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _crop_part2(self, frame: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """
        Crop and resize the arena region from the full frame.

        Returns:
            Tuple of (resized crop, crop x-offset in frame, crop y-offset in frame).
        """
        import cv2  # pylint: disable=import-outside-toplevel
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

    def _run_combo(self, part2: np.ndarray) -> np.ndarray:
        """
        Run both YOLO models on the arena crop and merge via NMS.

        Args:
            part2: resized arena crop (576×896 BGR).

        Returns:
            Nx6 float array: [x1, y1, x2, y2, conf, cls_idx].
        """
        import torch  # pylint: disable=import-outside-toplevel
        import torchvision  # pylint: disable=import-outside-toplevel

        preds = []
        for model in self._models:  # type: ignore[union-attr]
            results = model.predict(
                part2,
                conf=self.CONF_THRESHOLD,
                verbose=False,
                device=self._device,
            )
            if results and results[0].boxes is not None:
                boxes = results[0].boxes.data.clone()  # (N, 6): xyxy, conf, cls
                preds.append(boxes)

        if not preds:
            return np.zeros((0, 6))

        all_preds = torch.cat(preds, dim=0)
        keep = torchvision.ops.nms(all_preds[:, :4], all_preds[:, 4], self.IOU_THRESHOLD)
        return all_preds[keep].cpu().numpy()

    def _parse(
        self,
        raw: np.ndarray,
        part2_shape: Tuple[int, ...],
        crop_x: int,
        crop_y: int,
    ) -> List[Detection]:
        """
        Convert raw YOLO output rows to Detection objects in full-frame space.

        Args:
            raw: Nx7 array from _run_combo.
            part2_shape: shape of the resized arena crop (H, W, ...).
            crop_x: x-offset of the crop within the full frame.
            crop_y: y-offset of the crop within the full frame.

        Returns:
            List of Detection objects with tile coordinates.
        """
        detections: List[Detection] = []
        if len(raw) == 0:
            return detections

        x_left, x_right, y_top, y_bot = self._game_strip  # type: ignore[misc]
        crop_w, crop_h = self._part2_params[2], self._part2_params[3]  # type: ignore[index]
        part2_h, part2_w = part2_shape[:2]
        scale_x = (x_right - x_left) * crop_w / part2_w
        scale_y = (y_bot - y_top) * crop_h / part2_h

        from src.constants.game import PLAYER_SIDE_MIN_ROW  # pylint: disable=import-outside-toplevel

        for row in raw:
            bx1, by1, bx2, by2, conf, cls_idx = (
                float(row[0]), float(row[1]), float(row[2]),
                float(row[3]), float(row[4]), int(row[5]),
            )
            # Convert part2 pixels → full frame pixels
            fx1 = crop_x + bx1 * scale_x
            fy1 = crop_y + by1 * scale_y
            fx2 = crop_x + bx2 * scale_x
            fy2 = crop_y + by2 * scale_y

            # Centre pixel → tile (use game-strip-relative coords)
            cx = int((fx1 + fx2) / 2)
            cy = int((fy1 + fy2) / 2)
            tile_col, tile_row = self._mapper.pixel_to_tile(  # type: ignore[union-attr]
                cx - x_left, cy - y_top
            )

            # Units that spawn above the river belong to the opponent; they may
            # cross into the player's half later but their origin side is fixed.
            is_opponent = tile_row < PLAYER_SIDE_MIN_ROW

            names: Dict[int, str] = self._models[0].names  # type: ignore[index]
            raw_name = names.get(cls_idx, f"unit_{cls_idx}")
            card_name = _base(raw_name)

            detections.append(Detection(
                class_name=card_name,
                tile_x=tile_col,
                tile_y=tile_row,
                is_opponent=is_opponent,
                is_on_field=True,
                confidence=round(conf, 3),
                bbox_px=(int(fx1), int(fy1), int(fx2), int(fy2)),
            ))

        return detections
