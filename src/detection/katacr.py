"""
KataCR battlefield detector.

Wraps the two YOLOv8 models from KataCR (v0.7.13) to detect troops,
buildings, spells, and towers on the arena. Returns FrameDetections
with on_field populated; hand detection is handled separately.

Model weights are stored at data/models/katacr/ and downloaded on first
use from Google Drive if absent.

KataCR output format per detection:
  [x1, y1, x2, y2, conf, cls_idx, bel]
  bel: 0 = player (friend), 1 = opponent (enemy)
  Coordinates are in the part2 crop (576×896), not the full frame.

Screen partitioning for portrait-in-landscape (ratio ≈ 2.16):
  part2 crop: x=2.1%, y=7.3%, w=96%, h=70% of the game content strip
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.detection.result import Detection, FrameDetections
from src.grid.coordinate_mapper import CoordinateMapper

_WEIGHTS_DIR = Path(__file__).parents[2] / "data/models/katacr"
_GDRIVE_IDS = {
    "detector1_v0.7.13.pt": "1DMD-EYXa1qn8lN4JjPQ7UIuOMwaqS5w_",
    "detector2_v0.7.13.pt": "1yEq-6liLhs_pUfipJM1E-tMj6l4FSbxD",
}

# part2 crop params per aspect-ratio bucket (x, y, w, h as fractions of game strip)
_PART2_PARAMS = {
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
    return _SUFFIX_RE.sub("", name.lower()).strip()


def _download_weights() -> None:
    """Download KataCR weights from Google Drive if not present."""
    try:
        import gdown  # type: ignore
    except ImportError:
        raise RuntimeError(
            "pip install gdown   # required to auto-download KataCR weights"
        )
    _WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    for fname, gid in _GDRIVE_IDS.items():
        dest = _WEIGHTS_DIR / fname
        if not dest.exists():
            print(f"[KataCR] Downloading {fname} ...")
            gdown.download(id=gid, output=str(dest), quiet=False)


class KataCRDetector:
    """
    Battlefield detector using KataCR's dual YOLOv8 combo.

    Call calibrate(frame) once per video to set up the coordinate mapper,
    then call detect(frame) for each frame.
    """

    CONF_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.6

    def __init__(self, device: str = "auto"):
        self._models = None
        self._mapper: Optional[CoordinateMapper] = None
        self._game_strip: Optional[tuple] = None  # (x_left, x_right, y_top, y_bot)
        self._part2_params: Optional[tuple] = None

        if device == "auto":
            import torch
            if torch.backends.mps.is_available():
                self._device = "mps"
            elif torch.cuda.is_available():
                self._device = "cuda"
            else:
                self._device = "cpu"
        else:
            self._device = device

    def _load(self) -> None:
        if self._models is not None:
            return
        w1 = _WEIGHTS_DIR / "detector1_v0.7.13.pt"
        w2 = _WEIGHTS_DIR / "detector2_v0.7.13.pt"
        if not w1.exists() or not w2.exists():
            _download_weights()
        from ultralytics import YOLO  # type: ignore
        self._models = [
            YOLO(str(w1)).to(self._device),
            YOLO(str(w2)).to(self._device),
        ]
        print(f"[KataCR] Models loaded on {self._device}")

    def calibrate(self, frame: np.ndarray) -> None:
        """
        Detect game content bounds and set up coordinate mapper.
        Call once with a mid-game frame (not loading screen).
        """
        h, w = frame.shape[:2]
        gray = np.mean(frame, axis=2) if frame.ndim == 3 else frame
        cols = np.where(np.mean(gray, axis=0) > 30)[0]
        rows = np.where(np.mean(gray, axis=1) > 30)[0]

        if cols.size > 0 and (cols.max() - cols.min()) < w * 0.80:
            x_left, x_right = int(cols.min()), int(cols.max())
            y_top, y_bot = int(rows.min()), int(rows.max())
        else:
            x_left, y_top = 0, 0
            x_right, y_bot = w, h

        self._game_strip = (x_left, x_right, y_top, y_bot)
        game_h = max(1, y_bot - y_top)
        game_w = max(1, x_right - x_left)
        ratio = game_h / game_w

        # Pick the closest part2 params
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

    def detect(self, frame: np.ndarray) -> FrameDetections:
        """
        Run battlefield detection on one full video frame.
        Returns FrameDetections with on_field populated.
        """
        self._load()
        if self._mapper is None or self._game_strip is None:
            self.calibrate(frame)

        part2, crop_x, crop_y = self._crop_part2(frame)
        raw = self._run_combo(part2)
        detections = self._parse(raw, part2.shape, crop_x, crop_y, frame.shape)
        return FrameDetections(on_field=detections)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _crop_part2(self, frame: np.ndarray):
        """Crop the arena region from the full frame."""
        import cv2
        x_left, x_right, y_top, y_bot = self._game_strip
        gw = x_right - x_left
        gh = y_bot - y_top
        x, y, w, h = self._part2_params
        x1 = x_left + int(gw * x)
        y1 = y_top + int(gh * y)
        x2 = x_left + int(gw * (x + w))
        y2 = y_top + int(gh * (y + h))
        crop = frame[y1:y2, x1:x2]
        resized = cv2.resize(crop, (_PART2_TARGET_W, _PART2_TARGET_H))
        return resized, x1, y1

    def _run_combo(self, part2: np.ndarray) -> np.ndarray:
        """Run both YOLO models, merge with NMS, return Nx7 array."""
        import torch
        import torchvision

        preds = []
        for model in self._models:
            results = model.predict(
                part2,
                conf=self.CONF_THRESHOLD,
                verbose=False,
                device=self._device,
            )
            if results and results[0].boxes is not None:
                boxes = results[0].boxes.data.clone()  # (N, 6): xyxy, conf, cls
                # Append a dummy bel=0 column so we have 7 cols
                bel = torch.zeros(len(boxes), 1, device=boxes.device)
                boxes = torch.cat([boxes, bel], dim=1)
                preds.append(boxes)

        if not preds:
            return np.zeros((0, 7))

        preds = torch.cat(preds, dim=0)
        keep = torchvision.ops.nms(preds[:, :4], preds[:, 4], self.IOU_THRESHOLD)
        return preds[keep].cpu().numpy()

    def _parse(
        self,
        raw: np.ndarray,
        part2_shape: tuple,
        crop_x: int,
        crop_y: int,
        frame_shape: tuple,
    ) -> List[Detection]:
        """Convert raw YOLO output to Detection objects in full-frame space."""
        detections: List[Detection] = []
        if len(raw) == 0:
            return detections

        ph, pw = part2_shape[:2]
        scale_x = (self._game_strip[1] - self._game_strip[0]) * self._part2_params[2] / pw
        scale_y = (self._game_strip[3] - self._game_strip[2]) * self._part2_params[3] / ph

        for row in raw:
            bx1, by1, bx2, by2, conf, cls_idx, bel = (
                float(row[0]), float(row[1]), float(row[2]),
                float(row[3]), float(row[4]), int(row[5]), int(row[6]),
            )
            # Convert part2 pixels → full frame pixels
            fx1 = crop_x + bx1 * scale_x
            fy1 = crop_y + by1 * scale_y
            fx2 = crop_x + bx2 * scale_x
            fy2 = crop_y + by2 * scale_y

            # Centre pixel → tile
            cx = int((fx1 + fx2) / 2)
            cy = int((fy1 + fy2) / 2)
            # Adjust to game-strip-relative coords for tile mapping
            x_left, _, y_top, _ = self._game_strip
            tile_col, tile_row = self._mapper.pixel_to_tile(cx - x_left, cy - y_top)

            # Class name from model
            names = self._models[0].names
            raw_name = names.get(cls_idx, f"unit_{cls_idx}")
            card_name = _base(raw_name)

            detections.append(Detection(
                class_name=card_name,
                tile_x=tile_col,
                tile_y=tile_row,
                is_opponent=bool(bel),
                is_on_field=True,
                confidence=round(conf, 3),
                bbox_px=(int(fx1), int(fy1), int(fx2), int(fy2)),
            ))

        return detections
