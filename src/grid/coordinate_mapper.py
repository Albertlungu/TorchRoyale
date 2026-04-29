"""
Maps between pixel coordinates and the 18×32 tile grid.

Handles portrait games embedded in landscape frames (e.g. 1920×1080 with
black bars) by detecting game content bounds from an actual video frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from src.constants.game import GRID_COLS, GRID_ROWS


@dataclass
class ArenaBounds:
    """Pixel boundaries of the playable arena within a frame."""
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    image_width: int
    image_height: int

    @property
    def arena_width(self) -> int:
        return self.x_max - self.x_min

    @property
    def arena_height(self) -> int:
        return self.y_max - self.y_min


class CoordinateMapper:
    """
    Converts between pixel (x, y) and tile (col, row) coordinates.

    Reference calibration from 1170×2532 (portrait phone):
      arena origin: (27.6, 326.7)
      tile size:    62×50 px
    """

    _REF_W, _REF_H = 1170, 2532
    _REF_ORIGIN_X, _REF_ORIGIN_Y = 27.6, 326.7
    _REF_TILE_W, _REF_TILE_H = 62.0, 50.0

    def __init__(self):
        self.bounds: Optional[ArenaBounds] = None
        self.tile_w: float = 0.0
        self.tile_h: float = 0.0

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate_from_image(self, width: int, height: int) -> None:
        """Scale reference calibration to given image dimensions."""
        sx = width / self._REF_W
        sy = height / self._REF_H
        x_min = self._REF_ORIGIN_X * sx
        y_min = self._REF_ORIGIN_Y * sy
        tw = self._REF_TILE_W * sx
        th = self._REF_TILE_H * sy
        self.bounds = ArenaBounds(
            x_min=int(round(x_min)),
            y_min=int(round(y_min)),
            x_max=int(round(x_min + GRID_COLS * tw)),
            y_max=int(round(y_min + GRID_ROWS * th)),
            image_width=width,
            image_height=height,
        )
        self._recalc_tile_size()

    def calibrate_from_frame(self, frame: np.ndarray, black_thresh: int = 30) -> None:
        """
        Auto-detect game content bounds from a real video frame.

        For portrait games embedded in landscape frames (black bars on the
        sides), the column-average brightness identifies the game strip.
        Falls back to calibrate_from_image if no bars are detected.
        """
        h, w = frame.shape[:2]
        gray = np.mean(frame, axis=2) if frame.ndim == 3 else frame
        cols = np.where(np.mean(gray, axis=0) > black_thresh)[0]
        rows = np.where(np.mean(gray, axis=1) > black_thresh)[0]

        if cols.size == 0 or rows.size == 0:
            self.calibrate_from_image(w, h)
            return

        left, right = int(cols.min()), int(cols.max())
        top, bot = int(rows.min()), int(rows.max())
        game_w = max(1, right - left)

        if game_w >= w * 0.80:
            # No significant black bars — treat as native portrait
            self.calibrate_from_image(w, h)
            return

        game_h = max(1, bot - top)
        sx = game_w / self._REF_W
        sy = game_h / self._REF_H
        x_min = left + self._REF_ORIGIN_X * sx
        y_min = top + self._REF_ORIGIN_Y * sy
        tw = self._REF_TILE_W * sx
        th = self._REF_TILE_H * sy
        self.bounds = ArenaBounds(
            x_min=int(round(x_min)),
            y_min=int(round(y_min)),
            x_max=int(round(x_min + GRID_COLS * tw)),
            y_max=int(round(y_min + GRID_ROWS * th)),
            image_width=w,
            image_height=h,
        )
        self._recalc_tile_size()

    def _recalc_tile_size(self) -> None:
        if self.bounds is None:
            return
        self.tile_w = self.bounds.arena_width / GRID_COLS
        self.tile_h = self.bounds.arena_height / GRID_ROWS

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def pixel_to_tile(self, px: int, py: int) -> Tuple[int, int]:
        assert self.bounds is not None, "Call calibrate_from_* first"
        col = int((px - self.bounds.x_min) / self.tile_w)
        row = int((py - self.bounds.y_min) / self.tile_h)
        col = max(0, min(GRID_COLS - 1, col))
        row = max(0, min(GRID_ROWS - 1, row))
        return col, row

    def tile_to_pixel(self, col: int, row: int, center: bool = True) -> Tuple[int, int]:
        assert self.bounds is not None
        offset = 0.5 if center else 0.0
        px = int((col + offset) * self.tile_w + self.bounds.x_min)
        py = int((row + offset) * self.tile_h + self.bounds.y_min)
        return px, py

    def tile_bounds_pixels(self, col: int, row: int) -> Tuple[int, int, int, int]:
        assert self.bounds is not None
        x1 = int(col * self.tile_w + self.bounds.x_min)
        y1 = int(row * self.tile_h + self.bounds.y_min)
        x2 = int((col + 1) * self.tile_w + self.bounds.x_min)
        y2 = int((row + 1) * self.tile_h + self.bounds.y_min)
        return x1, y1, x2, y2
