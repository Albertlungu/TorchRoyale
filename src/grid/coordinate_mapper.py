"""
Maps between pixel coordinates and the 18×32 tile grid.

Handles portrait games embedded in landscape frames (e.g. 1920×1080 with
black bars) by detecting game content bounds from an actual video frame.

Public API:
  ArenaBounds      -- dataclass holding pixel boundaries of the playable arena
  CoordinateMapper -- call calibrate_from_* once, then pixel_to_tile / tile_to_pixel
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
        """Width of the arena region in pixels."""
        return self.x_max - self.x_min

    @property
    def arena_height(self) -> int:
        """Height of the arena region in pixels."""
        return self.y_max - self.y_min


class CoordinateMapper:
    """
    Converts between pixel (x, y) and tile (col, row) coordinates.

    Reference calibration from 1170×2532 (portrait phone):
      arena origin: (27.6, 326.7)
      tile size:    62×50 px
    """

    _REF_W: int = 1170
    _REF_H: int = 2532
    _REF_ORIGIN_X: float = 27.6
    _REF_ORIGIN_Y: float = 326.7
    _REF_TILE_W: float = 62.0
    _REF_TILE_H: float = 50.0

    def __init__(self) -> None:
        self.bounds: Optional[ArenaBounds] = None
        self.tile_w: float = 0.0
        self.tile_h: float = 0.0

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate_from_image(self, width: int, height: int) -> None:
        """
        Scale the reference calibration to the given image dimensions.

        Args:
            width:  image width in pixels.
            height: image height in pixels.
        """
        scale_x = width / self._REF_W
        scale_y = height / self._REF_H
        x_min = self._REF_ORIGIN_X * scale_x
        y_min = self._REF_ORIGIN_Y * scale_y
        tile_w = self._REF_TILE_W * scale_x
        tile_h = self._REF_TILE_H * scale_y
        self.bounds = ArenaBounds(
            x_min=int(round(x_min)),
            y_min=int(round(y_min)),
            x_max=int(round(x_min + GRID_COLS * tile_w)),
            y_max=int(round(y_min + GRID_ROWS * tile_h)),
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

        Args:
            frame:        BGR or greyscale image array.
            black_thresh: pixel brightness below which a column/row is considered black.
        """
        frame_h, frame_w = frame.shape[:2]
        gray = np.mean(frame, axis=2) if frame.ndim == 3 else frame
        cols = np.where(np.mean(gray, axis=0) > black_thresh)[0]
        rows = np.where(np.mean(gray, axis=1) > black_thresh)[0]

        if cols.size == 0 or rows.size == 0:
            self.calibrate_from_image(frame_w, frame_h)
            return

        left, right = int(cols.min()), int(cols.max())
        top, bot = int(rows.min()), int(rows.max())
        game_w = max(1, right - left)

        if game_w >= frame_w * 0.80:
            # No significant black bars — treat as native portrait
            self.calibrate_from_image(frame_w, frame_h)
            return

        game_h = max(1, bot - top)
        scale_x = game_w / self._REF_W
        scale_y = game_h / self._REF_H
        x_min = left + self._REF_ORIGIN_X * scale_x
        y_min = top + self._REF_ORIGIN_Y * scale_y
        tile_w = self._REF_TILE_W * scale_x
        tile_h = self._REF_TILE_H * scale_y
        self.bounds = ArenaBounds(
            x_min=int(round(x_min)),
            y_min=int(round(y_min)),
            x_max=int(round(x_min + GRID_COLS * tile_w)),
            y_max=int(round(y_min + GRID_ROWS * tile_h)),
            image_width=frame_w,
            image_height=frame_h,
        )
        self._recalc_tile_size()

    def _recalc_tile_size(self) -> None:
        """Recompute tile_w and tile_h from current bounds."""
        if self.bounds is None:
            return
        self.tile_w = self.bounds.arena_width / GRID_COLS
        self.tile_h = self.bounds.arena_height / GRID_ROWS

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def pixel_to_tile(self, pixel_x: int, pixel_y: int) -> Tuple[int, int]:
        """
        Convert game-strip-relative pixel coordinates to tile (col, row).

        Args:
            pixel_x: x position relative to the game strip left edge.
            pixel_y: y position relative to the game strip top edge.

        Returns:
            (col, row) clamped to valid grid bounds.

        Raises:
            AssertionError: if calibrate_from_* has not been called.
        """
        assert self.bounds is not None, "Call calibrate_from_* first"
        col = int((pixel_x - self.bounds.x_min) / self.tile_w)
        row = int((pixel_y - self.bounds.y_min) / self.tile_h)
        col = max(0, min(GRID_COLS - 1, col))
        row = max(0, min(GRID_ROWS - 1, row))
        return col, row

    def tile_to_pixel(self, col: int, row: int, center: bool = True) -> Tuple[int, int]:
        """
        Convert tile (col, row) to pixel coordinates.

        Args:
            col:    tile column (0-based).
            row:    tile row (0-based).
            center: if True, return the tile centre; otherwise the top-left corner.

        Returns:
            (px, py) in full-frame pixel space.

        Raises:
            AssertionError: if calibrate_from_* has not been called.
        """
        assert self.bounds is not None
        offset = 0.5 if center else 0.0
        pixel_x = int((col + offset) * self.tile_w + self.bounds.x_min)
        pixel_y = int((row + offset) * self.tile_h + self.bounds.y_min)
        return pixel_x, pixel_y

    def tile_bounds_pixels(self, col: int, row: int) -> Tuple[int, int, int, int]:
        """
        Return the pixel bounding box of a tile.

        Args:
            col: tile column (0-based).
            row: tile row (0-based).

        Returns:
            (x1, y1, x2, y2) pixel rectangle for the tile.

        Raises:
            AssertionError: if calibrate_from_* has not been called.
        """
        assert self.bounds is not None
        x1 = int(col * self.tile_w + self.bounds.x_min)
        y1 = int(row * self.tile_h + self.bounds.y_min)
        x2 = int((col + 1) * self.tile_w + self.bounds.x_min)
        y2 = int((row + 1) * self.tile_h + self.bounds.y_min)
        return x1, y1, x2, y2
