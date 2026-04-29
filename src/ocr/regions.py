"""
OCR region alignment for portrait games embedded in landscape frames.

Each align_*() method:
 - Checks if the static crop is black (game content is off-screen).
 - If black, scans the frame to find the real position using anchors
   specific to each UI element's corner of the screen.
 - Updates the region in-place and returns alignment debug info or None.

Public API:
  Region    -- (x_min, y_min, x_max, y_max) pixel rectangle
  UIRegions -- per-resolution set of timer, elixir, and multiplier regions
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class Region:
    """Axis-aligned pixel rectangle within a frame."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Return (x_min, y_min, x_max, y_max) as a plain tuple."""
        return self.x_min, self.y_min, self.x_max, self.y_max

    @property
    def width(self) -> int:
        """Width of the region in pixels."""
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        """Height of the region in pixels."""
        return self.y_max - self.y_min


class UIRegions:
    """All OCR regions for a given frame resolution."""

    def __init__(self, frame_w: int, frame_h: int) -> None:
        """
        Args:
            frame_w: full frame width in pixels.
            frame_h: full frame height in pixels.
        """
        self.w: int = frame_w
        self.h: int = frame_h
        # Declare region attributes before _build() populates them
        self.timer: Region = Region(0, 0, 0, 0)
        self.elixir_number: Region = Region(0, 0, 0, 0)
        self.multiplier_icon: Region = Region(0, 0, 0, 0)
        self._build()

    def _build(self) -> None:
        """Compute initial region positions from stored frame dimensions."""
        frame_w, frame_h = self.w, self.h

        if frame_w > frame_h:
            # Landscape export (portrait game centred in wider frame)
            self.timer = Region(
                x_min=int(frame_w * 0.954),
                y_min=int(frame_h * 0.068),
                x_max=int(frame_w * 0.96),
                y_max=int(frame_h * 0.095),
            )
        else:
            self.timer = Region(
                x_min=int(frame_w * 0.87),
                y_min=int(frame_h * 0.075),
                x_max=int(frame_w * 0.97),
                y_max=int(frame_h * 0.1),
            )

        self.elixir_number = Region(
            x_min=int(frame_w * 0.27),
            y_min=int(frame_h * 0.940),
            x_max=int(frame_w * 0.32),
            y_max=int(frame_h * 0.965),
        )

        self.multiplier_icon = Region(
            x_min=int(frame_w * 0.88),
            y_min=int(frame_h * 0.125),
            x_max=int(frame_w * 0.96),
            y_max=int(frame_h * 0.155),
        )

    # ------------------------------------------------------------------
    # Alignment helpers
    # ------------------------------------------------------------------

    def _is_black(self, frame: np.ndarray, region: Region, thresh: int = 10) -> bool:
        """Return True if the mean brightness of the region crop is at or below thresh."""
        x1, y1, x2, y2 = region.to_tuple()
        frame_h, frame_w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_w, x2), min(frame_h, y2)
        if x1 >= x2 or y1 >= y2:
            return True
        crop = frame[y1:y2, x1:x2]
        gray = np.mean(crop, axis=2) if crop.ndim == 3 else crop
        return float(np.mean(gray)) <= thresh

    def _rightmost_nonblack(self, frame: np.ndarray, thresh: int = 10) -> Optional[int]:
        """Return the x-coordinate of the rightmost non-black column, or None."""
        gray = np.mean(frame, axis=2) if frame.ndim == 3 else frame
        cols = np.where(np.mean(gray, axis=0) > thresh)[0]
        return int(cols.max()) if cols.size > 0 else None

    def align_timer(
        self,
        frame: np.ndarray,
        black_thresh: int = 10,
        padding: int = -7,
        width_ratio: float = 0.023,
        shift_down: int = 17,
    ) -> Optional[Tuple[int, Tuple[int, int, int, int]]]:
        """
        Realign the timer region if the static crop is black.

        Args:
            frame:       full BGR frame.
            black_thresh: brightness below which the crop is considered absent.
            padding:     pixel offset from the rightmost non-black column.
            width_ratio: timer width as a fraction of frame width.
            shift_down:  pixels to shift the region down from the original y_min.

        Returns:
            (rightmost_col, new_region_tuple) if realigned, else None.
        """
        if not self._is_black(frame, self.timer, black_thresh):
            return None
        rightmost = self._rightmost_nonblack(frame, black_thresh)
        if rightmost is None:
            return None
        frame_h, frame_w = frame.shape[:2]
        desired_w = max(8, int(frame_w * width_ratio))
        desired_h = max(4, self.timer.height)
        new_x_max = min(frame_w, rightmost + padding)
        new_x_min = max(0, new_x_max - desired_w)
        new_y_min = min(max(0, self.timer.y_min + shift_down), max(0, frame_h - desired_h))
        new_y_max = new_y_min + desired_h
        self.timer = Region(new_x_min, new_y_min, new_x_max, new_y_max)
        return (rightmost, self.timer.to_tuple())

    def align_multiplier(
        self,
        frame: np.ndarray,
        black_thresh: int = 10,
        padding: int = -17,
        width_ratio: float = 0.0219,
        shift_down: int = 17,
    ) -> Optional[Tuple[int, Tuple[int, int, int, int]]]:
        """
        Realign the multiplier icon region if the static crop is black.

        Args:
            frame:       full BGR frame.
            black_thresh: brightness below which the crop is considered absent.
            padding:     pixel offset from the rightmost non-black column.
            width_ratio: icon width as a fraction of frame width.
            shift_down:  pixels to shift the region down from the original y_min.

        Returns:
            (rightmost_col, new_region_tuple) if realigned, else None.
        """
        if not self._is_black(frame, self.multiplier_icon, black_thresh):
            return None
        rightmost = self._rightmost_nonblack(frame, black_thresh)
        if rightmost is None:
            return None
        frame_h, frame_w = frame.shape[:2]
        desired_w = max(8, int(frame_w * width_ratio))
        desired_h = max(4, self.multiplier_icon.height)
        new_x_max = min(frame_w, rightmost + padding)
        new_x_min = max(0, new_x_max - desired_w)
        new_y_min = min(
            max(0, self.multiplier_icon.y_min + shift_down), max(0, frame_h - desired_h)
        )
        new_y_max = new_y_min + desired_h
        self.multiplier_icon = Region(new_x_min, new_y_min, new_x_max, new_y_max)
        return (rightmost, self.multiplier_icon.to_tuple())

    def align_elixir(
        self,
        frame: np.ndarray,
        black_thresh: int = 10,
        x_offset: int = 126,
        box_w: int = 40,
        box_h: int = 27,
        y_offset: int = 10,
    ) -> Optional[Tuple[int, int, int, Tuple[int, int, int, int]]]:
        """
        Realign the elixir number region by finding the elixir bar's hue.

        Uses the pink/purple hue of the elixir bar (H=128–165 in OpenCV HSV)
        as an anchor, then positions the OCR box just above the bar.

        Args:
            frame:       full BGR frame.
            black_thresh: brightness threshold for detecting absence.
            x_offset:    horizontal offset from the leftmost game content column.
            box_w:       width of the OCR crop box in pixels.
            box_h:       height of the OCR crop box in pixels.
            y_offset:    pixels above the detected bar row where the box ends.

        Returns:
            (leftmost_col, bar_row, run_length, new_region_tuple) if realigned, else None.
        """
        import cv2  # pylint: disable=import-outside-toplevel
        if not self._is_black(frame, self.elixir_number, black_thresh):
            return None
        frame_h, frame_w = frame.shape[:2]
        gray = np.mean(frame, axis=2) if frame.ndim == 3 else frame
        cols = np.where(np.mean(gray, axis=0) > black_thresh)[0]
        if cols.size == 0:
            return None
        leftmost = int(cols.min())

        # Find elixir bar row by hue (pink/purple fill, H=128-165 in OpenCV HSV)
        search_top = int(frame_h * 0.88)
        roi = frame[search_top:, :]
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        bar_mask = cv2.inRange(
            roi_hsv, np.array([128, 80, 40]), np.array([165, 255, 255])
        )

        best_row: Optional[int] = None
        best_run: int = 0
        for row_idx in range(bar_mask.shape[0]):
            row_pixels = bar_mask[row_idx]
            run: int = 0
            in_run: bool = False
            for px in row_pixels:
                if px > 0:
                    in_run, run = True, run + 1
                elif in_run:
                    if run > best_run:
                        best_run, best_row = run, row_idx
                    in_run, run = False, 0
            if in_run and run > best_run:
                best_run, best_row = run, row_idx

        if best_row is None or best_run < 20:
            return None

        bar_row = search_top + best_row
        new_x_min = max(0, leftmost + x_offset)
        new_x_max = min(frame_w, new_x_min + box_w)
        new_y_max = min(frame_h, bar_row - y_offset + box_h // 2)
        new_y_min = max(0, new_y_max - box_h)
        self.elixir_number = Region(new_x_min, new_y_min, new_x_max, new_y_max)
        return (leftmost, bar_row, best_run, self.elixir_number.to_tuple())
