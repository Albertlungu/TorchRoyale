"""
OCR region alignment for portrait games embedded in landscape frames.

Each align_*() method:
 - Checks if the static crop is black (game content is off-screen).
 - If black, scans the frame to find the real position using anchors
   specific to each UI element's corner of the screen.
 - Updates the region in-place; returns alignment debug info or None.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class Region:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return self.x_min, self.y_min, self.x_max, self.y_max

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min


class UIRegions:
    """All OCR regions for a given frame resolution."""

    def __init__(self, w: int, h: int):
        self.w = w
        self.h = h
        self._build()

    def _build(self) -> None:
        w, h = self.w, self.h

        if w > h:
            # Landscape export (portrait game centred in wider frame)
            self.timer = Region(
                x_min=int(w * 0.954),
                y_min=int(h * 0.068),
                x_max=int(w * 0.96),
                y_max=int(h * 0.095),
            )
        else:
            self.timer = Region(
                x_min=int(w * 0.87),
                y_min=int(h * 0.075),
                x_max=int(w * 0.97),
                y_max=int(h * 0.1),
            )

        self.elixir_number = Region(
            x_min=int(w * 0.27),
            y_min=int(h * 0.940),
            x_max=int(w * 0.32),
            y_max=int(h * 0.965),
        )

        self.multiplier_icon = Region(
            x_min=int(w * 0.88),
            y_min=int(h * 0.125),
            x_max=int(w * 0.96),
            y_max=int(h * 0.155),
        )

    # ------------------------------------------------------------------
    # Alignment helpers
    # ------------------------------------------------------------------

    def _is_black(self, frame: np.ndarray, region: Region, thresh: int = 10) -> bool:
        x1, y1, x2, y2 = region.to_tuple()
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x1 >= x2 or y1 >= y2:
            return True
        crop = frame[y1:y2, x1:x2]
        gray = np.mean(crop, axis=2) if crop.ndim == 3 else crop
        return float(np.mean(gray)) <= thresh

    def _rightmost_nonblack(self, frame: np.ndarray, thresh: int = 10) -> Optional[int]:
        gray = np.mean(frame, axis=2) if frame.ndim == 3 else frame
        cols = np.where(np.mean(gray, axis=0) > thresh)[0]
        return int(cols.max()) if cols.size > 0 else None

    def align_timer(self, frame: np.ndarray, black_thresh: int = 10,
                    padding: int = -7, width_ratio: float = 0.023,
                    shift_down: int = 17) -> Optional[tuple]:
        if not self._is_black(frame, self.timer, black_thresh):
            return None
        rightmost = self._rightmost_nonblack(frame, black_thresh)
        if rightmost is None:
            return None
        h, w = frame.shape[:2]
        desired_w = max(8, int(w * width_ratio))
        desired_h = max(4, self.timer.height)
        new_x_max = min(w, rightmost + padding)
        new_x_min = max(0, new_x_max - desired_w)
        new_y_min = min(max(0, self.timer.y_min + shift_down), max(0, h - desired_h))
        new_y_max = new_y_min + desired_h
        self.timer = Region(new_x_min, new_y_min, new_x_max, new_y_max)
        return (rightmost, self.timer.to_tuple())

    def align_multiplier(self, frame: np.ndarray, black_thresh: int = 10,
                         padding: int = -17, width_ratio: float = 0.0219,
                         shift_down: int = 17) -> Optional[tuple]:
        if not self._is_black(frame, self.multiplier_icon, black_thresh):
            return None
        rightmost = self._rightmost_nonblack(frame, black_thresh)
        if rightmost is None:
            return None
        h, w = frame.shape[:2]
        desired_w = max(8, int(w * width_ratio))
        desired_h = max(4, self.multiplier_icon.height)
        new_x_max = min(w, rightmost + padding)
        new_x_min = max(0, new_x_max - desired_w)
        new_y_min = min(max(0, self.multiplier_icon.y_min + shift_down), max(0, h - desired_h))
        new_y_max = new_y_min + desired_h
        self.multiplier_icon = Region(new_x_min, new_y_min, new_x_max, new_y_max)
        return (rightmost, self.multiplier_icon.to_tuple())

    def align_elixir(self, frame: np.ndarray, black_thresh: int = 10,
                     x_offset: int = 126, box_w: int = 40, box_h: int = 27,
                     y_offset: int = 10) -> Optional[tuple]:
        import cv2
        if not self._is_black(frame, self.elixir_number, black_thresh):
            return None
        h, w = frame.shape[:2]
        gray = np.mean(frame, axis=2) if frame.ndim == 3 else frame
        cols = np.where(np.mean(gray, axis=0) > black_thresh)[0]
        if cols.size == 0:
            return None
        leftmost = int(cols.min())

        # Find elixir bar row by hue (pink/purple fill, H=128-165 OpenCV)
        search_top = int(h * 0.88)
        roi = frame[search_top:, :]
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        bar_mask = cv2.inRange(roi_hsv, np.array([128, 80, 40]), np.array([165, 255, 255]))

        best_row, best_run = None, 0
        for r in range(bar_mask.shape[0]):
            row = bar_mask[r]
            run, in_run = 0, False
            for px in row:
                if px > 0:
                    in_run, run = True, run + 1
                elif in_run:
                    if run > best_run:
                        best_run, best_row = run, r
                    in_run, run = False, 0
            if in_run and run > best_run:
                best_run, best_row = run, r

        if best_row is None or best_run < 20:
            return None

        bar_row = search_top + best_row
        new_x_min = max(0, leftmost + x_offset)
        new_x_max = min(w, new_x_min + box_w)
        new_y_max = min(h, bar_row - y_offset + box_h // 2)
        new_y_min = max(0, new_y_max - box_h)
        self.elixir_number = Region(new_x_min, new_y_min, new_x_max, new_y_max)
        return (leftmost, bar_row, best_run, self.elixir_number.to_tuple())
