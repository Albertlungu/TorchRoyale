"""
UI region definitions for Clash Royale screen elements.

Defines pixel regions (using ratios for resolution independence) for:
- Elixir display
- Timer
- Multiplier icons (x2/x3)
- Card hand slots
- Tower health bar regions

Default calibrated for 1080x2400 mobile resolution.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class UIRegion:
    """Defines a rectangular UI region in pixel coordinates.

    Attributes:
        x_min (int): Left pixel coordinate.
        y_min (int): Top pixel coordinate.
        x_max (int): Right pixel coordinate.
        y_max (int): Bottom pixel coordinate.
    """

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """
        Return as (x_min, y_min, x_max, y_max) tuple.

        Returns:
            (Tuple[int, int, int, int]) Tuple of pixel coordinates.
        """
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    def crop_from_image(self, image: np.ndarray) -> np.ndarray:
        """
        Extract this region from an image.

        Args:
            image (np.ndarray): numpy array (height, width, channels).

        Returns:
            (np.ndarray) Cropped region as numpy array.
        """
        return image[self.y_min : self.y_max, self.x_min : self.x_max]


class UIRegions:
    """
    UI element regions for Clash Royale.

    Uses ratios for resolution independence.
    Default calibrated for standard 1080x2400 mobile resolution.

    Screen layout (top to bottom):
    - 0-12%: Top UI (timer, player info, x2/x3 icons)
    - 12-80%: Arena (18x32 tile grid)
    - 80-100%: Bottom UI (card hand, elixir bar)

    Attributes:
        width (int): Screen width in pixels.
        height (int): Screen height in pixels.
        timer (UIRegion): Timer display region.
        multiplier_icon (UIRegion): Elixir multiplier icon region.
        elixir_bar (UIRegion): Elixir bar region.
        elixir_number (UIRegion): Elixir number display region.
        card_hand (UIRegion): Card hand region.
        card_slots (List[UIRegion]): Individual card slot regions.
        card_cost_regions (List[UIRegion]): Card cost display regions.
        next_card (UIRegion): Next card slot region.
        player_king_tower (UIRegion): Player king tower region.
        player_left_tower (UIRegion): Player left princess tower region.
        player_right_tower (UIRegion): Player right princess tower region.
        opponent_king_tower (UIRegion): Opponent king tower region.
        opponent_left_tower (UIRegion): Opponent left princess tower region.
        opponent_right_tower (UIRegion): Opponent right princess tower region.
    """

    def __init__(self, screen_width: int = 1080, screen_height: int = 2400) -> None:
        """
        Initialize UI regions for given screen dimensions.

        Args:
            screen_width (int): Screen width in pixels.
            screen_height (int): Screen height in pixels.

        Returns:
            None
        """
        self.width = screen_width
        self.height = screen_height
        self._build_regions()

    def _build_regions(self) -> None:
        """
        Build all UI regions based on screen dimensions.

        Returns:
            None
        """
        w, h = self.width, self.height

        # ============================================
        # TOP UI REGION (0-12% of screen height)
        # ============================================

        # Timer region (top center-right, shows MM:SS).
        # Use different ratios for portrait vs landscape videos.
        if w > h:
            # Landscape (desktop exports): place timer nearer the top-right
            # but slightly lower and narrower than the initial guess.
            self.timer = UIRegion(
                x_min=int(w * 0.954),
                y_min=int(h * 0.068),
                x_max=int(w * 0.96),
                y_max=int(h * 0.095),
            )
        else:
            # Portrait (mobile captures)
            self.timer = UIRegion(
                x_min=int(w * 0.87),
                y_min=int(h * 0.075),
                x_max=int(w * 0.97),
                y_max=int(h * 0.1),
            )

        # Multiplier icon region (x2/x3 indicator, appears top center)
        self.multiplier_icon = UIRegion(
            x_min=int(w * 0.88),
            y_min=int(h * 0.125),
            x_max=int(w * 0.96),
            y_max=int(h * 0.155),
        )

        def align_region_to_image(
            self,
            image,
            region_attr: str,
            black_thresh: int = 10,
            padding: int = 6,
            width_ratio: Optional[float] = None,
            shift_down: int = 0,
        ):
            """
            Generic aligner: if the named region crop is black, scan the image for
            the rightmost non-black column and place the region box relative to it.

            Args:
                image: Full frame as numpy array
                region_attr: Attribute name on this UIRegions instance (e.g. 'elixir_number')
                black_thresh: Brightness threshold to consider a crop black
                padding: pixels to pad to the right of detected column
                width_ratio: Fraction of screen width to use for width. If None,
                             use the current region width.
                shift_down: Pixels to move the bbox downward (positive = lower)

            Returns:
                (rightmost_column_index, (new_x_min,new_y_min,new_x_max,new_y_max)) or None
            """
            try:
                import numpy as _np
            except Exception:
                return None

            if not hasattr(self, region_attr):
                return None

            region: UIRegion = getattr(self, region_attr)
            x1, y1, x2, y2 = region.to_tuple()
            h, w = image.shape[:2]

            if x1 >= x2 or y1 >= y2 or x1 >= w or y1 >= h:
                return None

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                return None

            gray = _np.mean(crop, axis=2) if crop.ndim == 3 else crop
            if _np.mean(gray) > black_thresh:
                return None

            full_gray = _np.mean(image, axis=2) if image.ndim == 3 else image
            cols_nonblack = _np.where(_np.mean(full_gray, axis=0) > black_thresh)[0]
            if cols_nonblack.size == 0:
                return None

            rightmost = int(cols_nonblack.max())

            if width_ratio is None:
                desired_w = max(8, x2 - x1)
            else:
                desired_w = max(8, int(self.width * width_ratio))

            new_x_max = min(w, rightmost + padding)
            new_x_min = max(0, new_x_max - desired_w)

            desired_h = max(4, y2 - y1)
            new_y_min = min(max(0, y1 + shift_down), max(0, h - desired_h))
            new_y_max = new_y_min + desired_h

            # Update the region in-place
            region.x_min = int(new_x_min)
            region.x_max = int(new_x_max)
            region.y_min = int(new_y_min)
            region.y_max = int(new_y_max)

            return (
                rightmost,
                (int(new_x_min), int(new_y_min), int(new_x_max), int(new_y_max)),
            )

        # ============================================
        # BOTTOM UI REGION (80-100% of screen height)
        # ============================================

        # Elixir bar region (full purple bar)
        self.elixir_bar = UIRegion(
            x_min=int(w * 0.08),
            y_min=int(h * 0.835),
            x_max=int(w * 0.92),
            y_max=int(h * 0.855),
        )

        # Elixir number display (shows current elixir count 0-10)
        # Located at the left end of the elixir bar
        self.elixir_number = UIRegion(
            x_min=int(w * 0.27),
            y_min=int(h * 0.940),
            x_max=int(w * 0.32),
            y_max=int(h * 0.965),
        )

        # Card hand region (contains 4 cards)
        self.card_hand = UIRegion(
            x_min=int(w * 0.10),
            y_min=int(h * 0.86),
            x_max=int(w * 0.90),
            y_max=int(h * 0.98),
        )

        # Individual card slots (4 cards in hand)
        card_width = (0.90 - 0.10) / 4
        self.card_slots: List[UIRegion] = []
        for i in range(4):
            self.card_slots.append(
                UIRegion(
                    x_min=int(w * (0.10 + i * card_width)),
                    y_min=int(h * 0.86),
                    x_max=int(w * (0.10 + (i + 1) * card_width)),
                    y_max=int(h * 0.98),
                )
            )

        # Card elixir cost regions (small number on each card)
        # Located at bottom-left of each card
        self.card_cost_regions: List[UIRegion] = []
        for i in range(4):
            self.card_cost_regions.append(
                UIRegion(
                    x_min=int(w * (0.10 + i * card_width)),
                    y_min=int(h * 0.94),
                    x_max=int(w * (0.10 + i * card_width + 0.05)),
                    y_max=int(h * 0.98),
                )
            )

        # Next card slot (smaller, to the right of hand)
        self.next_card = UIRegion(
            x_min=int(w * 0.91),
            y_min=int(h * 0.88),
            x_max=int(w * 0.99),
            y_max=int(h * 0.96),
        )

        # ============================================
        # TOWER REGIONS (for health bar detection)
        # ============================================

        # Player towers (bottom half of arena)
        self.player_king_tower = UIRegion(
            x_min=int(w * 0.45),
            y_min=int(h * 0.75),
            x_max=int(w * 0.55),
            y_max=int(h * 0.77),
        )

        self.player_left_tower = UIRegion(
            x_min=int(w * 0.08),
            y_min=int(h * 0.58),
            x_max=int(w * 0.28),
            y_max=int(h * 0.61),
        )

        self.player_right_tower = UIRegion(
            x_min=int(w * 0.72),
            y_min=int(h * 0.58),
            x_max=int(w * 0.92),
            y_max=int(h * 0.61),
        )

        # Opponent towers (top half of arena)
        self.opponent_king_tower = UIRegion(
            x_min=int(w * 0.45),
            y_min=int(h * 0.08),
            x_max=int(w * 0.55),
            y_max=int(h * 0.10),
        )

        self.opponent_left_tower = UIRegion(
            x_min=int(w * 0.08),
            y_min=int(h * 0.22),
            x_max=int(w * 0.28),
            y_max=int(h * 0.25),
        )

        self.opponent_right_tower = UIRegion(
            x_min=int(w * 0.72),
            y_min=int(h * 0.22),
            x_max=int(w * 0.92),
            y_max=int(h * 0.25),
        )

    def scale_to_resolution(self, new_width: int, new_height: int) -> "UIRegions":
        """
        Create new UIRegions scaled to a different resolution.

        Args:
            new_width: New screen width
            new_height: New screen height

        Returns:
            New UIRegions instance for the given resolution
        """
        return UIRegions(new_width, new_height)

    def get_all_tower_regions(self) -> dict:
        """
        Get all tower regions as a dictionary.

        Returns:
            Dict mapping tower names to UIRegion objects
        """
        return {
            "player_king": self.player_king_tower,
            "player_left": self.player_left_tower,
            "player_right": self.player_right_tower,
            "opponent_king": self.opponent_king_tower,
            "opponent_left": self.opponent_left_tower,
            "opponent_right": self.opponent_right_tower,
        }

    def __repr__(self) -> str:
        return f"UIRegions(width={self.width}, height={self.height})"

    def align_elixir_to_image(
        self, image: np.ndarray, black_thresh: int = 10, box_w: int = 40, box_h: int = 27,
        x_offset: int = 126, y_offset: int = 10
    ) -> Optional[Tuple[int, int, int, Tuple[int, int, int, int]]]:
        """
        Anchor the elixir number box using two independent signals:
          X: leftmost non-black column (game content left edge) + x_offset
          Y: row with the longest horizontal run of saturated pixels in the
             bottom 40% of the frame (the elixir bar), shifted up by y_offset

        This separates the two axes so each can be independently reliable:
        x_offset is consistent across all videos, and the bar row is found
        adaptively so different arena styles and recording heights all work.

        Args:
            image (np.ndarray): Full frame as numpy array (H, W, 3).
            black_thresh (int): Pixel intensity threshold (default 10).
            box_w (int): Width of the elixir number box in pixels (default 40).
            box_h (int): Height of the elixir number box in pixels (default 27).
            x_offset (int): Pixels to offset from leftmost non-black column (default 126).
            y_offset (int): Pixels to shift up from the bar row (default 10).

        Returns:
            (Optional[Tuple[int, int, int, Tuple[int, int, int, int]]]) Tuple of (leftmost_column, bar_row, best_run_len, (x1, y1, x2, y2)), or None if alignment fails.
        """
        try:
            import numpy as _np
            import cv2 as _cv2
        except Exception:
            return

        if image is None or image.size == 0:
            return

        cur = self.elixir_number
        x1, y1, x2, y2 = cur.to_tuple()
        h, w = image.shape[:2]

        # --- X anchor: leftmost non-black column ---
        cx1 = max(0, min(x1, w - 1))
        cy1 = max(0, min(y1, h - 1))
        cx2 = max(0, min(x2, w))
        cy2 = max(0, min(y2, h))
        crop = image[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return
        gray = _np.mean(crop, axis=2) if crop.ndim == 3 else crop
        if _np.mean(gray) > black_thresh:
            return  # static region not black, no adjustment needed

        full_gray = _np.mean(image, axis=2) if image.ndim == 3 else image
        cols_nonblack = _np.where(_np.mean(full_gray, axis=0) > black_thresh)[0]
        if cols_nonblack.size == 0:
            return
        leftmost = int(cols_nonblack.min())

        # --- Y anchor: find bar row by detecting the elixir bar colour ---
        # The bar is in the bottom 12% of the frame. Actual HSV of the
        # pink/purple fill: H=130-165, S>80 (confirmed by sampling all games).
        search_top = int(h * 0.88)
        roi = image[search_top:, :]
        roi_hsv = _cv2.cvtColor(roi, _cv2.COLOR_BGR2HSV)
        bar_mask = _cv2.inRange(
            roi_hsv,
            _np.array([128, 80, 40]),
            _np.array([165, 255, 255]),
        )

        best_row_rel = None
        best_run_len = 0
        for row_rel in range(bar_mask.shape[0]):
            row = bar_mask[row_rel]
            in_run, run_len = False, 0
            for px in row:
                if px > 0:
                    in_run, run_len = True, run_len + 1
                elif in_run:
                    if run_len > best_run_len:
                        best_run_len, best_row_rel = run_len, row_rel
                    in_run, run_len = False, 0
            if in_run and run_len > best_run_len:
                best_run_len, best_row_rel = run_len, row_rel

        if best_row_rel is None or best_run_len < 20:
            return
        bar_row = search_top + best_row_rel

        # --- Place box ---
        # The digit sits ~y_offset px above the bar row.
        new_x_min = max(0, leftmost + x_offset)
        new_x_max = min(w, new_x_min + box_w)
        new_y_max = min(h, bar_row - y_offset + box_h // 2)
        new_y_min = max(0, new_y_max - box_h)

        self.elixir_number = UIRegion(
            x_min=new_x_min, y_min=new_y_min, x_max=new_x_max, y_max=new_y_max
        )

        return (leftmost, bar_row, best_run_len,
                (new_x_min, new_y_min, new_x_max, new_y_max))

    def align_timer_to_image(
        self, image: np.ndarray, black_thresh: int = 10, padding: int = 6, width_ratio: float = 0.04
    ) -> Optional[Tuple[int, Tuple[int, int, int, int]]]:
        """
        Heuristic: if the timer ROI appears black, scan the image for the
        rightmost non-black column and place the timer box relative to it.

        Args:
            image (np.ndarray): Full frame as a numpy array (H, W, 3).
            black_thresh (int): Pixel intensity threshold (default 10).
            padding (int): Pixels to pad to the right of detected column (default 6).
            width_ratio (float): Fraction of screen width for timer box (default 0.04).

        Returns:
            (Optional[Tuple[int, Tuple[int, int, int, int]]]) Tuple of (rightmost_column, (x1, y1, x2, y2)), or None if alignment fails.
        """
        try:
            import numpy as _np
        except Exception:
            return

        if image is None or image.size == 0:
            return

        # Current timer box
        cur = self.timer
        x1, y1, x2, y2 = cur.to_tuple()

        # If crop is empty, nothing to do
        h, w = image.shape[:2]
        if x1 >= x2 or y1 >= y2 or x1 >= w or y1 >= h:
            return

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return

        # Convert to grayscale and check brightness
        gray = _np.mean(crop, axis=2) if crop.ndim == 3 else crop
        if _np.mean(gray) > black_thresh:
            # Not black, nothing to change
            return

        # Find rightmost non-black column in full image
        full_gray = _np.mean(image, axis=2) if image.ndim == 3 else image
        cols_nonblack = _np.where(_np.mean(full_gray, axis=0) > black_thresh)[0]
        if cols_nonblack.size == 0:
            return None

        rightmost = int(cols_nonblack.max())

        # Desired width in pixels
        desired_w = max(8, int(self.width * width_ratio))

        new_x_max = min(w, rightmost + padding)
        new_x_min = max(0, new_x_max - desired_w)

        # Keep original height but move bbox lower for better alignment
        desired_h = max(4, y2 - y1)
        shift_down = 20  # shift lower by 20 pixels (user request)
        new_y_min = min(max(0, y1 + shift_down), max(0, h - desired_h))
        new_y_max = new_y_min + desired_h

        self.timer = UIRegion(
            x_min=new_x_min, y_min=new_y_min, x_max=new_x_max, y_max=new_y_max
        )

        # Return debug info for callers to log if desired
        return (rightmost, (new_x_min, new_y_min, new_x_max, new_y_max))

    def align_multiplier_to_image(
        self, image: np.ndarray, black_thresh: int = 10, padding: int = -17,
        width_ratio: float = 0.0219, shift_down: int = 17
    ) -> Optional[Tuple[int, Tuple[int, int, int, int]]]:
        """
        Mirror of align_timer_to_image for the x2/x3 multiplier icon.

        The icon sits just below and slightly left of the timer, so we use
        the same rightmost non-black column anchor but shift the Y down.

        Args:
            image (np.ndarray): Full frame as a numpy array (H, W, 3).
            black_thresh (int): Pixel intensity threshold (default 10).
            padding (int): Pixels to pad to the right of detected column (default -17).
            width_ratio (float): Fraction of screen width for icon box (default 0.0219).
            shift_down (int): Pixels to shift down from timer position (default 17).

        Returns:
            (Optional[Tuple[int, Tuple[int, int, int, int]]]) Tuple of (rightmost_column, (x1, y1, x2, y2)), or None if alignment fails.
        """
        try:
            import numpy as _np
        except Exception:
            return

        if image is None or image.size == 0:
            return

        cur = self.multiplier_icon
        x1, y1, x2, y2 = cur.to_tuple()

        h, w = image.shape[:2]
        if x1 >= x2 or y1 >= y2 or x1 >= w or y1 >= h:
            return

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return

        gray = _np.mean(crop, axis=2) if crop.ndim == 3 else crop
        if _np.mean(gray) > black_thresh:
            return

        full_gray = _np.mean(image, axis=2) if image.ndim == 3 else image
        cols_nonblack = _np.where(_np.mean(full_gray, axis=0) > black_thresh)[0]
        if cols_nonblack.size == 0:
            return

        rightmost = int(cols_nonblack.max())
        desired_w = max(8, int(self.width * width_ratio))

        new_x_max = min(w, rightmost + padding)
        new_x_min = max(0, new_x_max - desired_w)

        desired_h = max(4, y2 - y1)
        new_y_min = min(max(0, y1 + shift_down), max(0, h - desired_h))
        new_y_max = new_y_min + desired_h

        self.multiplier_icon = UIRegion(
            x_min=new_x_min, y_min=new_y_min, x_max=new_x_max, y_max=new_y_max
        )

        return (rightmost, (new_x_min, new_y_min, new_x_max, new_y_max))
