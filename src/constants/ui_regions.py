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
from typing import List, Tuple


@dataclass
class UIRegion:
    """Defines a rectangular UI region in pixel coordinates."""

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
        """Return as (x_min, y_min, x_max, y_max) tuple."""
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    def crop_from_image(self, image):
        """
        Extract this region from an image.

        Args:
            image: numpy array (height, width, channels)

        Returns:
            Cropped region as numpy array
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
    """

    def __init__(self, screen_width: int = 1080, screen_height: int = 2400):
        """
        Initialize UI regions for given screen dimensions.

        Args:
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
        """
        self.width = screen_width
        self.height = screen_height
        self._build_regions()

    def _build_regions(self):
        """Build all UI regions based on screen dimensions."""
        w, h = self.width, self.height

        # ============================================
        # TOP UI REGION (0-12% of screen height)
        # ============================================

        # Timer region (top center-right, shows MM:SS)
        self.timer = UIRegion(
            x_min=int(w * 0.40),
            y_min=int(h * 0.015),
            x_max=int(w * 0.60),
            y_max=int(h * 0.045),
        )

        # Multiplier icon region (x2/x3 indicator, appears top center)
        self.multiplier_icon = UIRegion(
            x_min=int(w * 0.42),
            y_min=int(h * 0.045),
            x_max=int(w * 0.58),
            y_max=int(h * 0.08),
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
            x_min=int(w * 0.28),
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
            x_min=int(w * 0.35),
            y_min=int(h * 0.70),
            x_max=int(w * 0.65),
            y_max=int(h * 0.73),
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
            x_min=int(w * 0.35),
            y_min=int(h * 0.14),
            x_max=int(w * 0.65),
            y_max=int(h * 0.17),
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
