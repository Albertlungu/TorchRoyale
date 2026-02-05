"""
Placement validity masks for different card types.

Each card type has different placement rules:
- Troops: Can only be placed on your side (not river, except bridges)
- Buildings: Can only be placed in building zones on your side
- Spells: Can be placed anywhere
- The Log: Acts like a troop (directional spell)
"""

from typing import Dict, List, Set, Tuple
from enum import Enum
import numpy as np

from .coordinate_mapper import CoordinateMapper

# Import elixir costs from constants module for convenience
from ..constants.game_constants import ELIXIR_COSTS, get_elixir_cost


class CardType(Enum):
    """Categories of cards with different placement rules."""
    TROOP = "troop"
    BUILDING = "building"
    SPELL = "spell"
    LOG_SPELL = "log_spell"  # The Log - directional, acts like troop placement


# Card name to card type mapping for the Hog 2.6 deck
CARD_TYPES: Dict[str, CardType] = {
    # Troops
    "hog-rider": CardType.TROOP,
    "musketeer": CardType.TROOP,
    "ice-golem": CardType.TROOP,
    "ice-spirit": CardType.TROOP,
    "skeletons": CardType.TROOP,

    # Buildings
    "cannon": CardType.BUILDING,

    # Spells
    "fireball": CardType.SPELL,

    # Special - The Log
    "the-log": CardType.LOG_SPELL,
}

# On-field versions (for detection, not placement)
ON_FIELD_CARDS: Set[str] = {
    "hog-rider-on-field",
    "musketeer_on_field",
    "ice-golem-on-field",
    "ice-spirit-evolution",
    "skeletons-evolution",
    "cannon_on_field",
    "cannon_evolution_on_field",
    "fireball-on-field",
    "the-log-on-field",
}


class PlacementValidator:
    """
    Validates and provides placement masks for different card types.

    Generates numpy arrays (32x18) where:
    - 1 = valid placement
    - 0 = invalid placement
    """

    def __init__(self, mapper: CoordinateMapper):
        """
        Initialize the placement validator.

        Args:
            mapper: CoordinateMapper instance for grid dimensions
        """
        self.mapper = mapper
        self.width = mapper.GRID_WIDTH
        self.height = mapper.GRID_HEIGHT

        # Pre-compute masks for each card type
        self._masks: Dict[CardType, np.ndarray] = {}
        self._build_masks()

    def _build_masks(self):
        """Pre-compute placement masks for each card type."""
        self._masks[CardType.TROOP] = self._build_troop_mask()
        self._masks[CardType.BUILDING] = self._build_building_mask()
        self._masks[CardType.SPELL] = self._build_spell_mask()
        self._masks[CardType.LOG_SPELL] = self._build_troop_mask()  # Same as troop

    def _build_troop_mask(self) -> np.ndarray:
        """
        Build placement mask for troops.

        Troops can be placed:
        - On your side (rows 17-31)
        - On bridges (specific tiles in river rows)

        Troops cannot be placed:
        - On enemy side (rows 0-14)
        - In the river (rows 15-16, except bridges)
        """
        mask = np.zeros((self.height, self.width), dtype=np.float32)

        for y in range(self.height):
            for x in range(self.width):
                # Your side
                if self.mapper.is_on_your_side(y):
                    mask[y, x] = 1.0
                # Bridges
                elif self.mapper.is_bridge(x, y):
                    mask[y, x] = 1.0

        return mask

    def _build_building_mask(self) -> np.ndarray:
        """
        Build placement mask for buildings.

        Buildings have more restricted placement:
        - Only on your side (rows 17-31)
        - Not on bridges
        - Not in certain edge columns (depends on game version)

        For simplicity, we'll allow rows 20-31 and columns 2-15.
        Fine-tune these values based on actual game testing.
        """
        mask = np.zeros((self.height, self.width), dtype=np.float32)

        # Building zone (approximate - adjust based on testing)
        building_row_min = 20
        building_row_max = 31
        building_col_min = 2
        building_col_max = 15

        for y in range(building_row_min, building_row_max + 1):
            for x in range(building_col_min, building_col_max + 1):
                mask[y, x] = 1.0

        return mask

    def _build_spell_mask(self) -> np.ndarray:
        """
        Build placement mask for spells.

        Spells can be placed anywhere on the arena.
        """
        return np.ones((self.height, self.width), dtype=np.float32)

    def get_mask(self, card_name: str) -> np.ndarray:
        """
        Get the placement validity mask for a card.

        Args:
            card_name: Name of the card (e.g., "hog-rider", "cannon")

        Returns:
            32x18 numpy array where 1=valid, 0=invalid
        """
        card_type = CARD_TYPES.get(card_name, CardType.TROOP)
        return self._masks[card_type].copy()

    def get_mask_by_type(self, card_type: CardType) -> np.ndarray:
        """Get the placement mask for a card type."""
        return self._masks[card_type].copy()

    def is_valid_placement(self, card_name: str, tile_x: int, tile_y: int) -> bool:
        """
        Check if a specific tile is valid for placing a card.

        Args:
            card_name: Name of the card
            tile_x: Tile x coordinate (0-17)
            tile_y: Tile y coordinate (0-31)

        Returns:
            True if placement is valid, False otherwise
        """
        if not (0 <= tile_x < self.width and 0 <= tile_y < self.height):
            return False

        mask = self.get_mask(card_name)
        return mask[tile_y, tile_x] > 0.5

    def get_valid_tiles(self, card_name: str) -> List[Tuple[int, int]]:
        """
        Get list of all valid tile coordinates for a card.

        Args:
            card_name: Name of the card

        Returns:
            List of (tile_x, tile_y) tuples
        """
        mask = self.get_mask(card_name)
        valid = np.where(mask > 0.5)
        return list(zip(valid[1], valid[0]))  # (x, y) format

    def apply_mask_to_heatmap(self, heatmap: np.ndarray, card_name: str) -> np.ndarray:
        """
        Apply validity mask to a probability heatmap.

        Sets invalid positions to 0 and renormalizes.

        Args:
            heatmap: 32x18 probability heatmap
            card_name: Name of the card

        Returns:
            Masked and renormalized heatmap
        """
        mask = self.get_mask(card_name)
        masked = heatmap * mask

        # Renormalize
        total = masked.sum()
        if total > 0:
            masked = masked / total

        return masked

    def visualize_mask(self, card_name: str) -> str:
        """
        Create a text visualization of the placement mask.

        Args:
            card_name: Name of the card

        Returns:
            String visualization of the mask
        """
        mask = self.get_mask(card_name)
        card_type = CARD_TYPES.get(card_name, CardType.TROOP)

        lines = [f"Placement mask for {card_name} ({card_type.value}):"]
        lines.append("-" * (self.width + 4))

        for y in range(self.height):
            row = f"{y:2d} |"
            for x in range(self.width):
                if mask[y, x] > 0.5:
                    if self.mapper.is_bridge(x, y):
                        row += "B"
                    else:
                        row += "#"
                else:
                    if self.mapper.is_river(x, y):
                        row += "~"
                    elif self.mapper.is_on_enemy_side(y):
                        row += "E"
                    else:
                        row += "."
            row += "|"
            lines.append(row)

        lines.append("-" * (self.width + 4))
        lines.append("    " + "".join(str(x % 10) for x in range(self.width)))
        lines.append("Legend: # = valid, . = invalid (your side), E = enemy, ~ = river, B = bridge")

        return "\n".join(lines)
