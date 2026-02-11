"""
Coordinate mapping between pixel coordinates and tile grid.

The Clash Royale arena is a 18x32 tile grid:
- 18 tiles wide (x-axis, left to right)
- 32 tiles tall (y-axis, top to bottom)

The arena does not cover the full screen - there's UI at top and bottom.
This module handles the conversion between pixel coordinates (from Roboflow)
and tile coordinates (for the strategy model).
"""

from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any
import json


@dataclass
class ArenaBounds:
    """
    Defines the pixel boundaries of the playable arena within a screenshot.

    Calibrated from reference resolution 1170x2532:
    - Arena origin: (27.6, 376.7)
    - Tile size: 62x50 px (width x height)
    - Arena: 18 tiles wide (1116 px), 32 tiles tall (1600 px)
    """
    # Top-left corner of the arena (pixel coordinates)
    x_min: int = 28
    y_min: int = 327

    # Bottom-right corner of the arena (pixel coordinates)
    x_max: int = 1144  # 28 + 18*62
    y_max: int = 1927  # 327 + 32*50

    # Full image dimensions (for validation)
    image_width: int = 1170
    image_height: int = 2532

    @property
    def arena_width(self) -> int:
        return self.x_max - self.x_min

    @property
    def arena_height(self) -> int:
        return self.y_max - self.y_min

    def contains_pixel(self, px: int, py: int) -> bool:
        """Check if a pixel coordinate is within the arena bounds."""
        return (self.x_min <= px <= self.x_max and
                self.y_min <= py <= self.y_max)


class CoordinateMapper:
    """
    Maps between pixel coordinates and tile grid coordinates.

    Grid layout:
    - Origin (0, 0) is top-left of the arena
    - X increases left to right (0-17)
    - Y increases top to bottom (0-31)

    Key rows:
    - Row 0-14: Enemy side
    - Row 15: River (top)
    - Row 16: River (bottom) / Bridges at specific columns
    - Row 17-31: Your side
    """

    GRID_WIDTH = 18   # tiles
    GRID_HEIGHT = 32  # tiles

    # Bridge tile positions (x coordinates where you can cross the river)
    BRIDGE_TILES_X = [3, 4, 13, 14]  # Left bridge and right bridge
    RIVER_ROWS = [15, 16]

    def __init__(self, bounds: Optional[ArenaBounds] = None):
        """
        Initialize the coordinate mapper.

        Args:
            bounds: Arena pixel boundaries. If None, uses default estimates.
        """
        self.bounds = bounds or ArenaBounds()
        self._calculate_tile_dimensions()

    def _calculate_tile_dimensions(self):
        """Calculate the pixel size of each tile."""
        self.tile_width = self.bounds.arena_width / self.GRID_WIDTH
        self.tile_height = self.bounds.arena_height / self.GRID_HEIGHT

    def pixel_to_tile(self, px: int, py: int) -> Tuple[int, int]:
        """
        Convert pixel coordinates to tile coordinates.

        Args:
            px: Pixel x coordinate (from Roboflow detection)
            py: Pixel y coordinate (from Roboflow detection)

        Returns:
            Tuple of (tile_x, tile_y) where:
            - tile_x is in range [0, 17]
            - tile_y is in range [0, 31]

        Note:
            Coordinates outside the arena are clamped to valid range.
        """
        # Adjust for arena offset
        arena_x = px - self.bounds.x_min
        arena_y = py - self.bounds.y_min

        # Convert to tile coordinates
        tile_x = int(arena_x / self.tile_width)
        tile_y = int(arena_y / self.tile_height)

        # Clamp to valid range
        tile_x = max(0, min(self.GRID_WIDTH - 1, tile_x))
        tile_y = max(0, min(self.GRID_HEIGHT - 1, tile_y))

        return (tile_x, tile_y)

    def tile_to_pixel(self, tile_x: int, tile_y: int, center: bool = True) -> Tuple[int, int]:
        """
        Convert tile coordinates to pixel coordinates.

        Args:
            tile_x: Tile x coordinate (0-17)
            tile_y: Tile y coordinate (0-31)
            center: If True, returns center of tile. If False, returns top-left.

        Returns:
            Tuple of (pixel_x, pixel_y)
        """
        offset = 0.5 if center else 0.0

        px = int((tile_x + offset) * self.tile_width + self.bounds.x_min)
        py = int((tile_y + offset) * self.tile_height + self.bounds.y_min)

        return (px, py)

    def get_tile_bounds_pixels(self, tile_x: int, tile_y: int) -> Tuple[int, int, int, int]:
        """
        Get the pixel bounding box for a tile.

        Returns:
            Tuple of (x_min, y_min, x_max, y_max) in pixels
        """
        x_min = int(tile_x * self.tile_width + self.bounds.x_min)
        y_min = int(tile_y * self.tile_height + self.bounds.y_min)
        x_max = int((tile_x + 1) * self.tile_width + self.bounds.x_min)
        y_max = int((tile_y + 1) * self.tile_height + self.bounds.y_min)

        return (x_min, y_min, x_max, y_max)

    def is_on_your_side(self, tile_y: int) -> bool:
        """Check if a tile row is on your side of the arena."""
        return tile_y >= 17

    def is_on_enemy_side(self, tile_y: int) -> bool:
        """Check if a tile row is on the enemy side of the arena."""
        return tile_y <= 14

    def is_river(self, tile_x: int, tile_y: int) -> bool:
        """Check if a tile is part of the river."""
        if tile_y not in self.RIVER_ROWS:
            return False
        # Bridges are not river
        if tile_x in self.BRIDGE_TILES_X:
            return False
        return True

    def is_bridge(self, tile_x: int, tile_y: int) -> bool:
        """Check if a tile is a bridge."""
        return tile_y in self.RIVER_ROWS and tile_x in self.BRIDGE_TILES_X

    # Reference values calibrated from 1170x2532 screenshots
    _REF_WIDTH = 1170
    _REF_HEIGHT = 2532
    _REF_ORIGIN_X = 27.6
    _REF_ORIGIN_Y = 326.7
    _REF_TILE_W = 62.0
    _REF_TILE_H = 50.0

    def calibrate_from_image(self, image_width: int, image_height: int) -> None:
        """
        Auto-calibrate arena bounds based on image dimensions.

        Scales from the reference resolution (1170x2532) where:
        - Arena origin: (27.6, 376.7)
        - Tile size: 62x50 px

        Args:
            image_width: Width of the screenshot in pixels
            image_height: Height of the screenshot in pixels
        """
        scale_x = image_width / self._REF_WIDTH
        scale_y = image_height / self._REF_HEIGHT

        x_min = self._REF_ORIGIN_X * scale_x
        y_min = self._REF_ORIGIN_Y * scale_y
        tile_w = self._REF_TILE_W * scale_x
        tile_h = self._REF_TILE_H * scale_y

        self.bounds = ArenaBounds(
            x_min=int(round(x_min)),
            y_min=int(round(y_min)),
            x_max=int(round(x_min + self.GRID_WIDTH * tile_w)),
            y_max=int(round(y_min + self.GRID_HEIGHT * tile_h)),
            image_width=image_width,
            image_height=image_height
        )
        self._calculate_tile_dimensions()

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration to dictionary."""
        return {
            "grid_width": self.GRID_WIDTH,
            "grid_height": self.GRID_HEIGHT,
            "tile_width_px": self.tile_width,
            "tile_height_px": self.tile_height,
            "bounds": {
                "x_min": self.bounds.x_min,
                "y_min": self.bounds.y_min,
                "x_max": self.bounds.x_max,
                "y_max": self.bounds.y_max,
                "image_width": self.bounds.image_width,
                "image_height": self.bounds.image_height
            }
        }

    def __repr__(self) -> str:
        return (f"CoordinateMapper(grid={self.GRID_WIDTH}x{self.GRID_HEIGHT}, "
                f"tile_size={self.tile_width:.1f}x{self.tile_height:.1f}px)")


def create_grid_visualization(mapper: CoordinateMapper) -> List[List[str]]:
    """
    Create a text visualization of the grid for debugging.

    Returns:
        2D list representing the grid with markers for special tiles.
    """
    grid = []
    for y in range(mapper.GRID_HEIGHT):
        row = []
        for x in range(mapper.GRID_WIDTH):
            if mapper.is_bridge(x, y):
                row.append("B")  # Bridge
            elif mapper.is_river(x, y):
                row.append("~")  # River
            elif mapper.is_on_enemy_side(y):
                row.append("E")  # Enemy side
            else:
                row.append(".")  # Your side
        grid.append(row)
    return grid


def print_grid(mapper: CoordinateMapper) -> None:
    """Print a visual representation of the grid."""
    grid = create_grid_visualization(mapper)
    print(f"Grid: {mapper.GRID_WIDTH}x{mapper.GRID_HEIGHT}")
    print("Legend: . = your side, E = enemy side, ~ = river, B = bridge")
    print("-" * (mapper.GRID_WIDTH + 4))
    for y, row in enumerate(grid):
        print(f"{y:2d} |{''.join(row)}|")
    print("-" * (mapper.GRID_WIDTH + 4))
    print("    " + "".join(str(x % 10) for x in range(mapper.GRID_WIDTH)))
