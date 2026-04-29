"""
Game timing, grid, and phase constants.

These values are referenced across detection, OCR, and feature-encoding modules.
Changing any constant here affects the entire pipeline, so treat them as read-only.

Public API:
  GAME_DURATION_S        -- regulation game length in seconds (3:00)
  DOUBLE_ELIXIR_START_S  -- seconds remaining when double-elixir begins
  OVERTIME_DURATION_S    -- overtime game length in seconds (2:00)
  TRIPLE_ELIXIR_START_S  -- seconds remaining in overtime when triple begins
  GRID_COLS / GRID_ROWS  -- arena tile dimensions (18 × 32)
  PLAYER_SIDE_MIN_ROW    -- first tile row on the player's half
  ENEMY_SIDE_MAX_ROW     -- last tile row on the opponent's half
  RIVER_ROWS             -- tile rows occupied by the river
  BRIDGE_COLS            -- tile columns containing bridge crossing points
  ELIXIR_MAX             -- maximum elixir a player can hold
  HAND_SIZE              -- number of cards shown in the player's hand
"""
from __future__ import annotations

from typing import Tuple

GAME_DURATION_S: int = 180        # 3:00 regulation
DOUBLE_ELIXIR_START_S: int = 60   # last 60s of regulation
OVERTIME_DURATION_S: int = 120    # 2:00 overtime
TRIPLE_ELIXIR_START_S: int = 60   # last 60s of overtime

GRID_COLS: int = 18
GRID_ROWS: int = 32
PLAYER_SIDE_MIN_ROW: int = 17     # rows 17-31 are player side
ENEMY_SIDE_MAX_ROW: int = 14      # rows 0-14 are enemy side
RIVER_ROWS: Tuple[int, int] = (15, 16)
BRIDGE_COLS: Tuple[int, int, int, int] = (3, 4, 13, 14)

ELIXIR_MAX: int = 10
HAND_SIZE: int = 4
