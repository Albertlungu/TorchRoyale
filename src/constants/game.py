"""Game timing and phase constants."""

GAME_DURATION_S: int = 180        # 3:00 regulation
DOUBLE_ELIXIR_START_S: int = 60   # last 60s of regulation
OVERTIME_DURATION_S: int = 120    # 2:00 overtime
TRIPLE_ELIXIR_START_S: int = 60   # last 60s of overtime

GRID_COLS: int = 18
GRID_ROWS: int = 32
PLAYER_SIDE_MIN_ROW: int = 17     # rows 17-31 are player side
ENEMY_SIDE_MAX_ROW: int = 14      # rows 0-14 are enemy side
RIVER_ROWS = (15, 16)
BRIDGE_COLS = (3, 4, 13, 14)

ELIXIR_MAX: int = 10
HAND_SIZE: int = 4
