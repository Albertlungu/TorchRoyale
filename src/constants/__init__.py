"""Game constants and configuration values."""

from .game_constants import (
    GamePhase,
    ElixirConstants,
    GameTimingConstants,
    ELIXIR_COSTS,
    TOWER_HP,
    get_elixir_cost,
    get_tower_max_hp,
    get_regen_rate,
)
from .ui_regions import UIRegion, UIRegions

__all__ = [
    "GamePhase",
    "ElixirConstants",
    "GameTimingConstants",
    "ELIXIR_COSTS",
    "TOWER_HP",
    "get_elixir_cost",
    "get_tower_max_hp",
    "get_regen_rate",
    "UIRegion",
    "UIRegions",
]
