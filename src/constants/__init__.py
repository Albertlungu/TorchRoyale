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
from .card_types import (
    CardType,
    get_card_type,
    get_cards_by_type,
    is_troop,
    is_spell,
    is_building,
    is_tower_troop,
    get_card_type_name,
)

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
    "CardType",
    "get_card_type",
    "get_cards_by_type",
    "is_troop",
    "is_spell",
    "is_building",
    "is_tower_troop",
    "get_card_type_name",
]
