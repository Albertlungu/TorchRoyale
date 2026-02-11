"""
Game constants for Clash Royale mechanics.

Contains elixir regeneration rates, game phase definitions,
tower HP tables, and card elixir costs.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple


class GamePhase(Enum):
    """Game phase based on timer and elixir multiplier."""
    SINGLE_ELIXIR = "single"       # 3:00 - 2:00 (normal elixir rate)
    DOUBLE_ELIXIR = "double"       # 2:00 - 0:00 or sudden death start
    TRIPLE_ELIXIR = "triple"       # Last 1:00 of sudden death
    SUDDEN_DEATH = "sudden_death"  # Overtime when tied
    GAME_OVER = "game_over"


@dataclass(frozen=True)
class ElixirConstants:
    """Elixir game mechanics constants."""
    MAX_ELIXIR: int = 10
    STARTING_ELIXIR: int = 5

    # Regeneration rates (seconds per 1 elixir)
    SINGLE_REGEN_RATE: float = 2.8
    DOUBLE_REGEN_RATE: float = 1.4
    TRIPLE_REGEN_RATE: float = 2.8 / 3  # ~0.933


@dataclass(frozen=True)
class GameTimingConstants:
    """Game timing constants in seconds."""
    TOTAL_GAME_TIME: int = 180         # 3 minutes
    DOUBLE_ELIXIR_START: int = 60      # Last 1 minute triggers x2
    SUDDEN_DEATH_DURATION: int = 180   # 3 minutes overtime max
    TRIPLE_ELIXIR_START: int = 60      # Last 1 min of sudden death triggers x3


# Tower HP by level: maps level -> (king_tower_hp, princess_tower_hp)
TOWER_HP: Dict[int, Tuple[int, int]] = {
    1:  (2400, 1400),
    2:  (2568, 1512),
    3:  (2736, 1624),
    4:  (2904, 1750),
    5:  (3096, 1890),
    6:  (3312, 2030),
    7:  (3528, 2184),
    8:  (3768, 2352),
    9:  (4008, 2534),
    10: (4411, 2786),
    11: (4824, 3052),
    12: (5304, 3346),
    13: (5832, 3668),
    14: (6408, 4032),
    15: (7056, 4424),
    16: (7776, 4858),
}


def get_tower_max_hp(level: int, is_king: bool) -> int:
    """
    Get max HP for a tower at a given level.

    Args:
        level: Tower level (1-16)
        is_king: True for king tower, False for princess tower

    Returns:
        Max HP value
    """
    king_hp, princess_hp = TOWER_HP.get(level, TOWER_HP[15])
    return king_hp if is_king else princess_hp


# Card elixir costs
# Maps card names (as detected by Roboflow model torchroyale/4) to elixir cost
ELIXIR_COSTS: Dict[str, int] = {
    # Hog 2.6 deck (player cards)
    "hog-rider": 4,
    "musketeer": 4,
    "ice-golem": 2,
    "ice-spirit": 1,
    "skeletons": 1,
    "skeleton": 1,
    "cannon": 3,
    "fireball": 4,
    "the-log": 2,
    "log": 2,

    # Evolution / hero variants (same cost as base card)
    "evo-musketeer": 4,
    "evo-skeletons": 1,
    "evo-skeleton": 1,
    "hero-ice-golem": 2,
    "hero-ice-golem-ability": 0,
    "hero-musketeer": 4,
    "hero-musketeer-ability": 0,
    "hero-wizard": 5,

    # Opponent cards detected by Roboflow model
    "base-bandit": 3,
    "boss-bandit": 3,
    "bowler": 5,
    "electro-dragon": 5,
    "electro-giant": 7,
    "electro-spirit": 1,
    "fire-spirit": 1,
    "golden-knight": 4,
    "mega-knight": 7,
    "mighty-miner": 4,
    "mini-pekka": 4,
    "minion": 3,
    "royal-ghost": 3,
    "royal-recruits": 7,
    "evo-cannon": 3,
    "evo-royal-ghost": 3,
    "evo-royal-recruits": 7,

    # Common cards (may appear as opponent detections)
    "knight": 3,
    "archers": 3,
    "minions": 3,
    "valkyrie": 4,
    "wizard": 5,
    "witch": 5,
    "pekka": 7,
    "giant": 5,
    "golem": 8,
    "royal-giant": 6,
    "hog": 4,
    "balloon": 5,
    "lava-hound": 7,
    "baby-dragon": 4,
    "electro-wizard": 4,
    "sparky": 6,
    "prince": 5,
    "dark-prince": 4,
    "goblin-barrel": 3,
    "goblin-gang": 3,
    "skeleton-army": 3,
    "minion-horde": 5,
    "three-musketeers": 9,
    "elixir-golem": 3,
    "battle-ram": 4,
    "ram-rider": 5,
    "bandit": 3,
    "inferno-dragon": 4,
    "lumberjack": 4,
    "night-witch": 4,
    "mother-witch": 4,
    "graveyard": 5,
    "poison": 4,
    "earthquake": 3,
    "lightning": 6,
    "rocket": 6,
    "freeze": 4,
    "rage": 2,
    "zap": 2,
    "arrows": 3,
    "tornado": 3,
    "barbarian-barrel": 2,
    "snowball": 2,
    "tesla": 4,
    "inferno-tower": 5,
    "bomb-tower": 4,
    "mortar": 4,
    "x-bow": 6,
    "goblin-hut": 5,
    "furnace": 4,
    "tombstone": 3,
    "elixir-collector": 6,
}


def get_elixir_cost(card_name: str) -> int:
    """
    Get elixir cost for a card, handling name variants.

    Strips common prefixes/suffixes from Roboflow detection names:
    - "opponent-" prefix
    - "-in-hand", "-next", "-on-field" suffixes
    - "_on_field" suffix
    - "-evolution", "_evolution" suffixes
    - "-ability" suffix

    Args:
        card_name: Card name as detected (may include prefixes/suffixes)

    Returns:
        Elixir cost (0 if card not found)
    """
    clean_name = card_name.lower()
    clean_name = clean_name.replace("opponent-", "")
    for suffix in ["-in-hand", "-next", "-on-field", "_on_field",
                   "-evolution", "_evolution", "-ability"]:
        clean_name = clean_name.replace(suffix, "")
    clean_name = clean_name.strip()

    return ELIXIR_COSTS.get(clean_name, 0)


def get_regen_rate(phase: GamePhase) -> float:
    """
    Get elixir regeneration rate for a game phase.

    Args:
        phase: Current game phase

    Returns:
        Seconds per 1 elixir regeneration
    """
    constants = ElixirConstants()

    if phase == GamePhase.TRIPLE_ELIXIR:
        return constants.TRIPLE_REGEN_RATE
    elif phase in (GamePhase.DOUBLE_ELIXIR, GamePhase.SUDDEN_DEATH):
        return constants.DOUBLE_REGEN_RATE
    else:
        return constants.SINGLE_REGEN_RATE
