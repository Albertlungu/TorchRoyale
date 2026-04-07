"""
Complete Clash Royale card elixir costs database.

This file contains all cards in Clash Royale (125+ cards as of 2026)
mapped to their elixir costs. Used for opponent elixir tracking.

Card names are normalized to lowercase with hyphens, matching
the Roboflow detection model output format.
"""

from typing import Dict

# Complete card elixir cost mapping
CARD_ELIXIR_COSTS: Dict[str, int] = {
    # 1 Elixir Cards
    "skeletons": 1,
    "skeleton": 1,
    "ice-spirit": 1,
    "fire-spirit": 1,
    "electro-spirit": 1,
    "heal-spirit": 1,

    # 2 Elixir Cards
    "ice-golem": 2,
    "the-log": 2,
    "log": 2,
    "zap": 2,
    "rage": 2,
    "snowball": 2,
    "barbarian-barrel": 2,
    "bats": 2,
    "spear-goblins": 2,
    "goblins": 2,
    "bomber": 2,
    "guards": 2,
    "larry": 2,

    # 3 Elixir Cards
    "knight": 3,
    "archers": 3,
    "arrows": 3,
    "goblin-barrel": 3,
    "goblin-gang": 3,
    "skeleton-army": 3,
    "minions": 3,
    "miner": 3,
    "cannon": 3,
    "tombstone": 3,
    "tornado": 3,
    "earthquake": 3,
    "elixir-golem": 3,
    "rascals": 3,
    "dart-goblin": 3,
    "bandit": 3,
    "base-bandit": 3,
    "boss-bandit": 3,
    "royal-ghost": 3,
    "ice-wizard": 3,
    "princess": 3,
    "magic-archer": 3,
    "miner": 3,
    "log-bait": 3,
    "wall-breakers": 3,
    "firecracker": 3,
    "royal-delivery": 3,
    "skeleton-king": 3,
    "little-prince": 3,

    # 4 Elixir Cards
    "hog-rider": 4,
    "hog": 4,
    "musketeer": 4,
    "mini-pekka": 4,
    "valkyrie": 4,
    "battle-ram": 4,
    "dark-prince": 4,
    "prince": 4,
    "baby-dragon": 4,
    "inferno-dragon": 4,
    "lumberjack": 4,
    "night-witch": 4,
    "mother-witch": 4,
    "electro-wizard": 4,
    "fireball": 4,
    "poison": 4,
    "freeze": 4,
    "tesla": 4,
    "bomb-tower": 4,
    "mortar": 4,
    "furnace": 4,
    "flying-machine": 4,
    "zappies": 4,
    "hunter": 4,
    "royal-hogs": 4,
    "barbarian-hut": 4,
    "mega-minion": 4,
    "hunter": 4,
    "golden-knight": 4,
    "mighty-miner": 4,
    "monk": 4,
    "phoenix": 4,

    # 5 Elixir Cards
    "giant": 5,
    "witch": 5,
    "wizard": 5,
    "balloon": 5,
    "prince": 5,
    "bowler": 5,
    "electro-dragon": 5,
    "inferno-tower": 5,
    "minion-horde": 5,
    "graveyard": 5,
    "goblin-hut": 5,
    "ram-rider": 5,
    "mega-knight": 5,
    "royal-recruits": 5,
    "executioner": 5,
    "cannon-cart": 5,
    "giant-skeleton": 5,
    "sparky": 5,
    "archer-queen": 5,
    "electro-giant": 5,

    # 6 Elixir Cards
    "giant-skeleton": 6,
    "elixir-collector": 6,
    "lightning": 6,
    "rocket": 6,
    "x-bow": 6,
    "royal-giant": 6,
    "elite-barbarians": 6,

    # 7 Elixir Cards
    "pekka": 7,
    "golem": 7,
    "lava-hound": 7,
    "mega-knight": 7,
    "electro-giant": 7,
    "royal-recruits": 7,

    # 8 Elixir Cards
    "golem": 8,

    # 9 Elixir Cards
    "three-musketeers": 9,

    # Evolution variants (same cost as base)
    "evo-skeletons": 1,
    "evo-skeleton": 1,
    "evo-ice-spirit": 1,
    "evo-fire-spirit": 1,
    "evo-electro-spirit": 1,
    "evo-knight": 3,
    "evo-archers": 3,
    "evo-mortar": 4,
    "evo-barbarians": 5,
    "evo-firecracker": 3,
    "evo-tesla": 4,
    "evo-cannon": 3,
    "evo-royal-giant": 6,
    "evo-royal-ghost": 3,
    "evo-royal-recruits": 7,
    "evo-musketeer": 4,
    "evo-valkyrie": 4,
    "evo-wizard": 5,
    "evo-mega-minion": 4,
    "evo-bats": 2,
    "evo-wall-breakers": 3,
    "evo-bomber": 2,
    "evo-pekka": 7,
    "evo-hog-rider": 4,
    "evo-ice-golem": 2,
    "evo-zap": 2,
    "evo-goblin-cage": 4,
    "evo-rocket": 6,

    # Hero variants (same cost as base)
    "hero-ice-golem": 2,
    "hero-ice-golem-ability": 0,
    "hero-musketeer": 4,
    "hero-musketeer-ability": 0,
    "hero-wizard": 5,
    "hero-wizard-ability": 0,
    "hero-knight": 3,
    "hero-knight-ability": 0,
    "hero-valkyrie": 4,
    "hero-valkyrie-ability": 0,
    "hero-pekka": 7,
    "hero-pekka-ability": 0,
    "hero-archer-queen": 5,
    "hero-archer-queen-ability": 0,
    "hero-skeleton-king": 3,
    "hero-skeleton-king-ability": 0,
    "hero-golden-knight": 4,
    "hero-golden-knight-ability": 0,
    "hero-mighty-miner": 4,
    "hero-mighty-miner-ability": 0,
    "hero-monk": 4,
    "hero-monk-ability": 0,
    "hero-phoenix": 4,
    "hero-phoenix-ability": 0,
    "hero-little-prince": 3,
    "hero-little-prince-ability": 0,

    # Towers (0 elixir - spawned at game start)
    "king-tower": 0,
    "princess-tower": 0,
    "crown-tower": 0,

    # Buildings
    "goblin-cage": 4,
    "goblin-drill": 4,

    # Troops
    "barbarians": 5,
    "royal-hogs": 4,
    "skeleton-dragons": 4,
    "electro-spirit": 1,
    "mother-witch": 4,
    "electro-giant": 7,

    # Champions (same as their base cost)
    "skeleton-king": 3,
    "archer-queen": 5,
    "golden-knight": 4,
    "mighty-miner": 4,
    "monk": 4,
    "phoenix": 4,
    "little-prince": 3,

    # Recent additions (2025-2026)
    "rune-giant": 7,
    "berserker": 4,
    "spirit-empress": 5,
    "vines": 3,
}


def get_card_cost(card_name: str) -> int:
    """
    Get elixir cost for any card, handling name variations.

    Strips common prefixes/suffixes:
    - "opponent-" prefix
    - "-in-hand", "-next", "-on-field", "_on_field" suffixes
    - "-evolution", "_evolution" suffixes
    - "-ability" suffix (abilities cost 0)

    Args:
        card_name: Card name as detected by RoboFlow

    Returns:
        Elixir cost (0 if unknown or ability)
    """
    clean_name = card_name.lower().strip()

    # Remove prefixes
    clean_name = clean_name.replace("opponent-", "")
    clean_name = clean_name.replace("player-", "")
    clean_name = clean_name.replace("friendly-", "")

    # Remove suffixes
    for suffix in ["-in-hand", "-next", "-on-field", "_on_field",
                   "-evolution", "_evolution", "-ability", "_ability"]:
        clean_name = clean_name.replace(suffix, "")

    clean_name = clean_name.strip()

    return CARD_ELIXIR_COSTS.get(clean_name, 0)


def get_all_cards_by_cost(cost: int) -> list[str]:
    """
    Get all cards that cost a specific amount of elixir.

    Args:
        cost: Elixir cost to filter by

    Returns:
        List of card names with that cost
    """
    return [name for name, c in CARD_ELIXIR_COSTS.items() if c == cost]


def is_valid_card(card_name: str) -> bool:
    """
    Check if a card name exists in the database.

    Args:
        card_name: Card name to check

    Returns:
        True if card exists, False otherwise
    """
    return get_card_cost(card_name) > 0 or card_name.lower() in ["king-tower", "princess-tower", "crown-tower"]
