"""
Card type categorization for Clash Royale cards.

This file categorizes all cards into major types for strategy reference:
- TROOP: Units that can move and attack
- SPELL: Temporary effects cast on the battlefield
- BUILDING: Stationary structures that decay over time
- TOWER_TROOP: Units that stay on crown towers

This helps players quickly identify:
- Win conditions (towers, heavy hitters)
- Defensive cards
- Support cards
- Spell synergy
"""

from enum import Enum
from typing import Dict, Set


class CardType(Enum):
    """
    Enum categorizing Clash Royale cards into types for strategy reference.

    Used to identify win conditions, defensive cards, spells, and buildings
    throughout the game logic and analysis.

    Attributes:
        TROOP (str): Units that can move and attack.
        SPELL (str): Temporary effects cast on the battlefield.
        BUILDING (str): Stationary structures that decay over time.
        TOWER_TROOP (str): Units that stay on crown towers.
    """
    TROOP = "troop"
    SPELL = "spell"
    BUILDING = "building"
    TOWER_TROOP = "tower_troop"


# Complete card type mapping
# All cards are categorized by their main type in Clash Royale
CARD_TYPES: Dict[str, CardType] = {
    # === TROOP CARDS ===
    # 1 Elixir
    "skeletons": CardType.TROOP,
    "skeleton": CardType.TROOP,
    "ice-spirit": CardType.TROOP,
    "fire-spirit": CardType.TROOP,
    "electro-spirit": CardType.TROOP,
    "heal-spirit": CardType.TROOP,

    # 2 Elixir
    "ice-golem": CardType.TROOP,
    "the-log": CardType.SPELL,
    "log": CardType.SPELL,
    "zap": CardType.SPELL,
    "rage": CardType.SPELL,
    "snowball": CardType.SPELL,
    "barbarian-barrel": CardType.SPELL,
    "bats": CardType.TROOP,
    "spear-goblins": CardType.TROOP,
    "goblins": CardType.TROOP,
    "bomber": CardType.TROOP,
    "guards": CardType.TROOP,
    "larry": CardType.TROOP,

    # 3 Elixir
    "knight": CardType.TROOP,
    "archers": CardType.TROOP,
    "arrows": CardType.SPELL,
    "goblin-barrel": CardType.SPELL,
    "goblin-gang": CardType.TROOP,
    "skeleton-army": CardType.TROOP,
    "minions": CardType.TROOP,
    "miner": CardType.TROOP,
    "cannon": CardType.BUILDING,
    "tombstone": CardType.BUILDING,
    "tornado": CardType.SPELL,
    "earthquake": CardType.SPELL,
    "elixir-golem": CardType.TROOP,
    "rascals": CardType.TROOP,
    "dart-goblin": CardType.TROOP,
    "bandit": CardType.TROOP,
    "base-bandit": CardType.TROOP,
    "boss-bandit": CardType.TROOP,
    "royal-ghost": CardType.TROOP,
    "ice-wizard": CardType.TROOP,
    "princess": CardType.TROOP,
    "magic-archer": CardType.TROOP,
    "log-bait": CardType.SPELL,
    "wall-breakers": CardType.TROOP,
    "firecracker": CardType.TROOP,
    "royal-delivery": CardType.SPELL,
    "skeleton-king": CardType.TROOP,
    "little-prince": CardType.TROOP,

    # 4 Elixir
    "hog-rider": CardType.TROOP,
    "hog": CardType.TROOP,
    "musketeer": CardType.TROOP,
    "mini-pekka": CardType.TROOP,
    "valkyrie": CardType.TROOP,
    "battle-ram": CardType.TROOP,
    "dark-prince": CardType.TROOP,
    "prince": CardType.TROOP,
    "baby-dragon": CardType.TROOP,
    "inferno-dragon": CardType.TROOP,
    "lumberjack": CardType.TROOP,
    "night-witch": CardType.TROOP,
    "mother-witch": CardType.TROOP,
    "electro-wizard": CardType.TROOP,
    "fireball": CardType.SPELL,
    "poison": CardType.SPELL,
    "freeze": CardType.SPELL,
    "tesla": CardType.BUILDING,
    "bomb-tower": CardType.BUILDING,
    "mortar": CardType.BUILDING,
    "furnace": CardType.BUILDING,
    "flying-machine": CardType.TROOP,
    "zappies": CardType.TROOP,
    "hunter": CardType.TROOP,
    "royal-hogs": CardType.TROOP,
    "barbarian-hut": CardType.BUILDING,
    "mega-minion": CardType.TROOP,
    "golden-knight": CardType.TROOP,
    "mighty-miner": CardType.TROOP,
    "monk": CardType.TROOP,
    "phoenix": CardType.TROOP,

    # 5 Elixir
    "giant": CardType.TROOP,
    "witch": CardType.TROOP,
    "wizard": CardType.TROOP,
    "balloon": CardType.TROOP,
    "bowler": CardType.TROOP,
    "electro-dragon": CardType.TROOP,
    "inferno-tower": CardType.BUILDING,
    "minion-horde": CardType.TROOP,
    "graveyard": CardType.SPELL,
    "goblin-hut": CardType.BUILDING,
    "ram-rider": CardType.TROOP,
    "mega-knight": CardType.TROOP,
    "royal-recruits": CardType.TROOP,
    "executioner": CardType.TROOP,
    "cannon-cart": CardType.TROOP,
    "giant-skeleton": CardType.TROOP,
    "sparky": CardType.TROOP,
    "archer-queen": CardType.TROOP,
    "electro-giant": CardType.TROOP,

    # 6 Elixir
    "elixir-collector": CardType.BUILDING,
    "lightning": CardType.SPELL,
    "rocket": CardType.SPELL,
    "x-bow": CardType.BUILDING,
    "royal-giant": CardType.TROOP,
    "elite-barbarians": CardType.TROOP,

    # 7 Elixir
    "pekka": CardType.TROOP,
    "golem": CardType.TROOP,
    "lava-hound": CardType.TROOP,

    # 8 Elixir
    # Golem already listed above

    # 9 Elixir
    "three-musketeers": CardType.TROOP,

    # === EVOLUTION VARIANTS ===
    "evo-skeletons": CardType.TROOP,
    "evo-skeleton": CardType.TROOP,
    "evo-ice-spirit": CardType.TROOP,
    "evo-fire-spirit": CardType.TROOP,
    "evo-electro-spirit": CardType.TROOP,
    "evo-knight": CardType.TROOP,
    "evo-archers": CardType.TROOP,
    "evo-mortar": CardType.BUILDING,
    "evo-barbarians": CardType.TROOP,
    "evo-firecracker": CardType.TROOP,
    "evo-tesla": CardType.BUILDING,
    "evo-cannon": CardType.BUILDING,
    "evo-royal-giant": CardType.TROOP,
    "evo-royal-ghost": CardType.TROOP,
    "evo-royal-recruits": CardType.TROOP,
    "evo-musketeer": CardType.TROOP,
    "evo-valkyrie": CardType.TROOP,
    "evo-wizard": CardType.TROOP,
    "evo-mega-minion": CardType.TROOP,
    "evo-bats": CardType.TROOP,
    "evo-wall-breakers": CardType.TROOP,
    "evo-bomber": CardType.TROOP,
    "evo-pekka": CardType.TROOP,
    "evo-hog-rider": CardType.TROOP,
    "evo-ice-golem": CardType.TROOP,
    "evo-zap": CardType.SPELL,
    "evo-goblin-cage": CardType.BUILDING,
    "evo-rocket": CardType.SPELL,

    # === HERO VARIANTS ===
    "hero-ice-golem": CardType.TROOP,
    "hero-ice-golem-ability": CardType.SPELL,
    "hero-musketeer": CardType.TROOP,
    "hero-musketeer-ability": CardType.SPELL,
    "hero-wizard": CardType.TROOP,
    "hero-wizard-ability": CardType.SPELL,
    "hero-knight": CardType.TROOP,
    "hero-knight-ability": CardType.SPELL,
    "hero-valkyrie": CardType.TROOP,
    "hero-valkyrie-ability": CardType.SPELL,
    "hero-pekka": CardType.TROOP,
    "hero-pekka-ability": CardType.SPELL,
    "hero-archer-queen": CardType.TROOP,
    "hero-archer-queen-ability": CardType.SPELL,
    "hero-skeleton-king": CardType.TROOP,
    "hero-skeleton-king-ability": CardType.SPELL,
    "hero-golden-knight": CardType.TROOP,
    "hero-golden-knight-ability": CardType.SPELL,
    "hero-mighty-miner": CardType.TROOP,
    "hero-mighty-miner-ability": CardType.SPELL,
    "hero-monk": CardType.TROOP,
    "hero-monk-ability": CardType.SPELL,
    "hero-phoenix": CardType.TROOP,
    "hero-phoenix-ability": CardType.SPELL,
    "hero-little-prince": CardType.TROOP,
    "hero-little-prince-ability": CardType.SPELL,

    # === TOWERS ===
    "king-tower": CardType.TOWER_TROOP,
    "princess-tower": CardType.TOWER_TROOP,
    "crown-tower": CardType.TOWER_TROOP,

    # === BUILDINGS ===
    "goblin-cage": CardType.BUILDING,
    "goblin-drill": CardType.BUILDING,

    # === TROOPS ===
    "barbarians": CardType.TROOP,
    "royal-hogs": CardType.TROOP,
    "skeleton-dragons": CardType.TROOP,
    "mother-witch": CardType.TROOP,
    "electro-giant": CardType.TROOP,

    # === CHAMPIONS ===
    "skeleton-king": CardType.TROOP,
    "archer-queen": CardType.TROOP,
    "golden-knight": CardType.TROOP,
    "mighty-miner": CardType.TROOP,
    "monk": CardType.TROOP,
    "phoenix": CardType.TROOP,
    "little-prince": CardType.TROOP,

    # === RECENT ADDITIONS ===
    "rune-giant": CardType.TROOP,
    "berserker": CardType.TROOP,
    "spirit-empress": CardType.TROOP,
    "vines": CardType.SPELL,
}


def get_card_type(card_name: str) -> CardType:
    """
    Get the type of a card by its name.

    Args:
        card_name: Card name as detected

    Returns:
        CardType enum value, defaults to TROOP if unknown
    """
    clean_name = card_name.lower().strip()

    clean_name = clean_name.replace("opponent-", "")
    clean_name = clean_name.replace("player-", "")
    clean_name = clean_name.replace("friendly-", "")

    for suffix in ["-in-hand", "-next", "-on-field", "_on_field",
                 "-evolution", "_evolution", "-ability", "_ability"]:
        clean_name = clean_name.replace(suffix, "")

    clean_name = clean_name.strip()

    return CARD_TYPES.get(clean_name, CardType.TROOP)


def get_cards_by_type(card_type: CardType) -> list[str]:
    """
    Get all cards of a specific type.

    Args:
        card_type: Type to filter by

    Returns:
        List of card names of that type
    """
    return [name for name, ctype in CARD_TYPES.items() if ctype == card_type and name]


def is_troop(card_name: str) -> bool:
    """
    Check if card is a troop.

    Args:
        card_name (str): Name of the card.

    Returns:
        (bool) True if the card is a troop, False otherwise.
    """
    return get_card_type(card_name) == CardType.TROOP


def is_spell(card_name: str) -> bool:
    """
    Check if card is a spell.

    Args:
        card_name (str): Name of the card.

    Returns:
        (bool) True if the card is a spell, False otherwise.
    """
    return get_card_type(card_name) == CardType.SPELL


def is_building(card_name: str) -> bool:
    """
    Check if card is a building.

    Args:
        card_name (str): Name of the card.

    Returns:
        (bool) True if the card is a building, False otherwise.
    """
    return get_card_type(card_name) == CardType.BUILDING


def is_tower_troop(card_name: str) -> bool:
    """
    Check if card is a tower troop.

    Args:
        card_name (str): Name of the card.

    Returns:
        (bool) True if the card is a tower troop, False otherwise.
    """
    return get_card_type(card_name) == CardType.TOWER_TROOP


# Card type names for display
CARD_TYPE_NAMES: Dict[CardType, str] = {
    CardType.TROOP: "Troop",
    CardType.SPELL: "Spell",
    CardType.BUILDING: "Building",
    CardType.TOWER_TROOP: "Tower Troop",
}


def get_card_type_name(card_name: str) -> str:
    """
    Get display name for card type.

    Args:
        card_name (str): Name of the card.

    Returns:
        (str) Display name for the card type (e.g., "Troop", "Spell").
    """
    return CARD_TYPE_NAMES.get(get_card_type(card_name), "Unknown")