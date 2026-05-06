"""
Deck archetype classification system.

Identifies deck types (Beatdown, Control, Cycle, Siege, Bridge Spam, Spell Bait)
based on card composition and provides matchup-specific strategies.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COSTS_PATH = _REPO_ROOT / "data" / "card_costs.json"


def _load_card_costs() -> Dict[str, int]:
    """
    Load card elixir costs from the card_costs.json data file.

    Args:
        None

    Returns:
        (Dict[str, int]): Map of normalized card names (lowercase, hyphens) to their elixir costs
    """
    cost_map: Dict[str, int] = {}
    if not _COSTS_PATH.exists():
        return cost_map
    with open(_COSTS_PATH, "r", encoding="utf-8") as handle:
        entries = json.load(handle)
    for entry in entries:
        name = entry["card_name"].lower().replace(" ", "-")
        cost_map[name] = entry["elixir_cost"]
    return cost_map


_COST_MAP = _load_card_costs()


class DeckArchetype(Enum):
    """Enum representing the six major Clash Royale deck archetypes plus an unclassified fallback.

    Attributes:
        BEATDOWN (DeckArchetype): Heavy tank-based decks 
        CONTROL (DeckArchetype): Defensive decks with chipping
        CYCLE (DeckArchetype): Low-cost decks to cycle to one win condition
        SIEGE (DeckArchetype): Building-focused decks 
        BRIDGE_SPAM (DeckArchetype): Aggressive decks 
        SPELL_BAIT (DeckArchetype): Swarm-heavy decks 
        UNKNOWN (DeckArchetype): Fallback 
    """

    BEATDOWN = "beatdown"
    CONTROL = "control"
    CYCLE = "cycle"
    SIEGE = "siege"
    BRIDGE_SPAM = "bridge_spam"
    SPELL_BAIT = "spell_bait"
    UNKNOWN = "unknown"


class DeckClassifier:
    """Classifies Clash Royale decks into archetypes and provides matchup-specific strategies

    Attributes:
        WIN_CONDITIONS (Dict[DeckArchetype, Set[str]]): Mapping of archetypes 
        TANKS (Set[str]): High-HP, tower-targeting 
        MINI_TANKS (Set[str]): Moderate-HP 
        CONTROL_DEFENSE (Set[str]): Defensive cards 
        CYCLE_CARDS (Set[str]): Very low elixir cost cards 
        SIEGE_BUILDINGS (Set[str]): Siege buildings 
        BRIDGE_SPAM_CARDS (Set[str]): bridge spam card
        BAIT_CARDS (Set[str]): Swarm and bait cards 
    """

    # Win condition cards for each archetype
    WIN_CONDITIONS = {
        DeckArchetype.BEATDOWN: {
            "golem",
            "lava-hound",
            "electro-giant",
            "giant",
            "goblin-giant",
        },
        DeckArchetype.CONTROL: {"graveyard", "miner", "royal-hogs"},
        DeckArchetype.CYCLE: {"hog-rider", "mortar", "miner", "royal-hogs"},
        DeckArchetype.SIEGE: {"x-bow", "mortar"},
        DeckArchetype.BRIDGE_SPAM: {"battle-ram", "ram-rider", "bandit", "royal-ghost"},
        DeckArchetype.SPELL_BAIT: {"goblin-barrel", "graveyard"},
    }

    # Tank cards (high HP, tower-targeting)
    TANKS = {
        "golem",
        "lava-hound",
        "electro-giant",
        "giant",
        "goblin-giant",
        "pekka",
        "mega-knight",
    }

    # Mini-tanks (moderate HP, versatile)
    MINI_TANKS = {
        "knight",
        "valkyrie",
        "mini-pekka",
        "ice-golem",
        "bandit",
        "royal-ghost",
    }

    # Control-style defensive cards
    CONTROL_DEFENSE = {
        "graveyard",
        "miner",
        "inferno-tower",
        "bowler",
        "executioner",
        "tornado",
    }

    # Cycle cards (very low elixir)
    CYCLE_CARDS = {
        "skeletons",
        "ice-spirit",
        "electro-spirit",
        "fire-spirit",
        "heal-spirit",
    }

    # Siege buildings
    SIEGE_BUILDINGS = {"x-bow", "mortar"}

    # Bridge spam cards (fast, aggressive)
    BRIDGE_SPAM_CARDS = {
        "bandit",
        "royal-ghost",
        "battle-ram",
        "ram-rider",
        "prince",
        "dark-prince",
    }

    # Spell bait cards (force small spells)
    BAIT_CARDS = {
        "goblin-gang",
        "goblin-barrel",
        "skeleton-army",
        "minion-horde",
        "bats",
        "princess",
    }

    @classmethod
    def classify_deck(cls, card_names: List[str]) -> DeckArchetype:
        """
        Classify a deck into an archetype based on its card composition.

        Args:
            card_names (List[str]): List of card names in the deck (case-insensitive, spaces allowed)

        Returns:
            (DeckArchetype): Classified deck archetype determined by card composition rules
        """
        cards_set = {card.lower().replace(" ", "-") for card in card_names}

        # Check for obvious archetype indicators

        # 1. Beatdown: Has heavy tank + support
        beatdown_tanks = cards_set & cls.TANKS
        if beatdown_tanks and len(cards_set & cls.CYCLE_CARDS) <= 2:
            return DeckArchetype.BEATDOWN

        # 2. Siege: Has siege building as win condition
        siege_cards = cards_set & cls.SIEGE_BUILDINGS
        if siege_cards:
            return DeckArchetype.SIEGE

        # 3. Cycle: Very low average elixir + cycle cards
        cycle_cards = cards_set & cls.CYCLE_CARDS
        if len(cycle_cards) >= 3 and len(cards_set & cls.TANKS) <= 1:
            return DeckArchetype.CYCLE

        # 4. Bridge Spam: Multiple bridge spam cards
        spam_cards = cards_set & cls.BRIDGE_SPAM_CARDS
        if len(spam_cards) >= 3:
            return DeckArchetype.BRIDGE_SPAM

        # 5. Spell Bait: Multiple bait cards + win condition
        bait_cards = cards_set & cls.BAIT_CARDS
        if len(bait_cards) >= 3:
            return DeckArchetype.SPELL_BAIT

        # 6. Control: Has control win condition + defensive cards
        control_wins = cards_set & cls.WIN_CONDITIONS[DeckArchetype.CONTROL]
        if control_wins and len(cards_set & cls.CONTROL_DEFENSE) >= 2:
            return DeckArchetype.CONTROL

        # Default to cycle if low elixir, otherwise control
        avg_elixir = cls._calculate_avg_elixir(card_names)
        if avg_elixir <= 3.2:
            return DeckArchetype.CYCLE

        return DeckArchetype.CONTROL

    @classmethod
    def _calculate_avg_elixir(cls, card_names: List[str]) -> float:
        """
        Calculate the average elixir cost of a deck

        Args:
            card_names (List[str]): List of card names in the deck

        Returns:
            (float): Average elixir cost of the deck
        """
        total_cost = 0
        for card in card_names:
            card_norm = card.lower().replace(" ", "-")
            total_cost += _COST_MAP.get(card_norm, 3)  # Default to 3

        return total_cost / len(card_names) if card_names else 3.0

    @classmethod
    def get_matchup_strategy(
        cls, our_archetype: DeckArchetype, opponent_archetype: DeckArchetype
    ) -> Dict[str, float]:
        """
        Get strategic parameters for a matchup between two deck archetypes.

        Args:
            our_archetype (DeckArchetype): Our deck's classified archetype
            opponent_archetype (DeckArchetype): Opponent's deck classified archetype

        Returns:
            (Dict[str, float]): Dictionary of strategic parameters including aggression_factor, defensive_weight, spell_conservatism, bridge_pressure, and cycle_speed
        """
        # Default neutral strategy
        strategy = {
            "aggression_factor": 0.5,
            "defensive_weight": 0.5,
            "spell_conservatism": 0.5,
            "bridge_pressure": 0.5,
            "cycle_speed": 1.0,  
        }

        # Matchup-specific adjustments (More can be added in the future)

        # Beatdown vs Control: Control should be defensive early
        if (
            our_archetype == DeckArchetype.CONTROL
            and opponent_archetype == DeckArchetype.BEATDOWN
        ):
            strategy["defensive_weight"] = 0.7
            strategy["aggression_factor"] = 0.3
            strategy["spell_conservatism"] = 0.8  

        # Cycle vs Beatdown: Cycle should pressure opposite lane
        elif (
            our_archetype == DeckArchetype.CYCLE
            and opponent_archetype == DeckArchetype.BEATDOWN
        ):
            strategy["aggression_factor"] = 0.6
            strategy["cycle_speed"] = 1.2  

        # Beatdown vs Cycle: Beatdown should be patient
        elif (
            our_archetype == DeckArchetype.BEATDOWN
            and opponent_archetype == DeckArchetype.CYCLE
        ):
            strategy["aggression_factor"] = 0.3
            strategy["defensive_weight"] = 0.6
            strategy["spell_conservatism"] = 0.7

        # Siege vs Anything: Siege should protect their building
        elif our_archetype == DeckArchetype.SIEGE:
            strategy["defensive_weight"] = 0.7
            strategy["spell_conservatism"] = 0.8  

        # Bridge Spam vs Control: Spam should be aggressive
        elif (
            our_archetype == DeckArchetype.BRIDGE_SPAM
            and opponent_archetype == DeckArchetype.CONTROL
        ):
            strategy["aggression_factor"] = 0.7
            strategy["bridge_pressure"] = 0.8

        # Spell Bait vs Anything: Bait should be patient
        elif our_archetype == DeckArchetype.SPELL_BAIT:
            strategy["aggression_factor"] = 0.4  
            strategy["spell_conservatism"] = 0.3  

        return strategy

    @classmethod
    def get_win_condition(cls, card_names: List[str]) -> Optional[str]:
        """
        Extract the primary win condition card from a deck.

        Args:
            card_names (List[str]): List of card names in the deck

        Returns:
            (Optional[str]): Normalized name of the primary win condition card if found, else None
        """
        cards_set = {card.lower().replace(" ", "-") for card in card_names}

        # Check each archetype's win conditions
        for archetype, win_cons in cls.WIN_CONDITIONS.items():
            possible_wins = cards_set & win_cons
            if possible_wins:
                return next(iter(possible_wins))

        return None
