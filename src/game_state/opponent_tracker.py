"""
Opponent state tracking for strategic decision making.

Tracks opponent elixir, card cycle, and hand state to enable
professional-level macro play decisions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.constants.cards import elixir_cost

# Load card costs from data file
_CARD_COSTS_PATH = Path(__file__).parents[2] / "card_costs.json"
_CARD_COSTS: Dict[str, int] = {}

if _CARD_COSTS_PATH.exists():
    with open(_CARD_COSTS_PATH, "r") as f:
        costs_data = json.load(f)
        _CARD_COSTS = {
            item["card_name"].lower(): item["elixir_cost"] for item in costs_data
        }


class OpponentTracker:
    """Tracks opponent elixir, deck composition, and hand state."""

    def __init__(self, initial_elixir: int = 5) -> None:
        """
        Args:
            initial_elixir: Starting elixir (usually 5 at game start)
        """
        self.elixir = float(initial_elixir)
        self.elixir_generation_rate = 1.0  # 1 elixir per 2.8 seconds in single
        self.last_update_time = 0.0

        # Deck tracking (8 cards, unknown at start)
        self.deck: List[str] = []  # Full 8-card deck in cycle order
        self.hand: List[Optional[str]] = [None, None, None, None]  # Current 4 cards
        self.known_cards: set[str] = set()  # Cards we've seen opponent play

        # Cycle tracking
        self.next_card_index = 0  # Position in deck cycle
        self.cards_played: List[Tuple[str, float]] = []  # (card, timestamp)

        # Elixir tracking
        self.elixir_spent: List[Tuple[str, int, float]] = []  # (card, cost, timestamp)
        self.total_elixir_generated = initial_elixir

    def reset(self) -> None:
        """Reset tracker between games."""
        self.__init__()

    def update_elixir(self, current_time: float, elixir_multiplier: int = 1) -> None:
        """
        Update opponent elixir based on time passed and multiplier.

        Args:
            current_time: Current game time in seconds
            elixir_multiplier: 1, 2, or 3 based on game phase
        """
        if self.last_update_time == 0:
            self.last_update_time = current_time
            return

        time_passed = current_time - self.last_update_time
        # Elixir generates every 2.8 seconds per multiplier
        generation_rate = elixir_multiplier / 2.8
        new_elixir = time_passed * generation_rate

        self.elixir = min(10.0, self.elixir + new_elixir)
        self.total_elixir_generated += new_elixir
        self.last_update_time = current_time

    def record_card_play(self, card_name: str, timestamp: float) -> None:
        """
        Record that opponent played a card.

        Args:
            card_name: Normalized card name (lowercase, spaces to dashes)
            timestamp: When the card was played
        """
        # Get elixir cost
        cost = _CARD_COSTS.get(card_name.lower())
        if cost is None:
            # Try without evolution suffix
            base_name = card_name.replace("-evolution", "")
            cost = _CARD_COSTS.get(base_name.lower())

        if cost is None:
            # Unknown card, estimate based on common costs
            cost = 3

        # Deduct elixir
        self.elixir = max(0.0, self.elixir - cost)

        # Track spending
        self.elixir_spent.append((card_name, cost, timestamp))
        self.cards_played.append((card_name, timestamp))

        # Add to known cards
        self.known_cards.add(card_name.lower())

        # Build deck if we have enough data
        if len(self.deck) < 8 and len(self.known_cards) >= 8:
            self._rebuild_deck()

        # Update hand (card leaves hand, next card enters)
        self._update_hand(card_name)

    def _rebuild_deck(self) -> None:
        """Rebuild opponent's 8-card deck from known cards."""
        # Take the 8 most frequently seen cards
        from collections import Counter

        card_counts = Counter([c.lower() for c, _ in self.cards_played])
        most_common = card_counts.most_common(8)

        self.deck = [card for card, _ in most_common]

        # Initialize hand (first 4 cards in deck)
        for i in range(min(4, len(self.deck))):
            self.hand[i] = self.deck[i]
        self.next_card_index = 4 % max(len(self.deck), 1)

    def _update_hand(self, played_card: str) -> None:
        """Update opponent hand after they play a card."""
        # Remove played card from hand if present
        normalized_played = played_card.lower()
        for i, hand_card in enumerate(self.hand):
            if hand_card and hand_card.lower() == normalized_played:
                self.hand[i] = None
                break

        # Fill empty slots from deck cycle
        for i in range(4):
            if self.hand[i] is None and self.deck:
                self.hand[i] = self.deck[self.next_card_index]
                self.next_card_index = (self.next_card_index + 1) % len(self.deck)

    def get_elixir_advantage(self, player_elixir: float) -> float:
        """Calculate elixir advantage (positive = we have more)."""
        return player_elixir - self.elixir

    def has_counter(self, our_card: str) -> bool:
        """
        Check if opponent likely has a counter to our card in hand.

        Args:
            our_card: Our card name (e.g., "hog-rider")

        Returns:
            True if opponent likely has a counter
        """
        if not self.hand or not any(self.hand):
            return False  # Unknown hand

        # Simple heuristic: check if any hand card hard counters our card
        # This can be expanded with a proper counter database
        counters = self._get_counters_for(our_card)

        for hand_card in self.hand:
            if hand_card and hand_card.lower() in [c.lower() for c in counters]:
                return True

        return False

    def _get_counters_for(self, card: str) -> List[str]:
        """Get list of cards that counter the given card."""
        # Basic counter mappings for Hog 2.6 deck
        counters = {
            "hog-rider": [
                "cannon",
                "tesla",
                "inferno-tower",
                "bomb-tower",
                "goblin-cage",
                "tombstone",
                "mini-pekka",
                "barbarians",
            ],
            "musketeer": ["fireball", "lightning", "poison", "rocket"],
            "ice-golem": ["knight", "mini-pekka", "musketeer"],
            "cannon": ["lightning", "earthquake", "fireball"],
        }

        return counters.get(card.lower(), [])

    def get_state_string(self) -> str:
        """Get formatted state for debugging/console output."""
        hand_str = ", ".join([c or "?" for c in self.hand])
        deck_str = ", ".join(self.deck) if self.deck else "Unknown"

        return (
            f"Opponent: {self.elixir:.1f} elixir | "
            f"Hand: [{hand_str}] | "
            f"Deck: [{deck_str}] | "
            f"Known: {len(self.known_cards)}/8 cards"
        )
