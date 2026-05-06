"""Detector orchestrator for live TorchRoyale gameplay."""

import time
from typing import Dict, List, Optional

from PIL import Image

from src.detection.card_detector import CardDetector
from src.detection.number_detector import NumberDetector
from src.detection.screen_detector import ScreenDetector
from src.detection.unit_detector import UnitDetector
from src.namespaces.cards import Card
from src.namespaces.screens import Screens
from src.namespaces.state import State


class LiveDetector:
    """
    Run the migrated detector stack over a live screenshot.

    Attributes:
        DECK_SIZE (int): Expected number of cards in a player's deck (8)
        cards (list): List of Card objects representing the player's deck
        card_detector (CardDetector): Slot-aware detector for the next card and the four visible hand cards
        number_detector (NumberDetector): Detector for on-screen numeric values
        unit_detector (UnitDetector): Detector for units on the battlefield
        screen_detector (ScreenDetector): Detector for the current game screen state
    """

    DECK_SIZE = 8

    def __init__(self, cards: List[Card]) -> None:
        """
        Initialise detectors for the given deck.

        Args:
            cards (List[Card]): All eight cards in the player's deck

        Raises:
            ValueError: If the number of cards provided is not exactly DECK_SIZE
        """
        if len(cards) != self.DECK_SIZE:
            raise ValueError(f"You must specify all {self.DECK_SIZE} deck cards.")

        self.cards = list(cards)
        self.card_detector = CardDetector(self.cards)
        self.number_detector = NumberDetector()
        self.unit_detector = UnitDetector(self.cards)
        self.screen_detector = ScreenDetector()

    def run(self, image: Image.Image) -> State:
        """
        Run all detectors on a single screenshot and return the combined state.

        Args:
            image (Image.Image): PIL screenshot of the current game frame
        Returns:
            State: Detected game state including hand cards, units, numbers, and current screen
        """
        cards, ready = self.card_detector.run(image)
        allies, enemies = self.unit_detector.run(image)
        numbers = self.number_detector.run(image)
        screen = self.screen_detector.run(image)
        return State(allies, enemies, numbers, tuple(cards), ready, screen)


class StateAdapter:
    """
    Adapt live detector state into the frame schema used by strategies.

    Attributes:
        _game_started_at (Optional[float]): Unix timestamp of when the current game started, or None when not in-game
    """

    def __init__(self) -> None:
        """
        Initialise the adapter with no active game session.

        Args:
            None
        Returns:
            None
        """
        self._game_started_at: Optional[float] = None

    @staticmethod
    def _normalize_name(name: str) -> str:
        """
        Convert underscore-separated unit names to hyphen-separated form.

        Args:
            name (str): Raw unit name using underscores
        Returns:
            str: Name with underscores replaced by hyphens
        """
        return name.replace("_", "-")

    def _time_remaining(self, state: State) -> Optional[int]:
        """
        Calculate the estimated seconds remaining in the current game.

        Args:
            state (State): Current detected game state
        Returns:
            Optional[int]: Seconds remaining, clamped to zero, or None if not currently in-game
        """
        if state.screen == Screens.IN_GAME and self._game_started_at is None:
            self._game_started_at = time.time()
        elif state.screen != Screens.IN_GAME:
            self._game_started_at = None

        if self._game_started_at is None:
            return None

        elapsed = int(time.time() - self._game_started_at)
        return max(0, 180 - elapsed)

    @staticmethod
    def _phase_from_time_remaining(time_remaining: Optional[int]) -> str:
        """
        Determine the current game phase from the time remaining.

        Args:
            time_remaining (Optional[int]): Seconds left in the game, or None
        Returns:
            str: "double" during overtime (≤60 s), "single" otherwise
        """
        if time_remaining is None:
            return "single"
        if time_remaining <= 60:
            return "double"
        return "single"

    def _detections(self, state: State) -> List[Dict[str, object]]:
        """
        Serialise ally and enemy unit detections into strategy-compatible dicts.

        Args:
            state (State): Current detected game state
        Returns:
            List[Dict[str, object]]: List of detection payload dicts with class_name, tile_x, tile_y, is_opponent, and is_on_field keys
        """
        payload = []
        for detection in state.allies:
            payload.append(
                {
                    "class_name": self._normalize_name(detection.unit.name),
                    "tile_x": detection.position.tile_x,
                    "tile_y": detection.position.tile_y,
                    "is_opponent": False,
                    "is_on_field": True,
                }
            )
        for detection in state.enemies:
            payload.append(
                {
                    "class_name": self._normalize_name(detection.unit.name),
                    "tile_x": detection.position.tile_x,
                    "tile_y": detection.position.tile_y,
                    "is_opponent": True,
                    "is_on_field": True,
                }
            )
        return payload

    def to_strategy_state(self, state: State) -> Dict[str, object]:
        """
        Convert a raw detector state into the frame dict expected by strategies.

        Args:
            state (State): Current detected game state
        Returns:
            Dict[str, object]: Frame dict with keys for timestamp, elixir, game phase, tower health, unit detections, and hand cards
        """
        time_remaining = self._time_remaining(state)
        return {
            "timestamp_ms": int(time.time() * 1000),
            "game_time_remaining": time_remaining,
            "game_phase": self._phase_from_time_remaining(time_remaining),
            "elixir_multiplier": 2
            if time_remaining is not None and time_remaining <= 60
            else 1,
            "player_elixir": int(round(state.numbers.elixir.number)),
            "opponent_elixir_estimated": 5.0,
            "detections": self._detections(state),
            "player_towers": {
                "player_left": {
                    "health_percent": round(
                        state.numbers.left_ally_princess_hp.number * 100, 2
                    )
                },
                "player_king": {"health_percent": 100.0},
                "player_right": {
                    "health_percent": round(
                        state.numbers.right_ally_princess_hp.number * 100, 2
                    )
                },
            },
            "opponent_towers": {
                "opponent_left": {
                    "health_percent": round(
                        state.numbers.left_enemy_princess_hp.number * 100, 2
                    )
                },
                "opponent_king": {"health_percent": 100.0},
                "opponent_right": {
                    "health_percent": round(
                        state.numbers.right_enemy_princess_hp.number * 100, 2
                    )
                },
            },
            "hand_cards": [
                f"{self._normalize_name(card.name)}-in-hand"
                for card in state.cards[1:5]
            ],
        }
