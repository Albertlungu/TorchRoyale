"""Detector orchestrator for live TorchRoyale gameplay."""

import time
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from src.detection.hand_classifier import HandClassifier
from src.detection.number_detector import NumberDetector
from src.detection.screen_detector import ScreenDetector
from src.detection.unit_detector import UnitDetector
from src.namespaces.cards import Card
from src.namespaces.cards import CARD_OBJECTS
from src.namespaces.cards import Cards
from src.namespaces.screens import Screens
from src.namespaces.state import State


class LiveDetector:
    """Run the migrated detector stack over a live screenshot
    
    Attributes:
        cards (List[Card]): List of Card objects in the player's deck
        card_detector (HandClassifier): YOLO model for identifying hand cards
        number_detector (NumberDetector): OCR model for reading UI numbers
        unit_detector (UnitDetector): YOLO model for detecting units on the board
        screen_detector (ScreenDetector): Classifier for determining the current game screen
    """

    DECK_SIZE = 8

    def __init__(self, cards: List[Card]) -> None:
        """
        Initialize detectors with the provided deck configuration.
        
        Args:
            cards (List[Card]): List of exactly 8 Card objects
        Returns:
            None
        """
        if len(cards) != self.DECK_SIZE:
            raise ValueError(f"You must specify all {self.DECK_SIZE} deck cards.")

        self.cards = list(cards)
        self.card_detector = HandClassifier()
        self.number_detector = NumberDetector()
        self.unit_detector = UnitDetector(self.cards)
        self.screen_detector = ScreenDetector()

    @staticmethod
    def _card_from_label(label: Optional[str]) -> Card:
        """
        Convert a raw classifier label into a canonical Card object.
        
        Args:
            label (Optional[str]): Raw label string from the hand classifier
        Returns:
            (Card): Matching Card object, or Cards.BLANK if unrecognized
        """
        if not label:
            return Cards.BLANK
        normalized = label.lower().replace("-", "_")
        return CARD_OBJECTS.get(normalized, Cards.BLANK)

    def run(self, image: Image.Image) -> State:
        """
        Run all detection models on a single screenshot and assemble the game state.
        
        Args:
            image (Image.Image): Screenshot captured from the Android device
        Returns:
            (State): Assembled game state
        """
        # HandClassifier expects an OpenCV-style ndarray rather than a PIL image.
        bgr_frame = np.asarray(image.convert("RGB"))[:, :, ::-1]
        hand_labels = self.card_detector.classify(bgr_frame)
        cards = tuple(
            [Cards.BLANK]
            + [self._card_from_label(label) for label in hand_labels]
        )
        ready = [
            index for index, card in enumerate(cards[1:5]) if card != Cards.BLANK
        ]
        allies, enemies = self.unit_detector.run(image)
        numbers = self.number_detector.run(image)
        screen = self.screen_detector.run(image)
        return State(allies, enemies, numbers, cards, ready, screen)


class StateAdapter:
    """Adapt live detector state into the frame schema used by strategies
    
    Attributes:
        _game_started_at (Optional[float]): Timestamp when the current game session began
    """

    def __init__(self) -> None:
        """
        Initialize the state adapter.
        
        Args:
            None
        Returns:
            None
        """
        self._game_started_at: Optional[float] = None

    @staticmethod
    def _normalize_name(name: str) -> str:
        """
        Convert internal card/unit names to canonical dash-separated format.
        
        Args:
            name (str): Raw name string using underscores
        Returns:
            (str): Normalized name string using dashes
        """
        return name.replace("_", "-")

    def _time_remaining(self, state: State) -> Optional[int]:
        """
        Calculate the remaining game time in seconds based on screen transitions.
        
        Args:
            state (State): Current game state
        Returns:
            (Optional[int]): Seconds remaining in the match, or None if not started
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
        Determine the elixir multiplier phase based on remaining time.
        
        Args:
            time_remaining (Optional[int]): Seconds left in the match
        Returns:
            (str): Game phase string, either "single" or "double"
        """
        if time_remaining is None:
            return "single"
        if time_remaining <= 60:
            return "double"
        return "single"

    def _detections(self, state: State) -> List[Dict[str, object]]:
        """
        Extract unit detection payloads for both allies and enemies
        Args:
            state (State): Current game state containing ally and enemy units
        Returns:
            (List[Dict[str, object]]): List of dictionaries with unit class and position data
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
        Convert live State into a dictionary matching the Strategy tab's expected schema.
        
        Args:
            state (State): Current game state from the live detection pipeline
        Returns:
            (Dict[str, object]): Strategy-compatible state dictionary
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
