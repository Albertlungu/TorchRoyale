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
    """Run the migrated detector stack over a live screenshot."""

    DECK_SIZE = 8

    def __init__(self, cards: List[Card]) -> None:
        if len(cards) != self.DECK_SIZE:
            raise ValueError(f"You must specify all {self.DECK_SIZE} deck cards.")

        self.cards = list(cards)
        self.card_detector = HandClassifier()
        self.number_detector = NumberDetector()
        self.unit_detector = UnitDetector(self.cards)
        self.screen_detector = ScreenDetector()

    @staticmethod
    def _card_from_label(label: Optional[str]) -> Card:
        if not label:
            return Cards.BLANK
        normalized = label.lower().replace("-", "_")
        return CARD_OBJECTS.get(normalized, Cards.BLANK)

    def run(self, image: Image.Image) -> State:
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
    """Adapt live detector state into the frame schema used by strategies."""

    def __init__(self) -> None:
        self._game_started_at: Optional[float] = None

    @staticmethod
    def _normalize_name(name: str) -> str:
        return name.replace("_", "-")

    def _time_remaining(self, state: State) -> Optional[int]:
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
        if time_remaining is None:
            return "single"
        if time_remaining <= 60:
            return "double"
        return "single"

    def _detections(self, state: State) -> List[Dict[str, object]]:
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
