"""Detector orchestrator for live TorchRoyale gameplay."""

import time
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from src.detection.hand_classifier import HandClassifier
from src.detection.number_detector import NumberDetector
from src.detection.screen_detector import ScreenDetector
from src.detection.unit_detector import UnitDetector
from src.game_state.building_tracker import BuildingPlacementTracker
from src.game_state.opponent_tracker import OpponentTracker
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
        self.opponent_tracker = OpponentTracker(initial_elixir=5)
        self.building_tracker = BuildingPlacementTracker()
        self._game_started_at: Optional[float] = None
        self._previous_enemy_counts: Dict[str, int] = {}

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
        # Track opponent cards and buildings
        self._track_opponent_state(enemies, screen)

        return State(allies, enemies, numbers, cards, ready, screen)

    def _game_time_elapsed(self, screen: Screens) -> Optional[float]:
        """Return elapsed game time in seconds for the current match."""
        if screen != Screens.IN_GAME:
            self._game_started_at = None
            self._previous_enemy_counts.clear()
            self.opponent_tracker.reset()
            self.building_tracker.reset()
            return None

        now = time.time()
        if self._game_started_at is None:
            self._game_started_at = now
            return 0.0

        return max(0.0, now - self._game_started_at)

    @staticmethod
    def _elixir_multiplier_for_elapsed(game_time_elapsed: float) -> int:
        """
        Return elixir multiplier from elapsed match time.

        0:00-2:00  -> 1x
        2:00-4:00  -> 2x (double + early overtime)
        4:00-5:00  -> 3x (late overtime)
        """
        if game_time_elapsed < 120:
            return 1
        if game_time_elapsed < 240:
            return 2
        return 3

    @staticmethod
    def _normalize_enemy_name(name: str) -> str:
        return name.lower().replace("_", "-")

    def _track_opponent_state(self, enemies: list, screen: Screens) -> None:
        """Track opponent cards, buildings, and update trackers."""
        game_time_elapsed = self._game_time_elapsed(screen)
        if game_time_elapsed is None:
            return

        # Update elixir with phase-aware multiplier.
        elixir_multiplier = self._elixir_multiplier_for_elapsed(game_time_elapsed)
        self.opponent_tracker.update_elixir(game_time_elapsed, elixir_multiplier)

        # Track newly appeared opponent cards using count delta per card name.
        enemy_counts = Counter(
            self._normalize_enemy_name(enemy.unit.name)
            for enemy in enemies
            if getattr(enemy, "unit", None) is not None
        )

        for card_name, count in enemy_counts.items():
            previous_count = self._previous_enemy_counts.get(card_name, 0)
            new_instances = max(0, count - previous_count)
            for _ in range(new_instances):
                self.opponent_tracker.record_card_play(card_name, game_time_elapsed)

        self._previous_enemy_counts = dict(enemy_counts)

        # Track buildings for prediction fireballing.
        for enemy in enemies:
            enemy_name = self._normalize_enemy_name(enemy.unit.name)

            if hasattr(enemy.unit, "building") or enemy_name in [
                "cannon",
                "tesla",
                "inferno-tower",
                "bomb-tower",
                "goblin-cage",
                "tombstone",
                "furnace",
            ]:
                self.building_tracker.record_building_placement(
                    enemy_name,
                    enemy.position.tile_x,
                    enemy.position.tile_y,
                    game_time_elapsed,
                )


class StateAdapter:
    """Adapt live detector state into the frame schema used by strategies."""

    def __init__(self) -> None:
        self._game_started_at: Optional[float] = None
        self._overtime_started_at: Optional[float] = None

    @staticmethod
    def _normalize_name(name: str) -> str:
        return name.replace("_", "-")

    def _time_remaining(self, state: State) -> Optional[int]:
        if state.screen == Screens.IN_GAME and self._game_started_at is None:
            self._game_started_at = time.time()
            self._overtime_started_at = None
        elif state.screen != Screens.IN_GAME:
            self._game_started_at = None
            self._overtime_started_at = None

        if self._game_started_at is None:
            return None

        elapsed = int(time.time() - self._game_started_at)
        if elapsed <= 180:
            return max(0, 180 - elapsed)

        overtime_elapsed = elapsed - 180
        if self._overtime_started_at is None:
            self._overtime_started_at = time.time() - overtime_elapsed
        return max(0, 120 - overtime_elapsed)

    def _phase_from_time_remaining(self, time_remaining: Optional[int]) -> str:
        if time_remaining is None:
            return "single"
        if self._overtime_started_at is None:
            return "double" if time_remaining <= 60 else "single"
        if time_remaining <= 0:
            return "game_over"
        return "triple" if time_remaining <= 60 else "double"

    def _elixir_multiplier_from_time_remaining(self, time_remaining: Optional[int]) -> int:
        if time_remaining is None:
            return 1
        if self._overtime_started_at is None:
            return 2 if time_remaining <= 60 else 1
        if time_remaining <= 0:
            return 1
        return 3 if time_remaining <= 60 else 2

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
            "elixir_multiplier": self._elixir_multiplier_from_time_remaining(
                time_remaining
            ),
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
