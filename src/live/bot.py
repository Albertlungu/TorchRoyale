"""Live TorchRoyale bot runtime for the migrated desktop UI."""

import threading
import time
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple

from src.capture.android_device import AndroidDevice
from src.live.detector import LiveDetector
from src.live.detector import StateAdapter
from src.live.visualizer import Visualizer
from src.namespaces.cards import CARD_OBJECTS
from src.namespaces.cards import Cards
from src.namespaces.screens import Screens
from src.recommendation.strategy import DTStrategy


DISPLAY_CARD_Y = 1067
DISPLAY_CARD_INIT_X = 164
DISPLAY_CARD_WIDTH = 117
DISPLAY_CARD_HEIGHT = 147
DISPLAY_CARD_DELTA_X = 136


class TorchRoyaleBot:
    """Live bot that captures device frames and executes strategy taps."""

    def __init__(
        self,
        config: Dict[str, Any],
        log_handler: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.config = config
        self.log_handler = log_handler
        self.pause_event = threading.Event()
        self.pause_event.set()
        self.should_run = True

        deck_cards = self._deck_from_config(config)
        self.device = AndroidDevice(
            device_serial=config.get("adb", {}).get("device_serial", ""),
            ip=config.get("adb", {}).get("ip", ""),
            adb_path=config.get("adb", {}).get("path"),
        )
        self.detector = LiveDetector(deck_cards)
        self.state_adapter = StateAdapter()
        visuals = config.get("visuals", {})
        self.visualizer = Visualizer(
            visuals.get("save_labels", False),
            visuals.get("save_images", False),
            visuals.get("show_images", False),
        )
        self.strategy = DTStrategy(
            checkpoint_path=config.get("strategy", {}).get("checkpoint"),
            device=config.get("strategy", {}).get("device", "cpu"),
        )
        self.auto_start = config.get("bot", {}).get("auto_start_game", False)
        self.load_deck = config.get("bot", {}).get("load_deck", False)
        self.loop_delay = float(config.get("ingame", {}).get("play_action", 1.0))
        self.last_action_at = 0.0

        if self.load_deck:
            self.device.load_deck([card.id_ for card in deck_cards])

    @staticmethod
    def _deck_from_config(config: Dict[str, Any]) -> list:
        names = config.get("bot", {}).get("deck")
        if names:
            cards = []
            for name in names:
                normalized = name.lower().replace("-", "_")
                card = CARD_OBJECTS.get(normalized)
                if card is None:
                    raise ValueError(f"Unknown card in bot.deck: {name}")
                cards.append(card)
            if len(cards) != 8:
                raise ValueError("bot.deck must contain exactly 8 cards.")
            return cards

        return [
            Cards.HOG_RIDER,
            Cards.MUSKETEER,
            Cards.ICE_GOLEM,
            Cards.ICE_SPIRIT,
            Cards.SKELETONS,
            Cards.CANNON,
            Cards.FIREBALL,
            Cards.THE_LOG,
        ]

    def _log(self, message: str) -> None:
        if self.log_handler:
            self.log_handler(message)

    @staticmethod
    def _get_card_centre(card_index: int) -> tuple[int, int]:
        x = (
            DISPLAY_CARD_INIT_X
            + DISPLAY_CARD_WIDTH / 2
            + card_index * DISPLAY_CARD_DELTA_X
        )
        y = DISPLAY_CARD_Y + DISPLAY_CARD_HEIGHT / 2
        return int(x), int(y)

    def _play_recommendation(
        self,
        strategy_state: Dict[str, Any],
        state,
    ) -> None:
        recommendation = self.strategy.recommend(strategy_state)
        if recommendation is None:
            self._log("No recommendation available.")
            return

        recommended_card, tile_x, tile_y = recommendation
        normalized = recommended_card.replace("-in-hand", "").replace("-", "_")
        hand_cards = list(state.cards[1:5])
        hand_index = None
        for index, card in enumerate(hand_cards):
            if card.name == normalized:
                hand_index = index
                break

        if hand_index is None:
            self._log(f"Recommended card not in hand: {recommended_card}")
            return

        mapper = self.strategy._mapper
        pixel_x, pixel_y = mapper.tile_to_pixel(tile_x, tile_y, center=True)
        card_x, card_y = self._get_card_centre(hand_index)

        self.device.click(card_x, card_y)
        time.sleep(0.15)
        self.device.click(pixel_x, pixel_y)
        self.last_action_at = time.time()
        self._log(
            f"Played {recommended_card} at tile ({tile_x}, {tile_y})."
        )

    def pause_or_resume(self) -> None:
        if self.pause_event.is_set():
            self.pause_event.clear()
            self._log("Bot paused.")
        else:
            self.pause_event.set()
            self._log("Bot resumed.")

    def stop(self) -> None:
        self.should_run = False

    def _handle_screen(self, state) -> bool:
        if state.screen == Screens.LOBBY and self.auto_start:
            click_xy = state.screen.click_xy
            if click_xy:
                self.device.click(*click_xy)
                self._log("Starting game from lobby.")
                time.sleep(2)
                return True
        elif state.screen in (Screens.END_OF_GAME, Screens.BYPASS_END_OF_GAME):
            click_xy = state.screen.click_xy
            if click_xy:
                self.device.click(*click_xy)
                self._log(f"Clicked {state.screen.name}.")
                time.sleep(2)
                self.strategy.reset_game()
                return True
        elif state.screen != Screens.IN_GAME:
            self._log(f"Current screen: {state.screen.name}")
            time.sleep(1)
            return True
        return False

    def run(self) -> None:
        started = False
        while self.should_run:
            if not self.pause_event.is_set():
                time.sleep(0.1)
                continue

            screenshot = self.device.take_screenshot()
            if not started:
                self._log("TorchRoyale live bot started.")
                started = True
            state = self.detector.run(screenshot)
            self.visualizer.run(screenshot, state)

            if self._handle_screen(state):
                continue

            strategy_state = self.state_adapter.to_strategy_state(state)
            if time.time() - self.last_action_at < self.loop_delay:
                time.sleep(0.1)
                continue

            self._play_recommendation(strategy_state, state)
            time.sleep(0.5)

        self._log("TorchRoyale live bot stopped.")
