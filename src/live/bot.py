"""Live TorchRoyale bot runtime for the migrated desktop UI."""

import threading
import time
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

from src.actions import ArchersAction
from src.actions import ArrowsAction
from src.actions import FireballAction
from src.actions import GiantAction
from src.actions import KnightAction
from src.actions import MinionsAction
from src.actions import MinipekkaAction
from src.actions import MusketeerAction
from src.capture.android_device import AndroidDevice
from src.live.detector import LiveDetector
from src.live.visualizer import Visualizer
from src.namespaces.cards import CARD_OBJECTS
from src.namespaces.cards import Cards
from src.namespaces.screens import Screens


DISPLAY_CARD_Y = 1067
DISPLAY_CARD_INIT_X = 164
DISPLAY_CARD_WIDTH = 117
DISPLAY_CARD_HEIGHT = 147
DISPLAY_CARD_DELTA_X = 136
DISPLAY_HEIGHT = 1280
TILE_HEIGHT = 27.6
TILE_WIDTH = 34
TILE_INIT_X = 52
TILE_INIT_Y = 296


class TorchRoyaleBot:
    """Live bot that captures device frames and executes strategy taps.
    
    Attributes:
        config (Dict[str, Any]): Application configuration dictionary
        log_handler (Optional[Callable[[str], None]]): Callback for logging to the UI
        pause_event (threading.Event): Threading event for pause/resume control
        should_run (bool): Flag to terminate the main loop
        actions (list): List of heuristic Action classes mapped to the deck
        device (AndroidDevice): Interface for ADB connection to the Android device
        detector (LiveDetector): Pipeline for unit, hand, and OCR detection
        visualizer (Visualizer): Draws annotations and emits frame updates to the UI
        auto_start (bool): automatically start the game
        load_deck (bool): Whether to load the configured deck on startup
        loop_delay (float): Seconds to wait after playing a card
    """

    def __init__(
        self,
        config: Dict[str, Any],
        log_handler: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        Initialize the bot with configuration and detection pipelines
        
        Args:
            config (Dict[str, Any]): Configuration for device and gameplay settings
            log_handler (Optional[Callable[[str], None]]): Callback function for UI logging
        Returns:
            None
        """
        self.config = config
        self.log_handler = log_handler
        self.pause_event = threading.Event()
        self.pause_event.set()
        self.should_run = True

        deck_cards = self._deck_from_config(config)
        self.actions = self._actions_for_deck(deck_cards)
        self.device = AndroidDevice(
            device_serial=config.get("adb", {}).get("device_serial", ""),
            ip=config.get("adb", {}).get("ip", ""),
            adb_path=config.get("adb", {}).get("path"),
        )
        self.detector = LiveDetector(deck_cards)
        visuals = config.get("visuals", {})
        self.visualizer = Visualizer(
            visuals.get("save_labels", False),
            visuals.get("save_images", False),
            visuals.get("show_images", False),
        )
        self.auto_start = config.get("bot", {}).get("auto_start_game", False)
        self.load_deck = config.get("bot", {}).get("load_deck", False)
        self.loop_delay = float(config.get("ingame", {}).get("play_action", 1.0))

        if self.load_deck:
            self.device.load_deck([card.id_ for card in deck_cards])

    @staticmethod
    def _deck_from_config(config: Dict[str, Any]) -> list:
        """
        Extract and validate the deck list from the configuration
        
        Args:
            config (Dict[str, Any]): Application configuration dictionary
        Returns:
            (list): List of Card objects representing the deck
        """
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
            Cards.MINIONS,
            Cards.GIANT,
            Cards.KNIGHT,
            Cards.ARROWS,
            Cards.ARCHERS,
            Cards.MUSKETEER,
            Cards.FIREBALL,
            Cards.MINIPEKKA,
        ]

    @staticmethod
    def _actions_for_deck(deck_cards: list):
        """
        Map deck cards to their corresponding heuristic Action classes
        
        Args:
            deck_cards (list): List of Card objects for the player's deck
        Returns:
            (list): List of Action classes ready to be instantiated
        """
        action_by_card_name = {
            Cards.MINIONS.name: MinionsAction,
            Cards.GIANT.name: GiantAction,
            Cards.KNIGHT.name: KnightAction,
            Cards.ARROWS.name: ArrowsAction,
            Cards.ARCHERS.name: ArchersAction,
            Cards.MUSKETEER.name: MusketeerAction,
            Cards.FIREBALL.name: FireballAction,
            Cards.MINIPEKKA.name: MinipekkaAction,
        }
        actions = []
        for card in deck_cards:
            action = action_by_card_name.get(card.name)
            if action is None:
                raise ValueError(f"No live action heuristic exists for card: {card.name}")
            actions.append(action)
        return actions

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

    @staticmethod
    def _get_tile_centre(tile_x: int, tile_y: int) -> tuple[int, int]:
        x = TILE_INIT_X + (tile_x + 0.5) * TILE_WIDTH
        y = DISPLAY_HEIGHT - TILE_INIT_Y - (tile_y + 0.5) * TILE_HEIGHT
        return int(x), int(y)

    def _get_actions(self, state):
        """
        Generate all valid heuristic actions for ready cards and board tiles
        
        Args:
            state (State): Current game state including ready slots and hand cards
        Returns:
            (list): List of instantiated Action objects
        """
        valid_actions = []
        for index in state.ready:
            card = state.cards[index + 1]
            if state.numbers.elixir.number < card.cost:
                continue
            action_factory = None
            for action in self.actions:
                if action.CARD.name == card.name:
                    action_factory = action
                    break
            if action_factory is None:
                continue
            for tile_x in range(18):
                for tile_y in range(32):
                    valid_actions.append(action_factory(index, tile_x, tile_y))
        return valid_actions

    def _play_best_action(self, state) -> None:
        """
        Score available actions and execute the highest-scoring tap sequence
        
        Args:
            state (State): Current game state for heuristic evaluation
        Returns:
            None
        """
        actions = self._get_actions(state)
        if not actions:
            self._log("No actions available.")
            time.sleep(self.loop_delay)
            return

        best_score = [0]
        best_action = None
        for action in actions:
            score = action.calculate_score(state)
            if score > best_score:
                best_score = score
                best_action = action

        if best_action is None or best_score[0] == 0:
            self._log("No good actions available.")
            time.sleep(self.loop_delay)
            return

        pixel_x, pixel_y = self._get_tile_centre(
            best_action.tile_x,
            best_action.tile_y,
        )
        card_x, card_y = self._get_card_centre(best_action.index)
        self.device.click(card_x, card_y)
        time.sleep(0.15)
        self.device.click(pixel_x, pixel_y)
        self._log(f"Played {best_action} with score {best_score}.")
        time.sleep(self.loop_delay)

    def pause_or_resume(self) -> None:
        """
        Toggle the pause/resume state of the bot.
        
        Args:
            None
        Returns:
            None
        """
        if self.pause_event.is_set():
            self.pause_event.clear()
            self._log("Bot paused.")
        else:
            self.pause_event.set()
            self._log("Bot resumed.")

    def stop(self) -> None:
        """
        Signal the main loop to terminate
        
        Args:
            None
        Returns:
            None
        """
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
            #capture
            screenshot = self.device.take_screenshot()
            if not started:
                self._log("TorchRoyale live bot started.")
                started = True
            #detect + OCR
            state = self.detector.run(screenshot)
            #Visualizer
            self.visualizer.run(screenshot, state)

            if self._handle_screen(state):
                continue
            #Score + Tap
            self._play_best_action(state)
            time.sleep(0.5)

        self._log("TorchRoyale live bot stopped.")
