"""Main window for the TorchRoyale desktop UI."""

from threading import Thread
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtWidgets import QWidget

from src.ui.animations import start_play_button_animation
from src.ui.layout_setup import setup_tabs
from src.ui.layout_setup import setup_top_bar
from src.ui.styles import set_styles


class MainWindow(QMainWindow):
    """
    Main application window for TorchRoyale.

    Attributes:
        config (Dict[str, Any]): Application configuration dictionary.
        actions (Optional[Any]): Action handlers for the bot (currently unused).
        bot_factory (Optional[Callable[..., Any]]): Factory to create bot instances.
        bot (Optional[Any]): The running bot instance.
        bot_thread (Optional[Thread]): Thread running the bot.
        is_running (bool): Whether the bot is currently running.
        log_message (pyqtSignal): Signal for emitting log messages to the UI.
    """

    log_message = pyqtSignal(str)

    def __init__(
        self,
        config: Dict[str, Any],
        actions: Optional[Any] = None,
        bot_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        """
        Initialize the main window with config and optional bot factory.

        Args:
            config (Dict[str, Any]): Application configuration.
            actions (Optional[Any]): Action handlers (unused, kept for compatibility).
            bot_factory (Optional[Callable[..., Any]]): Factory to create bot instances.
        Returns:
            None
        """
        super().__init__()
        self.config = config
        self.actions = actions
        self.bot_factory = bot_factory
        self.bot: Optional[Any] = None
        self.bot_thread: Optional[Thread] = None
        self.is_running = False

        self.setWindowTitle("TorchRoyale")
        self.setGeometry(100, 100, 1280, 820)

        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        top_bar = setup_top_bar(self)
        tab_widget = setup_tabs(self)

        main_layout.addWidget(top_bar)
        main_layout.addWidget(tab_widget)

        set_styles(self)
        start_play_button_animation(self)
        self.log_message.connect(self._append_log_message)
        self._configure_runtime_state()

    def _configure_runtime_state(self) -> None:
        """
        Reflect whether a live bot backend is available in this session.

        Disables UI controls and shows "UI only" status if no bot factory is provided.
        Args:
            None
        Returns:
            None
        """
        if self.bot_factory is not None:
            return

        self.start_stop_button.setEnabled(False)
        self.play_pause_button.setEnabled(False)
        self.start_stop_button.setToolTip(
            "Live bot backend has not been migrated into TorchRoyale yet."
        )
        self.server_id_label.setText("Status: UI only")
        self.append_log(
            "Live bot backend is not configured for TorchRoyale yet. "
            "UI controls are disabled."
        )
        if hasattr(self, "dashboard_status_value"):
            self.dashboard_status_value.setText("UI only")

    def log_handler_function(self, message: str) -> None:
        """
        Emit a log message to be displayed in the UI.

        Args:
            message (str): The log message to emit.
        Returns:
            None
        """
        self.log_message.emit(message)

    def _append_log_message(self, formatted_message: str) -> None:
        """
        Append a formatted log message to the runtime log display.

        Args:
            formatted_message (str): The formatted message to append.
        Returns:
            None
        """
        self.log_display.append(formatted_message)
        scrollbar = self.log_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def toggle_start_stop(self) -> None:
        """
        Toggle the bot between started and stopped states.

        If running, stops the bot and restarts the glow animation.
        If stopped, starts the bot and stops the glow animation.
        Args:
            None
        Returns:
            None
        """
        if self.is_running:
            self.stop_bot()
            self.glow_animation.start()
        else:
            self.start_bot()
            self.glow_animation.stop()

    def toggle_pause_resume_and_display(self) -> None:
        """
        Toggle the bot between paused and resumed states.

        Updates the button text between "Pause" and "Resume" accordingly.
        Args:
            None
        Returns:
            None
        """
        if not self.bot or not hasattr(self.bot, "pause_or_resume"):
            return
        pause_event = getattr(self.bot, "pause_event", None)
        if pause_event is not None and pause_event.is_set():
            self.play_pause_button.setText("Resume")
        else:
            self.play_pause_button.setText("Pause")
        self.bot.pause_or_resume()

    def start_bot(self) -> None:
        """
        Start the bot in a separate daemon thread.

        Updates UI controls to reflect running state.
        Args:
            None
        Returns:
            None
        """
        if self.is_running:
            return
        if self.bot_factory is None:
            return

        self.update_config()
        self.is_running = True
        self.bot_thread = Thread(target=self.bot_task, daemon=True)
        self.bot_thread.start()
        self.start_stop_button.setText("Stop")
        self.play_pause_button.show()
        self.server_id_label.setText("Status: Running")
        if hasattr(self, "dashboard_status_value"):
            self.dashboard_status_value.setText("Running")

    def stop_bot(self) -> None:
        """
        Stop the running bot and update UI controls to reflect stopped state.
        Args:
            None
        Returns:
            None
        """
        if self.bot and hasattr(self.bot, "stop"):
            self.bot.stop()
        self.is_running = False
        self.start_stop_button.setText("Start")
        self.play_pause_button.setText("Pause")
        self.play_pause_button.hide()
        self.server_id_label.setText("Status: Stopped")
        if hasattr(self, "dashboard_status_value"):
            self.dashboard_status_value.setText("Stopped")

    def restart_bot(self) -> None:
        """
        Restart the bot by stopping it then starting it again with current config.
        Args:
            None
        Returns:
            None
        """
        if self.is_running:
            self.stop_bot()
        self.update_config()
        self.start_bot()

    def update_config(self) -> Dict[str, Any]:
        """
        Update the config dictionary from current UI widget values.
        Args:
            None
        Returns:
            Dict[str, Any]: The updated configuration dictionary.
        """
        self.config.setdefault("visuals", {})
        self.config.setdefault("bot", {})
        self.config.setdefault("ingame", {})
        self.config.setdefault("adb", {})

        self.config["visuals"]["save_labels"] = self.save_labels_checkbox.isChecked()
        self.config["visuals"]["save_images"] = self.save_images_checkbox.isChecked()
        self.config["visuals"]["show_images"] = self.show_images_checkbox.isChecked()
        self.visualize_tab.update_active_state(self.config["visuals"]["show_images"])
        if hasattr(self, "dashboard_visualizer_value"):
            self.dashboard_visualizer_value.setText(
                "On" if self.config["visuals"]["show_images"] else "Off"
            )
        self.config["bot"]["load_deck"] = self.load_deck_checkbox.isChecked()
        self.config["bot"]["auto_start_game"] = (
            self.auto_start_game_checkbox.isChecked()
        )
        self.config["bot"]["log_level"] = self.log_level_dropdown.currentText()
        self.config["ingame"]["play_action"] = round(
            float(self.play_action_delay_input.value()), 2
        )
        self.config["adb"]["ip"] = self.adb_ip_input.text()
        self.config["adb"]["device_serial"] = self.device_serial_input.text()
        return self.config

    def _build_bot_instance(self) -> Any:
        """
        Build a bot instance using the bot factory.
        Args:
            None
        Returns:
            Any: The created bot instance.
        """
        try:
            return self.bot_factory(
                actions=self.actions,
                config=self.config,
                log_handler=self.log_handler_function,
            )
        except TypeError:
            return self.bot_factory(self.actions, self.config)

    def bot_task(self) -> None:
        """
        Run the bot task in a separate thread.

        Connects frame signals to the visualizer and runs the bot.
        Handles exceptions and ensures bot stops on completion.
        Args:
            None
        Returns:
            None
        """
        try:
            self.bot = self._build_bot_instance()
            visualizer = getattr(self.bot, "visualizer", None)
            frame_ready = getattr(visualizer, "frame_ready", None)
            if frame_ready is not None and hasattr(frame_ready, "connect"):
                frame_ready.connect(self.visualize_tab.update_frame)
            self.bot.run()
        except Exception as exc:
            self.append_log(f"Bot error: {exc}")
        finally:
            self.stop_bot()

    def append_log(self, message: str) -> None:
        """
        Append a log message to the runtime log display.

        Args:
            message (str): The message to append.
        Returns:
            None
        """
        self.log_display.append(message)    
