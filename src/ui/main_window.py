"""Main window for the TorchRoyale desktop UI."""

from collections.abc import Callable
from threading import Thread
from typing import Optional
from typing import Protocol

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QCheckBox
from PyQt6.QtWidgets import QComboBox
from PyQt6.QtWidgets import QDoubleSpinBox
from PyQt6.QtWidgets import QLineEdit
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtWidgets import QTextEdit
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtWidgets import QWidget

from src.ui.animations import start_play_button_animation
from src.ui.gameplay_widget import ImageStreamWindow
from src.ui.layout_setup import setup_tabs
from src.ui.layout_setup import setup_top_bar
from src.ui.styles import set_styles
from src.ui.utils import AppConfig


class BotProtocol(Protocol):
    """Protocol for bot instances created by bot_factory."""

    def run(self) -> None: ...
    def stop(self) -> None: ...
    def pause_or_resume(self) -> None: ...


class MainWindow(QMainWindow):
    """
    Main application window for TorchRoyale.

    Attributes:
        config (AppConfig): Application configuration dictionary.
        actions (Optional[list[type]]): Action handler classes for the bot (currently unused).
        bot_factory (Optional[Callable[..., Optional[BotProtocol]]]): Factory to create bot instances.
        bot (Optional[BotProtocol]): The running bot instance.
        bot_thread (Optional[Thread]): Thread running the bot.
        is_running (bool): Whether the bot is currently running.
        log_message (pyqtSignal): Signal for emitting log messages to the UI.

        start_stop_button (QPushButton): Button to start or stop the bot.
        play_pause_button (QPushButton): Button to pause or resume the bot.
        server_id_label (QLabel): Label displaying the current bot status.
        visualize_tab (ImageStreamWindow): Widget displaying live gameplay frames.
        log_display (QTextEdit): Read-only text edit for runtime log messages.
        dashboard_status_value (QLabel): Label showing bot status in the dashboard.
        dashboard_visualizer_value (QLabel): Label showing visualizer state in dashboard.
        adb_ip_input (QLineEdit): Input field for ADB IP address.
        device_serial_input (QLineEdit): Input field for device serial.
        log_level_dropdown (QComboBox): Dropdown for selecting log verbosity.
        play_action_delay_input (QDoubleSpinBox): Input for action delay in seconds.
        load_deck_checkbox (QCheckBox): Checkbox to load deck on startup.
        auto_start_game_checkbox (QCheckBox): Checkbox to auto-start games.
        show_images_checkbox (QCheckBox): Checkbox to enable live visualizer.
        save_images_checkbox (QCheckBox): Checkbox to save annotated frames.
        save_labels_checkbox (QCheckBox): Checkbox to save detector labels.
    """

    log_message = pyqtSignal(str)

    def __init__(
        self,
        config: AppConfig,
        actions: Optional[list[type]] = None,
        bot_factory: Optional[Callable[..., Optional[BotProtocol]]] = None,
    ) -> None:
        """
        Initialize the main window with config and optional bot factory.

        Args:
            config: Application configuration.
            actions: Action handler classes (unused, kept for compatibility).
            bot_factory: Factory to create bot instances.
        Returns:
            None
        """
        super().__init__()
        self.config: AppConfig = config
        self.actions: Optional[list[type]] = actions
        self.bot_factory: Optional[Callable[..., Optional[BotProtocol]]] = bot_factory
        self.bot: Optional[BotProtocol] = None
        self.bot_thread: Optional[Thread] = None
        self.is_running = False

        self.setWindowTitle("TorchRoyale")
        self.setGeometry(100, 100, 1280, 820)

        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        (
            top_bar,
            self.start_stop_button,
            self.play_pause_button,
            self.server_id_label,
        ) = setup_top_bar(self)

        (
            tab_widget,
            self.visualize_tab,
            self.log_display,
            self.dashboard_status_value,
            self.dashboard_visualizer_value,
            self.adb_ip_input,
            self.device_serial_input,
            self.log_level_dropdown,
            self.play_action_delay_input,
            self.load_deck_checkbox,
            self.auto_start_game_checkbox,
            self.show_images_checkbox,
            self.save_images_checkbox,
            self.save_labels_checkbox,
        ) = setup_tabs(self)

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

    def update_config(self) -> AppConfig:
        """
        Update the config dictionary from current UI widget values.
        Args:
            None
        Returns:
            AppConfig: The updated configuration dictionary.
        """
        self.config.setdefault("visuals", {})
        self.config.setdefault("bot", {})
        self.config.setdefault("ingame", {})
        self.config.setdefault("adb", {})

        self.config["visuals"]["save_labels"] = self.save_labels_checkbox.isChecked()
        self.config["visuals"]["save_images"] = self.save_images_checkbox.isChecked()
        self.config["visuals"]["show_images"] = self.show_images_checkbox.isChecked()
        self.visualize_tab.update_active_state(self.config["visuals"]["show_images"])
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

    def _build_bot_instance(self) -> Optional[BotProtocol]:
        """
        Build a bot instance using the bot factory.

        Args:
            None

        Returns:
            The created bot instance, or None if creation fails.
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
            self.bot: Optional[BotProtocol] = self._build_bot_instance()
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
