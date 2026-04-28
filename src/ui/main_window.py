"""
Main window for TorchRoyale GUI application.

Provides the primary application window with bot controls, logging,
visualization, and settings tabs using PyQt6.
"""

from typing import Any, Dict, Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtWidgets import QWidget

from src.ui.animations import start_play_button_animation
from src.ui.layout_setup import setup_tabs
from src.ui.layout_setup import setup_top_bar
from src.ui.styles import set_styles


class MainWindow(QMainWindow):
    """Main application window for TorchRoyale."""

    log_message = pyqtSignal(str)

    def __init__(self, config: Dict[str, Any], actions: Optional[Any] = None) -> None:
        """Initialize the main window."""
        super().__init__()
        self.config = config
        self.actions = actions
        self.bot: Optional[Any] = None
        self.bot_thread: Optional[Any] = None
        self.is_running = False

        self.setWindowTitle("TorchRoyale")
        self.setGeometry(100, 100, 900, 600)

        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        top_bar = setup_top_bar(self)
        tab_widget = setup_tabs(self)

        main_layout.addWidget(top_bar)
        main_layout.addWidget(tab_widget)

        set_styles(self)
        start_play_button_animation(self)
        self.log_message.connect(self._append_log_message)

    def log_handler_function(self, message: str) -> None:
        """Handle log messages from the bot."""
        self.log_message.emit(message)

    def _append_log_message(self, formatted_message: str) -> None:
        """Append a formatted log message to the log display."""
        self.log_display.append(formatted_message)
        self.log_display.verticalScrollBar().setValue(
            self.log_display.verticalScrollBar().maximum()
        )

    def toggle_start_stop(self) -> None:
        """Toggle between starting and stopping the bot."""
        if self.is_running:
            self.stop_bot()
            self.glow_animation.start()
        else:
            self.start_bot()
            self.glow_animation.stop()

    def toggle_pause_resume_and_display(self) -> None:
        """Toggle between pause and resume states."""
        if not self.bot:
            return
        if self.bot.pause_event.is_set():
            self.play_pause_button.setText("▶")
        else:
            self.play_pause_button.setText("⏸️")
        self.bot.pause_or_resume()

    def start_bot(self) -> None:
        """Start the bot in a separate thread."""
        if self.is_running:
            return
        self.update_config()
        self.is_running = True
        self.bot_thread = Thread(target=self.bot_task)
        self.bot_thread.daemon = True
        self.bot_thread.start()
        self.start_stop_button.setText("■")
        self.play_pause_button.show()
        self.server_id_label.setText("Status - Running")

    def stop_bot(self) -> None:
        """Stop the running bot."""
        if self.bot:
            self.bot.stop()
        self.is_running = False
        self.start_stop_button.setText("▶")
        self.play_pause_button.hide()
        self.server_id_label.setText("Status - Stopped")

    def restart_bot(self) -> None:
        """Restart the bot by stopping and starting again."""
        if self.is_running:
            self.stop_bot()
        self.update_config()
        self.start_bot()

    def update_config(self) -> Dict[str, Any]:
        """Update configuration from UI controls."""
        self.config["visuals"][
            "save_labels"
        ] = self.save_labels_checkbox.isChecked()
        self.config["visuals"][
            "save_images"
        ] = self.save_images_checkbox.isChecked()
        self.config["visuals"][
            "show_images"
        ] = self.show_images_checkbox.isChecked()
        self.visualize_tab.update_active_state(
            self.config["visuals"]["show_images"]
        )
        self.config["bot"]["load_deck"] = self.load_deck_checkbox.isChecked()
        self.config["bot"][
            "auto_start_game"
        ] = self.auto_start_game_checkbox.isChecked()
        self.config["bot"]["log_level"] = self.log_level_dropdown.currentText()
        self.config["ingame"]["play_action"] = round(
            float(self.play_action_delay_input.value()), 2
        )
        self.config["adb"]["ip"] = self.adb_ip_input.text()
        self.config["adb"]["device_serial"] = self.device_serial_input.text()
        return self.config

    def bot_task(self) -> None:
        """Run the bot task in a separate thread."""
        try:
            self.bot = Bot(actions=self.actions, config=self.config)
            self.bot.visualizer.frame_ready.connect(
                self.visualize_tab.update_frame
            )
            self.bot.run()
            self.stop_bot()
        except Exception as e:
            self.stop_bot()
            raise

    def append_log(self, message: str) -> None:
        """Append a message to the log display."""
        self.log_display.append(message)
