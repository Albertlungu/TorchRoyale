"""
Button animation effects for TorchRoyale GUI.
"""

from typing import TYPE_CHECKING

from PyQt6.QtCore import QEasingCurve
from PyQt6.QtCore import QPropertyAnimation
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGraphicsDropShadowEffect

if TYPE_CHECKING:
    from src.ui.main_window import MainWindow


def start_play_button_animation(window: "MainWindow") -> None:
    """
    Add a glowing effect animation to the start/stop button.

    Args:
        window: The main window containing the start/stop button.
    Returns:
        None
    """
    window.glow_effect = QGraphicsDropShadowEffect(window)
    window.glow_effect.setBlurRadius(10)
    window.glow_effect.setColor(Qt.GlobalColor.cyan)
    window.glow_effect.setOffset(0, 0)
    window.start_stop_button.setGraphicsEffect(window.glow_effect)

    _start_glow_animation(window)


def _start_glow_animation(window: "MainWindow") -> None:
    """
    Create and start the glow animation on the start/stop button.

    Args:
        window: The main window containing the glow effect.
    Returns:
        None
    """
    window.glow_animation = QPropertyAnimation(
        window.glow_effect, b"blurRadius"
    )
    window.glow_animation.setStartValue(0)
    window.glow_animation.setEndValue(25)
    window.glow_animation.setDuration(2000)
    window.glow_animation.setEasingCurve(QEasingCurve.Type.SineCurve)
    window.glow_animation.setLoopCount(-1)
    window.glow_animation.start()
