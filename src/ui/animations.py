"""
Button animation effects for TorchRoyale GUI.
"""

from PyQt6.QtCore import QEasingCurve
from PyQt6.QtCore import QPropertyAnimation
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGraphicsDropShadowEffect


def start_play_button_animation(window) -> None:
    """Add a glowing effect animation to the start/stop button."""
    window.glow_effect = QGraphicsDropShadowEffect(window)
    window.glow_effect.setBlurRadius(10)
    window.glow_effect.setColor(Qt.GlobalColor.cyan)
    window.glow_effect.setOffset(0, 0)
    window.start_stop_button.setGraphicsEffect(window.glow_effect)

    _start_glow_animation(window)


def _start_glow_animation(window) -> None:
    """Create and start the glow animation."""
    window.glow_animation = QPropertyAnimation(
        window.glow_effect, b"blurRadius"
    )
    window.glow_animation.setStartValue(0)
    window.glow_animation.setEndValue(25)
    window.glow_animation.setDuration(2000)
    window.glow_animation.setEasingCurve(QEasingCurve.Type.SineCurve)
    window.glow_animation.setLoopCount(-1)
    window.glow_animation.start()
