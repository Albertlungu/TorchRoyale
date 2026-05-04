"""
Image stream widget for displaying game frames.
"""

from PyQt6.QtGui import QImage
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtWidgets import QWidget


class ImageStreamWindow(QWidget):
    """
    Widget for displaying annotated game frames from the live bot.

    Attributes:
        image (QLabel): Label displaying the current game frame.
        inactiveIndicator (QLabel): Label shown when the visualizer is disabled.
    """

    def __init__(self) -> None:
        """
        Initialize the image stream window.

        Sets up the image display and inactive indicator with appropriate styling.
        """
        super().__init__()
        self.setStyleSheet("background-color: #181825; border: none;")

        self.image = QLabel(self)
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumHeight(420)
        self.image.setStyleSheet(
            "background-color: #1e1e2e; border: 1px solid #313244; border-radius: 14px;"
        )
        self.inactiveIndicator = QLabel(self)
        self.inactiveIndicator.setText(
            "Visualizer is disabled. Enable live visualizer in Settings."
        )
        self.inactiveIndicator.setWordWrap(True)
        self.inactiveIndicator.setStyleSheet(
            " ".join(
                [
                    "background-color: #313244;",
                    "color: #f9e2af;",
                    "padding: 10px 12px;",
                    "border: 1px solid #45475a;",
                    "border-radius: 10px;",
                    "height: fit-content;",
                    "width: fit-content;",
                ]
            )
        )
        self.inactiveIndicator.setMaximumHeight(48)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(self.inactiveIndicator)
        layout.addWidget(self.image)
        self.setLayout(layout)

    def update_frame(self, annotated_image) -> None:
        """
        Update the displayed frame with a new annotated image.

        Args:
            annotated_image (np.ndarray): The annotated image array (BGR/RGB format).
        Returns:
            None
        """
        height, width, _ = annotated_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(
            annotated_image.data.tobytes(),
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )

        pixmap = QPixmap.fromImage(q_image)
        self.image.setPixmap(
            pixmap.scaled(
                self.image.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def update_active_state(self, active: bool) -> None:
        """
        Update the visibility of the inactive indicator based on active state.

        Args:
            active (bool): Whether the visualizer is active (True = show image, False = show indicator).
        Returns:
            None
        """
        if not active:
            self.inactiveIndicator.show()
        else:
            self.inactiveIndicator.hide()
        self.image.clear()
