"""
Style definitions for the TorchRoyale GUI.
"""


def set_styles(window) -> None:
    """Apply a Catppuccin-inspired theme to the main window."""
    window.setStyleSheet(
        """
        QMainWindow {
            background-color: #11111b;
        }

        QWidget {
            background-color: #11111b;
            color: #cdd6f4;
            font-family: "SF Pro Text", "Inter", "Segoe UI", sans-serif;
            font-size: 13px;
        }

        QFrame#topBar {
            background-color: #181825;
            border: 1px solid #313244;
            border-radius: 18px;
        }

        QLabel#appTitle {
            color: #f5e0dc;
            font-size: 26px;
            font-weight: 700;
        }

        QLabel#appSubtitle {
            color: #a6adc8;
            font-size: 13px;
        }

        QLabel#statusLabel {
            color: #89b4fa;
            font-weight: 600;
        }

        QTabWidget::pane {
            border: none;
            top: 8px;
        }

        QTabBar::tab {
            background: #181825;
            color: #bac2de;
            border: 1px solid #313244;
            border-radius: 12px;
            padding: 10px 16px;
            margin-right: 8px;
            min-width: 110px;
        }

        QTabBar::tab:selected {
            background: #313244;
            color: #f5e0dc;
        }

        QTabBar::tab:hover:!selected {
            background: #1e1e2e;
        }

        QFrame#sectionCard {
            background-color: #181825;
            border: 1px solid #313244;
            border-radius: 18px;
        }

        QLabel#sectionTitle {
            color: #f5c2e7;
            font-size: 16px;
            font-weight: 700;
        }

        QLabel#sectionDescription {
            color: #a6adc8;
            line-height: 1.3em;
        }

        QLabel#metricLabel {
            color: #a6adc8;
            font-size: 12px;
            text-transform: uppercase;
            font-weight: 600;
        }

        QLabel#metricValue {
            color: #cdd6f4;
            font-size: 18px;
            font-weight: 700;
            background-color: #1e1e2e;
            border: 1px solid #313244;
            border-radius: 12px;
            padding: 12px 14px;
        }

        QTextEdit, QLineEdit, QComboBox, QDoubleSpinBox {
            background-color: #1e1e2e;
            color: #cdd6f4;
            border: 1px solid #45475a;
            border-radius: 12px;
            padding: 10px 12px;
            selection-background-color: #74c7ec;
            selection-color: #11111b;
        }

        QTextEdit#logDisplay {
            font-family: "SF Mono", "JetBrains Mono", "Menlo", monospace;
            font-size: 12px;
        }

        QComboBox::drop-down {
            border: none;
            width: 24px;
        }

        QComboBox QAbstractItemView {
            background-color: #1e1e2e;
            color: #cdd6f4;
            border: 1px solid #45475a;
            selection-background-color: #89b4fa;
            selection-color: #11111b;
        }

        QGroupBox {
            border: none;
        }

        QCheckBox {
            color: #cdd6f4;
            spacing: 10px;
        }

        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border-radius: 6px;
            border: 1px solid #6c7086;
            background-color: #1e1e2e;
        }

        QCheckBox::indicator:checked {
            background-color: #a6e3a1;
            border: 1px solid #a6e3a1;
        }

        QPushButton {
            border: 1px solid #45475a;
            background-color: #313244;
            color: #cdd6f4;
            padding: 10px 16px;
            border-radius: 12px;
            font-weight: 600;
        }

        QPushButton:hover {
            background-color: #45475a;
        }

        QPushButton:pressed {
            background-color: #585b70;
        }

        QPushButton#primaryControlButton,
        QPushButton#primaryButton {
            background-color: #89b4fa;
            color: #11111b;
            border: 1px solid #89b4fa;
        }

        QPushButton#primaryControlButton:hover,
        QPushButton#primaryButton:hover {
            background-color: #74c7ec;
            border: 1px solid #74c7ec;
        }

        QPushButton#secondaryControlButton,
        QPushButton#secondaryButton {
            background-color: #1e1e2e;
            color: #cdd6f4;
            border: 1px solid #585b70;
        }

        QScrollArea {
            border: none;
        }

        QScrollBar:vertical {
            background: #11111b;
            width: 10px;
            margin: 2px;
        }

        QScrollBar::handle:vertical {
            background: #585b70;
            border-radius: 5px;
            min-height: 24px;
        }

        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {
            height: 0px;
        }
        """
    )
