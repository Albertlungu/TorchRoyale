"""
UI layout setup for TorchRoyale main window.

Defines the top bar and the simplified dashboard/settings interface.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QCheckBox
from PyQt6.QtWidgets import QComboBox
from PyQt6.QtWidgets import QDoubleSpinBox
from PyQt6.QtWidgets import QFormLayout
from PyQt6.QtWidgets import QFrame
from PyQt6.QtWidgets import QGridLayout
from PyQt6.QtWidgets import QHBoxLayout
from PyQt6.QtWidgets import QLabel
from PyQt6.QtWidgets import QLineEdit
from PyQt6.QtWidgets import QPushButton
from PyQt6.QtWidgets import QScrollArea
from PyQt6.QtWidgets import QTabWidget
from PyQt6.QtWidgets import QTextEdit
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtWidgets import QWidget

from src.ui.gameplay_widget import ImageStreamWindow
from src.ui.utils import AppConfig
from src.ui.utils import save_config

if TYPE_CHECKING:
    from src.ui.main_window import MainWindow


def _wrap_section(title: str, description: str, widget: QWidget) -> QFrame:
    """
    Wrap a widget in a styled section card with title and description.

    Args:
        title (str): The section title displayed at the top.
        description (str): Optional description text below the title.
        widget (QWidget): The widget to embed inside the section.

    Returns:
        QFrame: A styled frame containing the title, description, and widget.
    """
    section = QFrame()
    section.setObjectName("sectionCard")
    layout = QVBoxLayout(section)
    layout.setContentsMargins(18, 18, 18, 18)
    layout.setSpacing(0)

    title_label = QLabel(title)
    title_label.setObjectName("sectionTitle")
    layout.addWidget(title_label)

    if description:
        description_label = QLabel(description)
        description_label.setObjectName("sectionDescription")
        description_label.setWordWrap(True)
        layout.addWidget(description_label)

    layout.addWidget(widget)
    return section


def _make_read_only_field(label_text: str, value_widget: QWidget) -> QWidget:
    """
    Create a read-only field with a label and value widget.

    Args:
        label_text (str): The label text displayed above the value.
        value_widget (QWidget): The widget displaying the value (e.g., QLabel).

    Returns:
        QWidget: A container widget with label and value arranged vertically.
    """
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)

    label = QLabel(label_text)
    label.setObjectName("metricLabel")
    layout.addWidget(label)
    layout.addWidget(value_widget)
    return container


def setup_top_bar(main_window: MainWindow) -> tuple[QFrame, QPushButton, QPushButton, QLabel]:  # type: ignore[type-arg]
    """
    Create the top bar with title, status, and transport controls.

    Args:
        main_window (MainWindow): The main window instance whose toggle methods
            are connected to button signals.

    Returns:
        tuple[QFrame, QPushButton, QPushButton, QLabel]:
            The configured top bar widget, start/stop button, play/pause button,
            and status label.
    """
    top_bar = QFrame()
    top_bar.setObjectName("topBar")
    top_bar_layout = QHBoxLayout(top_bar)
    top_bar_layout.setContentsMargins(20, 18, 20, 18)
    top_bar_layout.setSpacing(18)

    text_layout = QVBoxLayout()
    text_layout.setSpacing(2)

    server_name = QLabel("TorchRoyale")
    server_name.setObjectName("appTitle")
    text_layout.addWidget(server_name)

    server_details = QLabel("Minimal live control for Clash Royale inference")
    server_details.setObjectName("appSubtitle")
    text_layout.addWidget(server_details)

    server_id_label = QLabel("Status: Stopped")
    server_id_label.setObjectName("statusLabel")
    text_layout.addWidget(server_id_label)

    top_bar_layout.addLayout(text_layout)
    top_bar_layout.addStretch()

    button_layout = QHBoxLayout()
    button_layout.setSpacing(10)
    button_layout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

    play_pause_button = QPushButton("Pause")
    play_pause_button.setObjectName("secondaryControlButton")
    play_pause_button.setFont(QFont("Arial", 11))
    play_pause_button.clicked.connect(main_window.toggle_pause_resume_and_display)
    play_pause_button.hide()
    button_layout.addWidget(play_pause_button)

    start_stop_button = QPushButton("Start")
    start_stop_button.setObjectName("primaryControlButton")
    start_stop_button.setFont(QFont("Arial", 11))
    start_stop_button.clicked.connect(main_window.toggle_start_stop)
    button_layout.addWidget(start_stop_button)

    top_bar_layout.addLayout(button_layout)
    return top_bar, start_stop_button, play_pause_button, server_id_label


def _build_dashboard_tab(
    main_window: MainWindow,
) -> tuple[QWidget, ImageStreamWindow, QTextEdit, QLabel, QLabel]:
    """
    Build the Dashboard tab with live overlay, quick controls, and runtime log.

    Args:
        main_window (MainWindow): The main window instance whose config and
            update method are referenced.

    Returns:
        tuple[QWidget, ImageStreamWindow, QTextEdit, QLabel, QLabel]:
            The Dashboard tab widget, image stream widget, log display,
            status value label, and visualizer value label.
    """
    dashboard = QWidget()
    dashboard_layout = QGridLayout(dashboard)
    dashboard_layout.setContentsMargins(0, 0, 0, 0)
    dashboard_layout.setHorizontalSpacing(16)
    dashboard_layout.setVerticalSpacing(16)

    visualize_tab = ImageStreamWindow()
    visualize_tab.update_active_state(main_window.config["visuals"]["show_images"])
    visual_section = _wrap_section(
        "Live Overlay",
        "Annotated gameplay frames, including hand slots and on-field detections.",
        visualize_tab,
    )

    log_display = QTextEdit()
    log_display.setReadOnly(True)
    log_display.setObjectName("logDisplay")
    logs_section = _wrap_section(
        "Runtime Log",
        "Bot lifecycle, detector output, and runtime errors appear here.",
        log_display,
    )

    quick_actions = QWidget()
    quick_actions_layout = QGridLayout(quick_actions)
    quick_actions_layout.setContentsMargins(0, 0, 0, 0)
    quick_actions_layout.setHorizontalSpacing(12)
    quick_actions_layout.setVerticalSpacing(12)

    dashboard_status_value = QLabel("Stopped")
    dashboard_status_value.setObjectName("metricValue")
    dashboard_visualizer_value = QLabel(
        "On" if main_window.config["visuals"]["show_images"] else "Off"
    )
    dashboard_visualizer_value.setObjectName("metricValue")

    quick_actions_layout.addWidget(
        _make_read_only_field("Bot status", dashboard_status_value), 0, 0
    )
    quick_actions_layout.addWidget(
        _make_read_only_field("Visualizer", dashboard_visualizer_value), 0, 1
    )

    save_button = QPushButton("Save Current Settings")
    save_button.setObjectName("secondaryButton")
    save_button.clicked.connect(lambda: save_config(main_window.update_config()))
    quick_actions_layout.addWidget(save_button, 1, 0, 1, 2)

    quick_section = _wrap_section(
        "Quick Controls",
        "Persistent settings stay in the Settings tab. This keeps the main workflow uncluttered.",
        quick_actions,
    )

    dashboard_layout.addWidget(visual_section, 0, 0, 2, 2)
    dashboard_layout.addWidget(quick_section, 0, 2)
    dashboard_layout.addWidget(logs_section, 1, 2)
    dashboard_layout.setColumnStretch(0, 3)
    dashboard_layout.setColumnStretch(1, 2)
    dashboard_layout.setColumnStretch(2, 2)
    dashboard_layout.setRowStretch(0, 1)
    dashboard_layout.setRowStretch(1, 2)
    return (
        dashboard,
        visualize_tab,
        log_display,
        dashboard_status_value,
        dashboard_visualizer_value,
    )


def _build_settings_tab(
    main_window: MainWindow,
) -> tuple[
    QWidget,
    QLineEdit,
    QLineEdit,
    QComboBox,
    QDoubleSpinBox,
    QCheckBox,
    QCheckBox,
    QCheckBox,
    QCheckBox,
    QCheckBox,
]:
    """
    Build the Settings tab with connection, runtime, visualizer, and persistence sections.

    Args:
        main_window (MainWindow): The main window instance whose config and
            update method are referenced.

    Returns:
        tuple containing:
            - QWidget: The Settings tab widget with scrollable content.
            - QLineEdit: ADB IP input.
            - QLineEdit: Device serial input.
            - QComboBox: Log level dropdown.
            - QDoubleSpinBox: Action delay input.
            - QCheckBox: Load deck checkbox.
            - QCheckBox: Auto start game checkbox.
            - QCheckBox: Show images checkbox.
            - QCheckBox: Save images checkbox.
            - QCheckBox: Save labels checkbox.
    """
    settings_tab = QWidget()
    outer_layout = QVBoxLayout(settings_tab)
    outer_layout.setContentsMargins(0, 0, 0, 0)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)

    content = QWidget()
    settings_layout = QGridLayout(content)
    settings_layout.setContentsMargins(0, 0, 0, 0)
    settings_layout.setHorizontalSpacing(16)
    settings_layout.setVerticalSpacing(16)

    connection_form = QFormLayout()
    connection_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
    connection_form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
    connection_form.setSpacing(10)

    adb_ip_input = QLineEdit()
    adb_ip_input.setText(main_window.config["adb"]["ip"])
    device_serial_input = QLineEdit()
    device_serial_input.setText(main_window.config["adb"]["device_serial"])
    connection_form.addRow("ADB IP", adb_ip_input)
    connection_form.addRow("Device serial", device_serial_input)

    connection_widget = QWidget()
    connection_widget.setLayout(connection_form)
    connection_section = _wrap_section(
        "Connection",
        "Configure where the live bot should connect before starting a session.",
        connection_widget,
    )

    runtime_form = QFormLayout()
    runtime_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
    runtime_form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
    runtime_form.setSpacing(10)

    log_level_dropdown = QComboBox()
    log_level_dropdown.addItems(["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"])
    log_level_dropdown.setCurrentText(main_window.config["bot"]["log_level"])
    runtime_form.addRow("Log level", log_level_dropdown)

    play_action_delay_input = QDoubleSpinBox()
    play_action_delay_input.setRange(0.1, 5.0)
    play_action_delay_input.setSingleStep(0.1)
    play_action_delay_input.setValue(main_window.config["ingame"]["play_action"])
    runtime_form.addRow("Action delay (sec)", play_action_delay_input)

    load_deck_checkbox = QCheckBox("Load deck on startup")
    load_deck_checkbox.setChecked(main_window.config["bot"]["load_deck"])
    runtime_form.addRow(load_deck_checkbox)

    auto_start_game_checkbox = QCheckBox("Auto start games")
    auto_start_game_checkbox.setChecked(main_window.config["bot"]["auto_start_game"])
    runtime_form.addRow(auto_start_game_checkbox)

    runtime_widget = QWidget()
    runtime_widget.setLayout(runtime_form)
    runtime_section = _wrap_section(
        "Runtime",
        "Tune bot behavior without changing what the live session can do.",
        runtime_widget,
    )

    visuals_form = QFormLayout()
    visuals_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
    visuals_form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
    visuals_form.setSpacing(10)

    show_images_checkbox = QCheckBox("Enable live visualizer")
    show_images_checkbox.setChecked(main_window.config["visuals"]["show_images"])
    save_images_checkbox = QCheckBox("Save annotated frames to /debug")
    save_images_checkbox.setChecked(main_window.config["visuals"]["save_images"])
    save_labels_checkbox = QCheckBox("Save detector labels to /debug")
    save_labels_checkbox.setChecked(main_window.config["visuals"]["save_labels"])
    visuals_form.addRow(show_images_checkbox)
    visuals_form.addRow(save_images_checkbox)
    visuals_form.addRow(save_labels_checkbox)

    visuals_widget = QWidget()
    visuals_widget.setLayout(visuals_form)
    visuals_section = _wrap_section(
        "Visualizer",
        "Control the annotated live overlay and optional debug artifacts.",
        visuals_widget,
    )

    actions_row = QWidget()
    actions_layout = QHBoxLayout(actions_row)
    actions_layout.setContentsMargins(0, 0, 0, 0)
    actions_layout.setSpacing(12)

    save_config_button = QPushButton("Save Settings")
    save_config_button.setObjectName("primaryButton")
    save_config_button.clicked.connect(lambda: save_config(main_window.update_config()))
    actions_layout.addWidget(save_config_button)

    actions_layout.addStretch()
    actions_section = _wrap_section(
        "Persistence",
        "Write the current UI state back to the config file when you are ready.",
        actions_row,
    )

    settings_layout.addWidget(connection_section, 0, 0)
    settings_layout.addWidget(runtime_section, 0, 1)
    settings_layout.addWidget(visuals_section, 1, 0)
    settings_layout.addWidget(actions_section, 1, 1)
    settings_layout.setColumnStretch(0, 1)
    settings_layout.setColumnStretch(1, 1)
    settings_layout.setRowStretch(2, 1)

    scroll.setWidget(content)
    outer_layout.addWidget(scroll)
    return (
        settings_tab,
        adb_ip_input,
        device_serial_input,
        log_level_dropdown,
        play_action_delay_input,
        load_deck_checkbox,
        auto_start_game_checkbox,
        show_images_checkbox,
        save_images_checkbox,
        save_labels_checkbox,
    )


def setup_tabs(
    main_window: MainWindow,
) -> tuple[
    QTabWidget,
    ImageStreamWindow,
    QTextEdit,
    QLabel,
    QLabel,
    QLineEdit,
    QLineEdit,
    QComboBox,
    QDoubleSpinBox,
    QCheckBox,
    QCheckBox,
    QCheckBox,
    QCheckBox,
    QCheckBox,
]:
    """
    Create the simplified dashboard and settings tabs.

    Args:
        main_window (MainWindow): The main window instance whose config and
            methods are referenced by the tab builders.

    Returns:
        tuple containing:
            - QTabWidget: A tab widget containing Dashboard and Settings tabs.
            - ImageStreamWindow: The live image stream widget.
            - QTextEdit: The runtime log display.
            - QLabel: Dashboard status value label.
            - QLabel: Dashboard visualizer value label.
            - QLineEdit: ADB IP input.
            - QLineEdit: Device serial input.
            - QComboBox: Log level dropdown.
            - QDoubleSpinBox: Action delay input.
            - QCheckBox: Load deck checkbox.
            - QCheckBox: Auto start game checkbox.
            - QCheckBox: Show images checkbox.
            - QCheckBox: Save images checkbox.
            - QCheckBox: Save labels checkbox.
    """
    tab_widget = QTabWidget()

    (
        dashboard_tab,
        visualize_tab,
        log_display,
        dashboard_status_value,
        dashboard_visualizer_value,
    ) = _build_dashboard_tab(main_window)
    tab_widget.addTab(dashboard_tab, "Dashboard")

    (
        settings_tab,
        adb_ip_input,
        device_serial_input,
        log_level_dropdown,
        play_action_delay_input,
        load_deck_checkbox,
        auto_start_game_checkbox,
        show_images_checkbox,
        save_images_checkbox,
        save_labels_checkbox,
    ) = _build_settings_tab(main_window)
    tab_widget.addTab(settings_tab, "Settings")

    return (
        tab_widget,
        visualize_tab,
        log_display,
        dashboard_status_value,
        dashboard_visualizer_value,
        adb_ip_input,
        device_serial_input,
        log_level_dropdown,
        play_action_delay_input,
        load_deck_checkbox,
        auto_start_game_checkbox,
        show_images_checkbox,
        save_images_checkbox,
        save_labels_checkbox,
    )
