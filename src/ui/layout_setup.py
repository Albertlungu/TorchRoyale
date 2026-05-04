"""
UI layout setup for TorchRoyale main window.

Defines the top bar and the simplified dashboard/settings interface.
"""

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
from src.ui.utils import save_config


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


def setup_top_bar(main_window) -> QFrame:
    """
    Create the top bar with title, status, and transport controls.

    Args:
        main_window (MainWindow): The main window instance to attach controls to.

    Returns:
        QFrame: The configured top bar widget with title, status, and buttons.
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

    main_window.server_id_label = QLabel("Status: Stopped")
    main_window.server_id_label.setObjectName("statusLabel")
    text_layout.addWidget(main_window.server_id_label)

    top_bar_layout.addLayout(text_layout)
    top_bar_layout.addStretch()

    button_layout = QHBoxLayout()
    button_layout.setSpacing(10)
    button_layout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

    main_window.play_pause_button = QPushButton("Pause")
    main_window.play_pause_button.setObjectName("secondaryControlButton")
    main_window.play_pause_button.setFont(QFont("Arial", 11))
    main_window.play_pause_button.clicked.connect(
        main_window.toggle_pause_resume_and_display
    )
    main_window.play_pause_button.hide()
    button_layout.addWidget(main_window.play_pause_button)

    main_window.start_stop_button = QPushButton("Start")
    main_window.start_stop_button.setObjectName("primaryControlButton")
    main_window.start_stop_button.setFont(QFont("Arial", 11))
    main_window.start_stop_button.clicked.connect(
        main_window.toggle_start_stop
    )
    button_layout.addWidget(main_window.start_stop_button)

    top_bar_layout.addLayout(button_layout)
    return top_bar


def _build_dashboard_tab(main_window) -> QWidget:
    """
    Build the Dashboard tab with live overlay, quick controls, and runtime log.

    Args:
        main_window (MainWindow): The main window instance to attach widgets to.

    Returns:
        QWidget: The configured Dashboard tab widget.
    """
    dashboard = QWidget()
    dashboard_layout = QGridLayout(dashboard)
    dashboard_layout.setContentsMargins(0, 0, 0, 0)
    dashboard_layout.setHorizontalSpacing(16)
    dashboard_layout.setVerticalSpacing(16)

    main_window.visualize_tab = ImageStreamWindow()
    main_window.visualize_tab.update_active_state(
        main_window.config["visuals"]["show_images"]
    )
    visual_section = _wrap_section(
        "Live Overlay",
        "Annotated gameplay frames, including hand slots and on-field detections.",
        main_window.visualize_tab,
    )

    main_window.log_display = QTextEdit()
    main_window.log_display.setReadOnly(True)
    main_window.log_display.setObjectName("logDisplay")
    logs_section = _wrap_section(
        "Runtime Log",
        "Bot lifecycle, detector output, and runtime errors appear here.",
        main_window.log_display,
    )

    quick_actions = QWidget()
    quick_actions_layout = QGridLayout(quick_actions)
    quick_actions_layout.setContentsMargins(0, 0, 0, 0)
    quick_actions_layout.setHorizontalSpacing(12)
    quick_actions_layout.setVerticalSpacing(12)

    main_window.dashboard_status_value = QLabel("Stopped")
    main_window.dashboard_status_value.setObjectName("metricValue")
    main_window.dashboard_visualizer_value = QLabel(
        "On" if main_window.config["visuals"]["show_images"] else "Off"
    )
    main_window.dashboard_visualizer_value.setObjectName("metricValue")

    quick_actions_layout.addWidget(
        _make_read_only_field("Bot status", main_window.dashboard_status_value), 0, 0
    )
    quick_actions_layout.addWidget(
        _make_read_only_field("Visualizer", main_window.dashboard_visualizer_value), 0, 1
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
    return dashboard


def _build_settings_tab(main_window) -> QWidget:
    """
    Build the Settings tab with connection, runtime, visualizer, and persistence sections.

    Args:
        main_window (MainWindow): The main window instance to attach widgets to.

    Returns:
        QWidget: The configured Settings tab widget with scrollable content.
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

    main_window.adb_ip_input = QLineEdit()
    main_window.adb_ip_input.setText(main_window.config["adb"]["ip"])
    main_window.device_serial_input = QLineEdit()
    main_window.device_serial_input.setText(
        main_window.config["adb"]["device_serial"]
    )
    connection_form.addRow("ADB IP", main_window.adb_ip_input)
    connection_form.addRow("Device serial", main_window.device_serial_input)

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

    main_window.log_level_dropdown = QComboBox()
    main_window.log_level_dropdown.addItems(
        ["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"]
    )
    main_window.log_level_dropdown.setCurrentText(
        main_window.config["bot"]["log_level"]
    )
    runtime_form.addRow("Log level", main_window.log_level_dropdown)

    main_window.play_action_delay_input = QDoubleSpinBox()
    main_window.play_action_delay_input.setRange(0.1, 5.0)
    main_window.play_action_delay_input.setSingleStep(0.1)
    main_window.play_action_delay_input.setValue(
        main_window.config["ingame"]["play_action"]
    )
    runtime_form.addRow("Action delay (sec)", main_window.play_action_delay_input)

    main_window.load_deck_checkbox = QCheckBox("Load deck on startup")
    main_window.load_deck_checkbox.setChecked(
        main_window.config["bot"]["load_deck"]
    )
    runtime_form.addRow(main_window.load_deck_checkbox)

    main_window.auto_start_game_checkbox = QCheckBox("Auto start games")
    main_window.auto_start_game_checkbox.setChecked(
        main_window.config["bot"]["auto_start_game"]
    )
    runtime_form.addRow(main_window.auto_start_game_checkbox)

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

    main_window.show_images_checkbox = QCheckBox("Enable live visualizer")
    main_window.show_images_checkbox.setChecked(
        main_window.config["visuals"]["show_images"]
    )
    main_window.save_images_checkbox = QCheckBox("Save annotated frames to /debug")
    main_window.save_images_checkbox.setChecked(
        main_window.config["visuals"]["save_images"]
    )
    main_window.save_labels_checkbox = QCheckBox("Save detector labels to /debug")
    main_window.save_labels_checkbox.setChecked(
        main_window.config["visuals"]["save_labels"]
    )
    visuals_form.addRow(main_window.show_images_checkbox)
    visuals_form.addRow(main_window.save_images_checkbox)
    visuals_form.addRow(main_window.save_labels_checkbox)

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
    save_config_button.clicked.connect(
        lambda: save_config(main_window.update_config())
    )
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
    return settings_tab


def setup_tabs(main_window) -> QTabWidget:
    """
    Create the simplified dashboard and settings tabs.

    Args:
        main_window (MainWindow): The main window instance to attach tabs to.

    Returns:
        QTabWidget: A tab widget containing Dashboard and Settings tabs.
    """
    tab_widget = QTabWidget()
    tab_widget.addTab(_build_dashboard_tab(main_window), "Dashboard")
    tab_widget.addTab(_build_settings_tab(main_window), "Settings")
    return tab_widget
