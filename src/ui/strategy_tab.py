"""
Strategy reference tab for TorchRoyale.

Provides lookup tools for card counters, card metadata, and deck archetype
classification using the existing counter database and cost tables.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PyQt6.QtWidgets import QComboBox
from PyQt6.QtWidgets import QFrame
from PyQt6.QtWidgets import QGridLayout
from PyQt6.QtWidgets import QHBoxLayout
from PyQt6.QtWidgets import QLabel
from PyQt6.QtWidgets import QPushButton
from PyQt6.QtWidgets import QScrollArea
from PyQt6.QtWidgets import QTextEdit
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtWidgets import QWidget

from src.game_state.deck_classifier import DeckArchetype, DeckClassifier

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COUNTERS_PATH = _REPO_ROOT / "data" / "counters.json"
_COSTS_PATH = _REPO_ROOT / "data" / "card_costs.json"

# Loading Constants
def _load_counters() -> Dict[str, Any]:
    """Load the counter database from disk.

    Returns:
        (Dict[str, Any]): Dictionary mapping canonical card names to counter metadata
    """
    if not _COUNTERS_PATH.exists():
        return {}
    with open(_COUNTERS_PATH, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _load_card_costs() -> Tuple[Dict[str, int], Dict[str, str]]:
    """Load card elixir costs and types from disk.

    Returns:
        (Tuple[Dict[str, int], Dict[str, str]]): Tuple of (cost_map, type_map) keyed by normalised card name
    """
    cost_map: Dict[str, int] = {}
    type_map: Dict[str, str] = {}
    if not _COSTS_PATH.exists():
        return cost_map, type_map
    with open(_COSTS_PATH, "r", encoding="utf-8") as handle:
        entries = json.load(handle)
    for entry in entries:
        name = entry["card_name"].lower().replace(" ", "-")
        cost_map[name] = entry["elixir_cost"]
        type_map[name] = entry.get("card_type", "unknown")
    return cost_map, type_map

#Constants mappings
_COUNTERS = _load_counters()
_COST_MAP, _TYPE_MAP = _load_card_costs()

_DISPLAY_NAMES: Dict[str, str] = {
    "the-log": "The Log",
}

_ALIASES: Dict[str, str] = {
    "log": "the-log",
}


_CARDS_BY_DISPLAY: Dict[str, str] = {}
for _raw in sorted(set(list(_COUNTERS.keys()) + list(_COST_MAP.keys()))):
    canonical = _ALIASES.get(_raw, _raw)
    if canonical not in _CARDS_BY_DISPLAY:
        if canonical in _DISPLAY_NAMES:
            display = _DISPLAY_NAMES[canonical]
        else:
            display = canonical.replace("-", " ").title()
        _CARDS_BY_DISPLAY[display] = canonical

_ALL_CARD_NAMES = sorted(_CARDS_BY_DISPLAY.keys())


class StrategyTab(QWidget):
    """Strategy reference tab for TorchRoyale.

    Provides lookup tools for card counters, card metadata, and deck archetype
    classification using the existing counter database and cost tables.

    Attributes:
        _counters (Dict[str, Any]): Counter database (canonical names to counter metadata)
        _cost_map (Dict[str, int]): Card elixir costs 
        _type_map (Dict[str, str]): Card types 
        _display_names (Dict[str, str]): Mapping from normalised key to display name
        _aliases (Dict[str, str]): Alternative names mapping to canonical keys
        _cards_by_display (Dict[str, str]): Reverse mapping from display name to canonical key
        _all_card_names (List[str]): Sorted list of all card display names
        _classifier (DeckClassifier): Deck classifier instance
        _card_lookup_section (QWidget): Card lookup section widget
        _deck_analysis_section (QWidget): Deck analysis section widget
        _card_dropdown (QComboBox): Card lookup dropdown
        _card_result (QTextEdit): Card lookup result display
        _deck_inputs (List[QComboBox]): Deck card dropdowns
        _matchup_dropdown (QComboBox): Opponent archetype dropdown
        _deck_result (QTextEdit): Deck analysis result display
    """

    def __init__(self, main_window):
        super().__init__()
        _ = main_window

        self._counters = _COUNTERS
        self._cost_map = _COST_MAP
        self._type_map = _TYPE_MAP
        self._display_names = _DISPLAY_NAMES
        self._aliases = _ALIASES
        self._cards_by_display = _CARDS_BY_DISPLAY
        self._all_card_names = _ALL_CARD_NAMES
        self._classifier = DeckClassifier()

        self._card_lookup_section = None
        self._deck_analysis_section = None
        self._card_result = None
        self._deck_inputs = None
        self._matchup_dropdown = None
        self._deck_result = None

        self._build_ui()

    def _build_ui(self) -> None:
        """
        Build the complete strategy tab UI.
        Args:
            None
        Returns:
            None
        """
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        layout = QGridLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(16)
        layout.setVerticalSpacing(16)

        self._card_lookup_section, self._card_dropdown, self._card_result = self._build_card_lookup_section()
        self._deck_analysis_section, self._deck_inputs, self._matchup_dropdown, self._deck_result = self._build_deck_analysis_section()

        # Wire reactive updates (UI refreshing automatically)
        self._card_dropdown.currentTextChanged.connect(
            lambda _: self._display_card_result()
        )
        self._card_lookup_section.findChild(QPushButton, "primaryButton").clicked.connect(
            lambda: self._display_card_result()
        )

        for cb in self._deck_inputs:
            cb.currentTextChanged.connect(
                lambda _: self._display_deck_result()
            )
        self._matchup_dropdown.currentTextChanged.connect(
            lambda _: self._display_deck_result()
        )
        self._deck_analysis_section.findChild(QPushButton, "primaryButton").clicked.connect(
            lambda: self._display_deck_result()
        )

        layout.addWidget(self._card_lookup_section, 0, 0)
        layout.addWidget(self._deck_analysis_section, 0, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.setRowStretch(1, 1)

        scroll.setWidget(content)
        outer_layout.addWidget(scroll)

    def _create_card_dropdown(self) -> QComboBox:
        """Create a dropdown pre-populated with all known card display names.
        Args:
            None
        Returns:
            (QComboBox): ComboBox with a blank option followed by all card names sorted alphabetically
        """
        cb = QComboBox()
        cb.addItem("")
        for display_name in self._all_card_names:
            cb.addItem(display_name)
        return cb

    def _build_card_lookup_section(self) -> Tuple[QWidget, QComboBox, QTextEdit]:
        """Build the single-card counter lookup widget.
        Args:
            None
        Returns:
            (Tuple[QWidget, QComboBox, QTextEdit]): Tuple of (container, dropdown, result_display)
        """
        section = QFrame()
        section.setObjectName("sectionCard")
        layout = QVBoxLayout(section)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        title = QLabel("Card Lookup")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        description = QLabel(
            "Select a card to see its hard counters, counter-to-counter options, "
            "elixir cost, and card type."
        )
        description.setObjectName("sectionDescription")
        description.setWordWrap(True)
        layout.addWidget(description)

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(10)

        dropdown = self._create_card_dropdown()
        row_layout.addWidget(dropdown, stretch=3)

        search_button = QPushButton("Search")
        search_button.setObjectName("primaryButton")
        row_layout.addWidget(search_button, stretch=1)

        layout.addWidget(row)

        result_display = QTextEdit()
        result_display.setReadOnly(True)
        result_display.setObjectName("logDisplay")
        result_display.setMinimumHeight(280)
        layout.addWidget(result_display)

        return section, dropdown, result_display

    def _build_deck_analysis_section(self) -> Tuple[QWidget, List[QComboBox], QComboBox, QTextEdit]:
        """Build the 8-card deck analysis widget.
        Args:
            None
        Returns:
            (Tuple[QWidget, List[QComboBox], QComboBox, QTextEdit]): Tuple of (container, card_dropdowns, matchup_dropdown, result_display)
        """
        section = QFrame()
        section.setObjectName("sectionCard")
        layout = QVBoxLayout(section)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        title = QLabel("Deck Archetype Analysis")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        description = QLabel(
            "Select your 8 cards to classify the deck archetype, then choose an "
            "opponent archetype to see how the matchup plays out."
        )
        description.setObjectName("sectionDescription")
        description.setWordWrap(True)
        layout.addWidget(description)

        # 4x2 grid of card dropdowns
        deck_inputs: List[QComboBox] = []
        grid = QWidget()
        grid_layout = QGridLayout(grid)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(8)

        for i in range(4):
            for j in range(2):
                cb = self._create_card_dropdown()
                cb.setObjectName(f"deck_card_{i * 2 + j}")
                deck_inputs.append(cb)
                grid_layout.addWidget(cb, i, j)

        layout.addWidget(grid)

        matchup_label = QLabel("Simulate matchup against opponent archetype:")
        matchup_label.setObjectName("metricLabel")
        layout.addWidget(matchup_label)

        matchup_dropdown = QComboBox()
        matchup_dropdown.addItem("")
        for archetype in DeckArchetype:
            if archetype != DeckArchetype.UNKNOWN:
                display = archetype.value.replace("_", " ").title()
                matchup_dropdown.addItem(display, archetype.value)
        layout.addWidget(matchup_dropdown)

        analyze_button = QPushButton("Analyze Deck")
        analyze_button.setObjectName("primaryButton")
        layout.addWidget(analyze_button)

        result_display = QTextEdit()
        result_display.setReadOnly(True)
        result_display.setObjectName("logDisplay")
        result_display.setMinimumHeight(300)
        layout.addWidget(result_display)

        return section, deck_inputs, matchup_dropdown, result_display

    def _card_display_name(self, key: str) -> str:
        """Convert a canonical key to a human-readable display name.

        Args:
            key (str): Canonical card key

        Returns:
            (str): Human-readable title-case name
        """
        if key in self._display_names:
            return self._display_names[key]
        return key.replace("-", " ").title()

    def _normalise(self, name: str) -> str:
        """Normalise a user-entered card name to the canonical key format.

        Args:
            name (str): Raw card name entered by the user

        Returns:
            (str): Canonical key string
        """
        stripped = name.strip().lower().replace(" ", "-")
        return self._aliases.get(stripped, stripped)

    def _display_card_result(self) -> None:
        """
        Lookup and display counter information for the currently selected card.
        Args:
            None
        Returns:
            None
        """
        display_name = self._card_dropdown.currentText()
        if not display_name or display_name not in self._cards_by_display:
            self._card_result.setPlainText("Select a card to look up its counters.")
            return

        key = self._cards_by_display[display_name]
        # build deck analysis output
        lines: List[str] = []
        lines.append(f"Card: {self._card_display_name(key)}")
        lines.append("")

        cost = self._cost_map.get(key)
        card_type = self._type_map.get(key)
        if cost is not None:
            lines.append(f"  Elixir cost: {cost}")
        if card_type:
            lines.append(f"  Type:        {card_type}")

        counter_data = self._counters.get(key, {})
        counters = counter_data.get("counters", [])
        counter_to_counter = counter_data.get("counter_to_counter", {})

        lines.append("")
        if counters:
            lines.append("  Hard counters:")
            for c in counters:
                lines.append(f"    - {self._card_display_name(c)}")
        else:
            lines.append("  No hard counter data available.")

        lines.append("")
        if counter_to_counter:
            lines.append("  Counter-to-counter (what beats their counter):")
            for their_counter, our_responses in counter_to_counter.items():
                display_name = (
                    their_counter
                    if their_counter in ("any-dps", "tank+support", "any-troop", "splash", "spells")
                    else self._card_display_name(their_counter)
                )
                if our_responses:
                    responses = ", ".join(self._card_display_name(r) for r in our_responses)
                    lines.append(f"    If they play {display_name}: {responses}")
                else:
                    lines.append(f"    If they play {display_name}: (no direct counter)")
        else:
            lines.append("  No counter-to-counter data available.")

        self._card_result.setPlainText("\n".join(lines))

    def _display_deck_result(self) -> None:
        """
        Classify the entered deck and display archetype + matchup info
        Args:
            None
        Returns:
            None
        """
        card_names = []
        for cb in self._deck_inputs:
            display_name = cb.currentText().strip()
            if display_name and display_name in self._cards_by_display:
                card_names.append(self._cards_by_display[display_name])

        if not card_names:
            self._deck_result.setPlainText("Select at least one card to begin deck analysis.")
            return

        archetype = self._classifier.classify_deck(card_names)
        avg_elixir = self._classifier._calculate_avg_elixir(card_names)
        win_condition = self._classifier.get_win_condition(card_names)

        lines: List[str] = []
        lines.append("Deck Analysis")
        lines.append("")
        lines.append(f"  Archetype:     {archetype.value.title()}")
        lines.append(f"  Avg elixir:    {avg_elixir:.1f}")
        if win_condition:
            lines.append(f"  Win condition: {self._card_display_name(win_condition)}")

        lines.append("")
        lines.append(f"  Cards ({len(card_names)}):")
        for name in card_names:
            cost = self._cost_map.get(name)
            cost_str = str(cost) if cost is not None else "N/A"
            lines.append(f"    - {self._card_display_name(name)} ({cost_str})")

        matchup_raw = self._matchup_dropdown.currentData()
        if matchup_raw and archetype != DeckArchetype.UNKNOWN:
            try:
                opponent_archetype = DeckArchetype(matchup_raw)
            except ValueError:
                opponent_archetype = None

            if opponent_archetype:
                strategy = self._classifier.get_matchup_strategy(archetype, opponent_archetype)
                lines.append("")
                lines.append(
                    f"  Matchup: {archetype.value.title()} vs {opponent_archetype.value.title()}"
                )
                lines.append("")
                for param, value in strategy.items():
                    display_param = param.replace("_", " ").title()
                    #Unicode visual bar charts 
                    if isinstance(value, float):
                        if 0.0 <= value <= 1.0:
                            filled = int(value * 10)
                            bar = "\u2588" * filled + "\u2591" * (10 - filled)
                            lines.append(f"    {display_param:<25} [{bar}] {value:.2f}")
                        else:
                            lines.append(f"    {display_param:<25} {value:.2f}x")
                    else:
                        lines.append(f"    {display_param:<25} {value}")

        self._deck_result.setPlainText("\n".join(lines))


def _build_strategy_tab(main_window) -> QWidget:
    """Create the full Strategy reference tab.

    Args:
        main_window: Reference to the MainWindow for future signal wiring

    Returns:
        (QWidget): QWidget containing the strategy lookup and deck analysis sections
    """
    return StrategyTab(main_window)
