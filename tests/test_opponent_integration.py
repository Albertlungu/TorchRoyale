from __future__ import annotations

import importlib.util
import json
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.namespaces.screens import Screens


class _DummyAction:
    def __init__(self, _idx: int, tile_x: int, tile_y: int) -> None:
        self.tile_x = tile_x
        self.tile_y = tile_y

    def calculate_score(self, _state) -> list[float]:
        return [1.0]


_actions_stub = types.ModuleType("src.actions")
_actions_stub.ArchersAction = _DummyAction
_actions_stub.ArrowsAction = _DummyAction
_actions_stub.FireballAction = _DummyAction
_actions_stub.GiantAction = _DummyAction
_actions_stub.KnightAction = _DummyAction
_actions_stub.MinionsAction = _DummyAction
_actions_stub.MinipekkaAction = _DummyAction
_actions_stub.MusketeerAction = _DummyAction
_original_actions_module = sys.modules.get("src.actions")
sys.modules["src.actions"] = _actions_stub
from src.recommendation import heuristic_strategy as hs
if _original_actions_module is None:
    del sys.modules["src.actions"]
else:
    sys.modules["src.actions"] = _original_actions_module

_LIVE_DETECTOR_PATH = Path(__file__).parents[1] / "src" / "live" / "detector.py"
_LIVE_DETECTOR_SPEC = importlib.util.spec_from_file_location(
    "torchroyale_live_detector", _LIVE_DETECTOR_PATH
)
assert _LIVE_DETECTOR_SPEC is not None and _LIVE_DETECTOR_SPEC.loader is not None
_LIVE_DETECTOR_MODULE = importlib.util.module_from_spec(_LIVE_DETECTOR_SPEC)
_LIVE_DETECTOR_SPEC.loader.exec_module(_LIVE_DETECTOR_MODULE)
LiveDetector = _LIVE_DETECTOR_MODULE.LiveDetector
HybridStrategy = hs.HybridStrategy


def test_data_files_include_hog_26_cards_and_counter_mappings():
    repo_root = Path(__file__).parents[1]
    costs_path = repo_root / "data" / "card_costs.json"
    counters_path = repo_root / "data" / "counters.json"

    assert costs_path.exists()
    assert counters_path.exists()

    costs = json.loads(costs_path.read_text(encoding="utf-8"))
    cards_in_costs = {entry["card_name"].lower().replace(" ", "-") for entry in costs}

    expected_hog_26 = {
        "hog-rider",
        "musketeer",
        "ice-golem",
        "cannon",
        "skeletons",
        "ice-spirit",
        "fireball",
        "the-log",
    }
    assert expected_hog_26.issubset(cards_in_costs)

    counters = json.loads(counters_path.read_text(encoding="utf-8"))
    assert "hog-rider" in counters
    assert {"cannon", "tesla", "mini-pekka"}.issubset(
        set(counters["hog-rider"]["counters"])
    )
    assert "musketeer" in counters
    assert {"fireball", "lightning"}.issubset(set(counters["musketeer"]["counters"]))


def test_live_detector_records_only_new_enemy_instances():
    class DummyOpponentTracker:
        def __init__(self) -> None:
            self.recorded: list[str] = []
            self.elixir_updates: list[tuple[float, int]] = []

        def record_card_play(self, card_name: str, _timestamp: float) -> None:
            self.recorded.append(card_name)

        def update_elixir(self, current_time: float, multiplier: int) -> None:
            self.elixir_updates.append((current_time, multiplier))

        def reset(self) -> None:
            self.recorded.clear()
            self.elixir_updates.clear()

    class DummyBuildingTracker:
        def record_building_placement(self, *_args, **_kwargs) -> None:
            return

        def reset(self) -> None:
            return

    detector = LiveDetector.__new__(LiveDetector)
    detector.opponent_tracker = DummyOpponentTracker()
    detector.building_tracker = DummyBuildingTracker()
    detector._game_started_at = time.time() - 130  # double elixir window
    detector._previous_enemy_counts = {}

    def enemy(name: str):
        return SimpleNamespace(
            unit=SimpleNamespace(name=name),
            position=SimpleNamespace(tile_x=8, tile_y=10),
        )

    detector._track_opponent_state([enemy("hog_rider"), enemy("hog_rider")], Screens.IN_GAME)
    detector._track_opponent_state([enemy("hog_rider"), enemy("hog_rider")], Screens.IN_GAME)
    detector._track_opponent_state(
        [enemy("hog_rider"), enemy("hog_rider"), enemy("musketeer")], Screens.IN_GAME
    )

    assert detector.opponent_tracker.recorded == ["hog-rider", "hog-rider", "musketeer"]
    assert detector.opponent_tracker.elixir_updates
    assert detector.opponent_tracker.elixir_updates[0][1] == 2


def test_push_window_boosts_aggression_factor():
    hs._STRATEGIC_CONTEXT_CACHE.clear()
    embedding = np.zeros(128, dtype=np.float32)

    class FakeTracker:
        @staticmethod
        def get_elixir_advantage(_player_elixir: float) -> float:
            return 4.0

        @staticmethod
        def has_counter(_card: str) -> bool:
            return False

    context = hs._extract_strategic_context(
        embedding,
        {"player_elixir": 8, "detections": [], "hand_cards": []},
        FakeTracker(),
    )

    base = float(np.tanh(4.0 * 0.3))
    assert context["aggression_factor"] == pytest.approx(base * 1.5)
    assert context["opponent_lacks_hog_counter"] == 1.0


def test_heuristic_tile_window_scales_with_aggression():
    strategy = HybridStrategy.__new__(HybridStrategy)
    strategy._heuristic_count = 0
    strategy._score_action = lambda _action, _state, _detections: [1.0]

    state = {"hand_cards": ["knight-in-hand"], "detections": []}
    aggressive = strategy._evaluate_heuristic_actions(state, {"aggression_factor": 1.0})
    defensive = strategy._evaluate_heuristic_actions(state, {"aggression_factor": -1.0})

    aggressive_rows = {entry[2] for entry in aggressive}
    defensive_rows = {entry[2] for entry in defensive}

    assert aggressive_rows
    assert defensive_rows
    assert max(aggressive_rows) < min(defensive_rows)


def test_hybrid_strategy_logs_opponent_state_even_without_recommendation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    checkpoint = tmp_path / "dt.pt"
    checkpoint.write_bytes(b"placeholder")

    strategy = HybridStrategy(str(checkpoint))

    monkeypatch.setattr(strategy, "_run_dt_inference", lambda _state: ("knight", 9, 20, 0.5))
    monkeypatch.setattr(
        hs, "_compute_dt_embedding", lambda _state, _path: np.zeros(128, dtype=np.float32)
    )
    monkeypatch.setattr(strategy, "_evaluate_heuristic_actions", lambda _state, _ctx: [])

    result = strategy.recommend(
        {
            "timestamp_ms": 1,
            "player_elixir": 5,
            "game_time_remaining": 120,
            "elixir_multiplier": 1,
            "hand_cards": ["knight-in-hand"],
            "detections": [],
        }
    )

    output = capsys.readouterr().out
    assert result is None
    assert "[HybridStrategy]" in output
