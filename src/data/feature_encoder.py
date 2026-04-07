"""
Feature encoding for behavioral cloning model.

Converts a FrameState dictionary (as produced by VideoAnalyzer.to_dict())
into a fixed-size numpy float32 vector suitable for scikit-learn classifiers.

Feature vector layout (~97 dimensions):
    [0]      player_elixir / 10
    [1]      opponent_elixir / 10
    [2:6]    game_phase one-hot (single, double, triple, sudden_death)
    [6]      time_remaining / 300
    [7:10]   player tower HP ratios (left, king, right)
    [10:13]  opponent tower HP ratios (left, king, right)
    [13:93]  board detections: up to 20 slots x 4 values each
             (class_id/vocab_size, tile_x/17, tile_y/31, is_opponent)
    [93:97]  hand card IDs normalized by vocab size
"""

import numpy as np
from typing import Dict, Any, List, Optional

from ..constants.game_constants import ELIXIR_COSTS


# Build a stable card vocabulary from ELIXIR_COSTS.
# Index 0 is reserved for unknown/padding.
CARD_VOCAB: Dict[str, int] = {name: idx + 1 for idx, name in enumerate(sorted(ELIXIR_COSTS.keys()))}
VOCAB_SIZE: int = len(CARD_VOCAB) + 1  # +1 for the unknown/padding slot

PHASE_INDEX: Dict[str, int] = {
    "single": 0,
    "double": 1,
    "triple": 2,
    "sudden_death": 3,
}

MAX_DETECTIONS = 20
DETECTION_FEATURES = 4  # class_id, x, y, is_opponent
FEATURE_DIM = 1 + 1 + 4 + 1 + 3 + 3 + (MAX_DETECTIONS * DETECTION_FEATURES) + 4  # 97


def _card_name_to_id(name: str) -> int:
    """Map a card name to its vocabulary index (0 if unknown)."""
    clean = name.lower().replace("opponent-", "")
    for suffix in ["-in-hand", "-next", "-on-field", "_on_field",
                   "-evolution", "_evolution", "-ability"]:
        clean = clean.replace(suffix, "")
    clean = clean.strip()
    return CARD_VOCAB.get(clean, 0)


def _tower_hp_ratio(towers: Dict[str, Dict], key: str) -> float:
    """Extract HP ratio for a tower, returning 0.0 if missing or destroyed."""
    tower = towers.get(key, {})
    if tower.get("is_destroyed", False):
        return 0.0
    percent = tower.get("health_percent")
    if percent is not None:
        return float(percent) / 100.0
    return 1.0  # Assume full HP if not detected


def encode(state: Dict[str, Any]) -> np.ndarray:
    """
    Encode a single FrameState dict into a fixed-size feature vector.

    Args:
        state: A FrameState dictionary (from FrameState.to_dict()).

    Returns:
        numpy float32 array of shape (FEATURE_DIM,).
    """
    features = np.zeros(FEATURE_DIM, dtype=np.float32)
    idx = 0

    # Elixir (normalized to [0, 1])
    features[idx] = state.get("player_elixir", 5) / 10.0
    idx += 1
    features[idx] = state.get("opponent_elixir_estimated", 5.0) / 10.0
    idx += 1

    # Game phase one-hot
    phase = state.get("game_phase", "single")
    phase_idx = PHASE_INDEX.get(phase, 0)
    features[idx + phase_idx] = 1.0
    idx += 4

    # Time remaining (normalized; max regular game is 180s, overtime adds up to 180s more)
    time_remaining = state.get("game_time_remaining")
    features[idx] = (time_remaining / 300.0) if time_remaining is not None else 0.5
    idx += 1

    # Player tower HP ratios
    player_towers = state.get("player_towers", {})
    features[idx] = _tower_hp_ratio(player_towers, "player_left")
    features[idx + 1] = _tower_hp_ratio(player_towers, "player_king")
    features[idx + 2] = _tower_hp_ratio(player_towers, "player_right")
    idx += 3

    # Opponent tower HP ratios
    opponent_towers = state.get("opponent_towers", {})
    features[idx] = _tower_hp_ratio(opponent_towers, "opponent_left")
    features[idx + 1] = _tower_hp_ratio(opponent_towers, "opponent_king")
    features[idx + 2] = _tower_hp_ratio(opponent_towers, "opponent_right")
    idx += 3

    # Board detections (up to MAX_DETECTIONS, on-field only)
    detections = state.get("detections", [])
    on_field = [d for d in detections if d.get("is_on_field", False)]
    for i in range(MAX_DETECTIONS):
        if i < len(on_field):
            det = on_field[i]
            features[idx] = _card_name_to_id(det.get("class_name", "")) / VOCAB_SIZE
            features[idx + 1] = det.get("tile_x", 0) / 17.0
            features[idx + 2] = det.get("tile_y", 0) / 31.0
            features[idx + 3] = 1.0 if det.get("is_opponent", False) else 0.0
        # else: stays zero-padded
        idx += DETECTION_FEATURES

    # Hand cards (4 slots, normalized card IDs)
    hand_cards = state.get("hand_cards", [])
    for i in range(4):
        if i < len(hand_cards):
            features[idx] = _card_name_to_id(hand_cards[i]) / VOCAB_SIZE
        idx += 1

    return features


def encode_batch(states: List[Dict[str, Any]]) -> np.ndarray:
    """
    Encode a batch of FrameState dicts.

    Args:
        states: List of FrameState dictionaries.

    Returns:
        numpy float32 array of shape (len(states), FEATURE_DIM).
    """
    return np.array([encode(s) for s in states], dtype=np.float32)
