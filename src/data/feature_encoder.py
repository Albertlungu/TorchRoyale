"""
Encodes a frame state dict into a fixed-length float32 feature vector.

Layout (total = FEATURE_DIM = 33):
  [0]        player_elixir / 10
  [1]        game_time_remaining / 180  (0 if None)
  [2]        elixir_multiplier / 3
  [3:6]      player tower HP ratios  (left, king, right)
  [6:9]      opponent tower HP ratios
  [9:29]     top-20 on-field detections: tile_x/17, tile_y/31, is_opp, card_id/VOCAB
  [29:33]    hand card IDs / VOCAB  (4 slots)

Public API:
  FEATURE_DIM -- total length of the encoded vector
  encode()    -- convert a FrameDict to a numpy float32 array
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from src.constants.cards import VOCAB_SIZE, card_to_idx
from src.types import FrameDict, TowerDict

FEATURE_DIM: int = 33
_MAX_DETS: int = 20
_DET_FEATURES: int = 4  # tile_x, tile_y, is_opponent, card_id


def _tower_hp(towers: TowerDict, key: str) -> float:
    """
    Extract a normalised HP ratio for one tower from a towers dict.

    Args:
        towers: mapping of tower key to TowerDict (may be empty).
        key:    tower key, e.g. "player_left".

    Returns:
        Float in [0.0, 1.0]. Returns 0.0 if destroyed, 1.0 if unknown.
    """
    tower: Optional[TowerDict] = towers.get(key)  # type: ignore[assignment]
    if tower is None:
        return 1.0
    if tower.get("is_destroyed"):
        return 0.0
    pct: Optional[float] = tower.get("health_percent")
    return float(pct) / 100.0 if pct is not None else 1.0


def encode(state: FrameDict) -> np.ndarray:
    """
    Encode a frame state dict into a fixed-length float32 feature vector.

    Args:
        state: FrameDict from the analysis pipeline.

    Returns:
        numpy array of shape (FEATURE_DIM,) with dtype float32.
    """
    features = np.zeros(FEATURE_DIM, dtype=np.float32)
    idx: int = 0

    features[idx] = min(float(state.get("player_elixir") or 0), 10) / 10.0
    idx += 1
    timer = state.get("game_time_remaining")
    features[idx] = float(timer) / 180.0 if timer is not None else 0.0
    idx += 1
    features[idx] = float(state.get("elixir_multiplier") or 1) / 3.0
    idx += 1

    player_towers = state.get("player_towers", {})
    features[idx]     = _tower_hp(player_towers, "player_left")   # type: ignore[arg-type]
    features[idx + 1] = _tower_hp(player_towers, "player_king")   # type: ignore[arg-type]
    features[idx + 2] = _tower_hp(player_towers, "player_right")  # type: ignore[arg-type]
    idx += 3

    opp_towers = state.get("opponent_towers", {})
    features[idx]     = _tower_hp(opp_towers, "opponent_left")    # type: ignore[arg-type]
    features[idx + 1] = _tower_hp(opp_towers, "opponent_king")    # type: ignore[arg-type]
    features[idx + 2] = _tower_hp(opp_towers, "opponent_right")   # type: ignore[arg-type]
    idx += 3

    detections = state.get("detections", [])
    for det in (detections or [])[:_MAX_DETS]:
        features[idx]     = int(det.get("tile_x", 0)) / 17.0
        features[idx + 1] = int(det.get("tile_y", 0)) / 31.0
        features[idx + 2] = 1.0 if det.get("is_opponent") else 0.0
        features[idx + 3] = card_to_idx(det.get("class_name", "")) / VOCAB_SIZE
        idx += _DET_FEATURES
    idx += max(0, _MAX_DETS - len(detections or [])) * _DET_FEATURES

    hand = state.get("hand_cards", [])
    for slot in range(4):
        if slot < len(hand):
            # Strip -in-hand / -next suffix before vocabulary lookup
            name = hand[slot].replace("-in-hand", "").replace("-next", "").strip()
            features[idx] = card_to_idx(name) / VOCAB_SIZE
        idx += 1

    return features
