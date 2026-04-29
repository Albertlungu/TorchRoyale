"""
Encodes a frame state dict into a fixed-length float32 feature vector.

Layout (total = FEATURE_DIM):
  [0]        player_elixir / 10
  [1]        game_time_remaining / 180  (0 if None)
  [2]        elixir_multiplier / 3
  [3:6]      player tower HP ratios  (left, king, right)
  [6:9]      opponent tower HP ratios
  [9:29]     top-20 on-field detections: tile_x/17, tile_y/31, is_opp, card_id/VOCAB
  [29:33]    hand card IDs / VOCAB  (4 slots)
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from src.constants.cards import VOCAB_SIZE, card_to_idx

FEATURE_DIM = 33
_MAX_DETS = 20
_DET_FEATURES = 4  # tile_x, tile_y, is_opponent, card_id


def _tower_hp(towers: Dict[str, Any], key: str) -> float:
    t = towers.get(key, {})
    if t.get("is_destroyed"):
        return 0.0
    pct = t.get("health_percent")
    return float(pct) / 100.0 if pct is not None else 1.0


def encode(state: Dict[str, Any]) -> np.ndarray:
    features = np.zeros(FEATURE_DIM, dtype=np.float32)
    idx = 0

    features[idx] = min(float(state.get("player_elixir") or 0), 10) / 10.0
    idx += 1
    timer = state.get("game_time_remaining")
    features[idx] = float(timer) / 180.0 if timer is not None else 0.0
    idx += 1
    features[idx] = float(state.get("elixir_multiplier") or 1) / 3.0
    idx += 1

    pt = state.get("player_towers", {})
    features[idx]     = _tower_hp(pt, "player_left")
    features[idx + 1] = _tower_hp(pt, "player_king")
    features[idx + 2] = _tower_hp(pt, "player_right")
    idx += 3

    ot = state.get("opponent_towers", {})
    features[idx]     = _tower_hp(ot, "opponent_left")
    features[idx + 1] = _tower_hp(ot, "opponent_king")
    features[idx + 2] = _tower_hp(ot, "opponent_right")
    idx += 3

    for i, det in enumerate(state.get("detections", [])[:_MAX_DETS]):
        features[idx]     = int(det.get("tile_x", 0)) / 17.0
        features[idx + 1] = int(det.get("tile_y", 0)) / 31.0
        features[idx + 2] = 1.0 if det.get("is_opponent") else 0.0
        features[idx + 3] = card_to_idx(det.get("class_name", "")) / VOCAB_SIZE
        idx += _DET_FEATURES
    idx += max(0, _MAX_DETS - len(state.get("detections", []))) * _DET_FEATURES

    hand = state.get("hand_cards", [])
    for i in range(4):
        if i < len(hand):
            # Strip -in-hand suffix before lookup
            name = hand[i].replace("-in-hand", "").replace("-next", "").strip()
            features[idx] = card_to_idx(name) / VOCAB_SIZE
        idx += 1

    return features
