"""Episode dataclass and builder."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from src.data.feature_encoder import encode
import numpy as np


@dataclass
class Timestep:
    state: np.ndarray        # encoded feature vector
    action_card: int         # card vocab index
    action_pos: int          # flat tile index = tile_y * 18 + tile_x
    rtg: float               # return-to-go


@dataclass
class Episode:
    timesteps: List[Timestep] = field(default_factory=list)
    outcome: str = "unknown"  # "win" / "loss" / "unknown"

    @property
    def length(self) -> int:
        return len(self.timesteps)

    @property
    def returns_to_go(self) -> List[float]:
        return [t.rtg for t in self.timesteps]


def _base_name(name: str) -> str:
    import re
    return re.sub(r"-(in-hand|next|on-field|on_field)$", "", name.lower()).strip()


def _detect_placements(
    prev_dets: List[Dict], curr_dets: List[Dict], prev_tracked: Set[str]
) -> Tuple[List[Dict], Set[str]]:
    placements: List[Dict] = []
    current: Set[str] = set()
    for det in curr_dets:
        if det.get("is_opponent", True) or not det.get("is_on_field", False):
            continue
        ty = int(det.get("tile_y", 0))
        if ty < 17:
            continue
        name = det.get("class_name", "")
        tx = int(det.get("tile_x", 0))
        uid = f"{name}_{tx}_{ty}"
        current.add(uid)
        if uid not in prev_tracked:
            placements.append({"card_name": name, "tile_x": tx, "tile_y": ty})
    return placements, current


def build_episode(frames: List[Dict], outcome: str = "unknown") -> Episode:
    """Convert a list of frame dicts into an Episode."""
    from src.constants.cards import card_to_idx, VOCAB_SIZE
    from src.constants.game import GRID_COLS

    pairs: List[Dict] = []
    tracked: Set[str] = set()

    for i in range(1, len(frames)):
        placements, tracked = _detect_placements(
            frames[i - 1].get("detections", []),
            frames[i].get("detections", []),
            tracked,
        )
        for pl in placements:
            card_name = _base_name(pl["card_name"])
            hand = [_base_name(c) for c in frames[i - 1].get("hand_cards", [])]
            if card_name not in hand:
                continue  # card not in hand — skip
            pairs.append({"state": frames[i - 1], "action": pl})

    if not pairs:
        return Episode(outcome=outcome)

    # Compute returns-to-go (1 per timestep, summing to episode length)
    rtgs = list(range(len(pairs), 0, -1))

    timesteps = []
    for pair, rtg in zip(pairs, rtgs):
        state_vec = encode(pair["state"])
        card_name = _base_name(pair["action"]["card_name"])
        action_card = card_to_idx(card_name)
        tx = int(pair["action"]["tile_x"])
        ty = int(pair["action"]["tile_y"])
        action_pos = ty * GRID_COLS + tx
        timesteps.append(Timestep(state_vec, action_card, action_pos, float(rtg)))

    return Episode(timesteps=timesteps, outcome=outcome)
