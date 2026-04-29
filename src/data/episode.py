"""
Episode dataclass and builder.

An episode is a sequence of (state, action) pairs derived from a single
game replay. Actions are player card placements on the player's side of
the arena. Returns-to-go are computed as a simple countdown (episode length
down to 1) as a proxy for future value.

Public API:
  Timestep      -- single (state, action, rtg) tuple
  Episode       -- ordered sequence of Timesteps with an outcome label
  build_episode() -- convert a list of FrameDicts to an Episode
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import numpy as np

from src.data.feature_encoder import encode
from src.types import DetectionDict, FrameDict


@dataclass
class Timestep:
    """A single decision point in an episode."""

    state: np.ndarray     # encoded feature vector of shape (FEATURE_DIM,)
    action_card: int      # card vocabulary index
    action_pos: int       # flat tile index = tile_y * GRID_COLS + tile_x
    rtg: float            # return-to-go


@dataclass
class Episode:
    """A complete game episode as a list of Timesteps."""

    timesteps: List[Timestep] = field(default_factory=list)
    outcome: str = "unknown"  # "win" / "loss" / "unknown"

    @property
    def length(self) -> int:
        """Number of decision points in the episode."""
        return len(self.timesteps)

    @property
    def returns_to_go(self) -> List[float]:
        """RTG values for each timestep."""
        return [ts.rtg for ts in self.timesteps]


def _base_name(name: str) -> str:
    """Strip placement/label suffixes to get the canonical card name."""
    return re.sub(r"-(in-hand|next|on-field|on_field)$", "", name.lower()).strip()


def _detect_placements(
    _prev_dets: List[DetectionDict],
    curr_dets: List[DetectionDict],
    prev_tracked: Set[str],
) -> Tuple[List[Dict[str, object]], Set[str]]:
    """
    Find cards that were newly placed on the player's side between two frames.

    Args:
        _prev_dets:   detections from the previous frame (reserved for future use).
        curr_dets:    detections from the current frame.
        prev_tracked: set of unit IDs seen in the previous frame.

    Returns:
        (placements, current_ids) where placements is a list of dicts with
        card_name, tile_x, tile_y and current_ids is the updated tracked set.
    """
    placements: List[Dict[str, object]] = []
    current: Set[str] = set()
    for det in curr_dets:
        if det.get("is_opponent", True) or not det.get("is_on_field", False):
            continue
        tile_y = int(det.get("tile_y", 0))
        if tile_y < 17:
            continue
        name = det.get("class_name", "")
        tile_x = int(det.get("tile_x", 0))
        uid = f"{name}_{tile_x}_{tile_y}"
        current.add(uid)
        if uid not in prev_tracked:
            placements.append({"card_name": name, "tile_x": tile_x, "tile_y": tile_y})
    return placements, current


def build_episode(frames: List[FrameDict], outcome: str = "unknown") -> Episode:
    """
    Convert a list of frame dicts into an Episode.

    Only player placements (tile_y >= 17) of cards present in the hand are
    included as labelled actions.

    Args:
        frames:  list of FrameDicts from the analysis pipeline.
        outcome: game result label — "win", "loss", or "unknown".

    Returns:
        Episode with one Timestep per valid card placement.
    """
    from src.constants.cards import card_to_idx  # pylint: disable=import-outside-toplevel
    from src.constants.game import GRID_COLS     # pylint: disable=import-outside-toplevel

    pairs: List[Dict[str, object]] = []
    tracked: Set[str] = set()

    for frame_idx in range(1, len(frames)):
        placements, tracked = _detect_placements(
            frames[frame_idx - 1].get("detections", []),   # type: ignore[arg-type]
            frames[frame_idx].get("detections", []),        # type: ignore[arg-type]
            tracked,
        )
        for placement in placements:
            card_name = _base_name(str(placement["card_name"]))
            hand: List[str] = [
                _base_name(c) for c in frames[frame_idx - 1].get("hand_cards", [])
            ]
            if card_name not in hand:
                continue  # card not in hand — skip noisy detections
            pairs.append({"state": frames[frame_idx - 1], "action": placement})

    if not pairs:
        return Episode(outcome=outcome)

    # Compute returns-to-go: episode-length countdown (1 per timestep)
    rtgs: List[int] = list(range(len(pairs), 0, -1))

    timesteps: List[Timestep] = []
    for pair, rtg in zip(pairs, rtgs):
        state_vec = encode(pair["state"])  # type: ignore[arg-type]
        action: Dict[str, object] = pair["action"]  # type: ignore[assignment]
        card_name = _base_name(str(action["card_name"]))
        action_card = card_to_idx(card_name)
        tile_x = int(action["tile_x"])
        tile_y = int(action["tile_y"])
        action_pos = tile_y * GRID_COLS + tile_x
        timesteps.append(Timestep(state_vec, action_card, action_pos, float(rtg)))

    return Episode(timesteps=timesteps, outcome=outcome)
