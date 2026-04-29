"""
DTStrategy: wraps the inference engine and returns (card_name, tile_x, tile_y).

Only recommends cards that are actually in the player's hand (have the
-in-hand suffix), are affordable at the current elixir count, and target
a valid player-side tile.

Public API:
  DTStrategy -- load once, then call recommend(state) each frame
"""
from __future__ import annotations

from typing import List, Optional, Set, Tuple

from src.constants.cards import elixir_cost, IDX_TO_CARD
from src.constants.game import GRID_COLS, GRID_ROWS, PLAYER_SIDE_MIN_ROW
from src.transformer.inference import DTInference
from src.types import FrameDict


_VALID_PLAYER_ROWS: Set[int] = set(range(PLAYER_SIDE_MIN_ROW, GRID_ROWS))
_VALID_COLS: Set[int] = set(range(GRID_COLS))


def _affordable(card_name: str, elixir: int) -> bool:
    """Return True if the card costs > 0 and <= the available elixir."""
    cost = elixir_cost(card_name)
    return cost is not None and 0 < cost <= elixir


def _in_hand(card_name: str, hand: List[str]) -> bool:
    """
    Check if a base card name is present in the tracked hand.

    Args:
        card_name: canonical base name (no suffix).
        hand:      list of hand strings, each with "-in-hand" suffix.

    Returns:
        True if any hand entry resolves to card_name.
    """
    target = card_name.lower().strip()
    for entry in hand:
        base = entry.lower().replace("-in-hand", "").replace("-next", "").strip()
        if base == target:
            return True
    return False


def _is_valid_tile(col: int, row: int) -> bool:
    """Return True if (col, row) is on the player's side and within grid bounds."""
    return col in _VALID_COLS and row in _VALID_PLAYER_ROWS


class DTStrategy:
    """
    High-level strategy wrapper that filters model outputs to valid, affordable plays.

    When the model's top-ranked card is unavailable, falls back to the most
    expensive affordable card in hand placed at a default centre position.
    """

    def __init__(self, checkpoint_path: str, device: str = "cpu") -> None:
        """
        Args:
            checkpoint_path: path to a .pt checkpoint produced by Trainer.
            device:          PyTorch device string.
        """
        self._engine = DTInference(checkpoint_path, device)
        self._ready: bool = True

    @property
    def is_ready(self) -> bool:
        """True once the model is loaded and ready to produce recommendations."""
        return self._ready

    def reset_game(self) -> None:
        """Reset the inference engine context buffer. Call between games."""
        self._engine.reset()

    def recommend(self, state: FrameDict) -> Optional[Tuple[str, int, int]]:
        """
        Return a card placement recommendation for the current frame.

        Only recommends when the predicted card is in hand and affordable.
        Falls back to the most expensive affordable card if the model's
        primary recommendation cannot be used.

        Args:
            state: FrameDict from the analysis pipeline.

        Returns:
            (card_name, tile_x, tile_y), or None if no affordable card is available.
        """
        hand: List[str] = state.get("hand_cards", [])
        elixir: int = int(state.get("player_elixir") or 0)

        affordable_hand: List[str] = [
            entry for entry in hand
            if "-in-hand" in entry.lower()
            and _affordable(entry.lower().replace("-in-hand", "").strip(), elixir)
        ]
        if not affordable_hand:
            return None

        card_idx, pos_flat = self._engine.predict(state)
        tile_y = pos_flat // GRID_COLS
        tile_x = pos_flat % GRID_COLS

        predicted_name = IDX_TO_CARD.get(card_idx, "")

        # Validate: predicted card must be in hand and affordable
        if not _in_hand(predicted_name, hand) or not _affordable(predicted_name, elixir):
            # Fall back to most expensive affordable card in hand
            affordable_hand.sort(
                key=lambda entry: elixir_cost(entry.replace("-in-hand", "").strip()) or 0,
                reverse=True,
            )
            predicted_name = affordable_hand[0].replace("-in-hand", "").strip()
            tile_x, tile_y = GRID_COLS // 2, PLAYER_SIDE_MIN_ROW + 2

        # Validate tile position
        if not _is_valid_tile(tile_x, tile_y):
            tile_x = GRID_COLS // 2
            tile_y = PLAYER_SIDE_MIN_ROW + 2

        self._engine.update_action(card_idx, tile_y * GRID_COLS + tile_x)
        return predicted_name, tile_x, tile_y
