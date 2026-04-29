"""
DTStrategy: wraps inference engine and returns (card_name, tile_x, tile_y).

Only recommends cards that are actually in the player's hand (have the
-in-hand suffix), are affordable, and pass tile-validity checks.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.constants.cards import elixir_cost, IDX_TO_CARD
from src.constants.game import GRID_COLS, GRID_ROWS, PLAYER_SIDE_MIN_ROW
from src.transformer.inference import DTInference


_VALID_PLAYER_ROWS = set(range(PLAYER_SIDE_MIN_ROW, GRID_ROWS))
_VALID_COLS = set(range(GRID_COLS))


def _affordable(card_name: str, elixir: int) -> bool:
    cost = elixir_cost(card_name)
    return cost is not None and 0 < cost <= elixir


def _in_hand(card_name: str, hand: List[str]) -> bool:
    """Check if card_name (base) appears in the hand (which has -in-hand suffix)."""
    target = card_name.lower().strip()
    for h in hand:
        base = h.lower().replace("-in-hand", "").replace("-next", "").strip()
        if base == target:
            return True
    return False


def _is_valid_tile(col: int, row: int) -> bool:
    return col in _VALID_COLS and row in _VALID_PLAYER_ROWS


class DTStrategy:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self._engine = DTInference(checkpoint_path, device)
        self._ready = True

    @property
    def is_ready(self) -> bool:
        return self._ready

    def reset_game(self) -> None:
        self._engine.reset()

    def recommend(
        self, state: Dict[str, Any]
    ) -> Optional[Tuple[str, int, int]]:
        """
        Returns (card_name, tile_x, tile_y) or None.
        """
        hand = state.get("hand_cards", [])
        elixir = int(state.get("player_elixir") or 0)

        affordable_hand = [
            h for h in hand
            if "-in-hand" in h.lower()
            and _affordable(h.lower().replace("-in-hand", "").strip(), elixir)
        ]
        if not affordable_hand:
            return None

        card_idx, pos_flat = self._engine.predict(state)
        tile_y = pos_flat // GRID_COLS
        tile_x = pos_flat % GRID_COLS

        # Map model card index → card name
        predicted_name = IDX_TO_CARD.get(card_idx, "")

        # Validate: predicted card must be in hand and affordable
        if not _in_hand(predicted_name, hand) or not _affordable(predicted_name, elixir):
            # Fall back to most expensive affordable card
            affordable_hand.sort(key=lambda h: elixir_cost(h.replace("-in-hand","").strip()) or 0, reverse=True)
            predicted_name = affordable_hand[0].replace("-in-hand", "").strip()
            tile_x, tile_y = GRID_COLS // 2, PLAYER_SIDE_MIN_ROW + 2  # default centre

        # Validate tile position
        if not _is_valid_tile(tile_x, tile_y):
            tile_x = GRID_COLS // 2
            tile_y = PLAYER_SIDE_MIN_ROW + 2

        self._engine.update_action(card_idx, tile_y * GRID_COLS + tile_x)
        return predicted_name, tile_x, tile_y
