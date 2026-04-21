"""
Card recommendation strategies.

Provides two strategy implementations:
    MLStrategy  -- Two-stage RandomForest behavioral cloning (original).
    DTStrategy  -- Decision Transformer conditioned on desired returns.

Both expose the same recommend() interface.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import joblib

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.feature_encoder import encode, FEATURE_DIM
from src.constants.game_constants import get_elixir_cost
from src.grid.coordinate_mapper import CoordinateMapper
from src.grid.validity_masks import PlacementValidator
from src.utils.inference_config import InferenceConfig


DEFAULT_MODEL_DIR = Path(project_root) / "data" / "models"


class MLStrategy:
    """
    Behavioral cloning strategy that recommends card placements.

    Uses two Random Forest classifiers:
        Stage 1: Predicts which hand card to play (index 0-3).
        Stage 2: Predicts where to place it (tile position).

    Falls back to the most expensive affordable card at the center
    of the player's side if no models are loaded.
    """

    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize the strategy.

        Args:
            model_dir: Directory containing stage1_card.pkl and stage2_pos.pkl.
                       If None, uses data/models/ relative to project root.
        """
        self._model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self._stage1 = None
        self._stage2 = None
        self._models_loaded = False

        self._mapper = CoordinateMapper()
        self._validator = PlacementValidator(self._mapper)

        self._load_models()

    def _load_models(self):
        """Attempt to load trained models from disk."""
        stage1_path = self._model_dir / "stage1_card.pkl"
        stage2_path = self._model_dir / "stage2_pos.pkl"

        if stage1_path.exists() and stage2_path.exists():
            self._stage1 = joblib.load(stage1_path)
            self._stage2 = joblib.load(stage2_path)
            self._models_loaded = True

    @property
    def is_ready(self) -> bool:
        """Whether trained models are loaded and ready for inference."""
        return self._models_loaded

    def recommend(self, state: Dict[str, Any]) -> Optional[Tuple[str, int, int]]:
        """
        Recommend a card placement given the current game state.

        Args:
            state: A FrameState dictionary (from FrameState.to_dict()).

        Returns:
            (card_name, tile_x, tile_y) or None if no affordable card.
        """
        hand_cards = state.get("hand_cards", [])
        player_elixir = state.get("player_elixir", 0)

        if not hand_cards:
            return None

        # Filter to affordable cards in the 4 real hand slots (index 0-3).
        _MAX_HAND = 4
        affordable_indices = [
            i
            for i, card in enumerate(hand_cards)
            if i < _MAX_HAND
            and get_elixir_cost(card) <= player_elixir
            and get_elixir_cost(card) > 0
        ]

        if not affordable_indices:
            return None

        if self._models_loaded:
            return self._predict(state, hand_cards, affordable_indices)
        else:
            return self._fallback(hand_cards, affordable_indices)

    def _predict(
        self,
        state: Dict[str, Any],
        hand_cards: List[str],
        affordable_indices: List[int],
    ) -> Optional[Tuple[str, int, int]]:
        """Use trained models for prediction."""
        features = encode(state).reshape(1, -1)

        # Stage 1: predict card index
        card_probs = self._stage1.predict_proba(features)[0]

        # Zero out unaffordable cards and pick the best affordable one
        masked_probs = np.zeros_like(card_probs)
        for idx in affordable_indices:
            if idx < len(card_probs):
                masked_probs[idx] = card_probs[idx]

        if masked_probs.sum() == 0:
            # Model doesn't think any affordable card is good; fall back
            return self._fallback(hand_cards, affordable_indices)

        card_idx = int(np.argmax(masked_probs))
        card_name = hand_cards[card_idx]

        # Stage 2: predict position
        card_onehot = np.zeros((1, 4), dtype=np.float32)
        card_onehot[0, card_idx] = 1.0
        features_s2 = np.hstack([features, card_onehot])

        pos_probs = self._stage2.predict_proba(features_s2)[0]
        pos_classes = self._stage2.classes_

        # Build a full probability map and apply validity mask
        prob_map = np.zeros(32 * 18, dtype=np.float32)
        for prob, cls in zip(pos_probs, pos_classes):
            if 0 <= cls < 32 * 18:
                prob_map[cls] = prob

        # Apply validity mask
        validity = self._validator.get_mask(card_name)  # shape (32, 18)
        prob_map_2d = prob_map.reshape(32, 18)
        prob_map_2d *= validity

        if prob_map_2d.sum() == 0:
            # No valid position with positive probability; fall back to center
            return self._fallback(hand_cards, affordable_indices)

        best_pos = int(np.argmax(prob_map_2d))
        tile_x = best_pos // 18
        tile_y = best_pos % 18

        return (card_name, tile_x, tile_y)

    def _fallback(
        self,
        hand_cards: List[str],
        affordable_indices: List[int],
    ) -> Optional[Tuple[str, int, int]]:
        """
        Simple heuristic fallback when no model is available.

        Picks the most expensive affordable card and places it near
        the center of the player's side.
        """
        best_idx = max(affordable_indices, key=lambda i: get_elixir_cost(hand_cards[i]))
        card_name = hand_cards[best_idx]

        # Default placement: center of player's side
        tile_x = 9
        tile_y = 24

        # Adjust if not valid for this card type
        if not self._validator.is_valid_placement(card_name, tile_x, tile_y):
            valid_tiles = self._validator.get_valid_tiles(card_name)
            if valid_tiles:
                # Pick the valid tile closest to center
                center_x, center_y = 9, 24
                valid_tiles.sort(
                    key=lambda t: abs(t[0] - center_x) + abs(t[1] - center_y)
                )
                tile_x, tile_y = valid_tiles[0]
            else:
                return None

        return (card_name, tile_x, tile_y)


class DTStrategy:
    """
    Decision Transformer strategy for card recommendations.

    Uses a trained Decision Transformer that conditions on desired
    game outcomes to predict card placements. Maintains a rolling
    context window of recent game states and actions.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        target_return: Optional[float] = None,
        device: Optional[str] = None,
        temperature: Optional[float] = None,
        config_path: Optional[str] = None,
    ):
        """
        Initialize Decision Transformer strategy.

        Args:
            checkpoint_path: Override config checkpoint path
            target_return: Override config target return
            device: Override config device
            temperature: Override config temperature
            config_path: Path to inference config YAML (default: configs/inference.yaml)
        """
        # Load config
        self._config = InferenceConfig(config_path)

        # Apply overrides
        self._checkpoint_path = checkpoint_path or self._config.checkpoint_path
        self._target_return = target_return if target_return is not None else self._config.target_return
        self._device = device or self._config.device
        self._temperature = temperature if temperature is not None else self._config.temperature

        self._mapper = CoordinateMapper()
        self._validator = PlacementValidator(self._mapper)
        self._inference = None
        self._models_loaded = False

        self._load_model()

    def _load_model(self):
        if not Path(self._checkpoint_path).exists():
            return

        from src.transformer.inference import DTInference

        self._inference = DTInference(
            checkpoint_path=self._checkpoint_path,
            device=self._device,
            target_return=self._target_return,
            temperature=self._temperature,
            randomize_context_actions=self._config.randomize_context_actions,
        )
        self._models_loaded = True

    @property
    def is_ready(self) -> bool:
        return self._models_loaded

    def reset_game(self):
        """Call at the start of each new game to reset context."""
        if self._inference:
            self._inference.reset()

    def recommend(self, state: Dict[str, Any]) -> Optional[Tuple[str, int, int]]:
        """
        Recommend a card placement given the current game state.

        Same interface as MLStrategy.recommend().
        """
        hand_cards = state.get("hand_cards", [])
        player_elixir = state.get("player_elixir", 0)

        if not hand_cards:
            return None

        # The model was trained with exactly 4 hand positions (0-3).
        # Roboflow occasionally detects a 5th "next card" slot; ignore it.
        _MAX_HAND = 4
        affordable_indices = [
            i
            for i, card in enumerate(hand_cards)
            if i < _MAX_HAND
            and get_elixir_cost(card) <= player_elixir
            and get_elixir_cost(card) > 0
        ]

        if not affordable_indices:
            return None

        if not self._models_loaded:
            return self._fallback(hand_cards, affordable_indices)

        # Get DT prediction
        card_idx, pos_flat = self._inference.predict(state)

        # Validate card is affordable
        if card_idx not in affordable_indices:
            card_idx = affordable_indices[0]

        card_name = (
            hand_cards[card_idx] if card_idx < len(hand_cards) else hand_cards[0]
        )

        # Decode position
        tile_y = pos_flat // 18
        tile_x = pos_flat % 18

        # Validate placement
        if not self._validator.is_valid_placement(card_name, tile_x, tile_y):
            valid_tiles = self._validator.get_valid_tiles(card_name)
            if valid_tiles:
                valid_tiles.sort(key=lambda t: abs(t[0] - tile_x) + abs(t[1] - tile_y))
                tile_x, tile_y = valid_tiles[0]
            else:
                return None

        # Record action in context window
        self._inference.update_action(card_idx, tile_y * 18 + tile_x)

        return (card_name, tile_x, tile_y)

    def _fallback(
        self,
        hand_cards: List[str],
        affordable_indices: List[int],
    ) -> Optional[Tuple[str, int, int]]:
        best_idx = max(affordable_indices, key=lambda i: get_elixir_cost(hand_cards[i]))
        card_name = hand_cards[best_idx]

        tile_x = self._config.fallback_tile_x
        tile_y = self._config.fallback_tile_y

        if not self._validator.is_valid_placement(card_name, tile_x, tile_y):
            valid_tiles = self._validator.get_valid_tiles(card_name)
            if valid_tiles:
                valid_tiles.sort(key=lambda t: abs(t[0] - 9) + abs(t[1] - 24))
                tile_x, tile_y = valid_tiles[0]
            else:
                return None

        return (card_name, tile_x, tile_y)
