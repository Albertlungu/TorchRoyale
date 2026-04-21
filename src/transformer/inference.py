"""
Inference wrapper for the Decision Transformer.

Maintains a rolling context window of recent game states and actions,
and predicts the next card placement conditioned on desired return-to-go.
"""

from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import torch

from .config import DTConfig
from .model import DecisionTransformer
from src.data.feature_encoder import encode, FEATURE_DIM


class DTInference:
    """
    Wraps a trained Decision Transformer for real-time inference.

    Maintains history of past (state, action, RTG) tuples and predicts
    the next action given the current state and target return.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        target_return: Optional[float] = None,
        temperature: float = 1.0,
    ):
        """
        Args:
            checkpoint_path: Path to saved checkpoint (.pt file).
            device: Torch device string.
            target_return: Desired return-to-go for conditioning. If None,
                uses mean + 1 std of training RTG distribution (from checkpoint).
            temperature: Sampling temperature for action selection. Higher = more random.
                1.0 = sample from model distribution, 0.0 = greedy argmax.
        """
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )

        self.config: DTConfig = checkpoint["config"]
        self.rtg_mean: float = checkpoint.get("rtg_mean", 0.0)
        self.rtg_std: float = checkpoint.get("rtg_std", 1.0)
        self.device = torch.device(device)

        # Build and load model
        self.model = DecisionTransformer(self.config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.model.to(self.device)

        # Set target return
        if target_return is not None:
            self.target_return = target_return
        else:
            # Default: aim for the mean of winning games (mean + 1 std)
            self.target_return = self.rtg_mean + self.rtg_std

        # Sampling temperature
        self.temperature = temperature

        # Rolling context
        self.K = self.config.context_length
        self._reset_context()

    def _reset_context(self):
        self._states: List[np.ndarray] = []
        self._cards: List[int] = []
        self._positions: List[int] = []
        self._rtgs: List[float] = []
        self._current_rtg = self.target_return

    def reset(self, target_return: Optional[float] = None):
        """Reset context for a new game."""
        if target_return is not None:
            self.target_return = target_return
        self._reset_context()

    def predict(self, state_dict: Dict[str, Any]) -> Tuple[int, int]:
        """
        Predict the next action given current game state.

        Args:
            state_dict: FrameState dictionary (same format as feature_encoder.encode() expects).

        Returns:
            (card_index, tile_position) where card_index is 0-3
            and tile_position is 0-575 (tile_y * 18 + tile_x).
        """
        # Encode state
        state_vec = encode(state_dict)

        # Append to history
        self._states.append(state_vec)
        self._rtgs.append(self._current_rtg)

        # Build input tensors from last K timesteps
        T = len(self._states)
        K = min(T, self.K)
        start = T - K

        states = torch.zeros(1, K, FEATURE_DIM, dtype=torch.float32)
        actions_card = torch.zeros(1, K, dtype=torch.long)
        actions_pos = torch.zeros(1, K, dtype=torch.long)
        rtg = torch.zeros(1, K, 1, dtype=torch.float32)
        timesteps = torch.zeros(1, K, dtype=torch.long)
        mask = torch.ones(1, K, dtype=torch.float32)

        # Neutral default values for timesteps without recorded actions
        # Using center of player's side: tile_x=9, tile_y=24
        DEFAULT_POS = 24 * 18 + 9  # = 441

        for i in range(K):
            idx = start + i
            states[0, i] = torch.from_numpy(self._states[idx])
            rtg[0, i, 0] = (self._rtgs[idx] - self.rtg_mean) / max(self.rtg_std, 1e-6)
            timesteps[0, i] = idx

            # Past timesteps: use actual actions taken
            # Current timestep: use neutral default
            if idx < len(self._cards):
                actions_card[0, i] = self._cards[idx]
                actions_pos[0, i] = self._positions[idx]
            else:
                actions_pos[0, i] = DEFAULT_POS

        # Left-pad if K < context_length
        if K < self.K:
            pad = self.K - K
            states = torch.nn.functional.pad(states, (0, 0, pad, 0))
            actions_card = torch.nn.functional.pad(actions_card, (pad, 0))
            actions_pos = torch.nn.functional.pad(actions_pos, (pad, 0))
            rtg = torch.nn.functional.pad(rtg, (0, 0, pad, 0))
            timesteps = torch.nn.functional.pad(timesteps, (pad, 0))
            mask = torch.nn.functional.pad(mask, (pad, 0))
            # Fill padded positions with neutral default
            actions_pos[0, :pad] = DEFAULT_POS

        # Move to device
        states = states.to(self.device)
        actions_card = actions_card.to(self.device)
        actions_pos = actions_pos.to(self.device)
        rtg = rtg.to(self.device)
        timesteps = timesteps.to(self.device)
        mask = mask.to(self.device)

        # Forward pass
        with torch.no_grad():
            card_logits, pos_logits = self.model(
                states, actions_card, actions_pos, rtg, timesteps, mask
            )

        # Prediction from the last real position
        last_idx = self.K - 1

        # Sample with temperature to prevent autoregressive collapse
        if self.temperature > 0:
            card_probs = torch.softmax(card_logits[0, last_idx] / self.temperature, dim=-1)
            pos_probs = torch.softmax(pos_logits[0, last_idx] / self.temperature, dim=-1)
            card_pred = torch.multinomial(card_probs, 1).item()
            pos_pred = torch.multinomial(pos_probs, 1).item()
        else:
            # Greedy selection (deterministic)
            card_pred = card_logits[0, last_idx].argmax().item()
            pos_pred = pos_logits[0, last_idx].argmax().item()

        return card_pred, pos_pred

    def get_card_probs(self, state_dict: Dict[str, Any]) -> np.ndarray:
        """
        Get card selection probabilities without committing to an action.

        Useful for applying affordability masking before selecting.

        Returns:
            (4,) numpy array of probabilities.
        """
        card_pred, _ = self.predict(state_dict)
        # Pop the state we just added (this was just a probe)
        self._states.pop()
        self._rtgs.pop()

        # Rebuild for probs -- re-predict cleanly
        state_vec = encode(state_dict)
        self._states.append(state_vec)
        self._rtgs.append(self._current_rtg)

        T = len(self._states)
        K = min(T, self.K)
        start = T - K

        states = torch.zeros(1, K, FEATURE_DIM, dtype=torch.float32)
        actions_card = torch.zeros(1, K, dtype=torch.long)
        actions_pos = torch.zeros(1, K, dtype=torch.long)
        rtg = torch.zeros(1, K, 1, dtype=torch.float32)
        timesteps = torch.zeros(1, K, dtype=torch.long)
        mask = torch.ones(1, K, dtype=torch.float32)

        DEFAULT_POS = 24 * 18 + 9

        for i in range(K):
            idx = start + i
            states[0, i] = torch.from_numpy(self._states[idx])
            rtg[0, i, 0] = (self._rtgs[idx] - self.rtg_mean) / max(self.rtg_std, 1e-6)
            timesteps[0, i] = idx
            if idx < len(self._cards):
                actions_card[0, i] = self._cards[idx]
                actions_pos[0, i] = self._positions[idx]
            else:
                actions_pos[0, i] = DEFAULT_POS

        if K < self.K:
            pad = self.K - K
            states = torch.nn.functional.pad(states, (0, 0, pad, 0))
            actions_card = torch.nn.functional.pad(actions_card, (pad, 0))
            actions_pos = torch.nn.functional.pad(actions_pos, (pad, 0))
            rtg = torch.nn.functional.pad(rtg, (0, 0, pad, 0))
            timesteps = torch.nn.functional.pad(timesteps, (pad, 0))
            mask = torch.nn.functional.pad(mask, (pad, 0))
            actions_pos[0, :pad] = DEFAULT_POS

        states = states.to(self.device)
        actions_card = actions_card.to(self.device)
        actions_pos = actions_pos.to(self.device)
        rtg = rtg.to(self.device)
        timesteps = timesteps.to(self.device)
        mask = mask.to(self.device)

        with torch.no_grad():
            card_logits, _ = self.model(
                states, actions_card, actions_pos, rtg, timesteps, mask
            )

        probs = torch.softmax(card_logits[0, self.K - 1], dim=-1).cpu().numpy()

        # Pop the probe state
        self._states.pop()
        self._rtgs.pop()

        return probs

    def update_action(self, card_index: int, tile_position: int, reward: float = 0.0):
        """
        Record the action actually taken after predict().

        Must be called after predict() and before the next predict()
        to keep the context window consistent.

        Args:
            card_index: Hand card index that was played (0-3).
            tile_position: Flattened tile position (tile_y * 18 + tile_x).
            reward: Observed reward for this step (optional, decrements RTG).
        """
        self._cards.append(card_index)
        self._positions.append(tile_position)
        self._current_rtg -= reward
