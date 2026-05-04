"""
Decision Transformer inference engine.

Maintains a rolling context buffer of the last K (state, action, RTG) tuples
and produces card + position predictions for the current frame.

Public API:
  DTInference -- load a checkpoint, then call predict() each frame and
                 update_action() after acting on the recommendation
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from src.data.feature_encoder import encode, FEATURE_DIM
from src.transformer.model import DTConfig, DecisionTransformer
from src.types import FrameDict


class DTInference:
    """
    Rolling-context inference wrapper around DecisionTransformer.

    Buffers the last context_len (state, action, RTG) triples, predicts the
    next action, and updates the buffer after the action is taken.
    """

    def __init__(self, checkpoint_path: str, device: str = "cpu") -> None:
        """
        Args:
            checkpoint_path: path to a .pt checkpoint saved by Trainer.save().
            device:          PyTorch device string.
        """
        self.device = torch.device(device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        cfg_dict = ckpt.get("config", {})
        self.cfg = DTConfig(**{k: v for k, v in cfg_dict.items() if hasattr(DTConfig, k)})
        self.model = DecisionTransformer(self.cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        context_k = self.cfg.context_len
        self._states:   np.ndarray = np.zeros((context_k, FEATURE_DIM), dtype=np.float32)
        self._cards:    np.ndarray = np.zeros(context_k, dtype=np.int64)
        self._pos:      np.ndarray = np.zeros(context_k, dtype=np.int64)
        self._rtg:      np.ndarray = np.zeros((context_k, 1), dtype=np.float32)
        self._timestep: int = 0
        self._t_buf:    np.ndarray = np.zeros(context_k, dtype=np.int64)

    def reset(self) -> None:
        """Clear all context buffers. Call between games."""
        self._states[:] = 0
        self._cards[:] = 0
        self._pos[:] = 0
        self._rtg[:] = 0
        self._t_buf[:] = 0
        self._timestep = 0

    def predict(self, state: FrameDict, target_rtg: float = 1.0) -> Tuple[int, int]:
        """
        Predict the next card and placement from the current frame state.

        Args:
            state:      frame state dict passed to the feature encoder.
            target_rtg: desired normalised return-to-go (default 1.0 = best play).

        Returns:
            (card_idx, pos_flat) -- card vocabulary index and flat tile index
                                    (tile_y * GRID_COLS + tile_x).
        """
        context_k = self.cfg.context_len
        step = min(self._timestep, context_k - 1)

        # Shift rolling buffers left by one
        self._states = np.roll(self._states, -1, axis=0)
        self._cards  = np.roll(self._cards, -1)
        self._pos    = np.roll(self._pos, -1)
        self._rtg    = np.roll(self._rtg, -1, axis=0)
        self._t_buf  = np.roll(self._t_buf, -1)

        self._states[-1] = encode(state)
        self._rtg[-1]    = target_rtg
        self._t_buf[-1]  = self._timestep

        mask = np.zeros(context_k, dtype=np.float32)
        mask[max(0, context_k - step - 1):] = 1.0

        def to_t(arr: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
            """Wrap a numpy array in an unsqueezed batch tensor."""
            return torch.from_numpy(arr).unsqueeze(0).to(dtype=dtype, device=self.device)

        with torch.no_grad():
            card_logits, pos_logits = self.model(
                to_t(self._states),
                to_t(self._cards, torch.long),
                to_t(self._pos, torch.long),
                to_t(self._rtg),
                to_t(self._t_buf, torch.long),
                to_t(mask),
            )

        card_idx = int(card_logits[0, -1].argmax(-1).item())
        pos_flat  = int(pos_logits[0, -1].argmax(-1).item())
        return card_idx, pos_flat

    def update_action(self, card_idx: int, pos_flat: int) -> None:
        """
        Record the action that was taken after the last predict() call.

        Args:
            card_idx: card vocabulary index that was played.
            pos_flat: flat tile index (tile_y * GRID_COLS + tile_x).
        """
        self._cards[-1] = card_idx
        self._pos[-1]   = pos_flat
        self._timestep += 1
