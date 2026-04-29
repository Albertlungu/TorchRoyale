"""Decision Transformer inference engine."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from src.constants.game import GRID_COLS, GRID_ROWS
from src.data.feature_encoder import encode, FEATURE_DIM
from src.transformer.model import DTConfig, DecisionTransformer


class DTInference:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        cfg_dict = ckpt.get("config", {})
        self.cfg = DTConfig(**{k: v for k, v in cfg_dict.items() if hasattr(DTConfig, k)})
        self.model = DecisionTransformer(self.cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        K = self.cfg.context_len
        self._states   = np.zeros((K, FEATURE_DIM), dtype=np.float32)
        self._cards    = np.zeros(K, dtype=np.int64)
        self._pos      = np.zeros(K, dtype=np.int64)
        self._rtg      = np.zeros((K, 1), dtype=np.float32)
        self._timestep = 0
        self._t_buf    = np.zeros(K, dtype=np.int64)

    def reset(self) -> None:
        K = self.cfg.context_len
        self._states[:] = 0
        self._cards[:] = 0
        self._pos[:] = 0
        self._rtg[:] = 0
        self._t_buf[:] = 0
        self._timestep = 0

    def predict(self, state: Dict[str, Any], target_rtg: float = 1.0) -> Tuple[int, int]:
        """
        Args:
            state: frame state dict
            target_rtg: desired return-to-go (normalised)
        Returns:
            (card_idx, pos_flat) — card vocab index and flat tile index
        """
        K = self.cfg.context_len
        i = min(self._timestep, K - 1)

        # Shift buffers
        self._states = np.roll(self._states, -1, axis=0)
        self._cards  = np.roll(self._cards, -1)
        self._pos    = np.roll(self._pos, -1)
        self._rtg    = np.roll(self._rtg, -1, axis=0)
        self._t_buf  = np.roll(self._t_buf, -1)

        self._states[-1] = encode(state)
        self._rtg[-1]    = target_rtg
        self._t_buf[-1]  = self._timestep

        mask = np.zeros(K, dtype=np.float32)
        mask[max(0, K - i - 1):] = 1.0

        def to_t(arr, dtype=torch.float32):
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
        pos_flat = int(pos_logits[0, -1].argmax(-1).item())
        return card_idx, pos_flat

    def update_action(self, card_idx: int, pos_flat: int) -> None:
        self._cards[-1] = card_idx
        self._pos[-1]   = pos_flat
        self._timestep += 1
