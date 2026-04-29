"""PyTorch dataset for Decision Transformer training."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.episode import Episode
from src.data.feature_encoder import FEATURE_DIM
from src.constants.cards import VOCAB_SIZE
from src.constants.game import GRID_COLS, GRID_ROWS


class DTDataset(Dataset):
    def __init__(
        self,
        episodes: List[Episode],
        context_len: int = 30,
        state_noise_std: float = 0.0,
        rtg_noise_std: float = 0.0,
    ):
        self.episodes = episodes
        self.context_len = context_len
        self.state_noise_std = state_noise_std
        self.rtg_noise_std = rtg_noise_std

        # Build flat sample index: (episode_idx, start_timestep)
        self._samples: List[tuple] = []
        for ep_idx, ep in enumerate(episodes):
            for t in range(ep.length):
                self._samples.append((ep_idx, t))

        states = np.vstack([t.state for ep in episodes for t in ep.timesteps])
        self.rtg_mean = float(np.mean([t.rtg for ep in episodes for t in ep.timesteps]))
        self.rtg_std = max(float(np.std([t.rtg for ep in episodes for t in ep.timesteps])), 1.0)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        ep_idx, t = self._samples[idx]
        ep = self.episodes[ep_idx]
        K = self.context_len
        L = ep.length

        start = max(0, t - K + 1)
        end = t + 1
        seq = ep.timesteps[start:end]
        actual_len = len(seq)
        pad = K - actual_len

        states       = torch.zeros(K, FEATURE_DIM)
        actions_card = torch.zeros(K, dtype=torch.long)
        actions_pos  = torch.zeros(K, dtype=torch.long)
        rtg          = torch.zeros(K, 1)
        timesteps    = torch.zeros(K, dtype=torch.long)
        mask         = torch.zeros(K)

        for i, ts in enumerate(seq):
            p = pad + i
            states[p]       = torch.from_numpy(ts.state)
            actions_card[p] = ts.action_card
            actions_pos[p]  = ts.action_pos
            rtg[p, 0]       = float((ts.rtg - self.rtg_mean) / self.rtg_std)
            timesteps[p]    = start + i
            mask[p]         = 1.0

        if self.state_noise_std > 0:
            states += torch.randn_like(states) * self.state_noise_std * mask.unsqueeze(-1)
        if self.rtg_noise_std > 0:
            rtg += torch.randn_like(rtg) * self.rtg_noise_std * mask.unsqueeze(-1)

        return {
            "states":         states,
            "actions_card":   actions_card,
            "actions_pos":    actions_pos,
            "returns_to_go":  rtg,
            "timesteps":      timesteps,
            "attention_mask": mask,
        }


def load_dataset(
    path: str,
    context_len: int = 30,
    train_frac: float = 0.8,
) -> tuple[DTDataset, DTDataset]:
    with open(path, "rb") as f:
        episodes: List[Episode] = pickle.load(f)
    n = len(episodes)
    split = max(1, int(n * train_frac))
    train_eps = episodes[:split]
    val_eps   = episodes[split:]
    if not val_eps:
        val_eps = train_eps[-1:]
    return (
        DTDataset(train_eps, context_len=context_len, state_noise_std=0.01),
        DTDataset(val_eps,   context_len=context_len),
    )
