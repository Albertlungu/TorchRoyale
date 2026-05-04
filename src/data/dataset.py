"""
PyTorch dataset for Decision Transformer training.

DTDataset wraps a list of Episodes into windowed context sequences
suitable for the transformer's causal attention mechanism. Each sample
is a fixed-length context window ending at a particular timestep.

Public API:
  DTDataset   -- torch.utils.data.Dataset over episode timesteps
  load_dataset() -- load a pkl file and split into train/val DTDatasets
"""
from __future__ import annotations

import pickle
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.episode import Episode
from src.data.feature_encoder import FEATURE_DIM


class DTDataset(Dataset):
    """
    Windowed context dataset for Decision Transformer training.

    Each item is a dictionary of padded tensors covering a window of up to
    context_len timesteps ending at a sampled index.
    """

    def __init__(
        self,
        episodes: List[Episode],
        context_len: int = 30,
        state_noise_std: float = 0.0,
        rtg_noise_std: float = 0.0,
    ) -> None:
        """
        Args:
            episodes:        list of Episode objects.
            context_len:     maximum context window length (K).
            state_noise_std: gaussian noise standard deviation added to states during training.
            rtg_noise_std:   gaussian noise standard deviation added to RTG during training.
        """
        self.episodes: List[Episode] = episodes
        self.context_len: int = context_len
        self.state_noise_std: float = state_noise_std
        self.rtg_noise_std: float = rtg_noise_std

        # Build flat sample index: (episode_idx, timestep_idx)
        self._samples: List[Tuple[int, int]] = []
        for ep_idx, ep in enumerate(episodes):
            for step in range(ep.length):
                self._samples.append((ep_idx, step))

        all_rtgs: List[float] = [ts.rtg for ep in episodes for ts in ep.timesteps]
        self.rtg_mean: float = float(np.mean(all_rtgs)) if all_rtgs else 0.0
        self.rtg_std: float = max(float(np.std(all_rtgs)), 1.0) if all_rtgs else 1.0

    def __len__(self) -> int:
        """Return total number of (episode, timestep) samples."""
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Return a single padded context window as a dict of tensors.

        Args:
            idx: flat sample index.

        Returns:
            Dict with keys: states, actions_card, actions_pos, returns_to_go,
            timesteps, attention_mask.
        """
        ep_idx, step = self._samples[idx]
        ep = self.episodes[ep_idx]
        context_k = self.context_len

        start = max(0, step - context_k + 1)
        end = step + 1
        seq = ep.timesteps[start:end]
        actual_len = len(seq)
        pad = context_k - actual_len

        states: torch.Tensor = torch.zeros(context_k, FEATURE_DIM)
        actions_card: torch.Tensor = torch.zeros(context_k, dtype=torch.long)
        actions_pos: torch.Tensor = torch.zeros(context_k, dtype=torch.long)
        rtg: torch.Tensor = torch.zeros(context_k, 1)
        timesteps: torch.Tensor = torch.zeros(context_k, dtype=torch.long)
        mask: torch.Tensor = torch.zeros(context_k)

        for seq_i, ts in enumerate(seq):
            pos = pad + seq_i
            states[pos]       = torch.from_numpy(ts.state)
            actions_card[pos] = ts.action_card
            actions_pos[pos]  = ts.action_pos
            rtg[pos, 0]       = float((ts.rtg - self.rtg_mean) / self.rtg_std)
            timesteps[pos]    = start + seq_i
            mask[pos]         = 1.0

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
) -> Tuple[DTDataset, DTDataset]:
    """
    Load episodes from a pickle file and split into train/val datasets.

    Args:
        path:        path to a .pkl file containing a list of Episode objects.
        context_len: context window length passed to DTDataset.
        train_frac:  fraction of episodes used for training.

    Returns:
        (train_dataset, val_dataset) tuple.
    """
    with open(path, "rb") as pkl_file:
        episodes: List[Episode] = pickle.load(pkl_file)
    episode_count = len(episodes)
    split = max(1, int(episode_count * train_frac))
    train_eps = episodes[:split]
    val_eps: List[Episode] = episodes[split:]
    if not val_eps:
        val_eps = train_eps[-1:]
    return (
        DTDataset(train_eps, context_len=context_len, state_noise_std=0.01),
        DTDataset(val_eps,   context_len=context_len),
    )
