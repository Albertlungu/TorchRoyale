"""
PyTorch Dataset for Decision Transformer training.

Serves sliding windows of K timesteps from episodes. Short episodes
are left-padded with zeros. Long episodes yield multiple windows.
"""

import pickle
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.episode_builder import Episode
from src.data.feature_encoder import FEATURE_DIM


class DecisionTransformerDataset(Dataset):
    """
    Dataset that returns fixed-length context windows from game episodes.

    Each sample contains K consecutive timesteps with states, actions,
    returns-to-go, timestep indices, and attention masks.
    """

    def __init__(
        self,
        episodes: List[Episode],
        context_length: int = 20,
        state_noise_std: float = 0.0,
        rtg_noise_std: float = 0.0,
    ):
        """
        Args:
            episodes: List of Episode objects (from episode_builder).
            context_length: Number of timesteps per sample (K).
            state_noise_std: Gaussian noise std for state augmentation (0 = disabled).
            rtg_noise_std: Noise std for RTG augmentation (0 = disabled).
        """
        self.episodes = episodes
        self.context_length = context_length
        self.state_noise_std = state_noise_std
        self.rtg_noise_std = rtg_noise_std

        # Compute RTG normalization stats
        self.rtg_mean, self.rtg_std = self._compute_rtg_stats()

        # Build sampling index
        self.samples: List[Tuple[int, int, int]] = []
        self._build_index()

    def _compute_rtg_stats(self) -> Tuple[float, float]:
        """Compute global mean and std of returns-to-go across all episodes."""
        all_rtg = []
        for ep in self.episodes:
            if ep.returns_to_go is not None:
                all_rtg.extend(ep.returns_to_go.tolist())

        if not all_rtg:
            return 0.0, 1.0

        mean = float(np.mean(all_rtg))
        std = float(np.std(all_rtg))
        if std < 1e-6:
            std = 1.0
        return mean, std

    def _build_index(self):
        """Build (episode_idx, start, length) tuples for all valid windows."""
        K = self.context_length
        for ep_idx, ep in enumerate(self.episodes):
            T = ep.length
            if T == 0:
                continue
            if T <= K:
                self.samples.append((ep_idx, 0, T))
            else:
                for start in range(T - K + 1):
                    self.samples.append((ep_idx, start, K))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        ep_idx, start, length = self.samples[idx]
        ep = self.episodes[ep_idx]
        K = self.context_length

        # Initialize padded tensors
        states = torch.zeros(K, FEATURE_DIM, dtype=torch.float32)
        actions_card = torch.zeros(K, dtype=torch.long)
        actions_pos = torch.zeros(K, dtype=torch.long)
        returns_to_go = torch.zeros(K, 1, dtype=torch.float32)
        timesteps = torch.zeros(K, dtype=torch.long)
        attention_mask = torch.zeros(K, dtype=torch.float32)

        # Fill from the right (left-pad for short episodes)
        pad = K - length
        for i in range(length):
            ts = ep.timesteps[start + i]
            states[pad + i] = torch.from_numpy(ts.state_vec)
            actions_card[pad + i] = ts.action_card
            actions_pos[pad + i] = ts.action_pos
            timesteps[pad + i] = start + i
            attention_mask[pad + i] = 1.0

            # Normalized RTG
            raw_rtg = ep.returns_to_go[start + i]
            returns_to_go[pad + i, 0] = (float(raw_rtg) - self.rtg_mean) / self.rtg_std

        # Data augmentation (training only, controlled by std > 0)
        if self.state_noise_std > 0:
            noise = torch.randn_like(states) * self.state_noise_std
            states = states + noise * attention_mask.unsqueeze(-1)

        if self.rtg_noise_std > 0:
            noise = torch.randn_like(returns_to_go) * self.rtg_noise_std
            returns_to_go = returns_to_go + noise * attention_mask.unsqueeze(-1)

        return {
            "states": states,
            "actions_card": actions_card,
            "actions_pos": actions_pos,
            "returns_to_go": returns_to_go,
            "timesteps": timesteps,
            "attention_mask": attention_mask,
        }


def load_dataset(
    episodes_path: str,
    context_length: int = 20,
    state_noise_std: float = 0.0,
    rtg_noise_std: float = 0.0,
) -> DecisionTransformerDataset:
    """
    Load episodes from disk and create a dataset.

    Args:
        episodes_path: Path to pickled episodes file.
        context_length: Context window size K.
        state_noise_std: State augmentation noise.
        rtg_noise_std: RTG augmentation noise.

    Returns:
        DecisionTransformerDataset ready for DataLoader.
    """
    with open(episodes_path, "rb") as f:
        episodes = pickle.load(f)

    return DecisionTransformerDataset(
        episodes=episodes,
        context_length=context_length,
        state_noise_std=state_noise_std,
        rtg_noise_std=rtg_noise_std,
    )
