"""
Hyperparameter configuration for the Decision Transformer.
"""

from dataclasses import dataclass


@dataclass
class DTConfig:
    # Architecture
    state_dim: int = 97          # From feature_encoder.FEATURE_DIM
    card_action_dim: int = 4     # Hand card indices 0-3
    pos_action_dim: int = 576    # 32 * 18 tile positions
    n_heads: int = 4
    n_layers: int = 4
    embed_dim: int = 128
    dropout: float = 0.1
    context_length: int = 20     # K timesteps of history

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    max_epochs: int = 100
    grad_clip: float = 1.0

    # Loss weights
    card_loss_weight: float = 1.0
    pos_loss_weight: float = 1.0

    # Reward shaping
    tower_damage_weight: float = 0.5
    tower_destroy_bonus: float = 1.0
    king_tower_destroy_bonus: float = 2.0
    elixir_trade_weight: float = 0.1
    outcome_weight: float = 1.0

    # Data
    max_ep_len: int = 60         # Max card plays per game

    # Augmentation
    state_noise_std: float = 0.01
    rtg_noise_std: float = 0.05
