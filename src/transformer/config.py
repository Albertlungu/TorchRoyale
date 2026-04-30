"""
Hyperparameter configuration for the Decision Transformer.
"""

from dataclasses import dataclass


@dataclass
class DTConfig:
    """
    Configuration for the Decision Transformer model.

    Contains architecture hyperparameters, training settings, and reward
    shaping weights used for training the transformer-based agent.

    Attributes:
        state_dim (int): Feature dimension for the state vector (from feature_encoder.FEATURE_DIM).
        card_action_dim (int): Number of card actions (hand cards 0-3).
        pos_action_dim (int): Position action dimension (32x18 grid = 576 tiles).
        n_heads (int): Number of attention heads in transformer.
        n_layers (int): Number of transformer layers.
        embed_dim (int): Embedding dimension for transformer.
        dropout (float): Dropout rate for regularization.
        context_length (int): Context window size K (timesteps of history).
        batch_size (int): Training batch size.
        learning_rate (float): Optimizer learning rate.
        weight_decay (float): Weight decay for regularization.
        warmup_steps (int): Learning rate warmup steps.
        max_epochs (int): Maximum training epochs.
        grad_clip (float): Gradient clipping threshold.
        card_loss_weight (float): Weight for card selection loss.
        pos_loss_weight (float): Weight for position selection loss.
        tower_damage_weight (float): Reward weight for tower damage.
        tower_destroy_bonus (float): Bonus reward for destroying a tower.
        king_tower_destroy_bonus (float): Bonus reward for destroying king tower.
        elixir_trade_weight (float): Reward weight for elixir trades.
        outcome_weight (float): Reward weight for game outcome.
        max_ep_len (int): Maximum episode length (card plays per game).
        state_noise_std (float): State augmentation noise std.
        rtg_noise_std (float): Return-to-go augmentation noise std.
    """
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
