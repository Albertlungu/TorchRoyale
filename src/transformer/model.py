"""
Decision Transformer for Clash Royale.

Architecture: GPT-style causal transformer over (return, state, action)
triples. Predicts the next card to play and its placement tile.

Input sequence per timestep:
  [RTG token, State token, Action token]  ->  predicts next action

Public API:
  DTConfig           -- hyperparameter dataclass
  DecisionTransformer -- nn.Module implementing the full forward pass
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

from src.constants.cards import VOCAB_SIZE
from src.constants.game import GRID_COLS, GRID_ROWS
from src.data.feature_encoder import FEATURE_DIM


@dataclass
class DTConfig:
    """Hyperparameters for the Decision Transformer and training loop."""

    context_len: int = 30
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    dropout: float = 0.1
    card_loss_weight: float = 1.0
    pos_loss_weight: float = 1.0
    max_epochs: int = 400
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4

    @property
    def n_positions(self) -> int:
        """Total sequence length: RTG + state + action per context step."""
        return self.context_len * 3

    @property
    def n_cards(self) -> int:
        """Vocabulary size — number of distinct cards."""
        return VOCAB_SIZE

    @property
    def n_positions_tile(self) -> int:
        """Total number of grid tiles (flat position space)."""
        return GRID_COLS * GRID_ROWS


class DecisionTransformer(nn.Module):
    """
    Causal transformer that predicts card and placement from (RTG, state, action) sequences.

    The transformer decoder is used in a self-attention-only mode (memory == tgt)
    with a causal mask, following the Decision Transformer paper.
    """

    def __init__(self, cfg: DTConfig) -> None:
        """
        Args:
            cfg: DTConfig holding all architecture hyperparameters.
        """
        super().__init__()
        self.cfg = cfg
        d_model = cfg.d_model

        # Input projections
        self.state_proj = nn.Linear(FEATURE_DIM, d_model)
        self.rtg_proj   = nn.Linear(1, d_model)
        self.card_emb   = nn.Embedding(cfg.n_cards + 1, d_model)          # +1 for padding
        self.pos_emb    = nn.Embedding(cfg.n_positions_tile + 1, d_model)  # +1 for padding

        # Positional embedding over the interleaved sequence
        self.pos_enc = nn.Embedding(cfg.n_positions, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=cfg.n_head,
            dim_feedforward=d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=cfg.n_layer)

        self.ln = nn.LayerNorm(d_model)
        self.head_card = nn.Linear(d_model, cfg.n_cards)
        self.head_pos  = nn.Linear(d_model, cfg.n_positions_tile)
        self.drop = nn.Dropout(cfg.dropout)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Initialise linear and embedding weights with std=0.02."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        states: torch.Tensor,
        actions_card: torch.Tensor,
        actions_pos: torch.Tensor,
        rtg: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass over a batch of context sequences.

        Args:
            states:         (B, T, FEATURE_DIM) state feature vectors.
            actions_card:   (B, T) card vocabulary indices.
            actions_pos:    (B, T) flat tile position indices.
            rtg:            (B, T, 1) return-to-go values.
            timesteps:      (B, T) absolute timestep indices.
            attention_mask: (B, T) binary mask; 1 = valid token, 0 = padding.

        Returns:
            (card_logits, pos_logits) each of shape (B, T, n_cards/n_positions_tile).
        """
        batch_size, seq_len, _ = states.shape
        d_model = self.cfg.d_model

        state_tok = self.drop(self.state_proj(states))     # (B, T, d)
        rtg_tok   = self.drop(self.rtg_proj(rtg))          # (B, T, d)
        card_tok  = self.drop(self.card_emb(actions_card)) # (B, T, d)
        pos_tok   = self.drop(self.pos_emb(actions_pos))   # (B, T, d)
        act_tok   = card_tok + pos_tok

        # Interleave: RTG, State, Action -> shape (B, 3T, d)
        seq = torch.stack([rtg_tok, state_tok, act_tok], dim=2).view(
            batch_size, 3 * seq_len, d_model
        )
        pos_ids = torch.arange(3 * seq_len, device=states.device).unsqueeze(0)
        seq = seq + self.pos_enc(pos_ids)

        # Causal mask prevents attending to future positions
        causal_mask = torch.triu(
            torch.ones(3 * seq_len, 3 * seq_len, device=states.device), diagonal=1
        ).bool()

        # Expand (B, T) padding mask to (B, 3T)
        key_mask = attention_mask.repeat_interleave(3, dim=1).bool()

        out = self.transformer(
            tgt=seq,
            memory=seq,
            tgt_mask=causal_mask,
            memory_mask=causal_mask,
            tgt_key_padding_mask=~key_mask,
            memory_key_padding_mask=~key_mask,
        )
        out = self.ln(out)

        # Predict from state tokens (every 3rd token starting at index 1)
        state_out = out[:, 1::3]               # (B, T, d)
        card_logits = self.head_card(state_out) # (B, T, n_cards)
        pos_logits  = self.head_pos(state_out)  # (B, T, n_positions_tile)
        return card_logits, pos_logits
