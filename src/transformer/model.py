"""
Decision Transformer for Clash Royale.

Architecture: GPT-style causal transformer over (return, state, action)
triples. Predicts the next card to play and its placement tile.

Input sequence per timestep:
  [RTG token, State token, Action token]  →  predicts next action
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from src.constants.cards import VOCAB_SIZE
from src.constants.game import GRID_COLS, GRID_ROWS
from src.data.feature_encoder import FEATURE_DIM


@dataclass
class DTConfig:
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
        return self.context_len * 3  # RTG + state + action per step

    @property
    def n_cards(self) -> int:
        return VOCAB_SIZE

    @property
    def n_positions_tile(self) -> int:
        return GRID_COLS * GRID_ROWS


class DecisionTransformer(nn.Module):
    def __init__(self, cfg: DTConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # Input projections
        self.state_proj = nn.Linear(FEATURE_DIM, d)
        self.rtg_proj   = nn.Linear(1, d)
        self.card_emb   = nn.Embedding(cfg.n_cards + 1, d)        # +1 for padding
        self.pos_emb    = nn.Embedding(cfg.n_positions_tile + 1, d)  # +1 for padding

        # Positional embedding over the sequence
        self.pos_enc = nn.Embedding(cfg.n_positions, d)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=cfg.n_head,
            dim_feedforward=d * 4,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=cfg.n_layer)

        self.ln = nn.LayerNorm(d)
        self.head_card = nn.Linear(d, cfg.n_cards)
        self.head_pos  = nn.Linear(d, cfg.n_positions_tile)
        self.drop = nn.Dropout(cfg.dropout)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        states: torch.Tensor,          # (B, T, FEATURE_DIM)
        actions_card: torch.Tensor,    # (B, T)
        actions_pos: torch.Tensor,     # (B, T)
        rtg: torch.Tensor,             # (B, T, 1)
        timesteps: torch.Tensor,       # (B, T)
        attention_mask: torch.Tensor,  # (B, T)
    ):
        B, T, _ = states.shape
        d = self.cfg.d_model

        state_tok = self.drop(self.state_proj(states))        # (B, T, d)
        rtg_tok   = self.drop(self.rtg_proj(rtg))             # (B, T, d)
        card_tok  = self.drop(self.card_emb(actions_card))    # (B, T, d)
        pos_tok   = self.drop(self.pos_emb(actions_pos))      # (B, T, d)
        act_tok   = card_tok + pos_tok

        # Interleave: RTG, State, Action → shape (B, 3T, d)
        seq = torch.stack([rtg_tok, state_tok, act_tok], dim=2).view(B, 3 * T, d)
        pos_ids = torch.arange(3 * T, device=states.device).unsqueeze(0)
        seq = seq + self.pos_enc(pos_ids)

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(3 * T, 3 * T, device=states.device), diagonal=1
        ).bool()

        # Expand attention mask (B, T) → (B, 3T) then to (B*n_head, 3T, 3T)
        key_mask = attention_mask.repeat_interleave(3, dim=1).bool()  # (B, 3T)

        out = self.transformer(
            tgt=seq,
            memory=seq,
            tgt_mask=causal_mask,
            memory_mask=causal_mask,
            tgt_key_padding_mask=~key_mask,
            memory_key_padding_mask=~key_mask,
        )
        out = self.ln(out)

        # Predict from state tokens (positions 1, 4, 7, … = indices 1, 4, 7, …)
        state_out = out[:, 1::3]   # (B, T, d)

        card_logits = self.head_card(state_out)  # (B, T, n_cards)
        pos_logits  = self.head_pos(state_out)   # (B, T, n_pos)
        return card_logits, pos_logits
