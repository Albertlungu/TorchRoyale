"""
Decision Transformer for Clash Royale.

Treats card placement as a sequence modeling problem. The model receives
interleaved (return-to-go, state, action) tokens and predicts the next
action conditioned on desired future returns.

Architecture follows the original Decision Transformer paper
(Chen et al., 2021) with two-headed action prediction for card
selection and tile position.
"""

import torch
import torch.nn as nn

from .config import DTConfig


class DecisionTransformer(nn.Module):
    """
    Decision Transformer for Clash Royale card placement.

    Input sequence (interleaved per timestep):
        [RTG_1, State_1, Action_1, RTG_2, State_2, Action_2, ..., RTG_K, State_K, Action_K]

    Predicts actions from state token outputs using two heads:
        - Card head: which hand card to play (0-3)
        - Position head: where to place it (0-575, flattened 32x18 grid)
    """

    def __init__(self, config: DTConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim

        # --- Token embeddings ---

        self.state_embed = nn.Sequential(
            nn.Linear(config.state_dim, config.embed_dim),
            nn.Tanh(),
        )

        self.rtg_embed = nn.Sequential(
            nn.Linear(1, config.embed_dim),
            nn.Tanh(),
        )

        self.action_card_embed = nn.Embedding(config.card_action_dim, config.embed_dim)
        self.action_pos_embed = nn.Embedding(config.pos_action_dim, config.embed_dim)
        self.action_combine = nn.Linear(config.embed_dim * 2, config.embed_dim)

        # Timestep embedding (position within episode)
        self.timestep_embed = nn.Embedding(config.max_ep_len, config.embed_dim)

        # Token type embedding: 0=RTG, 1=state, 2=action
        self.token_type_embed = nn.Embedding(3, config.embed_dim)

        # Pre-transformer layer norm
        self.embed_ln = nn.LayerNorm(config.embed_dim)

        # --- Transformer ---

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.embed_dim,
            nhead=config.n_heads,
            dim_feedforward=config.embed_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.n_layers,
        )

        # --- Prediction heads ---

        self.card_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, config.card_action_dim),
        )

        self.pos_head = nn.Sequential(
            nn.Linear(config.embed_dim + config.card_action_dim, config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, config.pos_action_dim),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        states: torch.Tensor,
        actions_card: torch.Tensor,
        actions_pos: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """
        Forward pass.

        Args:
            states:        (B, K, 97)  float32
            actions_card:  (B, K)      long
            actions_pos:   (B, K)      long
            returns_to_go: (B, K, 1)   float32
            timesteps:     (B, K)      long
            attention_mask: (B, K)     float32 -- 1.0 real, 0.0 padding

        Returns:
            card_logits: (B, K, 4)
            pos_logits:  (B, K, 576)
        """
        B, K = states.shape[0], states.shape[1]
        device = states.device

        # Embed each token type -> (B, K, embed_dim)
        rtg_tokens = self.rtg_embed(returns_to_go)
        state_tokens = self.state_embed(states)

        card_emb = self.action_card_embed(actions_card)
        pos_emb = self.action_pos_embed(actions_pos)
        action_tokens = self.action_combine(
            torch.cat([card_emb, pos_emb], dim=-1)
        )

        # Add timestep embeddings
        timesteps_clamped = timesteps.clamp(0, self.config.max_ep_len - 1)
        time_emb = self.timestep_embed(timesteps_clamped)
        rtg_tokens = rtg_tokens + time_emb
        state_tokens = state_tokens + time_emb
        action_tokens = action_tokens + time_emb

        # Add token type embeddings
        type_ids = torch.arange(3, device=device)
        rtg_tokens = rtg_tokens + self.token_type_embed(type_ids[0])
        state_tokens = state_tokens + self.token_type_embed(type_ids[1])
        action_tokens = action_tokens + self.token_type_embed(type_ids[2])

        # Interleave: [RTG_1, S_1, A_1, RTG_2, S_2, A_2, ...] -> (B, 3K, embed_dim)
        seq = torch.stack([rtg_tokens, state_tokens, action_tokens], dim=2)
        seq = seq.reshape(B, 3 * K, self.embed_dim)

        # Expand attention mask: (B, K) -> (B, 3K)
        seq_mask = attention_mask.unsqueeze(-1).repeat(1, 1, 3).reshape(B, 3 * K)

        # Causal mask (upper triangular = -inf)
        causal_mask = torch.triu(
            torch.ones(3 * K, 3 * K, device=device) * float("-inf"),
            diagonal=1,
        )

        # Padding mask (True = ignore position)
        padding_mask = seq_mask == 0

        # Layer norm + transformer
        seq = self.embed_ln(seq)
        output = self.transformer(
            tgt=seq,
            memory=seq,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask,
            memory_key_padding_mask=padding_mask,
        )

        # Extract state token outputs at positions [1, 4, 7, ..., 3K-2]
        state_indices = torch.arange(K, device=device) * 3 + 1
        state_outputs = output[:, state_indices, :]

        # Card prediction
        card_logits = self.card_head(state_outputs)

        # Position prediction (conditioned on card)
        card_probs = torch.softmax(card_logits.detach(), dim=-1)
        pos_input = torch.cat([state_outputs, card_probs], dim=-1)
        pos_logits = self.pos_head(pos_input)

        return card_logits, pos_logits
