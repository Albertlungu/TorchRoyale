"""
Diagnostic script to trace Decision Transformer inference behavior.

Runs inference on synthetic states and logs:
- Raw model logits at each step
- Context window growth
- Action history accumulation
- Top-5 position predictions
"""

import torch
import numpy as np
from pathlib import Path

from src.transformer.inference import DTInference
from src.data.feature_encoder import FEATURE_DIM


def create_dummy_state(frame_idx: int) -> dict:
    """Create synthetic game state with variation."""
    return {
        "timestamp_ms": frame_idx * 100,
        "player_elixir": 5 + (frame_idx % 5),
        "opponent_elixir": 4,
        "player_tower_hp": [2500 - frame_idx * 10, 2500 - frame_idx * 10, 4000 - frame_idx * 20],
        "opponent_tower_hp": [2400, 2400, 3900],
        "hand_cards": ["knight", "archers", "fireball", "musketeer"],
        "next_card": "giant",
        "detections": [],
    }


def main():
    checkpoint_path = "output/models/best.pt"

    print("=" * 80)
    print(f"Loading checkpoint: {checkpoint_path}")
    print("=" * 80)

    inference = DTInference(checkpoint_path, device="cpu", target_return=3.0)

    print(f"\nModel config:")
    print(f"  context_length K = {inference.config.context_length}")
    print(f"  state_dim = {inference.config.state_dim}")
    print(f"  card_action_dim = {inference.config.card_action_dim}")
    print(f"  pos_action_dim = {inference.config.pos_action_dim}")
    print(f"\nInference settings:")
    print(f"  target_return = {inference.target_return:.3f}")
    print(f"  rtg_mean = {inference.rtg_mean:.3f}")
    print(f"  rtg_std = {inference.rtg_std:.3f}")
    print(f"  device = {inference.device}")
    print()

    # Run 15 predictions
    for i in range(15):
        state = create_dummy_state(i)

        print("=" * 80)
        print(f"STEP {i}")
        print("=" * 80)

        # Prediction
        card_idx, pos_flat = inference.predict(state)

        tile_y = pos_flat // 18
        tile_x = pos_flat % 18

        print(f"\n[PREDICTION]")
        print(f"  card_idx = {card_idx}")
        print(f"  pos_flat = {pos_flat}  ->  tile({tile_x}, {tile_y})")

        # Internal context state
        print(f"\n[CONTEXT STATE]")
        print(f"  _states length = {len(inference._states)}")
        print(f"  _cards length = {len(inference._cards)}")
        print(f"  _positions length = {len(inference._positions)}")
        print(f"  _rtgs length = {len(inference._rtgs)}")
        print(f"  _current_rtg = {inference._current_rtg:.3f}")

        # Reconstruct inputs to inspect
        T = len(inference._states)
        K = min(T, inference.K)
        start = T - K

        states = torch.zeros(1, K, FEATURE_DIM, dtype=torch.float32)
        actions_card = torch.zeros(1, K, dtype=torch.long)
        actions_pos = torch.zeros(1, K, dtype=torch.long)
        rtg = torch.zeros(1, K, 1, dtype=torch.float32)
        timesteps = torch.zeros(1, K, dtype=torch.long)
        mask = torch.ones(1, K, dtype=torch.float32)

        DEFAULT_POS = 24 * 18 + 9

        for j in range(K):
            idx = start + j
            states[0, j] = torch.from_numpy(inference._states[idx])
            rtg[0, j, 0] = (inference._rtgs[idx] - inference.rtg_mean) / max(inference.rtg_std, 1e-6)
            timesteps[0, j] = idx

            if idx < len(inference._cards):
                actions_card[0, j] = inference._cards[idx]
                actions_pos[0, j] = inference._positions[idx]
            else:
                actions_pos[0, j] = DEFAULT_POS

        if K < inference.K:
            pad = inference.K - K
            states = torch.nn.functional.pad(states, (0, 0, pad, 0))
            actions_card = torch.nn.functional.pad(actions_card, (pad, 0))
            actions_pos = torch.nn.functional.pad(actions_pos, (pad, 0))
            rtg = torch.nn.functional.pad(rtg, (0, 0, pad, 0))
            timesteps = torch.nn.functional.pad(timesteps, (pad, 0))
            mask = torch.nn.functional.pad(mask, (pad, 0))
            actions_pos[0, :pad] = DEFAULT_POS

        states = states.to(inference.device)
        actions_card = actions_card.to(inference.device)
        actions_pos = actions_pos.to(inference.device)
        rtg = rtg.to(inference.device)
        timesteps = timesteps.to(inference.device)
        mask = mask.to(inference.device)

        with torch.no_grad():
            card_logits, pos_logits = inference.model(
                states, actions_card, actions_pos, rtg, timesteps, mask
            )

        last_idx = inference.K - 1
        pos_probs = torch.softmax(pos_logits[0, last_idx], dim=-1)
        top5_pos = pos_probs.topk(5)

        print(f"\n[TOP-5 POSITION LOGITS]")
        for rank, (prob, pos_idx) in enumerate(zip(top5_pos.values, top5_pos.indices)):
            py = pos_idx.item() // 18
            px = pos_idx.item() % 18
            print(f"  {rank+1}. pos={pos_idx.item():3d}  tile({px:2d}, {py:2d})  prob={prob.item():.6f}")

        print(f"\n[ACTION HISTORY]")
        if len(inference._cards) == 0:
            print("  (empty - no actions recorded yet)")
        else:
            for j, (c, p) in enumerate(zip(inference._cards, inference._positions)):
                py = p // 18
                px = p % 18
                print(f"  {j}: card={c}  pos={p:3d}  tile({px:2d}, {py:2d})")

        # Update action
        inference.update_action(card_idx, pos_flat, reward=0.0)
        print(f"\n[UPDATED] Recorded action: card={card_idx}, pos={pos_flat}\n")

    print("=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
