"""
Quick placement validation test.

Loads the best checkpoint, runs predictions on training-data states at high
frame-skip, and asserts that predicted placements are never stuck at the
degenerate top-left corner (the placement bug).

Run:
    python tests/test_placement.py
"""

import pickle
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.transformer.inference import DTInference
from src.grid.validity_masks import PlacementValidator
from src.grid.coordinate_mapper import CoordinateMapper


CHECKPOINT = PROJECT_ROOT / "output" / "models" / "best.pt"
EPISODES_PKL = PROJECT_ROOT / "output" / "pkl" / "all_episodes.pkl"

# These are the degenerate positions the old NaN model always predicted
DEGENERATE_SPELLS = (0, 0)    # argmax(NaN) = 0 decoded to tile_x=0, tile_y=0
DEGENERATE_TROOPS = (0, 17)   # fallback from (0,0) for troops


def run():
    assert CHECKPOINT.exists(), f"Checkpoint not found: {CHECKPOINT}"
    assert EPISODES_PKL.exists(), f"Episodes not found: {EPISODES_PKL}"

    with open(EPISODES_PKL, "rb") as f:
        episodes = pickle.load(f)

    assert episodes, "No episodes in pkl file"

    infer = DTInference(checkpoint_path=str(CHECKPOINT), device="cpu")
    mapper = CoordinateMapper()
    validator = PlacementValidator(mapper)

    # Confirm no NaN in weights
    has_nan = any(
        torch.isnan(p).any().item() for p in infer.model.parameters()
    )
    assert not has_nan, "Model has NaN weights — training diverged"

    # Sample states at high frame-skip (every 30 timesteps) across all episodes
    FRAME_SKIP = 30
    predictions = []
    degenerate_count = 0
    total = 0

    for ep in episodes:
        infer.reset()
        for i in range(0, ep.length, FRAME_SKIP):
            ts = ep.timesteps[i]
            card_pred, pos_flat = infer.predict(ts.state)

            tile_y = pos_flat // 18
            tile_x = pos_flat % 18
            total += 1

            # Record whether prediction is degenerate
            if (tile_x, tile_y) in (DEGENERATE_SPELLS, DEGENERATE_TROOPS):
                degenerate_count += 1

            # Record action so context stays consistent
            infer.update_action(card_pred, pos_flat)
            predictions.append((tile_x, tile_y, card_pred))

    assert total > 0, "No predictions were made"

    # Count unique positions predicted (NaN model always produces 1-2 unique positions)
    unique_positions = len(set((tx, ty) for tx, ty, _ in predictions))
    degenerate_rate = degenerate_count / total

    print(f"Predictions: {total}")
    print(f"Unique positions predicted: {unique_positions}")
    print(f"Degenerate (top-left corner) rate: {degenerate_rate:.1%} ({degenerate_count}/{total})")
    print(f"card_pred distribution: {dict(sorted({c: sum(1 for _,_,cc in predictions if cc==c) for c in range(4)}.items()))}")

    # Sample some positions
    print("Sample predictions (tile_x, tile_y, card):")
    for tx, ty, c in predictions[:15]:
        print(f"  card={c}, tile_x={tx}, tile_y={ty}")

    # Count predictions that landed on the player's side (tile_y >= 17) —
    # all training placements are on the player's side, so a healthy model
    # must predict player-side tiles the overwhelming majority of the time.
    player_side_count = sum(1 for _, ty, _ in predictions if ty >= 17)
    player_side_rate = player_side_count / total

    print(f"Player-side placement rate: {player_side_rate:.1%} ({player_side_count}/{total})")

    # The NaN model produced 100% degenerate placements (argmax(NaN) = 0).
    # A fixed model must: predict multiple distinct positions and never collapse
    # to the two known-degenerate corners, and place on the player's side.
    assert unique_positions >= 3, (
        f"Only {unique_positions} unique positions — model may still be degenerate"
    )
    assert degenerate_rate < 0.50, (
        f"Degenerate placement rate {degenerate_rate:.1%} is too high (> 50%)"
    )
    assert player_side_rate >= 0.80, (
        f"Only {player_side_rate:.1%} of placements are on the player's side (expected >= 80%)"
    )

    print("\nPLACEMENT TEST PASSED")


if __name__ == "__main__":
    run()
