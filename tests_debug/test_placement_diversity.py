"""
Test script to validate placement diversity after temperature sampling fix.

Loads a cached video analysis and runs Decision Transformer inference,
then reports tile diversity statistics.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
from collections import Counter

from src.transformer.inference import DTInference
from src.recommendation.strategy import DTStrategy


def test_diversity_from_analysis(analysis_path: str, checkpoint_path: str):
    """Run inference on cached analysis and check tile diversity."""

    print(f"Loading analysis: {analysis_path}")
    with open(analysis_path) as f:
        analysis = json.load(f)

    frames = analysis["frames"]
    print(f"Loaded {len(frames)} frames\n")

    print(f"Loading checkpoint: {checkpoint_path}")
    strategy = DTStrategy(checkpoint_path=checkpoint_path)
    strategy.reset_game()

    if not strategy.is_ready:
        print("ERROR: Strategy not ready (checkpoint failed to load)")
        return

    print(f"Config loaded:")
    print(f"  Device: {strategy._device}")
    print(f"  Target return: {strategy._target_return}")
    print(f"  Temperature: {strategy._temperature}")
    print(f"  Inference temperature: {strategy._inference.temperature}\n")

    # Run inference on all frames
    recommendations = []
    tiles = []
    cards = []

    for i, frame in enumerate(frames):
        rec = strategy.recommend(frame)
        if rec is not None:
            card, tile_x, tile_y = rec
            recommendations.append((card, tile_x, tile_y))
            tiles.append((tile_x, tile_y))
            cards.append(card)

            if i < 10:  # Print first 10 for inspection
                print(f"Frame {i:3d}: card={card:20s} tile=({tile_x:2d}, {tile_y:2d})")

    print(f"\n{'='*60}")
    print("DIVERSITY STATISTICS")
    print(f"{'='*60}")

    print(f"\nTotal recommendations: {len(recommendations)}")
    print(f"Unique tiles: {len(set(tiles))}")
    print(f"Unique cards: {len(set(cards))}")

    if len(tiles) > 0:
        tile_counts = Counter(tiles)
        top_tile, top_count = tile_counts.most_common(1)[0]
        top_pct = 100.0 * top_count / len(tiles)

        print(f"\nMost frequent tile: ({top_tile[0]}, {top_tile[1]}) - {top_count}/{len(tiles)} ({top_pct:.1f}%)")

        print(f"\nTop 10 tiles:")
        for tile, count in tile_counts.most_common(10):
            pct = 100.0 * count / len(tiles)
            print(f"  tile({tile[0]:2d}, {tile[1]:2d}): {count:4d} times ({pct:5.1f}%)")

        print(f"\nCard distribution:")
        card_counts = Counter(cards)
        for card, count in sorted(card_counts.items()):
            pct = 100.0 * count / len(cards)
            print(f"  {card:20s}: {count:4d} times ({pct:5.1f}%)")

        print(f"\n{'='*60}")
        print("SUCCESS CRITERIA")
        print(f"{'='*60}")

        diversity_ok = len(set(tiles)) >= 20
        top_tile_ok = top_pct < 30.0

        print(f"Unique tiles >= 20:        {'✓ PASS' if diversity_ok else '✗ FAIL'} ({len(set(tiles))})")
        print(f"Top tile frequency < 30%:  {'✓ PASS' if top_tile_ok else '✗ FAIL'} ({top_pct:.1f}%)")

        if diversity_ok and top_tile_ok:
            print(f"\n{'='*60}")
            print("✓ ALL TESTS PASSED - Placement diversity is good!")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print("✗ TESTS FAILED - Placement still shows collapse")
            print(f"{'='*60}")


def main():
    analysis_path = "output/replay_runs/analysis/ScreenRecording_03-30-2026 12-13-45_1_analysis.json"
    checkpoint_path = "output/models/best.pt"

    test_diversity_from_analysis(analysis_path, checkpoint_path)


if __name__ == "__main__":
    main()
