#!/usr/bin/env python3
"""
Process a video containing multiple games into separate episodes.

Usage:
    python3 process_multi_game_video.py data/replays/ClashGameplay.mov --num-games 4
"""

import argparse
import pickle
import sys
from pathlib import Path

from src.data.episode_builder import build_episodes_from_multi_game_video
from src.data.outcome_detector import GameOutcome


def main():
    parser = argparse.ArgumentParser(
        description="Process multi-game video into episodes"
    )
    parser.add_argument("video", help="Path to video file")
    parser.add_argument(
        "--num-games", type=int, help="Expected number of games (for validation)"
    )
    parser.add_argument(
        "--outcome",
        default="win",
        choices=["win", "loss", "draw"],
        help="Outcome for all games (default: win)",
    )
    parser.add_argument(
        "--frame-skip", type=int, default=6, help="Frame skip (default: 6)"
    )
    parser.add_argument(
        "--output", default="output/pkl/multi_game_episodes.pkl", help="Output pkl path"
    )
    parser.add_argument(
        "--merge-with-existing",
        action="store_true",
        help="Merge with existing all_episodes.pkl",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        return 1

    # Map outcome string to enum
    outcome_map = {
        "win": GameOutcome.WIN,
        "loss": GameOutcome.LOSS,
        "draw": GameOutcome.DRAW,
    }
    outcome = outcome_map[args.outcome]

    print(f"Processing {video_path}")
    print(f"Expected: {args.num_games} games, all {args.outcome}s")
    print()

    # Build episodes
    episodes = build_episodes_from_multi_game_video(
        video_path=str(video_path),
        num_games=args.num_games,
        outcome=outcome,
        frame_skip=args.frame_skip,
        verbose=True,
    )

    if not episodes:
        print("\nERROR: No episodes created")
        return 1

    # Optionally merge with existing episodes
    if args.merge_with_existing:
        existing_path = Path("output/pkl/all_episodes.pkl")
        if existing_path.exists():
            print(f"\nMerging with existing {existing_path}")
            with open(existing_path, "rb") as f:
                existing_episodes = pickle.load(f)
            print(f"  Existing: {len(existing_episodes)} episodes")
            episodes = existing_episodes + episodes
            print(f"  Combined: {len(episodes)} episodes")
            args.output = str(existing_path)  # Write back to all_episodes.pkl

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(episodes, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nSaved {len(episodes)} episodes to {output_path}")

    # Summary
    total_timesteps = sum(ep.length for ep in episodes)
    wins = sum(1 for ep in episodes if ep.outcome == GameOutcome.WIN)
    losses = sum(1 for ep in episodes if ep.outcome == GameOutcome.LOSS)
    unknown = sum(1 for ep in episodes if ep.outcome == GameOutcome.UNKNOWN)

    print(f"\nSummary:")
    print(f"  Total episodes: {len(episodes)}")
    print(f"  Total timesteps: {total_timesteps}")
    print(f"  Wins: {wins}")
    print(f"  Losses: {losses}")
    print(f"  Unknown: {unknown}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
