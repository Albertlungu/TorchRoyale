#!/usr/bin/env python3
"""
View a replay with Decision Transformer recommendation overlay.

Opens an OpenCV window with the replay video and draws the suggested
card placement tile for each frame.

Usage:
  python view_replay.py data/replays/Game\\ 1.mov \\
      output/replay_runs/Game\\ 1_recommendations.jsonl
"""
from __future__ import annotations

import argparse

from src.overlay.player import VideoPlayer


def main() -> None:
    """Parse arguments and start the replay viewer."""
    parser = argparse.ArgumentParser(
        description="View a Clash Royale replay with DT recommendation overlay."
    )
    parser.add_argument("video", help="Path to the replay video.")
    parser.add_argument("jsonl", help="Path to the JSONL recommendations file.")
    parser.add_argument(
        "--production",
        action="store_true",
        help="Dim recommendations when the player cannot yet afford the card.",
    )
    args = parser.parse_args()

    player = VideoPlayer(args.video, args.jsonl, production=args.production)
    player.run()


if __name__ == "__main__":
    main()
