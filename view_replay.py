#!/usr/bin/env python3
"""
View a replay with DT recommendation overlay.

Usage:
  python view_replay.py data/replays/Game\ 1.mov output/replay_runs/Game\ 1_recommendations.jsonl
"""
import argparse
from src.overlay.player import VideoPlayer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("jsonl")
    parser.add_argument("--production", action="store_true")
    args = parser.parse_args()

    player = VideoPlayer(args.video, args.jsonl, production=args.production)
    player.run()


if __name__ == "__main__":
    main()
