#!/usr/bin/env python3
"""
Simple viewer for pre-existing inference results.

Usage:
    python3 view_replay.py <video_path> <recommendations_jsonl>
    python3 view_replay.py <video_path> <recommendations_jsonl> --production
"""

import argparse
import sys
from pathlib import Path

from src.overlay.video_player import VideoPlayer


def main():
    parser = argparse.ArgumentParser(
        description="View replay with DT recommendations overlay"
    )
    parser.add_argument("video", help="Path to replay video")
    parser.add_argument("jsonl", help="Path to recommendations JSONL file")
    parser.add_argument(
        "--production",
        action="store_true",
        help="Dim overlay when elixir is insufficient (default: always show)",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    jsonl_path = Path(args.jsonl)

    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        return 1

    if not jsonl_path.exists():
        print(f"ERROR: Recommendations file not found: {jsonl_path}")
        return 1

    wait_state = "production" if args.production else "debug"

    print(f"Video: {video_path}")
    print(f"Recommendations: {jsonl_path}")
    print(f"Wait state: {wait_state}")
    print()
    print("Controls:")
    print("  Space / k  -- play / pause")
    print("  h          -- toggle overlay on/off")
    print("  q / Escape -- quit")
    print()

    player = VideoPlayer(
        video_path=str(video_path),
        jsonl_path=str(jsonl_path),
        wait_state=wait_state,
    )
    player.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
