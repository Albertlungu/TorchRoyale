#!/usr/bin/env python3
"""
Convert analysis JSONs to a training episodes pickle file.

Optionally merges with an existing pickle if --merge-with is provided.
Each analysis JSON becomes one Episode; files with no valid placements are skipped.

Usage:
  python scripts/convert_to_episodes.py output/analysis \\
      --output output/pkl/episodes.pkl \\
      --merge-with output/pkl/existing.pkl
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.data.episode import Episode, build_episode


def main() -> None:
    """Parse arguments and convert analysis JSONs to an episodes pickle."""
    parser = argparse.ArgumentParser(
        description="Convert Clash Royale analysis JSONs to a training episodes pickle."
    )
    parser.add_argument("analyses_dir", help="Directory containing *_analysis.json files.")
    parser.add_argument("--output", default="output/pkl/episodes.pkl")
    parser.add_argument("--merge-with", default=None,
                        help="Existing .pkl file to merge new episodes into.")
    parser.add_argument(
        "--outcomes",
        default=None,
        help="JSON file mapping video stem to 'win'/'loss'/'unknown'.",
    )
    args = parser.parse_args()

    outcomes: Dict[str, str] = {}
    if args.outcomes:
        outcomes = json.loads(Path(args.outcomes).read_text(encoding="utf-8"))

    analyses = sorted(Path(args.analyses_dir).rglob("*_analysis.json"))
    if not analyses:
        print("No analysis files found")
        return

    episodes: List[Episode] = []
    for path in analyses:
        print(f"Converting {path.name} ...", end=" ")
        data = json.loads(path.read_text(encoding="utf-8"))
        frames = data.get("frames", [])
        if not frames:
            print("empty — skipped")
            continue

        stem = path.stem.replace("_analysis", "")
        outcome = outcomes.get(stem, "unknown")
        episode = build_episode(frames, outcome=outcome)
        if episode.length == 0:
            print("0 timesteps — skipped")
            continue

        episodes.append(episode)
        print(f"{episode.length} timesteps")

    existing: List[Episode] = []
    if args.merge_with and Path(args.merge_with).exists():
        with open(args.merge_with, "rb") as existing_file:
            existing = pickle.load(existing_file)
        total = len(existing) + len(episodes)
        print(f"Merging {len(existing)} existing + {len(episodes)} new = {total} total")

    combined = existing + episodes
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as out_file:
        pickle.dump(combined, out_file)
    print(f"\nWrote {len(combined)} episodes -> {out_path}")


if __name__ == "__main__":
    main()
