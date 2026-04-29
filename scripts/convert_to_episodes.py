#!/usr/bin/env python3
"""
Convert analysis JSONs to a training episodes pkl.
Merges with an existing pkl if --merge-with is provided.

Usage:
  python scripts/convert_to_episodes.py output/analysis \
      --output output/pkl/episodes.pkl \
      --merge-with output/pkl/existing.pkl
"""
import argparse, json, pickle, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.data.episode import build_episode, Episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("analyses_dir")
    parser.add_argument("--output", default="output/pkl/episodes.pkl")
    parser.add_argument("--merge-with", default=None)
    parser.add_argument("--outcomes", default=None,
                        help="JSON file mapping video stem -> win/loss/unknown")
    args = parser.parse_args()

    outcomes = {}
    if args.outcomes:
        outcomes = json.loads(Path(args.outcomes).read_text())

    analyses = sorted(Path(args.analyses_dir).rglob("*_analysis.json"))
    if not analyses:
        print("No analysis files found"); return

    episodes = []
    for path in analyses:
        print(f"Converting {path.name} ...", end=" ")
        data = json.loads(path.read_text())
        frames = data.get("frames", [])
        if not frames:
            print("empty — skipped"); continue

        stem = path.stem.replace("_analysis", "")
        outcome = outcomes.get(stem, "unknown")
        ep = build_episode(frames, outcome=outcome)
        if ep.length == 0:
            print(f"0 timesteps — skipped"); continue

        episodes.append(ep)
        print(f"{ep.length} timesteps")

    existing: list = []
    if args.merge_with and Path(args.merge_with).exists():
        with open(args.merge_with, "rb") as f:
            existing = pickle.load(f)
        print(f"Merging {len(existing)} existing + {len(episodes)} new = {len(existing)+len(episodes)} total")

    combined = existing + episodes
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(combined, f)
    print(f"\nWrote {len(combined)} episodes → {out}")


if __name__ == "__main__":
    main()
