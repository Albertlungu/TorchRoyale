#!/usr/bin/env python3
"""
Run DT inference on a replay video using a pre-computed analysis JSON.

Produces a JSONL recommendations file that can be visualised with view_replay.py.

Usage:
  python run_inference.py data/replays/Game\\ 1.mov --checkpoint data/models/dt/best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.overlay.inference_runner import InferenceRunner


def main() -> None:
    """Parse arguments and run the inference pass."""
    parser = argparse.ArgumentParser(
        description="Run Decision Transformer inference on a cached replay analysis."
    )
    parser.add_argument("video", help="Path to the replay video.")
    parser.add_argument("--checkpoint", default="data/models/dt/best.pt")
    parser.add_argument("--output", default=None)
    parser.add_argument("--analysis-dir", default="output/analysis")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    video = Path(args.video)
    output: str = args.output or f"output/replay_runs/{video.stem}_recommendations.jsonl"

    print(f"Video:      {video}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output:     {output}")
    print()

    runner = InferenceRunner(
        video_path=str(video),
        checkpoint_path=args.checkpoint,
        output_jsonl=output,
        analysis_dir=args.analysis_dir,
        device=args.device,
    )
    runner.run()


if __name__ == "__main__":
    main()
