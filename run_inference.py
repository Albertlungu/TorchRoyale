#!/usr/bin/env python3
"""
Quick script to run inference on a replay video.

Usage:
    python3 run_inference.py <video_path> [--frame-skip SKIP] [--checkpoint PATH]
"""

import argparse
from pathlib import Path
from src.overlay.inference_runner import InferenceRunner


def main():
    parser = argparse.ArgumentParser(description="Run DT inference on replay video")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--frame-skip", type=int, default=6, help="Frame skip (default: 6)")
    parser.add_argument("--checkpoint", default="output/models/best.pt", help="Model checkpoint")
    parser.add_argument("--output", help="Output JSONL path (default: auto-generated)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        return 1

    output_path = args.output or f"output/replay_runs/{video_path.stem}_recommendations.jsonl"

    print(f"Video: {video_path}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Frame skip: {args.frame_skip}")
    print(f"Output: {output_path}")
    print()

    runner = InferenceRunner(
        video_path=str(video_path),
        checkpoint_path=args.checkpoint,
        output_jsonl=output_path,
        analysis_output_dir="output/analysis",
        frame_skip=args.frame_skip,
    )

    result_path = runner.run()
    print(f"\nDone! Recommendations written to: {result_path}")
    return 0


if __name__ == "__main__":
    main()
