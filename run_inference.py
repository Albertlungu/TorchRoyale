#!/usr/bin/env python3
"""
Quick script to run inference on a replay video.

Usage:
    python3 run_inference.py <video_path> [--frame-skip SKIP] [--checkpoint PATH]
    python3 run_inference.py <video_path> --no-cache  # Force re-analysis
    python3 run_inference.py <video_path> --cached-analysis path/to/analysis.json
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

    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument("--no-cache", action="store_true",
                             help="Force re-analysis, ignore cached analysis")
    cache_group.add_argument("--cached-analysis",
                             help="Use specific cached analysis JSON file")

    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        return 1

    # Auto-generate output path if not provided
    if args.output:
        output_path = args.output
    else:
        output_path = f"output/replay_runs/{video_path.stem}_recommendations.jsonl"

    # Handle cached analysis
    analysis_output_dir = "output/analysis"
    if args.no_cache:
        # Delete cached analysis if it exists
        cached_json = Path(analysis_output_dir) / f"{video_path.stem}_analysis.json"
        if cached_json.exists():
            print(f"Removing cached analysis: {cached_json}")
            cached_json.unlink()
    elif args.cached_analysis:
        # Copy specified cached analysis to expected location
        import shutil
        cached_source = Path(args.cached_analysis)
        if not cached_source.exists():
            print(f"ERROR: Cached analysis not found: {cached_source}")
            return 1
        cached_dest = Path(analysis_output_dir) / f"{video_path.stem}_analysis.json"
        cached_dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"Using cached analysis: {cached_source}")
        shutil.copy(cached_source, cached_dest)

    print(f"Video: {video_path}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Frame skip: {args.frame_skip}")
    print(f"Output: {output_path}")
    print()

    runner = InferenceRunner(
        video_path=str(video_path),
        checkpoint_path=args.checkpoint,
        output_jsonl=output_path,
        analysis_output_dir=analysis_output_dir,
        frame_skip=args.frame_skip,
    )

    result_path = runner.run()
    print(f"\nDone! Recommendations written to: {result_path}")
    return 0


if __name__ == "__main__":
    main()
