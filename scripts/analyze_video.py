#!/usr/bin/env python3
"""
Analyze a replay video using KataCR detection.

Usage:
  python scripts/analyze_video.py data/replays/Game\ 1.mov
  python scripts/analyze_video.py data/replays/Game\ 1.mov --frame-skip 6 --device mps
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.overlay.analyzer import VideoAnalyzer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--output-dir", default="output/analysis")
    parser.add_argument("--frame-skip", type=int, default=6)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--preload-ocr", action="store_true")
    args = parser.parse_args()

    analyzer = VideoAnalyzer(
        output_dir=args.output_dir,
        frame_skip=args.frame_skip,
        device=args.device,
        preload_ocr=args.preload_ocr,
    )
    analyzer.analyze_video(args.video)


if __name__ == "__main__":
    main()
