"""
CLI entry point for replay-based inference and overlay.

Modes
-----
preprocess   Analyze the full video, write a JSONL recommendation log,
             then open the video in a Tkinter window with the overlay.
headless     Analyze the video and write the JSONL only (no display).

Usage
-----
  python -m src.overlay.replay_runner \\
      --video data/replays/game.MP4 \\
      --checkpoint output/models/epoch_200.pt \\
      --mode preprocess \\
      --wait-state production

  python -m src.overlay.replay_runner \\
      --video data/replays/game.MP4 \\
      --checkpoint output/models/epoch_200.pt \\
      --mode headless
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="TorchRoyale replay inference + overlay",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to replay video (MP4).",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained .pt model checkpoint.",
    )
    parser.add_argument(
        "--mode",
        choices=["preprocess", "headless"],
        default="preprocess",
        help=(
            "preprocess: analyze video then show it with the overlay. "
            "headless: write JSONL only."
        ),
    )
    parser.add_argument(
        "--wait-state",
        choices=["debug", "production"],
        default="debug",
        help=(
            "debug: always show the recommendation. "
            "production: dim the overlay when elixir is too low."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="output/replay_runs",
        help="Directory for JSONL output and video analysis cache.",
    )
    args = parser.parse_args()

    video_stem = Path(args.video).stem
    output_dir = Path(args.output_dir)
    jsonl_path = output_dir / f"{video_stem}_recommendations.jsonl"

    # --- Inference ---
    from src.overlay.inference_runner import InferenceRunner

    runner = InferenceRunner(
        video_path=args.video,
        checkpoint_path=args.checkpoint,
        output_jsonl=str(jsonl_path),
        analysis_output_dir=str(output_dir / "analysis"),
    )
    runner.run()

    if args.mode == "headless":
        print(f"Done. Recommendations written to {jsonl_path}")
        return

    # --- Replay with overlay ---
    from src.overlay.video_player import VideoPlayer

    print(f"\nStarting overlay player (wait_state={args.wait_state!r})")
    print("Controls: [Space] pause  [H] toggle overlay  [Q] quit\n")

    player = VideoPlayer(
        video_path=args.video,
        jsonl_path=str(jsonl_path),
        wait_state=args.wait_state,
    )
    player.run()


if __name__ == "__main__":
    main()
