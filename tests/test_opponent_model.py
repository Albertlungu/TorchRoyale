"""
Opponent model test: enemy card detection only.

Runs a YOLOv8 opponent model on sampled frames and saves annotated images to
output/test_opponent_model/ with red bounding boxes.

Usage:
  python tests/test_opponent_model.py data/replays/Game_23.mp4
  python tests/test_opponent_model.py data/replays/Game_23.mp4 --start 1000 --stride 120 --conf 0.1
  python tests/test_opponent_model.py --weights data/models/onfield/torchroyale-enemies-best.pt data/replays/Game_23.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parents[1]))

from ultralytics import YOLO

_OUTPUT_DIR = Path("output/test_opponent_model")
_DEFAULT_WEIGHTS = Path("data/models/onfield/torchroyale-enemies-best.pt")
_COLOUR = (0, 0, 220)  # red


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Opponent model smoke test.")
    parser.add_argument("video", help="Path to input video file.")
    parser.add_argument(
        "--weights",
        default=str(_DEFAULT_WEIGHTS),
        help="Path to opponent .pt weights file.",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stride", type=int, default=120)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument(
        "--class-filter",
        default=None,
        help="Optional class name filter (e.g. 'hero-knight').",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    video_path = Path(args.video)
    weights_path = Path(args.weights)

    if not video_path.exists():
        print(f"Error: video not found: {video_path}")
        sys.exit(1)
    if not weights_path.exists():
        print(f"Error: weights not found: {weights_path}")
        sys.exit(1)

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading opponent model from {weights_path}...")
    model = YOLO(str(weights_path))
    print(f"Classes: {list(model.names.values())}")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {vid_w}x{vid_h} @ {fps}fps, {total} frames")

    saved = 0
    frame_idx = args.start
    class_filter = args.class_filter.lower().strip() if args.class_filter else None

    while saved < args.count:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(frame, conf=args.conf, verbose=False)[0]
        out = frame.copy()

        count = 0
        if results.boxes is not None:
            for box in results.boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls = box
                name = model.names[int(cls)]
                if class_filter and name.lower() != class_filter:
                    continue
                cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), _COLOUR, 2)
                cv2.putText(
                    out,
                    f"{name} {conf:.2f}",
                    (int(x1), max(int(y1) - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    _COLOUR,
                    1,
                    cv2.LINE_AA,
                )
                count += 1

        out_path = _OUTPUT_DIR / f"frame_{frame_idx}.jpg"
        cv2.imwrite(str(out_path), out)
        if class_filter:
            print(f"Saved: {out_path}  ({count} '{class_filter}' detections)")
        else:
            print(f"Saved: {out_path}  ({count} detections)")

        saved += 1
        frame_idx += args.stride

    cap.release()
    print(f"\nDone! Saved {saved} frames to {_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
