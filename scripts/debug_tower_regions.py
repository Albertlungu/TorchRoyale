"""
Debug visualizer for tower HP OCR regions.

Draws the 6 tower OCR bounding boxes on sampled frames so you can verify
and adjust region positions in src/ocr/regions.py.

Usage:
  python scripts/debug_tower_regions.py data/replays/Game_1.mp4
  python scripts/debug_tower_regions.py data/replays/Game_1.mp4 --start 1000 --stride 300
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.ocr.regions import UIRegions

_OUTPUT_DIR = Path("output/debug_tower_regions")

# (label, attr_name, colour BGR)
_REGIONS = [
    ("opp_left",  "opponent_tower_left",  (0, 0, 220)),    # red
    ("opp_king",  "opponent_tower_king",  (0, 140, 255)),  # orange
    ("opp_right", "opponent_tower_right", (0, 0, 220)),    # red
    ("pl_left",   "player_tower_left",    (0, 200, 0)),    # green
    ("pl_king",   "player_tower_king",    (200, 200, 0)),  # cyan
    ("pl_right",  "player_tower_right",   (0, 200, 0)),    # green
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--start",  type=int, default=500)
    parser.add_argument("--stride", type=int, default=300)
    parser.add_argument("--count",  type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: {video_path} not found")
        sys.exit(1)

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ui = UIRegions(vid_w, vid_h)

    frame_idx = args.start
    saved = 0
    while saved < args.count:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break

        out = frame.copy()
        for label, attr, colour in _REGIONS:
            region = getattr(ui, attr)
            x1, y1, x2, y2 = region.to_tuple()
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(out, label, (x1, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)

        out_path = _OUTPUT_DIR / f"frame_{frame_idx}.jpg"
        cv2.imwrite(str(out_path), out)
        print(f"Saved: {out_path}")

        saved += 1
        frame_idx += args.stride

    cap.release()
    print(f"\nDone. Adjust fractions in src/ocr/regions.py and rerun.")


if __name__ == "__main__":
    main()
