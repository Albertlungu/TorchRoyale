"""
Tower region debug test.

Draws all 6 tower OCR bboxes on sampled frames and prints raw OCR + processed
HP/level for each tower. Saves annotated frames to output/test_towers/.

Usage:
  python tests/test_tower_regions.py data/replays/Game_23.mp4
  python tests/test_tower_regions.py data/replays/Game_23.mp4 --start 1000 --stride 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.ocr.detector import DigitDetector
from src.ocr.regions import UIRegions
from src.ocr.tower_tracker import TowerTracker, determine_outcome_from_readings

_OUTPUT_DIR = Path("output/test_towers")

# (label, region_attr, colour BGR, is_king)
_TOWER_DEFS = [
    ("opp_left",  "opponent_tower_left",  (0, 0, 220),   False),
    ("opp_king",  "opponent_tower_king",  (0, 140, 255), True),
    ("opp_right", "opponent_tower_right", (0, 0, 220),   False),
    ("pl_left",   "player_tower_left",    (0, 200, 0),   False),
    ("pl_king",   "player_tower_king",    (200, 200, 0), True),
    ("pl_right",  "player_tower_right",   (0, 200, 0),   False),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tower region OCR debug test.")
    parser.add_argument("video")
    parser.add_argument("--start",  type=int, default=500)
    parser.add_argument("--stride", type=int, default=200)
    parser.add_argument("--count",  type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: {video_path} not found")
        sys.exit(1)

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ocr = DigitDetector()
    cap = cv2.VideoCapture(str(video_path))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ui = UIRegions(vid_w, vid_h)

    # One tracker per tower, shared across all sampled frames to accumulate state
    trackers = {
        label: TowerTracker(is_king=is_king)
        for label, _, _, is_king in _TOWER_DEFS
    }

    # Track last HP reading per tower for outcome determination
    last_hp = {label: None for label, *_ in _TOWER_DEFS}

    print(f"Video: {vid_w}x{vid_h}, {total} frames")
    print(f"Sampling {args.count} frames from frame {args.start} every {args.stride}\n")

    frame_idx = args.start
    saved = 0

    while saved < args.count:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break

        out = frame.copy()
        print(f"--- frame {frame_idx} ---")

        # Pass 1: read princess towers only to establish levels
        for label, attr, colour, is_king in _TOWER_DEFS:
            if not is_king:
                raw = ocr.detect_tower_raw(frame, getattr(ui, attr).to_tuple())
                trackers[label].read(raw)

        # Propagate princess level to king before reading kings
        for king_lbl, left_lbl, right_lbl in [
            ("pl_king", "pl_left", "pl_right"),
            ("opp_king", "opp_left", "opp_right"),
        ]:
            for src in [left_lbl, right_lbl]:
                if trackers[src].level is not None:
                    trackers[king_lbl].set_level(trackers[src].level)
                    break

        # Pass 2: read all towers and render
        readings = {}
        for label, attr, colour, is_king in _TOWER_DEFS:
            region = getattr(ui, attr)
            x1, y1, x2, y2 = region.to_tuple()
            raw = ocr.detect_tower_raw(
                frame, region.to_tuple(),
                bottom_fraction=0.7 if is_king else 1.0,
                invert=is_king,
            )
            reading = trackers[label].read(raw)
            readings[label] = (raw, reading, x1, y1, x2, y2, colour)

        for label, (raw, reading, x1, y1, x2, y2, colour) in readings.items():
            if reading.hp is not None:
                last_hp[label] = reading.hp

            if reading.destroyed:
                status = "DESTROYED"
            elif reading.at_max:
                status = f"lvl={reading.level} HP=MAX({reading.hp})"
            else:
                status = f"lvl={reading.level} HP={reading.hp}"

            print(f"  {label:<12s}  raw='{raw}'  →  {status}")

            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(out, f"{label}: {raw}", (x1, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)
            cv2.putText(out, status, (x1, y2 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1, cv2.LINE_AA)

        out_path = _OUTPUT_DIR / f"frame_{frame_idx}.jpg"
        cv2.imwrite(str(out_path), out)
        print(f"  → saved {out_path}\n")

        saved += 1
        frame_idx += args.stride

    cap.release()

    # Final outcome
    print("=== Final tower state ===")
    player_labels  = ["pl_left", "pl_king", "pl_right"]
    opponent_labels = ["opp_left", "opp_king", "opp_right"]

    p_hps  = [last_hp[l] for l in player_labels]
    o_hps  = [last_hp[l] for l in opponent_labels]
    p_dead = [trackers[l].destroyed for l in player_labels]
    o_dead = [trackers[l].destroyed for l in opponent_labels]

    for label in player_labels + opponent_labels:
        t = trackers[label]
        print(f"  {label:<12s}  level={t.level}  destroyed={t.destroyed}  last_hp={last_hp[label]}")

    outcome = determine_outcome_from_readings(p_hps, p_dead, o_hps, o_dead)
    print(f"\nInferred outcome: {outcome.upper()}")
    print(f"Saved {saved} frames to {_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
