"""
Full pipeline test: cicadas, visionbot, and hand classifier.

Runs detection every --stride frames starting at --start on a specified video.
Saves annotated frames to output/test_frames/ with:
  - Green boxes: player cards (Cicadas model)
  - Red boxes:   opponent cards (Vision Bot model)
  - Blue boxes:  hand card slots (Hand Classifier)

Usage:
  python tests/test_full_pipeline.py data/replays/Game_23.mp4
  python tests/test_full_pipeline.py data/replays/Game_23.mp4 --start 1000 --stride 120
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.detection.dual_model_detector import DualModelDetector
from src.detection.hand_classifier import HandClassifier, get_next_bbox
from src.detection.result import Detection

_OUTPUT_DIR = Path("output/test_frames")

# Bbox colours (BGR)
_CICADAS_COLOUR = (0, 200, 0)  # green: player cards
_VISIONBOT_COLOUR = (0, 0, 220)  # red: opponent cards
_HAND_COLOUR = (200, 140, 0)  # blue: hand slots


def _draw_detections(
    frame: np.ndarray,
    on_field: List[Detection],
    hand_labels: List[Optional[str]],
    game_strip: Optional[Tuple[int, int]],
) -> np.ndarray:
    out = frame.copy()
    frame_h, frame_w = out.shape[:2]

    # On-field detections
    for det in on_field:
        if not det.bbox_px:
            continue
        x1, y1, x2, y2 = det.bbox_px
        colour = _VISIONBOT_COLOUR if det.is_opponent else _CICADAS_COLOUR
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(
            out,
            label,
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            colour,
            1,
            cv2.LINE_AA,
        )

    # Hand slots
    if game_strip is not None:
        x_left, x_right = game_strip
        game_w = x_right - x_left
        from src.detection.hand_classifier import (
            _INSET_FRAC,
            _NEXT_LEFT_FRAC,
            _NEXT_RIGHT_FRAC,
            _SLOT_OFFSET_FRACS,
            _Y_BOT_FRAC,
            _Y_TOP_FRAC,
        )

        next_end = x_left + int(game_w * _NEXT_RIGHT_FRAC)
        y_top = int(frame_h * _Y_TOP_FRAC)
        y_bot = int(frame_h * _Y_BOT_FRAC)
        cards_w = x_right - next_end
        slot_w = cards_w // 4
        inset = int(slot_w * _INSET_FRAC)

        for i, (label, offset_frac) in enumerate(zip(hand_labels, _SLOT_OFFSET_FRACS)):
            offset = int(slot_w * offset_frac)
            sx1 = next_end + i * slot_w + inset + offset
            sx2 = sx1 + slot_w - 2 * inset
            sx1 = max(0, sx1)
            sx2 = min(frame_w, sx2)
            cv2.rectangle(out, (sx1, y_top), (sx2, y_bot), _HAND_COLOUR, 2)
            if label:
                cv2.putText(
                    out,
                    label,
                    (sx1, y_top - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    _HAND_COLOUR,
                    1,
                    cv2.LINE_AA,
                )

        # Next-card box
        nx1, ny1, nx2, ny2 = get_next_bbox(frame_h, frame_w, x_left, x_right)
        cv2.rectangle(out, (nx1, ny1), (nx2, ny2), _HAND_COLOUR, 1)

    return out


def _detect_game_strip(frame: np.ndarray) -> Optional[Tuple[int, int]]:
    gray = np.mean(frame, axis=2) if frame.ndim == 3 else frame
    cols = np.where(np.mean(gray, axis=0) > 30)[0]
    if cols.size == 0:
        return None
    return int(cols.min()), int(cols.max())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full pipeline smoke test.")
    parser.add_argument("video", help="Path to input video file.")
    parser.add_argument("--start", type=int, default=0, help="First frame to test.")
    parser.add_argument(
        "--stride", type=int, default=120, help="Frame stride between samples."
    )
    parser.add_argument(
        "--count", type=int, default=20, help="Number of frames to process."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.45,
        help="Confidence threshold for on-field detections (default 0.1).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: video not found: {video_path}")
        sys.exit(1)

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Initializing detectors...")
    detector = DualModelDetector(conf_threshold=args.conf)
    hand_clf = HandClassifier()

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {vid_w}x{vid_h} @ {fps}fps, {total} frames")

    # Seek to start frame and calibrate
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    ok, cal_frame = cap.read()
    if not ok:
        print(f"Error: could not read frame {args.start}")
        sys.exit(1)

    print(f"Calibrating from frame {args.start}...")
    detector.calibrate(cal_frame)

    saved = 0
    frame_idx = args.start

    while saved < args.count:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break

        game_strip = _detect_game_strip(frame)
        gs_tuple = game_strip  # (x_left, x_right) or None

        field_result = detector.detect(frame)
        hand_labels = hand_clf.classify(frame, game_strip=gs_tuple)

        annotated = _draw_detections(
            frame, field_result.on_field, hand_labels, gs_tuple
        )

        out_path = _OUTPUT_DIR / f"frame_{frame_idx}.jpg"
        cv2.imwrite(str(out_path), annotated)

        player_dets = sum(1 for d in field_result.on_field if not d.is_opponent)
        opp_dets = sum(1 for d in field_result.on_field if d.is_opponent)
        hand_str = ", ".join(l or "?" for l in hand_labels)
        print(
            f"Saved: {out_path}  "
            f"({player_dets} player, {opp_dets} opponent on-field)  "
            f"hand=[{hand_str}]"
        )

        saved += 1
        frame_idx += args.stride

    cap.release()
    print(f"\nDone! Saved {saved} frames to {_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
