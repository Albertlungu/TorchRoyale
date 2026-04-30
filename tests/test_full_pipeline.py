"""
Comprehensive test for the full detection pipeline.

Runs detection every 120 frames starting at frame 1000 on a specified video.
Outputs annotated frames showing:
  1. Player cards (green boxes - from Cicadas model)
  2. Opponent cards (red boxes - from Vision Bot model)
  3. OCR regions with detected values (timer, elixir, multiplier)
  4. Hand cards with evo status

Usage:
  python tests/test_full_pipeline.py data/replays/Game_23.mp4 --start 1000 --stride 120
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.detection.dual_model_detector import DualModelDetector
from src.detection.hand_classifier import HandClassifier
from src.detection.hand_tracker import HandTracker
from src.ocr.detector import DigitDetector, OCRResult
from src.ocr.regions import UIRegions
from src.types import DetectionDict


def draw_onfield_detections(frame: np.ndarray, detections: list, mapper) -> np.ndarray:
    """
    Draw on-field detections with color-coded bounding boxes.

    Args:
        frame: BGR image array.
        detections: list of Detection objects.
        mapper: CoordinateMapper for tile grid overlay.

    Returns:
        Annotated frame.
    """
    result = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det.bbox_px
        # Green=player, Red=opponent
        color = (0, 255, 0) if not det.is_opponent else (0, 0, 255)
        thickness = 2

        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

        # Label with class name, confidence, and tile coords
        label = f"{det.class_name} {det.confidence:.2f} ({det.tile_x},{det.tile_y})"
        cv2.putText(result, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return result


def draw_hand_cards(
    frame: np.ndarray, hand: list, game_strip: Optional[Tuple[int, int]]
) -> np.ndarray:
    """
    Draw hand card slots with card names and evo status.

    Args:
        frame: BGR image array.
        hand: List of card name strings from HandTracker.
        game_strip: (x_left, x_right) pixel bounds.

    Returns:
        Annotated frame.
    """
    result = frame.copy()
    frame_h, frame_w = frame.shape[:2]

    # Hand area position (from hand_classifier.py)
    y_top = int(frame_h * 0.845)
    y_bot = int(frame_h * 0.965)

    if game_strip is None:
        x_left = 0
        x_right = frame_w
    else:
        x_left, x_right = game_strip

    next_end = x_left + int((x_right - x_left) * 0.115)
    cards_w = x_right - next_end
    slot_w = cards_w // 4

    for i, card in enumerate(hand):
        if card is None:
            continue

        x1 = next_end + i * slot_w
        x2 = x1 + slot_w

        # Determine color based on evo status
        if "evolution" in card:
            color = (255, 165, 0)  # Orange for evo
        else:
            color = (255, 255, 0)  # Cyan for normal

        # Draw slot rectangle
        cv2.rectangle(result, (x1, y_top), (x2, y_bot), color, 2)

        # Clean up card name for display
        display_name = card.replace("-in-hand", "").replace("-evolution", "").replace("-", " ")
        if "evolution" in card:
            display_name += " [EVO]"

        cv2.putText(
            result, display_name, (x1 + 5, y_top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )

    return result


def draw_ocr_regions(
    frame: np.ndarray,
    ui_regions: UIRegions,
    timer_secs: Optional[int],
    elixir: Optional[int],
    multiplier: int,
) -> np.ndarray:
    """
    Draw OCR regions with detected values.

    Args:
        frame: BGR image array.
        ui_regions: UIRegions object with timer, elixir, multiplier regions.
        timer_secs: Detected timer value in seconds.
        elixir: Detected elixir count.
        multiplier: Detected multiplier (1, 2, or 3).

    Returns:
        Annotated frame.
    """
    result = frame.copy()

    # Timer region (blue)
    t = ui_regions.timer
    cv2.rectangle(result, (t.x_min, t.y_min), (t.x_max, t.y_max), (255, 0, 0), 2)
    timer_str = f"Timer: {timer_secs}s" if timer_secs is not None else "Timer: N/A"
    cv2.putText(
        result, timer_str, (t.x_min, t.y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
    )

    # Elixir region (yellow)
    e = ui_regions.elixir_number
    cv2.rectangle(result, (e.x_min, e.y_min), (e.x_max, e.y_max), (0, 255, 255), 2)
    elixir_str = f"Elixir: {elixir}" if elixir is not None else "Elixir: N/A"
    cv2.putText(
        result, elixir_str, (e.x_min, e.y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
    )

    # Multiplier region (purple)
    m = ui_regions.multiplier_icon
    cv2.rectangle(result, (m.x_min, m.y_min), (m.x_max, m.y_max), (255, 0, 255), 2)
    cv2.putText(
        result,
        f"Mult: {multiplier}x",
        (m.x_min, m.y_min - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 255),
        2,
    )

    return result


def main() -> None:
    """Run the full pipeline test."""
    parser = argparse.ArgumentParser(description="Test full detection pipeline with visualization")
    parser.add_argument("video", help="Path to the replay video file")
    parser.add_argument("--start", type=int, default=1000, help="Start frame (default: 1000)")
    parser.add_argument("--stride", type=int, default=120, help="Frame stride (default: 120)")
    parser.add_argument("--count", type=int, default=20, help="Number of frames to process (default: 20)")
    parser.add_argument("--output-dir", default="output/test_frames", help="Output directory for frames")
    parser.add_argument("--device", default="auto", help="PyTorch device (default: auto)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize detectors
    print("Initializing detectors...")
    dual_detector = DualModelDetector(device=args.device)
    hand_clf = HandClassifier()
    ocr = DigitDetector()
    hand_tracker = HandTracker()

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {vid_w}x{vid_h} @ {fps}fps, {total_frames} frames")

    # Calibrate from start frame
    print(f"Calibrating from frame {args.start}...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    ret, calib_frame = cap.read()
    if ret:
        dual_detector.calibrate(calib_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    hand_tracker.reset()

    # Process frames
    frame_num = args.start
    saved = 0

    while saved < args.count:
        # Skip frames
        for _ in range(args.stride - 1):
            cap.grab()

        ret, frame = cap.read()
        if not ret:
            break

        # Rebuild UI regions per frame
        ui_regions = UIRegions(vid_w, vid_h)
        ui_regions.align_timer(frame)
        ui_regions.align_elixir(frame)
        ui_regions.align_multiplier(frame)

        # OCR detection
        timer_secs = ocr.detect_timer(frame, ui_regions.timer.to_tuple())
        elixir_result = ocr.detect_elixir(frame, ui_regions.elixir_number.to_tuple())
        multiplier = ocr.detect_multiplier(frame, ui_regions.multiplier_icon.to_tuple())
        player_elixir = elixir_result.value if elixir_result.detected else None

        # On-field detection
        field_result = dual_detector.detect(frame)

        # Hand detection
        x_left, x_right = (
            dual_detector._game_strip[:2] if dual_detector._game_strip else (None, None)
        )
        game_strip = (x_left, x_right) if x_left is not None else None

        classified = hand_clf.classify(frame, game_strip=game_strip)

        # Update hand tracker
        on_field_dets: List[DetectionDict] = [
            DetectionDict(
                class_name=det.class_name,
                tile_x=det.tile_x,
                tile_y=det.tile_y,
                is_opponent=det.is_opponent,
                is_on_field=True,
                confidence=det.confidence,
            )
            for det in field_result.on_field
        ]

        hand_dets: List[DetectionDict] = []
        for card_name in classified:
            if card_name:
                hand_dets.append(
                    DetectionDict(
                        class_name=f"{card_name}-in-hand",
                        tile_x=0,
                        tile_y=31,
                        is_opponent=False,
                        is_on_field=False,
                        confidence=1.0,
                    )
                )

        all_dets = on_field_dets + hand_dets
        tracked_hand = hand_tracker.update(all_dets, frame=frame, game_strip=game_strip)

        # Draw annotations
        annotated = frame.copy()
        annotated = draw_onfield_detections(annotated, field_result.on_field, dual_detector._mapper)
        annotated = draw_hand_cards(annotated, tracked_hand, game_strip)
        annotated = draw_ocr_regions(annotated, ui_regions, timer_secs, player_elixir, multiplier)

        # Add frame info
        info_text = f"Frame: {frame_num} | Timer: {timer_secs}s | Elixir: {player_elixir} | Mult: {multiplier}x"
        cv2.putText(
            annotated, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        # Save frame
        output_path = output_dir / f"frame_{frame_num}.jpg"
        cv2.imwrite(str(output_path), annotated)
        print(f"Saved: {output_path} ({len(field_result.on_field)} on-field detections)")

        frame_num += args.stride
        saved += 1

    cap.release()
    print(f"\nDone! Saved {saved} frames to {output_dir}")


if __name__ == "__main__":
    main()
