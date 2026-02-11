#!/usr/bin/env python3
"""
Test script for elixir tracking system.

Usage:
    # Test on a screenshot
    python test_elixir_tracking.py screenshot path/to/image.png

    # Test on a video (first 10 seconds)
    python test_elixir_tracking.py video path/to/video.mp4

    # Full video analysis
    python test_elixir_tracking.py video path/to/video.mp4 --full
"""

import sys
import cv2
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ocr import DigitDetector
from src.constants import UIRegions, GamePhase
from src.game_state import GamePhaseTracker, TowerHealthDetector
from src.recommendation.elixir_manager import OpponentElixirTracker
from detection_test import DetectionPipeline


def test_screenshot(image_path: str):
    """Test detection on a single screenshot."""
    print(f"\n{'='*60}")
    print(f"Testing on: {image_path}")
    print('='*60)

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image: {image_path}")
        return

    h, w = img.shape[:2]
    print(f"Image size: {w}x{h}")

    # Initialize components
    print("\nInitializing OCR (first run downloads model ~100MB)...")
    detector = DigitDetector()
    regions = UIRegions(w, h)
    health_detector = TowerHealthDetector()

    # Test elixir detection
    print("\n--- Elixir Detection ---")
    elixir_region = regions.elixir_number.to_tuple()
    print(f"Elixir region: {elixir_region}")
    result = detector.detect_elixir(img, elixir_region)
    print(f"Detected elixir: {result.value}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Raw text: '{result.raw_text}'")

    # Test timer detection
    print("\n--- Timer Detection ---")
    timer_region = regions.timer.to_tuple()
    print(f"Timer region: {timer_region}")
    timer = detector.detect_timer(img, timer_region)
    if timer is not None:
        minutes = timer // 60
        seconds = timer % 60
        print(f"Detected time: {minutes}:{seconds:02d} ({timer} seconds)")
    else:
        print("Timer not detected")

    # Test multiplier icon
    print("\n--- Multiplier Icon ---")
    mult_region = regions.multiplier_icon.to_tuple()
    print(f"Multiplier region: {mult_region}")
    mult = detector.detect_multiplier_icon(img, mult_region)
    print(f"Detected multiplier: x{mult}")

    # Run Roboflow detection for tower positions
    print("\n--- Roboflow Detection ---")
    pipeline = DetectionPipeline()
    rf_results = pipeline.process_image(image_path)
    tower_detections = []
    for det in rf_results["detections"]:
        if "tower" in det.class_name:
            tower_det = {
                "class_name": det.class_name,
                "pixel_x": det.pixel_x,
                "pixel_y": det.pixel_y,
                "pixel_width": det.pixel_width,
                "pixel_height": det.pixel_height,
                "is_opponent": det.is_opponent,
            }
            tower_detections.append(tower_det)
            side = "OPP" if det.is_opponent else "YOU"
            print(f"  {det.class_name} [{side}] @ ({det.pixel_x}, {det.pixel_y}) "
                  f"size {det.pixel_width}x{det.pixel_height}")

    # Detect tower levels from king towers
    print("\n--- Tower Level Detection ---")
    player_level = 15
    opponent_level = 15
    for det in tower_detections:
        if "king" in det["class_name"]:
            level = health_detector.detect_tower_level(
                img,
                det["pixel_x"], det["pixel_y"],
                det["pixel_width"], det["pixel_height"],
                det["is_opponent"],
            )
            if det["is_opponent"]:
                opponent_level = level
                print(f"  Opponent king tower level: {level}")
            else:
                player_level = level
                print(f"  Player king tower level: {level}")

    # Test tower health via OCR
    print("\n--- Tower Health (OCR) ---")
    tower_health = health_detector.detect_all_towers(
        img, tower_detections,
        player_level=player_level,
        opponent_level=opponent_level,
    )
    for name, th in tower_health.items():
        if th.detected:
            print(f"  {name}: {th.hp_current}/{th.hp_max} ({th.health_percent:.1f}%) "
                  f"raw='{th.raw_text}'")
        else:
            print(f"  {name}: FULL HP ({th.hp_max} max)")

    # Save debug image with regions drawn
    debug_img = img.copy()

    # Draw elixir region (green)
    x1, y1, x2, y2 = elixir_region
    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(debug_img, "Elixir", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw timer region (blue)
    x1, y1, x2, y2 = timer_region
    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(debug_img, "Timer", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Draw multiplier region (red)
    x1, y1, x2, y2 = mult_region
    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(debug_img, "Mult", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Draw tower HP search regions (cyan for player, magenta for opponent)
    for det in tower_detections:
        is_opp = det["is_opponent"]
        is_king = "king" in det["class_name"]
        color = (255, 0, 255) if is_opp else (255, 255, 0)
        hp_region = health_detector.get_hp_region(
            det["pixel_x"], det["pixel_y"],
            det["pixel_width"], det["pixel_height"],
            is_opp, w, h, is_king=is_king,
        )
        if hp_region:
            x1, y1, x2, y2 = hp_region
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
            label = f"{'opp' if is_opp else 'you'}-{det['class_name']}"
            cv2.putText(debug_img, label, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Save debug image
    output_path = Path(image_path).stem + "_debug.png"
    cv2.imwrite(output_path, debug_img)
    print(f"\nDebug image saved: {output_path}")
    print("(Shows detected regions - check if they align with actual UI)")


def test_video(video_path: str, full: bool = False):
    """Test detection on a video file."""
    print(f"\n{'='*60}")
    print(f"Testing on video: {video_path}")
    print('='*60)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {width}x{height}, {fps:.1f} FPS, {duration:.1f}s")

    # Initialize components
    print("\nInitializing components...")
    detector = DigitDetector()
    regions = UIRegions(width, height)
    phase_tracker = GamePhaseTracker()
    opponent_tracker = OpponentElixirTracker()

    # Process frames
    frame_skip = 6  # Every 6th frame
    max_frames = total_frames if full else int(fps * 10)  # 10 seconds unless --full

    print(f"\nProcessing {'all' if full else 'first 10 seconds of'} frames (every {frame_skip}th)...")
    print("-" * 60)

    frame_num = 0
    processed = 0

    while frame_num < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % frame_skip == 0:
            timestamp_ms = int((frame_num / fps) * 1000)

            # Detect elixir
            elixir_result = detector.detect_elixir(
                frame, regions.elixir_number.to_tuple()
            )

            # Detect multiplier
            mult = detector.detect_multiplier_icon(
                frame, regions.multiplier_icon.to_tuple()
            )

            # Update phase tracker
            phase = phase_tracker.update(multiplier_detected=mult)

            # Update opponent tracker (no detections in this simple test)
            opp_elixir = opponent_tracker.update(
                timestamp_ms=timestamp_ms,
                game_phase=phase,
                opponent_detections=[]
            )

            # Print status every second
            if processed % int(fps / frame_skip) == 0:
                time_str = f"{timestamp_ms/1000:.1f}s"
                print(
                    f"Time: {time_str:>6} | "
                    f"Player Elixir: {elixir_result.value:>2} | "
                    f"Opp Elixir: {opp_elixir:>5.1f} | "
                    f"Phase: {phase.value:>12} (x{phase_tracker.elixir_multiplier})"
                )

            processed += 1

        frame_num += 1

    cap.release()
    print("-" * 60)
    print(f"Processed {processed} frames")
    print(f"Opponent cards detected: {opponent_tracker.total_cards_played}")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print("\nAvailable test images:")
        test_dir = Path("tests/data")
        if test_dir.exists():
            for f in sorted(test_dir.glob("*.png"))[:5]:
                print(f"  - {f}")
        return

    mode = sys.argv[1]
    path = sys.argv[2]
    full = "--full" in sys.argv

    if mode == "screenshot":
        test_screenshot(path)
    elif mode == "video":
        test_video(path, full=full)
    else:
        print(f"Unknown mode: {mode}")
        print("Use 'screenshot' or 'video'")


if __name__ == "__main__":
    main()
