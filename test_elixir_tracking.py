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

    # Run Roboflow detection
    print("\n--- Roboflow Detection ---")
    pipeline = DetectionPipeline()
    rf_results = pipeline.process_image(image_path)
    all_detections = rf_results["detections"]
    mapper = rf_results["mapper"]

    tower_detections = []
    friendly_cards = []
    opponent_cards = []

    for det in all_detections:
        side = "OPP" if det.is_opponent else "YOU"
        status = "field" if det.is_on_field else "hand"
        print(f"  {det.class_name:25s} [{side}] tile=({det.tile_x:2d},{det.tile_y:2d}) "
              f"pixel=({det.pixel_x:4d},{det.pixel_y:4d}) "
              f"conf={det.confidence:.2f} [{status}]")

        if "tower" in det.class_name:
            tower_detections.append({
                "class_name": det.class_name,
                "pixel_x": det.pixel_x,
                "pixel_y": det.pixel_y,
                "pixel_width": det.pixel_width,
                "pixel_height": det.pixel_height,
                "is_opponent": det.is_opponent,
            })
        elif det.is_opponent:
            opponent_cards.append(det)
        else:
            friendly_cards.append(det)

    print(f"\n  Summary: {len(all_detections)} total "
          f"({len(tower_detections)} towers, {len(friendly_cards)} friendly, "
          f"{len(opponent_cards)} opponent)")

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
        if th.is_destroyed:
            print(f"  {name}: DESTROYED (0/{th.hp_max})")
        elif th.detected:
            print(f"  {name}: {th.hp_current}/{th.hp_max} ({th.health_percent:.1f}%) "
                  f"raw='{th.raw_text}'")
        elif th.health_percent == 100.0:
            print(f"  {name}: FULL HP ({th.hp_max} max)")
        else:
            print(f"  {name}: OCR FAILED (hp unknown) raw='{th.raw_text}'")

    # Save debug image with all overlays
    debug_img = img.copy()

    # Draw grid overlay (green, semi-transparent)
    overlay = debug_img.copy()
    for gx in range(mapper.GRID_WIDTH + 1):
        px = int(gx * mapper.tile_width + mapper.bounds.x_min)
        cv2.line(overlay,
                 (px, mapper.bounds.y_min), (px, mapper.bounds.y_max),
                 (0, 255, 0), 2)
    for gy in range(mapper.GRID_HEIGHT + 1):
        py = int(gy * mapper.tile_height + mapper.bounds.y_min)
        cv2.line(overlay,
                 (mapper.bounds.x_min, py), (mapper.bounds.x_max, py),
                 (0, 255, 0), 2)
    debug_img = cv2.addWeighted(overlay, 0.5, debug_img, 0.5, 0)

    # Draw all Roboflow detection bounding boxes
    for det in all_detections:
        bx1 = det.pixel_x - det.pixel_width // 2
        by1 = det.pixel_y - det.pixel_height // 2
        bx2 = det.pixel_x + det.pixel_width // 2
        by2 = det.pixel_y + det.pixel_height // 2
        color = (0, 100, 255) if det.is_opponent else (255, 200, 0)
        cv2.rectangle(debug_img, (bx1, by1), (bx2, by2), color, 2)
        label = f"{det.class_name} ({det.tile_x},{det.tile_y})"
        cv2.putText(debug_img, label, (bx1, by1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # Draw OCR regions: elixir (green), timer (blue), multiplier (red)
    for region, label, color in [
        (elixir_region, "Elixir", (0, 255, 0)),
        (timer_region, "Timer", (255, 0, 0)),
        (mult_region, "Mult", (0, 0, 255)),
    ]:
        x1, y1, x2, y2 = region
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(debug_img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

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
            label = f"HP:{'opp' if is_opp else 'you'}-{'king' if is_king else 'princess'}"
            cv2.putText(debug_img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Draw summary text overlay at top-left
    info_lines = [
        f"Elixir: {result.value}  Timer: {timer}s  Mult: x{mult}",
    ]
    for name in ["player_left", "player_king", "player_right",
                 "opponent_left", "opponent_king", "opponent_right"]:
        th = tower_health.get(name)
        if th is None:
            info_lines.append(f"{name}: ?")
        elif th.is_destroyed:
            info_lines.append(f"{name}: DESTROYED")
        elif th.detected:
            info_lines.append(f"{name}: {th.hp_current}/{th.hp_max}")
        elif th.health_percent == 100.0:
            info_lines.append(f"{name}: FULL ({th.hp_max})")
        else:
            info_lines.append(f"{name}: OCR FAIL")

    for i, line in enumerate(info_lines):
        y_pos = 30 + i * 25
        cv2.putText(debug_img, line, (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(debug_img, line, (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Save debug image
    output_path = Path(image_path).stem + "_debug.png"
    cv2.imwrite(output_path, debug_img)
    print(f"\nDebug image saved: {output_path}")
    print("(Grid + detections + OCR regions + tower HP)")


def test_video(video_path: str, full: bool = False):
    """Test full pipeline on a video file."""
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
    health_detector = TowerHealthDetector()
    pipeline = DetectionPipeline()

    # Tower level tracking
    player_level = 15
    opponent_level = 15
    levels_detected = False
    last_tower_hp = {}

    # Process frames
    frame_skip = 6  # Every 6th frame
    max_frames = total_frames if full else int(fps * 10)  # 10 seconds unless --full

    print(f"\nProcessing {'all' if full else 'first 10 seconds of'} frames (every {frame_skip}th)...")
    print("-" * 100)

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

            # Run Roboflow detection
            try:
                rf_results = pipeline.process_image_array(frame)
                raw_dets = rf_results.get("detections", [])
            except Exception:
                raw_dets = []

            # Extract tower detections and opponent on-field cards
            tower_dets = []
            opponent_on_field = []
            for det in raw_dets:
                if "tower" in det.class_name:
                    tower_dets.append({
                        "class_name": det.class_name,
                        "pixel_x": det.pixel_x,
                        "pixel_y": det.pixel_y,
                        "pixel_width": det.pixel_width,
                        "pixel_height": det.pixel_height,
                        "is_opponent": det.is_opponent,
                    })
                if det.is_opponent and det.is_on_field:
                    opponent_on_field.append(det)

            # Detect tower levels (once)
            if not levels_detected and tower_dets:
                for td in tower_dets:
                    if "king" in td["class_name"]:
                        level = health_detector.detect_tower_level(
                            frame,
                            td["pixel_x"], td["pixel_y"],
                            td["pixel_width"], td["pixel_height"],
                            td["is_opponent"],
                        )
                        if td["is_opponent"]:
                            opponent_level = level
                        else:
                            player_level = level
                levels_detected = True

            # Detect tower health
            tower_health = {}
            if tower_dets:
                tower_health = health_detector.detect_all_towers(
                    frame, tower_dets,
                    player_level=player_level,
                    opponent_level=opponent_level,
                )

            # Use last known HP for OCR failures on princess towers
            for name, result in tower_health.items():
                if result.health_percent is not None:
                    last_tower_hp[name] = result
                elif name in last_tower_hp:
                    tower_health[name] = last_tower_hp[name]

            # Update opponent tracker
            opp_elixir = opponent_tracker.update(
                timestamp_ms=timestamp_ms,
                game_phase=phase,
                opponent_detections=opponent_on_field
            )

            # Print status every second
            if processed % max(1, int(fps / frame_skip)) == 0:
                time_str = f"{timestamp_ms/1000:.1f}s"

                # Build tower HP summary
                tower_strs = []
                for name in ["player_left", "player_king", "player_right",
                             "opponent_left", "opponent_king", "opponent_right"]:
                    th = tower_health.get(name)
                    if th is None:
                        tower_strs.append(f"{name}: ?")
                    elif th.is_destroyed:
                        tower_strs.append(f"{name}: DEAD")
                    elif th.detected:
                        tower_strs.append(f"{name}: {th.hp_current}")
                    else:
                        tower_strs.append(f"{name}: FULL" if th.health_percent == 100.0 else f"{name}: ?")

                print(
                    f"Time: {time_str:>6} | "
                    f"Elixir: {elixir_result.value:>2} | "
                    f"Opp: {opp_elixir:>5.1f} | "
                    f"Phase: {phase.value:>12} (x{phase_tracker.elixir_multiplier})"
                )
                print(f"  Towers: {' | '.join(tower_strs)}")

            processed += 1

        frame_num += 1

    cap.release()
    print("-" * 100)
    print(f"Processed {processed} frames")
    print(f"Opponent cards detected: {opponent_tracker.total_cards_played}")
    print(f"Tower levels - Player: {player_level}, Opponent: {opponent_level}")


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
