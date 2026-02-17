#!/usr/bin/env python3
"""
Test pipeline for TorchRoyale with full debug output.

This pipeline prioritizes accuracy and debugging over speed:
- Full console output
- Debug image generation for every frame
- Comprehensive logging
- All detection operations run every frame

Usage:
    python pipeline_test.py screenshot path/to/image.png
    python pipeline_test.py video path/to/video.mp4
    python pipeline_test.py video path/to/video.mp4 --full
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

# Suppress optional model dependency warnings
os.environ.setdefault("QWEN_2_5_ENABLED", "False")
os.environ.setdefault("QWEN_3_ENABLED", "False")
os.environ.setdefault("CORE_MODEL_SAM_ENABLED", "False")
os.environ.setdefault("CORE_MODEL_SAM3_ENABLED", "False")
os.environ.setdefault("CORE_MODEL_GAZE_ENABLED", "False")
os.environ.setdefault("CORE_MODEL_YOLO_WORLD_ENABLED", "False")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from detection_test import DetectionPipeline
from src.constants import UIRegions
from src.game_state import GamePhaseTracker, TowerHealthDetector
from src.ocr import DigitDetector
from src.recommendation.elixir_manager import OpponentElixirTracker

# Debug output root
DEBUG_DIR = Path("debug")


def _draw_annotated_frame(
    frame,
    mapper,
    all_detections,
    tower_detections,
    health_detector,
    tower_health,
    elixir_result,
    timer,
    mult,
    w,
    h,
):
    """Draw all debug overlays on a frame and return the annotated image."""
    debug_img = frame.copy()

    # Draw grid overlay (green, semi-transparent)
    if mapper is not None:
        overlay = debug_img.copy()
        for gx in range(mapper.GRID_WIDTH + 1):
            px = int(gx * mapper.tile_width + mapper.bounds.x_min)
            cv2.line(
                overlay,
                (px, mapper.bounds.y_min),
                (px, mapper.bounds.y_max),
                (0, 255, 0),
                2,
            )
        for gy in range(mapper.GRID_HEIGHT + 1):
            py = int(gy * mapper.tile_height + mapper.bounds.y_min)
            cv2.line(
                overlay,
                (mapper.bounds.x_min, py),
                (mapper.bounds.x_max, py),
                (0, 255, 0),
                2,
            )
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
        cv2.putText(
            debug_img, label, (bx1, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1
        )

    # Draw tower HP search regions
    for det in tower_detections:
        is_opp = det["is_opponent"]
        is_king = "king" in det["class_name"]
        color = (255, 0, 255) if is_opp else (255, 255, 0)
        hp_region = health_detector.get_hp_region(
            det["pixel_x"],
            det["pixel_y"],
            det["pixel_width"],
            det["pixel_height"],
            is_opp,
            w,
            h,
            is_king=is_king,
        )
        if hp_region:
            x1, y1, x2, y2 = hp_region
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
            label = (
                f"HP:{'opp' if is_opp else 'you'}-{'king' if is_king else 'princess'}"
            )
            cv2.putText(
                debug_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )

    # Draw summary text overlay
    info_lines = [
        f"Elixir: {elixir_result.value if elixir_result else '?'} | Timer: {timer}s | Mult: x{mult}",
    ]

    for name in [
        "player_left",
        "player_king",
        "player_right",
        "opponent_left",
        "opponent_king",
        "opponent_right",
    ]:
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
        cv2.putText(
            debug_img, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3
        )
        cv2.putText(
            debug_img,
            line,
            (10, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    return debug_img


def test_screenshot(image_path: str):
    """Test detection on a single screenshot with full debug output."""
    print(f"\n{'=' * 60}")
    print(f"Testing screenshot: {image_path}")
    print("=" * 60)

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image: {image_path}")
        return

    h, w = img.shape[:2]
    print(f"Image size: {w}x{h}")

    # Initialize components
    print("\nInitializing components...")
    detector = DigitDetector()
    regions = UIRegions(w, h)
    health_detector = TowerHealthDetector()
    pipeline = DetectionPipeline()

    # Test elixir detection
    print("\n--- Elixir Detection ---")
    elixir_region = regions.elixir_number.to_tuple()
    print(f"Elixir region: {elixir_region}")
    elixir_result = detector.detect_elixir(img, elixir_region)
    print(f"Detected elixir: {elixir_result.value}")
    print(f"Confidence: {elixir_result.confidence:.2f}")
    print(f"Raw text: '{elixir_result.raw_text}'")

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
    rf_results = pipeline.process_image(image_path)
    all_detections = rf_results["detections"]
    mapper = rf_results["mapper"]

    tower_detections = []
    for det in all_detections:
        side = "OPP" if det.is_opponent else "YOU"
        status = "field" if det.is_on_field else "hand"
        print(
            f"  {det.class_name:25s} [{side}] tile=({det.tile_x:2d},{det.tile_y:2d}) "
            f"pixel=({det.pixel_x:4d},{det.pixel_y:4d}) conf={det.confidence:.2f} [{status}]"
        )

        if "tower" in det.class_name:
            tower_detections.append(
                {
                    "class_name": det.class_name,
                    "pixel_x": det.pixel_x,
                    "pixel_y": det.pixel_y,
                    "pixel_width": det.pixel_width,
                    "pixel_height": det.pixel_height,
                    "is_opponent": det.is_opponent,
                }
            )

    # Tower health detection
    print("\n--- Tower Health Detection ---")
    tower_health = {}
    if tower_detections:
        tower_health = health_detector.detect_all_towers(
            img, tower_detections, player_level=15, opponent_level=15
        )

    for name, th in tower_health.items():
        if th.is_destroyed:
            print(f"  {name}: DESTROYED (0/{th.hp_max})")
        elif th.detected:
            print(
                f"  {name}: {th.hp_current}/{th.hp_max} ({th.health_percent:.1f}%) "
                f"raw='{th.raw_text}'"
            )
        elif th.health_percent == 100.0:
            print(f"  {name}: FULL HP ({th.hp_max} max)")
        else:
            print(f"  {name}: OCR FAILED raw='{th.raw_text}'")

    # Save debug image
    out_dir = DEBUG_DIR / "screenshots"
    out_dir.mkdir(parents=True, exist_ok=True)

    debug_img = _draw_annotated_frame(
        img,
        mapper,
        all_detections,
        tower_detections,
        health_detector,
        tower_health,
        elixir_result,
        timer,
        mult,
        w,
        h,
    )

    # Draw OCR regions
    for region, label, color in [
        (elixir_region, "Elixir", (0, 255, 0)),
        (timer_region, "Timer", (255, 0, 0)),
        (mult_region, "Mult", (0, 0, 255)),
    ]:
        x1, y1, x2, y2 = region
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            debug_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )

    output_path = out_dir / f"{Path(image_path).stem}_debug.png"
    cv2.imwrite(str(output_path), debug_img)
    print(f"\nDebug image saved: {output_path}")


def test_video(video_path: str, full: bool = False):
    """Test full pipeline on a video with comprehensive debug output."""
    print(f"\n{'=' * 60}")
    print(f"Testing video: {video_path}")
    print("=" * 60)

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

    # Create debug output directory
    video_name = Path(video_path).stem
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = DEBUG_DIR / "videos" / f"{video_name}_{run_timestamp}"
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    print(f"Debug output: {run_dir}/")

    # Initialize components
    print("\nInitializing components...")
    detector = DigitDetector()
    regions = UIRegions(width, height)
    phase_tracker = GamePhaseTracker()
    opponent_tracker = OpponentElixirTracker()
    health_detector = TowerHealthDetector()
    pipeline = DetectionPipeline()

    # Processing parameters - prioritize accuracy over speed
    frame_skip = 3  # Process every 3rd frame for thorough testing
    max_frames = total_frames if full else int(fps * 5)  # 5 seconds unless --full
    save_interval = max(1, int(fps / frame_skip))  # Save ~1 frame per second

    print(
        f"\nProcessing {'all' if full else 'first 5 seconds of'} frames (every {frame_skip}th)..."
    )
    print(f"Saving debug frames every ~1 second to {frames_dir}/")
    print("-" * 100)

    frame_num = 0
    processed = 0
    mapper = None

    # Game state tracking
    player_level = 15
    opponent_level = 15
    levels_detected = False
    last_tower_hp = {}

    # Performance tracking
    processing_times = []
    start_time = time.time()

    while frame_num < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % frame_skip == 0:
            frame_start_time = time.time()
            timestamp_ms = int((frame_num / fps) * 1000)

            # OCR operations (every frame for testing)
            elixir_result = detector.detect_elixir(
                frame, regions.elixir_number.to_tuple()
            )
            mult = detector.detect_multiplier_icon(
                frame, regions.multiplier_icon.to_tuple()
            )
            timer = detector.detect_timer(frame, regions.timer.to_tuple())

            # Update phase tracker
            phase = phase_tracker.update(multiplier_detected=mult)

            # Run Roboflow detection
            all_dets = []
            try:
                rf_results = pipeline.process_image_array(frame)
                all_dets = rf_results.get("detections", [])
                mapper = rf_results.get("mapper", mapper)
            except Exception as e:
                print(f"Warning: Detection failed on frame {frame_num}: {e}")
                all_dets = []

            # Extract tower detections and opponent cards
            tower_dets = []
            opponent_on_field = []
            for det in all_dets:
                if "tower" in det.class_name:
                    tower_dets.append(
                        {
                            "class_name": det.class_name,
                            "pixel_x": det.pixel_x,
                            "pixel_y": det.pixel_y,
                            "pixel_width": det.pixel_width,
                            "pixel_height": det.pixel_height,
                            "is_opponent": det.is_opponent,
                        }
                    )
                if det.is_opponent and det.is_on_field:
                    opponent_on_field.append(det)

            # Detect tower levels (once)
            if not levels_detected and tower_dets:
                for td in tower_dets:
                    if "king" in td["class_name"]:
                        level = health_detector.detect_tower_level(
                            frame,
                            td["pixel_x"],
                            td["pixel_y"],
                            td["pixel_width"],
                            td["pixel_height"],
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
                    frame,
                    tower_dets,
                    player_level=player_level,
                    opponent_level=opponent_level,
                )

            # Use last known HP for failures
            for name, res in tower_health.items():
                if res.health_percent is not None:
                    last_tower_hp[name] = res
                elif name in last_tower_hp:
                    tower_health[name] = last_tower_hp[name]

            # Update opponent tracker
            opp_elixir = opponent_tracker.update(
                timestamp_ms=timestamp_ms,
                game_phase=phase,
                opponent_detections=opponent_on_field,
            )

            frame_processing_time = time.time() - frame_start_time
            processing_times.append(frame_processing_time)

            # Console output every frame for testing
            time_str = f"{timestamp_ms / 1000:.1f}s"
            print(
                f"Frame {frame_num:6d} ({time_str:>6}) | "
                f"Elixir: {elixir_result.value:>2} | "
                f"Opp: {opp_elixir:>5.1f} | "
                f"Phase: {phase.value:>12} | "
                f"Towers: {len(tower_dets)} | "
                f"Process: {frame_processing_time * 1000:.1f}ms"
            )

            # Save debug frame only at save_interval (~1 per second)
            if mapper is not None and processed % save_interval == 0:
                debug_frame = _draw_annotated_frame(
                    frame,
                    mapper,
                    all_dets,
                    tower_dets,
                    health_detector,
                    tower_health,
                    elixir_result,
                    timer,
                    mult,
                    width,
                    height,
                )
                frame_path = frames_dir / f"frame_{frame_num:06d}_{timestamp_ms}ms.png"
                cv2.imwrite(str(frame_path), debug_frame)

            processed += 1

        frame_num += 1

    cap.release()

    # Performance summary
    total_time = time.time() - start_time
    avg_processing_time = (
        sum(processing_times) / len(processing_times) if processing_times else 0
    )

    print("-" * 100)
    print("TEST COMPLETE")
    print(f"Processed {processed} frames in {total_time:.2f}s")
    print(f"Average processing time per frame: {avg_processing_time * 1000:.1f}ms")
    print(
        f"Theoretical max FPS: {1 / avg_processing_time:.1f}"
        if avg_processing_time > 0
        else "N/A"
    )
    print(f"Debug frames saved to: {frames_dir}/")
    print(f"Tower levels - Player: {player_level}, Opponent: {opponent_level}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python pipeline_test.py screenshot <image_path>")
        print("  python pipeline_test.py video <video_path>")
        print("  python pipeline_test.py video <video_path> --full")
        sys.exit(1)

    mode = sys.argv[1]
    path = sys.argv[2]

    if mode == "screenshot":
        test_screenshot(path)
    elif mode == "video":
        full = len(sys.argv) > 3 and sys.argv[3] == "--full"
        test_video(path, full=full)
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
