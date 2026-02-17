#!/usr/bin/env python3
"""
Live pipeline testing - Tests the real-time pipeline performance.

This script tests the LivePipeline class to measure actual performance
and validate that it can achieve real-time processing speeds.

Usage:
    python pipeline_live_test.py <video_path>
    python pipeline_live_test.py <video_path> --target-fps 30
    python pipeline_live_test.py <video_path> --save-debug
"""

import argparse
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

from pipeline_live import LivePipeline

# Debug output root
DEBUG_DIR = Path("debug")


def draw_performance_overlay(frame, stats, game_state):
    """Draw performance and game state overlay on frame."""
    overlay = frame.copy()

    # Performance stats (top-left, green background)
    perf_lines = [
        f"FPS Target: {stats['target_fps']:.1f}",
        f"Process Time: {game_state['processing_time'] * 1000:.1f}ms",
        f"Frame: {game_state['frame_count']}",
        f"Budget: {1000 / stats['target_fps']:.1f}ms",
    ]

    # Draw background for performance stats
    cv2.rectangle(overlay, (10, 10), (300, 130), (0, 100, 0), -1)
    for i, line in enumerate(perf_lines):
        y = 35 + i * 25
        cv2.putText(
            overlay, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

    # Game state (top-right, blue background)
    game_lines = [
        f"Elixir: {game_state['elixir']}",
        f"Opp Elixir: {game_state['opponent_elixir']:.1f}",
        f"Timer: {game_state['timer']}s" if game_state["timer"] else "Timer: ?",
        f"Phase: {game_state['game_phase'].value}"
        if game_state["game_phase"]
        else "Phase: ?",
    ]

    # Draw background for game state
    w = frame.shape[1]
    cv2.rectangle(overlay, (w - 350, 10), (w - 10, 130), (100, 0, 0), -1)
    for i, line in enumerate(game_lines):
        y = 35 + i * 25
        cv2.putText(
            overlay,
            line,
            (w - 340, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    # Tower health (bottom, semi-transparent)
    if game_state["tower_health"]:
        tower_y_start = frame.shape[0] - 200
        cv2.rectangle(
            overlay,
            (10, tower_y_start),
            (w - 10, frame.shape[0] - 10),
            (50, 50, 50),
            -1,
        )

        tower_lines = []
        for name in [
            "player_left",
            "player_king",
            "player_right",
            "opponent_left",
            "opponent_king",
            "opponent_right",
        ]:
            th = game_state["tower_health"].get(name)
            if th is None:
                tower_lines.append(f"{name}: ?")
            elif th.is_destroyed:
                tower_lines.append(f"{name}: DESTROYED")
            elif th.detected:
                tower_lines.append(f"{name}: {th.hp_current}/{th.hp_max}")
            elif th.health_percent == 100.0:
                tower_lines.append(f"{name}: FULL")
            else:
                tower_lines.append(f"{name}: ?")

        # Draw tower health in 2 columns
        for i, line in enumerate(tower_lines):
            col = i % 2
            row = i // 2
            x = 20 + col * 400
            y = tower_y_start + 30 + row * 25
            cv2.putText(
                overlay, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

    # Blend overlay
    alpha = 0.7
    return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)


def test_live_pipeline_performance(
    video_path: str,
    target_fps: float = 30.0,
    save_debug: bool = False,
    max_duration: float = 60.0,
):
    """Test the live pipeline performance on a video."""

    print(f"\n{'=' * 80}")
    print("LIVE PIPELINE PERFORMANCE TEST")
    print(f"Video: {video_path}")
    print(f"Target FPS: {target_fps}")
    print(f"Max Duration: {max_duration}s")
    print(f"Debug Output: {'Enabled' if save_debug else 'Disabled'}")
    print("=" * 80)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(
        f"Video Info: {width}x{height}, {video_fps:.1f} FPS, {duration:.1f}s, {total_frames} frames"
    )

    # Setup debug output if requested
    debug_dir = None
    if save_debug:
        video_name = Path(video_path).stem
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_dir = DEBUG_DIR / "live_test" / f"{video_name}_{run_timestamp}"
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"Debug frames will be saved to: {debug_dir}/")

    # Initialize live pipeline
    print("\nInitializing live pipeline...")
    pipeline = LivePipeline(target_fps=target_fps)

    # Performance tracking
    frame_times = []
    processing_times = []
    missed_deadlines = 0
    frame_budget = 1.0 / target_fps

    max_frames = int(min(duration, max_duration) * target_fps)

    print(f"\nProcessing up to {max_frames} frames...")
    print(f"Frame budget: {frame_budget * 1000:.1f}ms per frame")
    print("-" * 80)

    frame_count = 0
    start_time = time.time()
    last_print_time = start_time

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        frame_start = time.time()
        game_state = pipeline.process_frame(frame)
        processing_time = time.time() - frame_start

        processing_times.append(processing_time)

        # Check if we missed our frame deadline
        if processing_time > frame_budget:
            missed_deadlines += 1

        # Console output every second
        current_time = time.time()
        if current_time - last_print_time >= 1.0:
            elapsed = current_time - start_time
            actual_fps = frame_count / elapsed if elapsed > 0 else 0
            avg_process_time = sum(processing_times[-int(target_fps) :]) / min(
                len(processing_times), int(target_fps)
            )

            print(
                f"Frame {frame_count:5d} | "
                f"Actual FPS: {actual_fps:5.1f} | "
                f"Process: {processing_time * 1000:5.1f}ms | "
                f"Avg: {avg_process_time * 1000:5.1f}ms | "
                f"Deadline Misses: {missed_deadlines:3d} | "
                f"Elixir: {game_state['elixir']:2d}"
            )

            last_print_time = current_time

        # Save debug frame if requested
        if save_debug and frame_count % int(target_fps) == 0:  # Save every second
            debug_frame = draw_performance_overlay(
                frame, pipeline.get_performance_stats(), game_state
            )
            debug_path = debug_dir / f"frame_{frame_count:06d}.png"
            cv2.imwrite(str(debug_path), debug_frame)

        frame_count += 1

        # Simulate real-time by sleeping if we processed too fast
        # (Only if we're under budget - don't slow down already slow processing)
        if processing_time < frame_budget:
            sleep_time = frame_budget - processing_time
            time.sleep(sleep_time)

        frame_times.append(time.time() - frame_start)

    cap.release()

    # Final performance analysis
    total_time = time.time() - start_time
    stats = pipeline.get_performance_stats()

    avg_processing_time = (
        sum(processing_times) / len(processing_times) if processing_times else 0
    )
    max_processing_time = max(processing_times) if processing_times else 0
    min_processing_time = min(processing_times) if processing_times else 0

    # Calculate percentiles
    processing_times.sort()
    p95_time = (
        processing_times[int(0.95 * len(processing_times))] if processing_times else 0
    )
    p99_time = (
        processing_times[int(0.99 * len(processing_times))] if processing_times else 0
    )

    deadline_miss_rate = (
        (missed_deadlines / frame_count * 100) if frame_count > 0 else 0
    )

    print("\n" + "=" * 80)
    print("PERFORMANCE RESULTS")
    print("=" * 80)
    print(f"Frames Processed: {frame_count}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Target FPS: {target_fps:.1f}")
    print(f"Actual FPS: {stats['average_fps']:.2f}")
    print(f"Performance Ratio: {stats['average_fps'] / target_fps * 100:.1f}%")
    print()
    print("Processing Time Statistics:")
    print(f"  Average: {avg_processing_time * 1000:.2f}ms")
    print(f"  Minimum: {min_processing_time * 1000:.2f}ms")
    print(f"  Maximum: {max_processing_time * 1000:.2f}ms")
    print(f"  95th Percentile: {p95_time * 1000:.2f}ms")
    print(f"  99th Percentile: {p99_time * 1000:.2f}ms")
    print(f"  Frame Budget: {frame_budget * 1000:.2f}ms")
    print()
    print(f"Deadline Misses: {missed_deadlines} ({deadline_miss_rate:.1f}%)")
    print()

    # Performance verdict
    if stats["average_fps"] >= target_fps * 0.95:
        print("✅ PASS: Pipeline achieves target FPS")
    elif stats["average_fps"] >= target_fps * 0.8:
        print("⚠️  MARGINAL: Pipeline almost achieves target FPS")
    else:
        print("❌ FAIL: Pipeline too slow for real-time processing")

    if deadline_miss_rate < 5:
        print("✅ STABLE: Low deadline miss rate")
    elif deadline_miss_rate < 15:
        print("⚠️  UNSTABLE: Moderate deadline miss rate")
    else:
        print("❌ UNSTABLE: High deadline miss rate")

    if save_debug:
        print(f"\nDebug frames saved to: {debug_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Test live pipeline performance")
    parser.add_argument("video_path", help="Path to test video")
    parser.add_argument(
        "--target-fps",
        type=float,
        default=30.0,
        help="Target processing FPS (default: 30.0)",
    )
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save debug frames with performance overlay",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Maximum test duration in seconds (default: 30.0)",
    )

    args = parser.parse_args()

    if not Path(args.video_path).exists():
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)

    test_live_pipeline_performance(
        args.video_path,
        target_fps=args.target_fps,
        save_debug=args.save_debug,
        max_duration=args.max_duration,
    )


if __name__ == "__main__":
    main()
