"""
Label extraction pipeline for behavioral cloning.

Processes professional Clash Royale gameplay videos and extracts
(game_state, action) pairs where an action is a card placement event.

A card placement event is detected when a new card appears on the
player's side of the field between consecutive analyzed frames.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Set, Optional

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.video.video_analyzer import VideoAnalyzer


def _detect_player_placements(
    prev_detections: List[Dict[str, Any]],
    curr_detections: List[Dict[str, Any]],
    prev_tracked: Set[str],
) -> tuple[List[Dict[str, Any]], Set[str]]:
    """
    Detect new card placements on the player's side between two frames.

    Uses position-signature tracking (same logic as OpponentElixirTracker)
    but for the player's cards.

    Args:
        prev_detections: Detections from the previous frame.
        curr_detections: Detections from the current frame.
        prev_tracked: Set of card_id strings tracked from the previous frame.

    Returns:
        Tuple of (list of new placement dicts, updated tracked set).
    """
    new_placements = []
    current_tracked: Set[str] = set()

    for det in curr_detections:
        # Only player cards that are on the field
        if det.get("is_opponent", True):
            continue
        if not det.get("is_on_field", False):
            continue

        tile_y = det.get("tile_y", 0)
        # Must be on the player's side (y >= 17)
        if tile_y < 17:
            continue

        card_name = det.get("class_name", "")
        tile_x = det.get("tile_x", 0)
        card_id = f"{card_name}_{tile_x}_{tile_y}"
        current_tracked.add(card_id)

        if card_id not in prev_tracked:
            new_placements.append({
                "card_name": card_name,
                "tile_x": tile_x,
                "tile_y": tile_y,
            })

    return new_placements, current_tracked


def extract_labels_from_video(
    video_path: str,
    frame_skip: int = 6,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Process a single video and extract (state, action) training pairs.

    Each training pair consists of the game state just BEFORE a card was
    played, and the action (card_name, tile_x, tile_y) that was played.

    Args:
        video_path: Path to the gameplay video.
        frame_skip: Process every Nth frame (lower = more granular detection
                    but slower; 6 is ~5 FPS from 30 FPS source).
        verbose: Print progress info.

    Returns:
        List of training examples, each a dict with "state" and "action" keys.
    """
    analyzer = VideoAnalyzer(
        frame_skip=frame_skip,
        save_annotated_frames=False,
        verbose=verbose,
    )

    if verbose:
        print(f"Analyzing video: {video_path}")

    result = analyzer.analyze_video(video_path)
    frames = result.get("frames", [])

    if verbose:
        print(f"Extracted {len(frames)} frames from video.")

    training_pairs = []
    tracked: Set[str] = set()

    for i in range(1, len(frames)):
        prev_frame = frames[i - 1]
        curr_frame = frames[i]

        placements, tracked = _detect_player_placements(
            prev_frame.get("detections", []),
            curr_frame.get("detections", []),
            tracked,
        )

        for placement in placements:
            training_pairs.append({
                "state": prev_frame,
                "action": placement,
            })

    if verbose:
        print(f"Found {len(training_pairs)} card placement events.")

    return training_pairs


def extract_labels_from_videos(
    video_paths: List[str],
    output_path: str = "data/training_data.json",
    frame_skip: int = 6,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Process multiple videos and save combined training data.

    Args:
        video_paths: List of paths to gameplay videos.
        output_path: Where to save the combined JSON dataset.
        frame_skip: Process every Nth frame.
        verbose: Print progress info.

    Returns:
        Combined list of all training examples.
    """
    all_pairs = []

    for idx, video_path in enumerate(video_paths):
        if verbose:
            print(f"\n--- Video {idx + 1}/{len(video_paths)} ---")

        pairs = extract_labels_from_video(video_path, frame_skip, verbose)
        all_pairs.extend(pairs)

    # Save to disk
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        json.dump(all_pairs, f, indent=2)

    if verbose:
        print(f"\nTotal training examples: {len(all_pairs)}")
        print(f"Saved to: {out}")

    return all_pairs


def main():
    """CLI entry point for label extraction."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract (state, action) training pairs from Clash Royale gameplay videos."
    )
    parser.add_argument(
        "videos",
        nargs="+",
        help="Paths to gameplay video files",
    )
    parser.add_argument(
        "--output",
        default="data/training_data.json",
        help="Output JSON path (default: data/training_data.json)",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=6,
        help="Process every Nth frame (default: 6)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()
    extract_labels_from_videos(
        args.videos,
        output_path=args.output,
        frame_skip=args.frame_skip,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
