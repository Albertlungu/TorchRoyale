"""
Episode construction for Decision Transformer training.

Converts video analysis output (from VideoAnalyzer) into Episode objects:
sequences of (state, action, reward) tuples suitable for training.

Reward shaping includes tower damage, tower destruction bonuses,
and game outcome signals.
"""

import json
import os
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.constants.game_constants import get_elixir_cost
from src.data.feature_encoder import encode
from src.data.label_extractor import extract_labels_from_video
from src.data.outcome_detector import GameOutcome
from src.transformer.config import DTConfig


@dataclass
class Timestep:
    """A single card-placement event with state, action, and reward."""

    state: Dict[str, Any]  # Raw FrameState dict
    state_vec: np.ndarray  # (97,) encoded feature vector
    action_card: int  # 0-3 hand card index
    action_pos: int  # 0-575 flattened tile position
    reward: float = 0.0
    timestamp_ms: int = 0


@dataclass
class Episode:
    """A full game as a sequence of card-placement timesteps."""

    timesteps: List[Timestep] = field(default_factory=list)
    outcome: GameOutcome = GameOutcome.UNKNOWN
    returns_to_go: Optional[np.ndarray] = None  # (T,) computed after rewards
    video_path: str = ""

    @property
    def total_return(self) -> float:
        return sum(ts.reward for ts in self.timesteps)

    @property
    def length(self) -> int:
        return len(self.timesteps)


def _compute_tower_reward(
    prev_state: Dict[str, Any],
    curr_state: Dict[str, Any],
    config: DTConfig,
) -> float:
    """
    Compute reward from tower health changes between two states.

    Positive reward for damaging opponent towers.
    Negative reward for own towers taking damage.
    """
    reward = 0.0

    # Opponent towers (damage dealt = positive)
    for tower_key in ["opponent_left", "opponent_king", "opponent_right"]:
        prev_hp = _get_tower_hp_ratio(prev_state, "opponent_towers", tower_key)
        curr_hp = _get_tower_hp_ratio(curr_state, "opponent_towers", tower_key)

        delta = prev_hp - curr_hp
        if delta > 0:
            reward += delta * config.tower_damage_weight

            # Destruction bonus
            if curr_hp == 0.0 and prev_hp > 0.0:
                if "king" in tower_key:
                    reward += config.king_tower_destroy_bonus
                else:
                    reward += config.tower_destroy_bonus

    # Player towers (damage taken = negative)
    for tower_key in ["player_left", "player_king", "player_right"]:
        prev_hp = _get_tower_hp_ratio(prev_state, "player_towers", tower_key)
        curr_hp = _get_tower_hp_ratio(curr_state, "player_towers", tower_key)

        delta = prev_hp - curr_hp
        if delta > 0:
            reward -= delta * config.tower_damage_weight

            if curr_hp == 0.0 and prev_hp > 0.0:
                if "king" in tower_key:
                    reward -= config.king_tower_destroy_bonus
                else:
                    reward -= config.tower_destroy_bonus

    return reward


def _get_tower_hp_ratio(
    state: Dict[str, Any],
    tower_group: str,
    tower_key: str,
) -> float:
    """Extract HP ratio for a tower from a state dict."""
    towers = state.get(tower_group, {})
    tower = towers.get(tower_key, {})
    if tower.get("is_destroyed", False):
        return 0.0
    percent = tower.get("health_percent")
    if percent is not None:
        return float(percent) / 100.0
    return 1.0


def _compute_returns_to_go(rewards: List[float]) -> np.ndarray:
    """Compute return-to-go at each timestep (reverse cumulative sum)."""
    rtg = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running += rewards[t]
        rtg[t] = running
    return rtg


def build_episode(
    training_pairs: List[Dict[str, Any]],
    outcome: GameOutcome,
    video_path: str = "",
    config: Optional[DTConfig] = None,
) -> Episode:
    """
    Build an Episode from extracted (state, action) training pairs.

    Args:
        training_pairs: Output of label_extractor.extract_labels_from_video().
            Each dict has "state" (FrameState dict) and "action" (card_name, tile_x, tile_y).
        outcome: Game outcome (win/loss/unknown).
        video_path: Source video path for metadata.
        config: DTConfig for reward weights. Uses defaults if None.

    Returns:
        Episode with computed rewards and returns-to-go.
    """
    if config is None:
        config = DTConfig()

    timesteps = []

    for pair in training_pairs:
        state = pair["state"]
        action = pair["action"]

        card_name = action["card_name"]
        tile_x = action["tile_x"]
        tile_y = action["tile_y"]

        # Find card index in hand
        hand_cards = state.get("hand_cards", [])
        card_idx = _find_card_in_hand(card_name, hand_cards)
        if card_idx is None:
            continue

        # Encode state
        state_vec = encode(state)

        # Flatten tile position
        action_pos = tile_y * 18 + tile_x

        timesteps.append(
            Timestep(
                state=state,
                state_vec=state_vec,
                action_card=card_idx,
                action_pos=action_pos,
                timestamp_ms=state.get("timestamp_ms", 0),
            )
        )

    if not timesteps:
        return Episode(outcome=outcome, video_path=video_path)

    # Compute shaped rewards
    for i in range(len(timesteps)):
        reward = 0.0

        # Tower damage reward (compare with next timestep's state)
        if i < len(timesteps) - 1:
            reward += _compute_tower_reward(
                timesteps[i].state,
                timesteps[i + 1].state,
                config,
            )

        timesteps[i].reward = reward

    # Add game outcome to final timestep
    if outcome == GameOutcome.WIN:
        timesteps[-1].reward += config.outcome_weight
    elif outcome == GameOutcome.LOSS:
        timesteps[-1].reward -= config.outcome_weight

    # Compute returns-to-go
    rewards = [ts.reward for ts in timesteps]
    rtg = _compute_returns_to_go(rewards)

    return Episode(
        timesteps=timesteps,
        outcome=outcome,
        returns_to_go=rtg,
        video_path=video_path,
    )


def _find_card_in_hand(card_name: str, hand_cards: List[str]) -> Optional[int]:
    """Find the index of a card in the player's hand, handling name variants."""
    clean = card_name.lower()
    for suffix in [
        "-in-hand",
        "-next",
        "-on-field",
        "_on_field",
        "-evolution",
        "_evolution",
        "-ability",
    ]:
        clean = clean.replace(suffix, "")
    clean = clean.strip()

    for i, hand_card in enumerate(hand_cards):
        hand_clean = hand_card.lower()
        for suffix in [
            "-in-hand",
            "-next",
            "-on-field",
            "_on_field",
            "-evolution",
            "_evolution",
            "-ability",
        ]:
            hand_clean = hand_clean.replace(suffix, "")
        hand_clean = hand_clean.strip()
        if hand_clean == clean:
            return i

    return None


def _detect_match_over_frames(video_path: str, verbose: bool = True) -> List[int]:
    """
    Scan video for frames containing "Match Over" text using OCR.

    Returns list of timestamps (in milliseconds) where "Match Over" appears.
    """
    if verbose:
        print("Scanning video for 'Match Over' text...")

    # Initialize easyOCR reader
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    except ImportError:
        if verbose:
            print("  WARNING: easyocr not available, skipping OCR detection")
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        if verbose:
            print(f"  ERROR: Could not open video: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    match_over_timestamps = []
    last_match_over_time = -10000  # Track to avoid duplicates from consecutive frames

    # Sample once per second
    frame_skip = int(fps) if fps > 0 else 30
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        # Only check every Nth frame
        if frame_idx % frame_skip == 0:
            try:
                # Use easyOCR to extract text from entire frame
                results = reader.readtext(frame, paragraph=False)

                # Check all detected text
                for (bbox, text, confidence) in results:
                    text_lower = text.lower().strip()
                    # Look for "Match Over" or "Match" and "Over" nearby
                    if ("match" in text_lower and "over" in text_lower) or "matchover" in text_lower:
                        # Avoid duplicates from consecutive frames (within 5 seconds)
                        if timestamp_ms - last_match_over_time > 5000:
                            match_over_timestamps.append(timestamp_ms)
                            last_match_over_time = timestamp_ms
                            if verbose:
                                print(f"  Found 'Match Over' at {timestamp_ms / 1000:.1f}s (text: '{text}')")
                            break
            except Exception as e:
                if verbose:
                    print(f"  OCR error at frame {frame_idx}: {e}")

        frame_idx += 1

        # Progress indicator
        if verbose and frame_idx % (frame_skip * 5) == 0:
            progress = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
            print(f"  Scanning... {progress:.0f}% complete", end='\r')

    cap.release()

    if verbose:
        print(f"\n  Found {len(match_over_timestamps)} 'Match Over' instances")

    return match_over_timestamps


def _detect_game_boundaries(
    training_pairs: List[Dict[str, Any]],
    match_over_timestamps: Optional[List[int]] = None,
) -> List[int]:
    """
    Detect boundaries between multiple games in a single video.

    Returns list of indices where new games start (first game starts at 0).

    Detects new games by looking for:
    - "Match Over" text detected via OCR
    - All towers resetting to full health
    - Elixir resetting to starting value
    """
    if not training_pairs:
        return [0]

    boundaries = [0]

    # Convert match_over_timestamps to a set for quick lookup
    match_over_set = set(match_over_timestamps) if match_over_timestamps else set()

    for i in range(1, len(training_pairs)):
        prev_state = training_pairs[i - 1]["state"]
        curr_state = training_pairs[i]["state"]
        curr_timestamp = curr_state.get("timestamp_ms", 0)

        # Check for "Match Over" detection within 5 seconds of this timestamp
        match_over_detected = False
        if match_over_set:
            for mo_time in match_over_set:
                if abs(curr_timestamp - mo_time) < 5000:  # Within 5 seconds
                    match_over_detected = True
                    break

        # Check if all player towers reset to full health
        player_towers_reset = _all_towers_full(curr_state, "player_towers")
        opponent_towers_reset = _all_towers_full(curr_state, "opponent_towers")

        # Check if both sets of towers went from damaged to full
        prev_player_damaged = not _all_towers_full(prev_state, "player_towers")
        prev_opponent_damaged = not _all_towers_full(prev_state, "opponent_towers")

        tower_reset = (player_towers_reset and opponent_towers_reset and
                      (prev_player_damaged or prev_opponent_damaged))

        # New game detected if we see "Match Over" OR tower reset
        if match_over_detected or tower_reset:
            # Avoid duplicate boundaries
            if not boundaries or i - boundaries[-1] > 10:  # At least 10 timesteps apart
                boundaries.append(i)

    return boundaries


def _all_towers_full(state: Dict[str, Any], tower_group: str) -> bool:
    """Check if all towers in a group are at full health."""
    towers = state.get(tower_group, {})
    for tower_key in ["left", "king", "right"]:
        tower_key_full = f"player_{tower_key}" if "player" in tower_group else f"opponent_{tower_key}"
        tower = towers.get(tower_key_full, {})

        if tower.get("is_destroyed", False):
            return False

        hp_percent = tower.get("health_percent")
        if hp_percent is not None and hp_percent < 95:  # Allow small margin
            return False

    return True


def build_episodes_from_multi_game_video(
    video_path: str,
    num_games: Optional[int] = None,
    outcome: GameOutcome = GameOutcome.WIN,
    config: Optional[DTConfig] = None,
    frame_skip: int = 6,
    verbose: bool = True,
) -> List[Episode]:
    """
    Process a video containing multiple games into separate episodes.

    Args:
        video_path: Path to video file containing multiple games.
        num_games: Expected number of games (for validation, optional).
        outcome: Outcome to assign to all games.
        config: DTConfig for reward weights.
        frame_skip: Frame skip for video analysis.
        verbose: Print progress.

    Returns:
        List of Episode objects, one per game detected.
    """
    if verbose:
        print(f"Processing multi-game video: {video_path}")
        print(f"Expected games: {num_games if num_games else 'auto-detect'}")
        print(f"Outcome for all games: {outcome.value}")
        print()

    # First, scan video for "Match Over" text
    match_over_timestamps = _detect_match_over_frames(video_path, verbose=verbose)

    if verbose:
        print()

    # Extract all (state, action) pairs from the full video
    training_pairs = extract_labels_from_video(
        video_path,
        frame_skip=frame_skip,
        verbose=verbose,
    )

    if not training_pairs:
        if verbose:
            print("No training pairs extracted from video")
        return []

    # Detect game boundaries using both "Match Over" and tower reset signals
    boundaries = _detect_game_boundaries(training_pairs, match_over_timestamps)

    if verbose:
        print(f"\nDetected {len(boundaries)} games at indices: {boundaries}")

    # Split into separate episodes
    episodes = []
    for game_idx in range(len(boundaries)):
        start_idx = boundaries[game_idx]
        end_idx = boundaries[game_idx + 1] if game_idx + 1 < len(boundaries) else len(training_pairs)

        game_pairs = training_pairs[start_idx:end_idx]

        if not game_pairs:
            continue

        ep = build_episode(
            training_pairs=game_pairs,
            outcome=outcome,
            video_path=f"{video_path}#game{game_idx + 1}",
            config=config,
        )

        if ep.length > 0:
            episodes.append(ep)
            if verbose:
                print(f"  Game {game_idx + 1}: {ep.length} timesteps, return={ep.total_return:.2f}")

    if num_games is not None and len(episodes) != num_games:
        if verbose:
            print(f"WARNING: Expected {num_games} games but detected {len(episodes)}")

    return episodes


def build_episode_from_video(
    video_path: str,
    outcome: Optional[GameOutcome] = None,
    config: Optional[DTConfig] = None,
    frame_skip: int = 6,
    verbose: bool = True,
) -> Episode:
    """
    Full pipeline: video -> episode.

    Args:
        video_path: Path to gameplay video.
        outcome: Pre-determined outcome. Ignored in favor of WIN for local runs.
        config: DTConfig for reward weights.
        frame_skip: Frame skip for video analysis.
        verbose: Print progress.

    Returns:
        Complete Episode with rewards and returns-to-go.
    """
    # Force all locally processed videos to WIN.
    outcome = GameOutcome.WIN
    if verbose:
        print("Forced outcome: win")

    # Extract (state, action) pairs
    training_pairs = extract_labels_from_video(
        video_path,
        frame_skip=frame_skip,
        verbose=verbose,
    )

    return build_episode(
        training_pairs=training_pairs,
        outcome=outcome,
        video_path=video_path,
        config=config,
    )


def _load_outcome_from_analysis_json(video_path: str) -> Optional[GameOutcome]:
    """
    Try to load outcome from corresponding analysis JSON file.

    Looks for {video_stem}_analysis.json in the output directory.

    Args:
        video_path: Path to the video file.

    Returns:
        GameOutcome if found in JSON, otherwise None.
    """
    video_stem = Path(video_path).stem
    possible_json_paths = [
        Path("output") / f"{video_stem}_analysis.json",
        Path(video_path).parent / f"{video_stem}_analysis.json",
    ]

    for json_path in possible_json_paths:
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                outcome_str = data.get("outcome")
                if outcome_str:
                    return GameOutcome(outcome_str)
            except (json.JSONDecodeError, ValueError, KeyError):
                continue

    return None


def build_episodes_from_videos(
    video_paths: List[str],
    output_path: str = "data/episodes.pkl",
    outcomes: Optional[Dict[str, GameOutcome]] = None,
    config: Optional[DTConfig] = None,
    frame_skip: int = 6,
    verbose: bool = True,
    allow_empty_output: bool = False,
) -> List[Episode]:
    """
    Process multiple videos into episodes and save to disk.

    Args:
        video_paths: List of video file paths.
        output_path: Where to save pickled episodes.
        outcomes: Optional dict mapping video path -> GameOutcome.
                  If not provided, will attempt to load from analysis JSON files.
        config: DTConfig for reward weights.
        frame_skip: Frame skip for video analysis.
        verbose: Print progress.
        allow_empty_output: If True, writes an empty pickle when no episodes are built.

    Returns:
        List of Episode objects.
    """
    episodes = []

    for idx, vpath in enumerate(video_paths):
        if verbose:
            print(f"\n=== Video {idx + 1}/{len(video_paths)}: {vpath} ===")

        outcome = None
        if outcomes:
            path_obj = Path(vpath)
            lookup_keys = (
                vpath,
                path_obj.name,
                path_obj.stem,
            )
            for key in lookup_keys:
                if key in outcomes:
                    outcome = outcomes[key]
                    break

        # If no outcome from mapping, try to load from analysis JSON
        if outcome is None:
            outcome = _load_outcome_from_analysis_json(vpath)
            if outcome is not None and verbose:
                print(f"  Loaded outcome from analysis JSON: {outcome.value}")

        try:
            ep = build_episode_from_video(
                vpath,
                outcome=outcome,
                config=config,
                frame_skip=frame_skip,
                verbose=verbose,
            )
            if ep.length > 0:
                episodes.append(ep)
                if verbose:
                    print(
                        f"  -> {ep.length} timesteps, return={ep.total_return:.2f}, outcome={ep.outcome.value}"
                    )
            else:
                if verbose:
                    print("  -> Skipped (no valid timesteps)")
        except Exception as e:
            if verbose:
                print(f"  -> Error: {e}")

    if not episodes and not allow_empty_output:
        raise RuntimeError(
            "No non-empty episodes were built. Output file was not written. "
            "This usually means no valid placements were detected or no played cards "
            "were found in hand."
        )

    # Save to disk (atomic write to avoid partial/corrupted files)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out.with_suffix(out.suffix + ".tmp")
    with open(tmp_out, "wb") as f:
        pickle.dump(episodes, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_out, out)

    if verbose:
        print(f"\nSaved {len(episodes)} episodes to {out}")
        total_ts = sum(ep.length for ep in episodes)
        print(f"Total timesteps: {total_ts}")
        wins = sum(1 for ep in episodes if ep.outcome == GameOutcome.WIN)
        losses = sum(1 for ep in episodes if ep.outcome == GameOutcome.LOSS)
        print(
            f"Outcomes: {wins} wins, {losses} losses, {len(episodes) - wins - losses} unknown"
        )

    return episodes


def main() -> int:
    """CLI entry point for episode building."""
    import argparse
    import glob

    parser = argparse.ArgumentParser(
        description="Build Decision Transformer episodes from Clash Royale replay videos."
    )
    parser.add_argument(
        "videos",
        nargs="+",
        help="Paths to video files or glob patterns",
    )
    parser.add_argument(
        "--output",
        default="data/episodes.pkl",
        help="Output pickle path (default: data/episodes.pkl)",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=6,
        help="Process every Nth frame (default: 6)",
    )
    parser.add_argument(
        "--outcomes",
        default=None,
        help="JSON file mapping video filenames to 'win'/'loss'",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow writing an empty episodes pickle",
    )

    args = parser.parse_args()

    # Resolve glob patterns
    video_paths = []
    unmatched_patterns = []
    for pattern in args.videos:
        matched = glob.glob(pattern)
        if matched:
            video_paths.extend(sorted(matched))
        elif Path(pattern).is_file():
            video_paths.append(pattern)
        else:
            unmatched_patterns.append(pattern)

    if unmatched_patterns and not args.quiet:
        print(f"Warning: no files matched: {', '.join(unmatched_patterns)}")

    if not video_paths:
        print("No input videos found. Nothing to process.")
        return 2

    # Deduplicate while preserving order
    video_paths = list(dict.fromkeys(video_paths))

    # Load outcomes if provided
    outcomes_dict = None
    if args.outcomes:
        with open(args.outcomes, "r", encoding="utf-8") as f:
            raw = json.load(f)
        outcomes_dict = {k: GameOutcome(v) for k, v in raw.items()}

    try:
        build_episodes_from_videos(
            video_paths=video_paths,
            output_path=args.output,
            outcomes=outcomes_dict,
            frame_skip=args.frame_skip,
            verbose=not args.quiet,
            allow_empty_output=args.allow_empty,
        )
    except Exception as exc:
        print(f"Episode build failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
