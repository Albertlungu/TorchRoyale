"""
Episode construction for Decision Transformer training.

Converts video analysis output (from VideoAnalyzer) into Episode objects:
sequences of (state, action, reward) tuples suitable for training.

Reward shaping includes tower damage, tower destruction bonuses,
and game outcome signals.
"""

import json
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.feature_encoder import encode
from src.data.outcome_detector import GameOutcome, OutcomeDetector
from src.data.label_extractor import extract_labels_from_video
from src.constants.game_constants import get_elixir_cost
from src.transformer.config import DTConfig


@dataclass
class Timestep:
    """A single card-placement event with state, action, and reward."""
    state: Dict[str, Any]      # Raw FrameState dict
    state_vec: np.ndarray      # (97,) encoded feature vector
    action_card: int           # 0-3 hand card index
    action_pos: int            # 0-575 flattened tile position
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

        timesteps.append(Timestep(
            state=state,
            state_vec=state_vec,
            action_card=card_idx,
            action_pos=action_pos,
            timestamp_ms=state.get("timestamp_ms", 0),
        ))

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
    for suffix in ["-in-hand", "-next", "-on-field", "_on_field",
                   "-evolution", "_evolution", "-ability"]:
        clean = clean.replace(suffix, "")
    clean = clean.strip()

    for i, hand_card in enumerate(hand_cards):
        hand_clean = hand_card.lower()
        for suffix in ["-in-hand", "-next", "-on-field", "_on_field",
                       "-evolution", "_evolution", "-ability"]:
            hand_clean = hand_clean.replace(suffix, "")
        hand_clean = hand_clean.strip()
        if hand_clean == clean:
            return i

    return None


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
        outcome: Pre-determined outcome. If None, auto-detects from video.
        config: DTConfig for reward weights.
        frame_skip: Frame skip for video analysis.
        verbose: Print progress.

    Returns:
        Complete Episode with rewards and returns-to-go.
    """
    # Auto-detect outcome if not provided
    if outcome is None:
        detector = OutcomeDetector()
        outcome = detector.detect_from_video(video_path)
        if verbose:
            print(f"Detected outcome: {outcome.value}")

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


def build_episodes_from_videos(
    video_paths: List[str],
    output_path: str = "data/episodes.pkl",
    outcomes: Optional[Dict[str, GameOutcome]] = None,
    config: Optional[DTConfig] = None,
    frame_skip: int = 6,
    verbose: bool = True,
) -> List[Episode]:
    """
    Process multiple videos into episodes and save to disk.

    Args:
        video_paths: List of video file paths.
        output_path: Where to save pickled episodes.
        outcomes: Optional dict mapping video path -> GameOutcome.
        config: DTConfig for reward weights.
        frame_skip: Frame skip for video analysis.
        verbose: Print progress.

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
                    print(f"  -> {ep.length} timesteps, return={ep.total_return:.2f}, outcome={ep.outcome.value}")
            else:
                if verbose:
                    print("  -> Skipped (no valid timesteps)")
        except Exception as e:
            if verbose:
                print(f"  -> Error: {e}")

    # Save to disk
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(episodes, f)

    if verbose:
        print(f"\nSaved {len(episodes)} episodes to {out}")
        total_ts = sum(ep.length for ep in episodes)
        print(f"Total timesteps: {total_ts}")
        wins = sum(1 for ep in episodes if ep.outcome == GameOutcome.WIN)
        losses = sum(1 for ep in episodes if ep.outcome == GameOutcome.LOSS)
        print(f"Outcomes: {wins} wins, {losses} losses, {len(episodes) - wins - losses} unknown")

    return episodes


def main():
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

    args = parser.parse_args()

    # Resolve glob patterns
    video_paths = []
    for pattern in args.videos:
        matched = glob.glob(pattern)
        video_paths.extend(sorted(matched) if matched else [pattern])

    # Load outcomes if provided
    outcomes_dict = None
    if args.outcomes:
        with open(args.outcomes) as f:
            raw = json.load(f)
        outcomes_dict = {
            k: GameOutcome(v) for k, v in raw.items()
        }

    build_episodes_from_videos(
        video_paths=video_paths,
        output_path=args.output,
        outcomes=outcomes_dict,
        frame_skip=args.frame_skip,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
