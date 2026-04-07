"""
Convert JSON datasets into Decision Transformer episodes pickle files.

Supports two input JSON shapes:
1) Video analysis JSON from VideoAnalyzer (contains "frames").
2) Training data JSON from label_extractor (list of {"state", "action"}).

By default, this script will refuse to overwrite the output when zero
non-empty episodes are built.

Examples:
    python scripts/convert_json_to_episodes.py output/example.json --output data/episodes.pkl
    python scripts/convert_json_to_episodes.py data/training_data.json --format training --output data/episodes.pkl
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.episode_builder import Episode, build_episode
from src.data.outcome_detector import GameOutcome
from src.transformer.config import DTConfig


@dataclass
class FileConversionResult:
    """Summary for one JSON input file conversion."""

    source_path: Path
    source_format: str
    training_pairs: int
    episode_length: int
    skipped: bool
    skip_reason: str = ""


def _resolve_paths(patterns: Sequence[str]) -> List[Path]:
    """Resolve explicit paths and glob patterns into existing file paths."""
    resolved: List[Path] = []
    seen: Set[Path] = set()

    for pattern in patterns:
        matches = glob.glob(pattern)
        candidates = matches if matches else [pattern]
        for candidate in candidates:
            path = Path(candidate)
            if not path.exists() or not path.is_file():
                continue
            abs_path = path.resolve()
            if abs_path in seen:
                continue
            seen.add(abs_path)
            resolved.append(abs_path)

    return resolved


def _load_outcomes(outcomes_path: Optional[str]) -> Dict[str, GameOutcome]:
    """Load optional outcome mapping from JSON."""
    if outcomes_path is None:
        return {}

    with open(outcomes_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError(
            "Outcome mapping must be a JSON object of key -> win/loss/unknown."
        )

    parsed: Dict[str, GameOutcome] = {}
    for key, value in raw.items():
        parsed[str(key)] = GameOutcome(str(value))
    return parsed


def _guess_format(payload: Any, forced_format: str) -> str:
    """Infer input format unless explicitly forced."""
    if forced_format != "auto":
        return forced_format

    if isinstance(payload, dict) and isinstance(payload.get("frames"), list):
        return "analysis"
    if isinstance(payload, list):
        return "training"

    raise ValueError(
        "Could not infer JSON format. Use --format analysis or --format training."
    )


def _detect_player_placements(
    prev_detections: List[Dict[str, Any]],
    curr_detections: List[Dict[str, Any]],
    prev_tracked: Set[str],
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """Detect newly placed player cards between consecutive frames."""
    new_placements: List[Dict[str, Any]] = []
    current_tracked: Set[str] = set()

    for det in curr_detections:
        if det.get("is_opponent", True):
            continue
        if not det.get("is_on_field", False):
            continue

        tile_y = int(det.get("tile_y", 0))
        if tile_y < 17:
            continue

        card_name = str(det.get("class_name", ""))
        tile_x = int(det.get("tile_x", 0))
        card_id = f"{card_name}_{tile_x}_{tile_y}"
        current_tracked.add(card_id)

        if card_id not in prev_tracked:
            new_placements.append(
                {
                    "card_name": card_name,
                    "tile_x": tile_x,
                    "tile_y": tile_y,
                }
            )

    return new_placements, current_tracked


def _extract_pairs_from_analysis_json(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert analysis JSON into training pairs."""
    frames = payload.get("frames", [])
    if not isinstance(frames, list):
        raise ValueError("Analysis JSON field 'frames' must be a list.")

    pairs: List[Dict[str, Any]] = []
    tracked: Set[str] = set()

    for idx in range(1, len(frames)):
        prev_frame = frames[idx - 1]
        curr_frame = frames[idx]

        prev_detections = prev_frame.get("detections", [])
        curr_detections = curr_frame.get("detections", [])
        placements, tracked = _detect_player_placements(
            prev_detections,
            curr_detections,
            tracked,
        )

        for placement in placements:
            pairs.append({"state": prev_frame, "action": placement})

    return pairs


def _extract_pairs_from_training_json(payload: Any) -> List[Dict[str, Any]]:
    """Validate and return training pairs from label_extractor JSON."""
    if not isinstance(payload, list):
        raise ValueError("Training JSON must be a list of {'state', 'action'} objects.")

    validated: List[Dict[str, Any]] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Training JSON item {idx} is not an object.")
        if "state" not in item or "action" not in item:
            raise ValueError(
                f"Training JSON item {idx} is missing 'state' or 'action'."
            )
        validated.append(item)

    return validated


def _lookup_outcome(
    source_path: Path,
    outcomes: Dict[str, GameOutcome],
    video_path: Optional[str] = None,
) -> GameOutcome:
    """Find outcome using source path and optional original video path."""
    if not outcomes:
        return GameOutcome.UNKNOWN

    keys: List[str] = [
        str(source_path),
        source_path.name,
        source_path.stem,
    ]

    if video_path:
        video_obj = Path(video_path)
        keys.extend([video_path, video_obj.name, video_obj.stem])

    for key in keys:
        if key in outcomes:
            return outcomes[key]

    return GameOutcome.UNKNOWN


def _write_pickle_atomic(episodes: List[Episode], output_path: Path) -> None:
    """Write pickle via temp file then replace target path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    with open(tmp_path, "wb") as f:
        pickle.dump(episodes, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, output_path)


def convert_json_files(
    input_paths: Sequence[Path],
    forced_format: str,
    outcomes: Dict[str, GameOutcome],
    config: Optional[DTConfig] = None,
) -> Tuple[List[Episode], List[FileConversionResult]]:
    """Convert one or more JSON files into Episode objects."""
    if config is None:
        config = DTConfig()

    episodes: List[Episode] = []
    results: List[FileConversionResult] = []

    for path in input_paths:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        source_format = _guess_format(payload, forced_format)

        if source_format == "analysis":
            assert isinstance(payload, dict)
            pairs = _extract_pairs_from_analysis_json(payload)
            original_video_path = payload.get("video_info", {}).get("path")
            outcome = _lookup_outcome(path, outcomes, original_video_path)
            episode = build_episode(
                training_pairs=pairs,
                outcome=outcome,
                video_path=str(original_video_path or path),
                config=config,
            )
        else:
            pairs = _extract_pairs_from_training_json(payload)
            outcome = _lookup_outcome(path, outcomes)
            episode = build_episode(
                training_pairs=pairs,
                outcome=outcome,
                video_path=str(path),
                config=config,
            )

        skipped = episode.length == 0
        reason = "no valid timesteps after card-in-hand filtering" if skipped else ""
        results.append(
            FileConversionResult(
                source_path=path,
                source_format=source_format,
                training_pairs=len(pairs),
                episode_length=episode.length,
                skipped=skipped,
                skip_reason=reason,
            )
        )

        if not skipped:
            episodes.append(episode)

    return episodes, results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert analysis/training JSON into Decision Transformer episodes.pkl"
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="JSON files or glob patterns",
    )
    parser.add_argument(
        "--output",
        default="data/episodes.pkl",
        help="Output pickle path (default: data/episodes.pkl)",
    )
    parser.add_argument(
        "--format",
        choices=["auto", "analysis", "training"],
        default="auto",
        help="Input JSON format (default: auto)",
    )
    parser.add_argument(
        "--outcomes",
        default=None,
        help="JSON mapping of key -> win/loss/unknown",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow writing an empty episode list",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-file logs",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    input_paths = _resolve_paths(args.inputs)
    if not input_paths:
        print("No input JSON files were found.")
        return 2

    try:
        outcomes = _load_outcomes(args.outcomes)
        episodes, results = convert_json_files(
            input_paths=input_paths,
            forced_format=args.format,
            outcomes=outcomes,
        )
    except Exception as exc:
        print(f"Conversion failed: {exc}")
        return 1

    if not args.quiet:
        for item in results:
            status = "SKIP" if item.skipped else "OK"
            line = (
                f"[{status}] {item.source_path} | format={item.source_format} "
                f"pairs={item.training_pairs} timesteps={item.episode_length}"
            )
            if item.skip_reason:
                line += f" | reason={item.skip_reason}"
            print(line)

    if not episodes and not args.allow_empty:
        print("No non-empty episodes were built. Refusing to overwrite output.")
        print("Use --allow-empty if you explicitly want an empty episodes pickle.")
        return 2

    output_path = Path(args.output)
    try:
        _write_pickle_atomic(episodes, output_path)
    except Exception as exc:
        print(f"Failed to write pickle: {exc}")
        return 1

    total_steps = sum(ep.length for ep in episodes)
    print(
        f"Wrote {len(episodes)} episode(s), {total_steps} timestep(s) to {output_path.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
