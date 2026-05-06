"""
InferenceRunner: loads a cached analysis JSON and runs the DT on each frame,
producing a JSONL recommendations file for the overlay.

The runner expects the analysis JSON to already exist (produced by VideoAnalyzer).
If the file is absent it raises FileNotFoundError rather than re-running analysis.

Public API:
  InferenceRunner -- construct with paths, call run() to produce the JSONL file
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from src.constants.cards import elixir_cost
from src.recommendation.heuristic_strategy import HeuristicStrategy
from src.types import FrameDict, RecommendationDict


class InferenceRunner:
    """
    Offline inference pass over a pre-computed analysis JSON.

    For each frame in the analysis, the Decision Transformer predicts a card
    placement. Results are written one-per-line as JSON (JSONL format).
    """

    def __init__(
        self,
        video_path: str,
        checkpoint_path: str,
        output_jsonl: str,
        analysis_dir: str = "output/analysis",
        device: str = "cpu",
    ) -> None:
        """
        Args:
            video_path:      path to the source replay video (used to locate the analysis JSON).
            checkpoint_path: path to the DT model checkpoint.
            output_jsonl:    path where the JSONL output will be written.
            analysis_dir:    directory containing <stem>_analysis.json files.
            device:          PyTorch device for the model.
        """
        self._video_path = video_path
        self._checkpoint = checkpoint_path
        self._output = Path(output_jsonl)
        self._analysis_dir = Path(analysis_dir)
        self._device = device

    def run(self) -> Path:
        """
        Run inference over all frames and write the JSONL output file.

        Returns:
            Path to the written JSONL file.

        Raises:
            FileNotFoundError: if the cached analysis JSON does not exist.
        """
        stem = Path(self._video_path).stem
        cached = self._analysis_dir / f"{stem}_analysis.json"

        if cached.exists():
            print(f"Loading cached analysis: {cached}")
            with open(cached, encoding="utf-8") as analysis_file:
                result = json.load(analysis_file)
        else:
            print(f"No cached analysis for {stem}. Run analyze_video.py first.")
            raise FileNotFoundError(cached)

        frames: List[FrameDict] = result["frames"]
        print(f"Loaded {len(frames)} frames. Running inference ...")

        strategy = HeuristicStrategy(self._checkpoint, device=self._device)
        strategy.reset_game()

        self._output.parent.mkdir(parents=True, exist_ok=True)
        n_recs: int = 0

        with open(self._output, "w", encoding="utf-8") as out_file:
            for frame in frames:
                ts: int = frame.get("timestamp_ms", 0)  # Use .get() for safety
                elx = frame.get("player_elixir", 0)
                rec = strategy.recommend(frame)

                entry: RecommendationDict
                if rec is not None:
                    card, tile_x, tile_y = rec
                    cost: int = elixir_cost(card) or 0
                    entry = RecommendationDict(
                        timestamp_ms=ts,
                        player_elixir=elx,
                        has_recommendation=True,
                        card=card,
                        tile_x=tile_x,
                        tile_y=tile_y,
                        elixir_required=cost,
                        detections=frame.get("detections", []),
                    )
                    n_recs += 1
                else:
                    entry = RecommendationDict(
                        timestamp_ms=ts,
                        player_elixir=elx,
                        has_recommendation=False,
                        detections=frame.get("detections", []),
                    )
                out_file.write(json.dumps(dict(entry)) + "\n")

        print(
            f"Wrote {len(frames)} frames ({n_recs} recommendations) -> {self._output}"
        )
        return self._output
