"""
InferenceRunner: loads a cached analysis JSON and runs the DT on each frame,
producing a JSONL recommendations file for the overlay.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.constants.cards import elixir_cost
from src.recommendation.strategy import DTStrategy


class InferenceRunner:
    def __init__(
        self,
        video_path: str,
        checkpoint_path: str,
        output_jsonl: str,
        analysis_dir: str = "output/analysis",
        device: str = "cpu",
    ):
        self._video_path = video_path
        self._checkpoint = checkpoint_path
        self._output = Path(output_jsonl)
        self._analysis_dir = Path(analysis_dir)
        self._device = device

    def run(self) -> Path:
        stem = Path(self._video_path).stem
        cached = self._analysis_dir / f"{stem}_analysis.json"

        if cached.exists():
            print(f"Loading cached analysis: {cached}")
            with open(cached) as f:
                result = json.load(f)
        else:
            print(f"No cached analysis for {stem}. Run analyze_video.py first.")
            raise FileNotFoundError(cached)

        frames = result["frames"]
        print(f"Loaded {len(frames)} frames. Running inference ...")

        strategy = DTStrategy(self._checkpoint, device=self._device)
        strategy.reset_game()

        self._output.parent.mkdir(parents=True, exist_ok=True)
        n_recs = 0

        with open(self._output, "w") as f:
            for frame in frames:
                ts  = frame["timestamp_ms"]
                elx = frame.get("player_elixir", 0)
                rec = strategy.recommend(frame)

                if rec is not None:
                    card, tx, ty = rec
                    cost = elixir_cost(card) or 0
                    entry = {
                        "timestamp_ms": ts,
                        "player_elixir": elx,
                        "has_recommendation": True,
                        "card": card,
                        "tile_x": tx,
                        "tile_y": ty,
                        "elixir_required": cost,
                        "detections": frame.get("detections", []),
                    }
                    n_recs += 1
                else:
                    entry = {
                        "timestamp_ms": ts,
                        "player_elixir": elx,
                        "has_recommendation": False,
                        "detections": frame.get("detections", []),
                    }
                f.write(json.dumps(entry) + "\n")

        print(f"Wrote {len(frames)} frames ({n_recs} recommendations) → {self._output}")
        return self._output
