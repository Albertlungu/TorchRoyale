"""
Processes a replay video through the full pipeline and writes per-frame
recommendations to a JSONL file.

Each line is one of:
  {"timestamp_ms": ..., "player_elixir": ..., "has_recommendation": true,
   "card": "...", "tile_x": ..., "tile_y": ..., "elixir_required": ...}
  {"timestamp_ms": ..., "player_elixir": ..., "has_recommendation": false}
"""

import json
import sys
from pathlib import Path
from typing import Optional

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.video.video_analyzer import VideoAnalyzer
from src.recommendation.strategy import DTStrategy
from src.constants.game_constants import get_elixir_cost


class InferenceRunner:
    """
    Runs VideoAnalyzer + DTStrategy on a replay video and writes a JSONL
    recommendation log that the overlay can read.
    """

    def __init__(
        self,
        video_path: str,
        checkpoint_path: str,
        output_jsonl: str,
        analysis_output_dir: str = "output/analysis",
        frame_skip: int = 6,
    ):
        self._video_path = video_path
        self._checkpoint_path = checkpoint_path
        self._output_jsonl = Path(output_jsonl)
        self._analysis_output_dir = analysis_output_dir
        self._frame_skip = frame_skip

    def run(self) -> Path:
        """
        Process the video end-to-end, run inference on every frame, and write
        the recommendation JSONL. Returns the path to the written file.

        If a VideoAnalyzer output JSON already exists for this video, it is
        loaded directly and the Roboflow analysis step is skipped.
        """
        video_stem = Path(self._video_path).stem
        cached_json = Path(self._analysis_output_dir) / f"{video_stem}_analysis.json"

        if cached_json.exists():
            print(f"Loading cached analysis: {cached_json}")
            import json as _json
            with open(cached_json) as _f:
                result = _json.load(_f)
        else:
            print(f"Analyzing video: {self._video_path}")
            analyzer = VideoAnalyzer(
                output_dir=self._analysis_output_dir,
                frame_skip=self._frame_skip,
                verbose=True,
            )
            result = analyzer.analyze_video(self._video_path)

        frames = result["frames"]
        print(f"Video analysis complete: {len(frames)} frames extracted.")

        print(f"Loading DT checkpoint: {self._checkpoint_path}")
        strategy = DTStrategy(
            checkpoint_path=self._checkpoint_path,
            device="cpu",
        )
        strategy.reset_game()

        if not strategy.is_ready:
            print(
                "Warning: DTStrategy could not load the checkpoint. "
                "All frames will be written with has_recommendation=false."
            )

        self._output_jsonl.parent.mkdir(parents=True, exist_ok=True)

        n_recommendations = 0
        with open(self._output_jsonl, "w") as f:
            for frame in frames:
                timestamp_ms: int = frame["timestamp_ms"]
                player_elixir: int = frame.get("player_elixir", 0)

                rec = strategy.recommend(frame)

                # Carry detections through so the player can log them.
                detections = frame.get("detections", [])

                if rec is not None:
                    card, tile_x, tile_y = rec
                    entry = {
                        "timestamp_ms": timestamp_ms,
                        "player_elixir": player_elixir,
                        "has_recommendation": True,
                        "card": card,
                        "tile_x": int(tile_x),
                        "tile_y": int(tile_y),
                        "elixir_required": int(get_elixir_cost(card)),
                        "detections": detections,
                    }
                    n_recommendations += 1
                else:
                    entry = {
                        "timestamp_ms": timestamp_ms,
                        "player_elixir": player_elixir,
                        "has_recommendation": False,
                        "detections": detections,
                    }

                f.write(json.dumps(entry) + "\n")

        print(
            f"Wrote {len(frames)} frames ({n_recommendations} with recommendations) "
            f"to {self._output_jsonl}"
        )
        return self._output_jsonl
