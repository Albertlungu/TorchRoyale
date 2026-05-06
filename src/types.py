"""
Centralised TypedDict definitions for structured data shared across modules.

All dicts that represent frame state, detection results, video metadata, or
tower state are defined here and imported throughout the codebase.

Public API:
  DetectionDict   -- a single on-field or in-hand unit detection
  TowerDict       -- health state for one tower
  FrameDict       -- complete per-frame state from the analysis pipeline
  VideoInfoDict   -- metadata header of an analysis JSON file
  RecommendationDict -- one entry from the InferenceRunner JSONL output
"""
from __future__ import annotations

from typing import List, Optional

from typing_extensions import TypedDict


class DetectionDict(TypedDict, total=False):
    """A single unit detection from the battlefield or hand."""

    class_name: str
    tile_x: int
    tile_y: int
    is_opponent: bool
    is_on_field: bool
    confidence: float


class HandSlotDict(TypedDict, total=False):
    """A single hand-card classification with an explicit 1-based slot."""

    slot: int
    class_name: str
    confidence: float


class TowerDict(TypedDict, total=False):
    """Health state for one tower (king or arena towers)."""

    health_percent: Optional[float]
    is_destroyed: bool


class _PlayerTowers(TypedDict, total=False):
    player_left: TowerDict
    player_king: TowerDict
    player_right: TowerDict


class _OpponentTowers(TypedDict, total=False):
    opponent_left: TowerDict
    opponent_king: TowerDict
    opponent_right: TowerDict


class FrameDict(TypedDict, total=False):
    """Complete per-frame state produced by the analysis pipeline."""

    timestamp_ms: int
    frame_number: int
    game_time_remaining: Optional[int]
    elixir_multiplier: int
    game_phase: Optional[str]
    player_elixir: Optional[int]
    opponent_elixir_estimated: Optional[int]
    detections: List[DetectionDict]
    hand_cards: List[str]
    hand_slots: List[HandSlotDict]
    player_towers: _PlayerTowers
    opponent_towers: _OpponentTowers


class VideoInfoDict(TypedDict):
    """Metadata header stored at the top of every analysis JSON."""

    path: str
    width: int
    height: int
    fps: float
    duration_seconds: float
    frame_skip: int
    total_frames_processed: int


class RecommendationDict(TypedDict, total=False):
    """One line of the JSONL file produced by InferenceRunner."""

    timestamp_ms: int
    player_elixir: Optional[int]
    has_recommendation: bool
    card: str
    tile_x: int
    tile_y: int
    elixir_required: int
    detections: List[DetectionDict]
