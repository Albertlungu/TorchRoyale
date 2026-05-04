"""
Shared detection result dataclasses used by all detection backends.

Public API:
  Detection      -- a single on-field unit with tile coordinates and metadata
  HandCard       -- a single card detected in the player's hand
  FrameDetections -- container holding all detections for one video frame
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Detection:
    """A single on-field unit detected by the battlefield detector."""

    class_name: str       # canonical card/unit name (no suffix)
    tile_x: int
    tile_y: int
    is_opponent: bool
    is_on_field: bool
    confidence: float
    bbox_px: Tuple[int, int, int, int] = field(
        default_factory=tuple  # type: ignore[arg-type]
    )  # (x1, y1, x2, y2) in full-frame pixels


@dataclass
class HandCard:
    """A card detected in the player's hand."""

    class_name: str    # canonical card name
    slot: int          # 0-based slot index (0–3)
    confidence: float = 1.0


@dataclass
class FrameDetections:
    """All detections produced for a single video frame."""

    on_field: List[Detection] = field(default_factory=list)
    hand: List[HandCard] = field(default_factory=list)
