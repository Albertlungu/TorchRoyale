"""Shared detection result dataclass used by all detection backends."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Detection:
    class_name: str        # canonical card/unit name (no suffix)
    tile_x: int
    tile_y: int
    is_opponent: bool
    is_on_field: bool
    confidence: float
    bbox_px: tuple = field(default_factory=tuple)  # (x1,y1,x2,y2) in full frame


@dataclass
class HandCard:
    class_name: str        # canonical card name
    slot: int              # 0-3
    confidence: float = 1.0


@dataclass
class FrameDetections:
    on_field: List[Detection] = field(default_factory=list)
    hand: List[HandCard] = field(default_factory=list)
