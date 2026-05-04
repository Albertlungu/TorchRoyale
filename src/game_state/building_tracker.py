"""
Building placement pattern analyzer for prediction fireballing.

Tracks opponent building placements over time to learn preferred positions
and calculate confidence scores for prediction plays.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Buildings that are commonly predicted with fireball
_PREDICTABLE_BUILDINGS = {
    "cannon",
    "tesla",
    "inferno-tower",
    "bomb-tower",
    "goblin-cage",
    "tombstone",
    "furnace",
    "elixir-collector",
}


class BuildingPlacementTracker:
    """Tracks opponent building placements to enable prediction plays."""

    def __init__(self) -> None:
        # Maps (tile_x, tile_y) -> {count, last_seen, confidence}
        self.placement_patterns: Dict[Tuple[int, int], Dict] = defaultdict(
            lambda: {"count": 0, "last_seen": 0.0, "confidence": 0.0}
        )

        # Track recent placements for temporal analysis
        self.recent_placements: List[Dict] = []

        # Track building types at each position
        self.building_types_at_position: Dict[Tuple[int, int], set] = defaultdict(set)

        # Learning parameters
        self.min_placements_for_prediction = 3
        self.confidence_threshold = 0.6
        self.decay_rate = 0.95  # Confidence decays over time

    def record_building_placement(
        self, building_name: str, tile_x: int, tile_y: int, game_time: float
    ) -> None:
        """
        Record that opponent placed a building at a specific position.

        Args:
            building_name: Normalized building name
            tile_x, tile_y: Placement coordinates
            game_time: Current game time in seconds
        """
        if building_name.lower() not in _PREDICTABLE_BUILDINGS:
            return

        pos_key = (tile_x, tile_y)

        # Update pattern tracking
        self.placement_patterns[pos_key]["count"] += 1
        self.placement_patterns[pos_key]["last_seen"] = game_time

        # Track which building types appear at this position
        self.building_types_at_position[pos_key].add(building_name.lower())

        # Add to recent placements (keep last 20)
        self.recent_placements.append(
            {"building": building_name.lower(), "position": pos_key, "time": game_time}
        )

        if len(self.recent_placements) > 20:
            self.recent_placements.pop(0)

        # Recalculate confidence for this position
        self._recalculate_confidence(pos_key, game_time)

    def _recalculate_confidence(
        self, position: Tuple[int, int], current_time: float
    ) -> None:
        """Recalculate confidence score for a position based on placement history."""
        pattern = self.placement_patterns[position]
        count = pattern["count"]
        last_seen = pattern["last_seen"]

        if count < self.min_placements_for_prediction:
            pattern["confidence"] = 0.0
            return

        # Base confidence on count (more placements = higher confidence)
        base_confidence = min(1.0, count / 5.0)  # Cap at 5 placements

        # Recency bonus (more recent placements are more reliable)
        time_since_last = max(0.0, current_time - last_seen)
        recency_bonus = max(
            0.0, 1.0 - (time_since_last / 60.0)
        )  # Decay over 60 seconds

        # Consistency bonus (same building type = higher confidence)
        building_types = len(self.building_types_at_position[position])
        consistency_bonus = (
            1.0 if building_types == 1 else 0.7 if building_types == 2 else 0.5
        )

        # Calculate final confidence
        confidence = base_confidence * (0.5 + recency_bonus * 0.5) * consistency_bonus

        # Apply temporal decay from previous confidence
        if pattern["confidence"] > 0:
            confidence = max(confidence, pattern["confidence"] * self.decay_rate)

        pattern["confidence"] = min(1.0, confidence)

    def get_prediction_target(
        self, current_time: float
    ) -> Optional[Tuple[int, int, float]]:
        """
        Get the best prediction target for fireball.

        Returns:
            (tile_x, tile_y, confidence) or None if no good prediction
        """
        best_position = None
        best_confidence = 0.0

        for position, pattern in self.placement_patterns.items():
            confidence = pattern["confidence"]

            # Skip if confidence too low or position too old
            if confidence < self.confidence_threshold:
                continue

            time_since_last = current_time - pattern["last_seen"]
            if time_since_last > 30:  # Too long since last placement
                continue

            # Prefer positions with higher confidence
            if confidence > best_confidence:
                best_confidence = confidence
                best_position = position

        if best_position:
            return (*best_position, best_confidence)

        return None

    def get_state_string(self) -> str:
        """Get formatted state for debugging."""
        if not self.placement_patterns:
            return "No building placement data yet"

        # Sort by confidence
        sorted_positions = sorted(
            self.placement_patterns.items(),
            key=lambda x: x[1]["confidence"],
            reverse=True,
        )

        lines = ["Building Placement Patterns:"]
        for pos, pattern in sorted_positions[:3]:  # Top 3
            if pattern["confidence"] > 0.1:
                building_types = ", ".join(self.building_types_at_position[pos])
                lines.append(
                    f"  Pos({pos[0]},{pos[1]}): {pattern['confidence']:.2f} confidence, "
                    f"{pattern['count']} placements, Buildings: {building_types}"
                )

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset tracker between games."""
        self.placement_patterns.clear()
        self.recent_placements.clear()
        self.building_types_at_position.clear()
