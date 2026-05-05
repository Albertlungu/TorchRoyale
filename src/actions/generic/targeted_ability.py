"""Targeted ability module for spell coordination."""

import math

from src.actions.generic.action import Action
from src.namespaces.units import Units


class TargetedAbility(Action):
    """
    Strategic spell deployment based on target value analysis.

    Attributes:
        CARD (Card): The card associated with this action.
        RADIUS (float): Effective blast radius of the spell in tiles.
        MIN_SCORE (int): Minimum hit-value threshold required to recommend the spell.
        UNIT_TO_SCORE (dict): Mapping of unit type to hit-value contribution.
        tile_x (int): Grid column for the placement.
        tile_y (int): Grid row for the placement.
    """

    RADIUS = None
    MIN_SCORE = 5
    UNIT_TO_SCORE = {Units.SKELETON: 1}

    def calculate_score(self, state) -> list:
        """
        Score spell placement by the total value of enemies caught in the blast radius.

        Args:
            state: Current game state with enemy detections.

        Returns:
            list: Three-element list containing a binary viability flag,
                the total hit score, and the negative minimum distance to any
                hit enemy.
        """
        hit_score = 0
        max_distance = float("inf")
        for detection in state.enemies:
            distance = math.hypot(
                self.tile_x - detection.position.tile_x,
                self.tile_y - detection.position.tile_y + 2,
            )
            if distance <= self.RADIUS - 1:
                hit_score += self.UNIT_TO_SCORE.get(detection.unit, 2)
                max_distance = min(max_distance, -distance)
        return [1 if hit_score >= self.MIN_SCORE else 0, hit_score, max_distance]
