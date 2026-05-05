"""Aerial coordination module for air unit deployment."""

import math

from src.actions.generic.action import Action


class AerialCoordination(Action):
    """
    Strategic air unit deployment based on battlefield analysis.

    Attributes:
        CARD (Card): The card associated with this action.
        tile_x (int): Grid column for the placement.
        tile_y (int): Grid row for the placement.
    """

    def calculate_score(self, state) -> list:
        """
        Score aerial placement by proximity to enemy units.

        Args:
            state: Current game state with enemy detections and elixir count.

        Returns:
            list: Score components where the first element is the primary score
                and the second (when present) is the negative distance to the
                nearest enemy.
        """
        score = [0.5] if state.numbers.elixir.number == 10 else [0]
        for detection in state.enemies:
            distance = math.hypot(
                detection.position.tile_x - self.tile_x,
                detection.position.tile_y - self.tile_y,
            )
            if distance < 1:
                score = [1, -distance]
        return score
