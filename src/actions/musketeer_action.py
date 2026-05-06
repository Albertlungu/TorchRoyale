"""Musketeer strategic coordination module."""

import math

from src.actions.generic.defensive_positioning import DefensivePositioning
from src.namespaces.cards import Cards


class MusketeerAction(DefensivePositioning):
    """
    Heuristic action for placing the Musketeer card.

    Attributes:
        CARD (Card): Always ``Cards.MUSKETEER``.
    """

    CARD = Cards.MUSKETEER

    def calculate_score(self, state) -> list:
        """
        Score Musketeer placement based on distance to the nearest enemy.

        The Musketeer is most effective when placed just outside her attack range
        so that she can fire at enemies without being immediately targeted.

        Args:
            state: Current game state with enemy detections.

        Returns:
            list: ``[1]`` when the nearest enemy is between 5 and 6 tiles away,
                ``[0]`` otherwise.
        """
        for detection in state.enemies:
            distance = math.hypot(
                detection.position.tile_x - self.tile_x,
                detection.position.tile_y - self.tile_y,
            )
            if 5 < distance < 6:
                return [1]
            if distance < 5:
                return [0]
        return [0]
