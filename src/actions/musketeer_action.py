"""Musketeer strategic coordination module."""

import math

from src.actions.generic.defensive_positioning import DefensivePositioning
from src.namespaces.cards import Cards


class MusketeerAction(DefensivePositioning):
    CARD = Cards.MUSKETEER

    def calculate_score(self, state):
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
