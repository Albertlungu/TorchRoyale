"""Aerial coordination module for air unit deployment."""

import math

from src.actions.generic.action import Action


class AerialCoordination(Action):
    """Strategic air unit deployment based on battlefield analysis."""

    def calculate_score(self, state):
        score = [0.5] if state.numbers.elixir.number == 10 else [0]
        for detection in state.enemies:
            distance = math.hypot(
                detection.position.tile_x - self.tile_x,
                detection.position.tile_y - self.tile_y,
            )
            if distance < 1:
                score = [1, -distance]
        return score
