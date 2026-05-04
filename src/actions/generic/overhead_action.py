"""Air troop drop heuristics."""

import math

from src.actions.generic.action import Action


class OverheadAction(Action):
    """Drop directly over enemy troops, or cycle at 10 elixir."""

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
