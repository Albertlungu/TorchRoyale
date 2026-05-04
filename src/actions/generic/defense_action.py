"""Defensive placement heuristics."""

from src.actions.generic.action import Action


class DefenseAction(Action):
    """Play in fixed defensive tiles when enemies cross the bridge."""

    def calculate_score(self, state):
        if (self.tile_x, self.tile_y) not in {(8, 9), (9, 9)}:
            return [0]

        left_count = 0
        right_count = 0
        for detection in state.enemies:
            if detection.position.tile_y > 16:
                continue
            if detection.position.tile_x >= 9:
                right_count += 1
            else:
                left_count += 1

        if left_count == right_count == 0:
            return [0]
        if left_count >= right_count and self.tile_x == 9:
            return [0]
        return [1]
