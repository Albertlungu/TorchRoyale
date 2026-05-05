"""Defensive positioning module for strategic defense coordination."""

from src.actions.generic.action import Action


class DefensivePositioning(Action):
    """
    Strategic defensive placement based on threat assessment.

    Attributes:
        CARD (Card): The card associated with this action.
        tile_x (int): Grid column for the placement.
        tile_y (int): Grid row for the placement.
    """

    def calculate_score(self, state) -> list:
        """
        Score defensive placement based on the lane with the most enemy pressure.

        Args:
            state: Current game state with enemy detections and tile coordinates.

        Returns:
            list: ``[1]`` when the placement counters the dominant threat lane,
                ``[0]`` otherwise.
        """
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
