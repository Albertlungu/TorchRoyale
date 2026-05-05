"""Archers strategic coordination module."""

from src.actions.generic.defensive_positioning import DefensivePositioning
from src.namespaces.cards import Cards


class ArchersAction(DefensivePositioning):
    """
    Heuristic action for placing the Archers card.

    Attributes:
        CARD (Card): Always ``Cards.ARCHERS``.
    """

    CARD = Cards.ARCHERS

    def calculate_score(self, state) -> list:
        """
        Score Archers placement based on elixir and nearby enemy positions.

        Args:
            state: Current game state with enemy detections and elixir count.

        Returns:
            list: ``[1, distance_bonus]`` when an enemy is within range on the
                correct lane, ``[0.5]`` at full elixir, or ``[0]`` otherwise.
        """
        score = [0.5] if state.numbers.elixir.number == 10 else [0]
        for detection in state.enemies:
            left_side = detection.position.tile_x <= 8 and self.tile_x == 7
            right_side = detection.position.tile_x > 8 and self.tile_x == 10
            if self.tile_y < detection.position.tile_y <= 14 and (
                left_side or right_side
            ):
                score = [1, self.tile_y - detection.position.tile_y]
        return score
