"""Archers strategic coordination module."""

from src.actions.generic.defensive_positioning import DefensivePositioning
from src.namespaces.cards import Cards


class ArchersAction(DefensivePositioning):
    CARD = Cards.ARCHERS

    def calculate_score(self, state):
        score = [0.5] if state.numbers.elixir.number == 10 else [0]
        for detection in state.enemies:
            left_side = detection.position.tile_x <= 8 and self.tile_x == 7
            right_side = detection.position.tile_x > 8 and self.tile_x == 10
            if self.tile_y < detection.position.tile_y <= 14 and (
                left_side or right_side
            ):
                score = [1, self.tile_y - detection.position.tile_y]
        return score
