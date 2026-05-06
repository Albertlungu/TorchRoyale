"""Giant strategic coordination module."""

from src.actions.generic.offensive_tactics import OffensiveTactics
from src.namespaces.cards import Cards


class GiantAction(OffensiveTactics):
    """
    Heuristic action for placing the Giant card.

    Attributes:
        CARD (Card): Always ``Cards.GIANT``.
    """

    CARD = Cards.GIANT

    def calculate_score(self, state) -> list:
        """
        Score Giant placement at bridge positions based on elixir and tower health.

        Args:
            state: Current game state with elixir count and enemy tower health.

        Returns:
            list: ``[1, tower_alive, weaker_lane]`` when conditions favour a push,
                or ``[0]`` when elixir is insufficient or the tile is not a bridge
                position.
        """
        if state.numbers.elixir.number != 10:
            return [0]

        left_hp = state.numbers.left_enemy_princess_hp.number
        right_hp = state.numbers.right_enemy_princess_hp.number

        if (self.tile_x, self.tile_y) == (3, 15):
            return [1, left_hp > 0, left_hp <= right_hp]
        if (self.tile_x, self.tile_y) == (14, 15):
            return [1, right_hp > 0, right_hp <= left_hp]
        return [0]
