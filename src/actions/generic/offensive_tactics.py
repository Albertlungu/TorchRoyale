"""Offensive tactics module for strategic unit placement."""

from src.actions.generic.action import Action


class OffensiveTactics(Action):
    """
    Strategic offensive positioning based on game state analysis.

    Attributes:
        CARD (Card): The card associated with this action.
        tile_x (int): Grid column for the placement.
        tile_y (int): Grid row for the placement.
    """

    def calculate_score(self, state) -> list:
        """
        Score bridge-push placement based on elixir and enemy tower health.

        Args:
            state: Current game state with elixir count and tower health values.

        Returns:
            list: Score components favouring the lane with the weaker enemy tower,
                or ``[0]`` when the placement is invalid or elixir is insufficient.
        """
        if (self.tile_x, self.tile_y) not in {(3, 15), (14, 15)}:
            return [0]

        if state.numbers.elixir.number != 10:
            return [0]

        left_hp = state.numbers.left_enemy_princess_hp.number
        right_hp = state.numbers.right_enemy_princess_hp.number
        if self.tile_x == 3:
            return [1, left_hp > 0, left_hp <= right_hp]
        return [1, right_hp > 0, right_hp <= left_hp]
