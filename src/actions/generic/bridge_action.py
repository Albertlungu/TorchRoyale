"""Bridge play heuristics."""

from src.actions.generic.action import Action


class BridgeAction(Action):
    """Play at bridge only when capped elixir favors pressure."""

    def calculate_score(self, state):
        if (self.tile_x, self.tile_y) not in {(3, 15), (14, 15)}:
            return [0]

        if state.numbers.elixir.number != 10:
            return [0]

        left_hp = state.numbers.left_enemy_princess_hp.number
        right_hp = state.numbers.right_enemy_princess_hp.number
        if self.tile_x == 3:
            return [1, left_hp > 0, left_hp <= right_hp]
        return [1, right_hp > 0, right_hp <= left_hp]
