"""
Unit tests for game_constants module.
"""

import pytest
from src.constants.game_constants import (
    GamePhase,
    ElixirConstants,
    GameTimingConstants,
    TOWER_HP,
    get_tower_max_hp,
    get_elixir_cost,
    get_regen_rate,
)


class TestGamePhase:
    """Tests for GamePhase enum."""

    def test_phase_values(self):
        assert GamePhase.SINGLE_ELIXIR.value == "single"
        assert GamePhase.DOUBLE_ELIXIR.value == "double"
        assert GamePhase.TRIPLE_ELIXIR.value == "triple"
        assert GamePhase.SUDDEN_DEATH.value == "sudden_death"
        assert GamePhase.GAME_OVER.value == "game_over"


class TestElixirConstants:
    """Tests for ElixirConstants."""

    def test_max_elixir(self):
        assert ElixirConstants.MAX_ELIXIR == 10

    def test_starting_elixir(self):
        assert ElixirConstants.STARTING_ELIXIR == 5

    def test_single_regen_rate(self):
        assert ElixirConstants.SINGLE_REGEN_RATE == 2.8

    def test_double_regen_rate(self):
        assert ElixirConstants.DOUBLE_REGEN_RATE == 1.4

    def test_triple_regen_rate(self):
        assert ElixirConstants.TRIPLE_REGEN_RATE == pytest.approx(2.8 / 3)


class TestGameTimingConstants:
    """Tests for GameTimingConstants."""

    def test_total_game_time(self):
        assert GameTimingConstants.TOTAL_GAME_TIME == 180

    def test_double_elixir_start(self):
        assert GameTimingConstants.DOUBLE_ELIXIR_START == 60

    def test_sudden_death_duration(self):
        assert GameTimingConstants.SUDDEN_DEATH_DURATION == 180


class TestTowerHP:
    """Tests for tower HP dictionary."""

    def test_contains_level_1(self):
        assert 1 in TOWER_HP

    def test_contains_max_level(self):
        assert 16 in TOWER_HP

    def test_king_hp_higher_than_princess(self):
        king_hp, princess_hp = TOWER_HP[11]
        assert king_hp > princess_hp

    def test_hp_increases_with_level(self):
        hp_lvl10 = TOWER_HP[10][0]
        hp_lvl11 = TOWER_HP[11][0]
        assert hp_lvl11 > hp_lvl10


class TestGetTowerMaxHP:
    """Tests for get_tower_max_hp function."""

    def test_king_tower_level_11(self):
        hp = get_tower_max_hp(11, is_king=True)
        assert hp == 4824

    def test_princess_tower_level_11(self):
        hp = get_tower_max_hp(11, is_king=False)
        assert hp == 3052

    def test_unknown_level_defaults_to_15(self):
        hp = get_tower_max_hp(99, is_king=True)
        expected = TOWER_HP[15][0]
        assert hp == expected


class TestGetElixirCost:
    """Tests for get_elixir_cost function."""

    def test_basic_troops(self):
        assert get_elixir_cost("hog-rider") == 4
        assert get_elixir_cost("musketeer") == 4

    def test_spells(self):
        assert get_elixir_cost("fireball") == 4
        assert get_elixir_cost("the-log") == 2

    def test_buildings(self):
        assert get_elixir_cost("cannon") == 3

    def test_evolution_cards(self):
        assert get_elixir_cost("evo-musketeer") == 4
        assert get_elixir_cost("evo-skeletons") == 1

    def test_hero_cards(self):
        assert get_elixir_cost("hero-ice-golem") == 2

    def test_hero_ability_resolves_to_base(self):
        assert get_elixir_cost("hero-musketeer-ability") == 4


class TestGetRegenRate:
    """Tests for get_regen_rate function."""

    def test_single_elixir(self):
        rate = get_regen_rate(GamePhase.SINGLE_ELIXIR)
        assert rate == 2.8

    def test_double_elixir(self):
        rate = get_regen_rate(GamePhase.DOUBLE_ELIXIR)
        assert rate == 1.4

    def test_triple_elixir(self):
        rate = get_regen_rate(GamePhase.TRIPLE_ELIXIR)
        assert rate == pytest.approx(2.8 / 3)

    def test_sudden_death(self):
        rate = get_regen_rate(GamePhase.SUDDEN_DEATH)
        assert rate == 1.4

    def test_game_over(self):
        rate = get_regen_rate(GamePhase.GAME_OVER)
        assert rate == 2.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])