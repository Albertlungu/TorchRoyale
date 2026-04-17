"""
Unit tests for card_costs module.
"""

import pytest
from src.constants.card_costs import (
    CARD_ELIXIR_COSTS,
    get_card_cost,
    get_all_cards_by_cost,
    is_valid_card,
)


class TestCardElixirCosts:
    """Tests for CARD_ELIXIR_COSTS dictionary."""

    def test_contains_basic_cards(self):
        assert "knight" in CARD_ELIXIR_COSTS
        assert "archers" in CARD_ELIXIR_COSTS
        assert "giant" in CARD_ELIXIR_COSTS

    def test_contains_all_rarities(self):
        assert "skeletons" in CARD_ELIXIR_COSTS  # Common
        assert "musketeer" in CARD_ELIXIR_COSTS  # Rare
        assert "witch" in CARD_ELIXIR_COSTS  # Epic
        assert "princess" in CARD_ELIXIR_COSTS  # Legendary

    def test_contains_spells(self):
        assert "fireball" in CARD_ELIXIR_COSTS
        assert "zap" in CARD_ELIXIR_COSTS
        assert "arrows" in CARD_ELIXIR_COSTS

    def test_contains_buildings(self):
        assert "cannon" in CARD_ELIXIR_COSTS
        assert "tesla" in CARD_ELIXIR_COSTS
        assert "goblin-hut" in CARD_ELIXIR_COSTS


class TestGetCardCost:
    """Tests for get_card_cost function."""

    def test_basic_troops(self):
        assert get_card_cost("knight") == 3
        assert get_card_cost("archers") == 3
        assert get_card_cost("giant") == 5

    def test_spells(self):
        assert get_card_cost("fireball") == 4
        assert get_card_cost("zap") == 2
        assert get_card_cost("arrows") == 3

    def test_buildings(self):
        assert get_card_cost("cannon") == 3
        assert get_card_cost("tesla") == 4
        assert get_card_cost("goblin-hut") == 5

    def test_legendary_cards(self):
        assert get_card_cost("princess") == 3
        assert get_card_cost("the-log") == 2

    def test_evolution_cards(self):
        assert get_card_cost("evo-skeletons") == 1
        assert get_card_cost("evo-knight") == 3
        assert get_card_cost("evo-musketeer") == 4

    def test_hero_cards(self):
        assert get_card_cost("hero-knight") == 3
        assert get_card_cost("hero-musketeer") == 4

    def test_hero_ability_strips_suffix(self):
        assert get_card_cost("hero-ice-golem-ability") == 2
        assert get_card_cost("hero-musketeer-ability") == 4

    def test_towers(self):
        assert get_card_cost("king-tower") == 0
        assert get_card_cost("princess-tower") == 0


class TestNameNormalization:
    """Tests for name normalization in get_card_cost."""

    def test_opponent_prefix(self):
        assert get_card_cost("opponent-hog-rider") == 4
        assert get_card_cost("opponent-knight") == 3

    def test_player_prefix(self):
        assert get_card_cost("player-giant") == 5

    def test_in_hand_suffix(self):
        assert get_card_cost("hog-rider-in-hand") == 4

    def test_next_suffix(self):
        assert get_card_cost("knight-next") == 3

    def test_on_field_suffix(self):
        assert get_card_cost("cannon-on-field") == 3

    def test_evolution_suffix(self):
        assert get_card_cost("skeletons-evolution") == 1


class TestCaseInsensitivity:
    """Tests for case insensitivity."""

    def test_uppercase(self):
        assert get_card_cost("KNIGHT") == 3
        assert get_card_cost("GIANT") == 5

    def test_mixed_case(self):
        assert get_card_cost("Knight") == 3
        assert get_card_cost("FireBall") == 4


class TestGetAllCardsByCost:
    """Tests for get_all_cards_by_cost function."""

    def test_returns_list(self):
        cards = get_all_cards_by_cost(1)
        assert isinstance(cards, list)

    def test_contains_expected(self):
        cards = get_all_cards_by_cost(1)
        assert "skeletons" in cards

    def test_multiple_costs(self):
        cards_1 = get_all_cards_by_cost(1)
        cards_2 = get_all_cards_by_cost(2)
        assert len(cards_1) > 0
        assert len(cards_2) > 0


class TestIsValidCard:
    """Tests for is_valid_card function."""

    def test_valid_troop(self):
        assert is_valid_card("knight") is True

    def test_valid_spell(self):
        assert is_valid_card("fireball") is True

    def test_valid_building(self):
        assert is_valid_card("cannon") is True

    def test_valid_tower(self):
        assert is_valid_card("king-tower") is True
        assert is_valid_card("princess-tower") is True

    def test_invalid_card(self):
        assert is_valid_card("not-a-card") is False
        assert is_valid_card("") is False


class TestCardCostRange:
    """Tests for valid elixir cost ranges."""

    def test_all_costs_in_valid_range(self):
        for card, cost in CARD_ELIXIR_COSTS.items():
            assert 0 <= cost <= 10, f"Invalid cost for {card}: {cost}"

    def test_has_1_cost_cards(self):
        assert get_card_cost("skeletons") == 1
        assert get_card_cost("ice-spirit") == 1

    def test_has_9_cost_cards(self):
        assert get_card_cost("three-musketeers") == 9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])