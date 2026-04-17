"""
Unit tests for card_types module.
"""

import pytest
from src.constants.card_types import (
    CardType,
    get_card_type,
    get_cards_by_type,
    is_troop,
    is_spell,
    is_building,
    is_tower_troop,
    get_card_type_name,
    CARD_TYPES,
)


class TestCardTypeEnum:
    """Tests for CardType enum."""

    def test_card_type_values(self):
        assert CardType.TROOP.value == "troop"
        assert CardType.SPELL.value == "spell"
        assert CardType.BUILDING.value == "building"
        assert CardType.TOWER_TROOP.value == "tower_troop"


class TestGetCardType:
    """Tests for get_card_type function."""

    def test_troop_cards(self):
        assert get_card_type("knight") == CardType.TROOP
        assert get_card_type("archers") == CardType.TROOP
        assert get_card_type("giant") == CardType.TROOP
        assert get_card_type("hog-rider") == CardType.TROOP
        assert get_card_type("mini-pekka") == CardType.TROOP
        assert get_card_type("prince") == CardType.TROOP
        assert get_card_type("balloon") == CardType.TROOP

    def test_spell_cards(self):
        assert get_card_type("fireball") == CardType.SPELL
        assert get_card_type("arrows") == CardType.SPELL
        assert get_card_type("zap") == CardType.SPELL
        assert get_card_type("lightning") == CardType.SPELL
        assert get_card_type("poison") == CardType.SPELL
        assert get_card_type("rocket") == CardType.SPELL
        assert get_card_type("the-log") == CardType.SPELL
        assert get_card_type("tornado") == CardType.SPELL
        assert get_card_type("freeze") == CardType.SPELL

    def test_building_cards(self):
        assert get_card_type("cannon") == CardType.BUILDING
        assert get_card_type("tesla") == CardType.BUILDING
        assert get_card_type("inferno-tower") == CardType.BUILDING
        assert get_card_type("x-bow") == CardType.BUILDING
        assert get_card_type("mortar") == CardType.BUILDING
        assert get_card_type("goblin-hut") == CardType.BUILDING
        assert get_card_type("bomb-tower") == CardType.BUILDING

    def test_tower_troops(self):
        assert get_card_type("king-tower") == CardType.TOWER_TROOP
        assert get_card_type("princess-tower") == CardType.TOWER_TROOP
        assert get_card_type("crown-tower") == CardType.TOWER_TROOP

    def test_name_normalization(self):
        assert get_card_type("opponent-hog-rider") == CardType.TROOP
        assert get_card_type("player-knight") == CardType.TROOP
        assert get_card_type("friendly-giant") == CardType.TROOP

    def test_suffix_removal(self):
        assert get_card_type("hog-rider-in-hand") == CardType.TROOP
        assert get_card_type("knight-next") == CardType.TROOP
        assert get_card_type("tesla-on-field") == CardType.BUILDING

    def test_evolution_cards(self):
        assert get_card_type("evo-skeletons") == CardType.TROOP
        assert get_card_type("evo-knight") == CardType.TROOP
        assert get_card_type("evo-musketeer") == CardType.TROOP

    def test_hero_cards(self):
        assert get_card_type("hero-knight") == CardType.TROOP
        assert get_card_type("hero-musketeer") == CardType.TROOP
        assert get_card_type("hero-pekka") == CardType.TROOP

    def test_hero_abilities_resolve_to_source(self):
        assert get_card_type("hero-knight-ability") == CardType.TROOP
        assert get_card_type("hero-musketeer-ability") == CardType.TROOP


class TestIsFunctions:
    """Tests for is_* helper functions."""

    def test_is_troop(self):
        assert is_troop("knight") is True
        assert is_troop("giant") is True
        assert is_troop("hog-rider") is True
        assert is_troop("fireball") is False
        assert is_troop("cannon") is False

    def test_is_spell(self):
        assert is_spell("fireball") is True
        assert is_spell("zap") is True
        assert is_spell("arrows") is True
        assert is_spell("knight") is False
        assert is_spell("cannon") is False

    def test_is_building(self):
        assert is_building("cannon") is True
        assert is_building("tesla") is True
        assert is_building("x-bow") is True
        assert is_building("knight") is False
        assert is_building("fireball") is False

    def test_is_tower_troop(self):
        assert is_tower_troop("king-tower") is True
        assert is_tower_troop("princess-tower") is True
        assert is_tower_troop("cannon") is False


class TestGetCardTypeName:
    """Tests for get_card_type_name function."""

    def test_returns_display_name(self):
        assert get_card_type_name("knight") == "Troop"
        assert get_card_type_name("fireball") == "Spell"
        assert get_card_type_name("cannon") == "Building"
        assert get_card_type_name("king-tower") == "Tower Troop"


class TestGetCardsByType:
    """Tests for get_cards_by_type function."""

    def test_returns_list(self):
        troops = get_cards_by_type(CardType.TROOP)
        assert isinstance(troops, list)
        assert len(troops) > 0

    def test_contains_expected_cards(self):
        troops = get_cards_by_type(CardType.TROOP)
        assert "knight" in troops
        assert "giant" in troops

        spells = get_cards_by_type(CardType.SPELL)
        assert "fireball" in spells
        assert "zap" in spells

        buildings = get_cards_by_type(CardType.BUILDING)
        assert "cannon" in buildings
        assert "tesla" in buildings


class TestCardTypeCoverage:
    """Tests that verify complete card coverage."""

    def test_all_cards_typed(self):
        assert len(CARD_TYPES) > 100, "Should have 100+ cards categorized"

    def test_all_troop_cards_have_type(self):
        common_troops = ["knight", "archers", "barbarians", "goblins", "skeletons"]
        for card in common_troops:
            assert get_card_type(card) == CardType.TROOP

    def test_all_spell_cards_have_type(self):
        common_spells = ["fireball", "arrows", "zap", "lightning", "rocket"]
        for card in common_spells:
            assert get_card_type(card) == CardType.SPELL

    def test_all_building_cards_have_type(self):
        common_buildings = ["cannon", "tesla", "mortar", "goblin-hut"]
        for card in common_buildings:
            assert get_card_type(card) == CardType.BUILDING


class TestCaseInsensitivity:
    """Tests for case insensitivity."""

    def test_uppercase(self):
        assert get_card_type("KNIGHT") == CardType.TROOP
        assert get_card_type("GIANT") == CardType.TROOP
        assert get_card_type("FIREBALL") == CardType.SPELL

    def test_mixed_case(self):
        assert get_card_type("Knight") == CardType.TROOP
        assert get_card_type("FireBall") == CardType.SPELL
        assert get_card_type("X-Bow") == CardType.BUILDING


if __name__ == "__main__":
    pytest.main([__file__, "-v"])