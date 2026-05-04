"""Fireball play heuristics."""

from src.actions.generic.spell_action import SpellAction
from src.namespaces.cards import Cards


class FireballAction(SpellAction):
    CARD = Cards.FIREBALL
    RADIUS = 2.5
