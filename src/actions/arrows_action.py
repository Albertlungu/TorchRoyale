"""Arrows play heuristics."""

from src.actions.generic.spell_action import SpellAction
from src.namespaces.cards import Cards


class ArrowsAction(SpellAction):
    CARD = Cards.ARROWS
    RADIUS = 4
