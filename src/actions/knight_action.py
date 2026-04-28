"""Knight play heuristics."""

from src.actions.generic.defense_action import DefenseAction
from src.namespaces.cards import Cards


class KnightAction(DefenseAction):
    CARD = Cards.KNIGHT
