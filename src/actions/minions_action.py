"""Minions play heuristics."""

from src.actions.generic.overhead_action import OverheadAction
from src.namespaces.cards import Cards


class MinionsAction(OverheadAction):
    CARD = Cards.MINIONS
