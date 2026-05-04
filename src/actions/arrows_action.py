"""Arrows strategic coordination module."""

from src.actions.generic.targeted_ability import TargetedAbility
from src.namespaces.cards import Cards


class ArrowsAction(TargetedAbility):
    CARD = Cards.ARROWS
    RADIUS = 4
