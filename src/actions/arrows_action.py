"""Arrows strategic coordination module."""

from src.actions.generic.targeted_ability import TargetedAbility
from src.namespaces.cards import Cards


class ArrowsAction(TargetedAbility):
    """
    Heuristic action for placing the Arrows spell card.

    Attributes:
        CARD (Card): Always ``Cards.ARROWS``.
        RADIUS (float): Effective blast radius of the Arrows spell (4 tiles).
    """

    CARD = Cards.ARROWS
    RADIUS = 4
