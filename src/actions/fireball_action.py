"""Fireball strategic coordination module."""

from src.actions.generic.targeted_ability import TargetedAbility
from src.namespaces.cards import Cards


class FireballAction(TargetedAbility):
    """
    Heuristic action for placing the Fireball spell card.

    Attributes:
        CARD (Card): Always ``Cards.FIREBALL``.
        RADIUS (float): Effective blast radius of the Fireball spell (2.5 tiles).
    """

    CARD = Cards.FIREBALL
    RADIUS = 2.5
