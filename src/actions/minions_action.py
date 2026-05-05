"""Minions strategic coordination module."""

from src.actions.generic.aerial_coordination import AerialCoordination
from src.namespaces.cards import Cards


class MinionsAction(AerialCoordination):
    """
    Heuristic action for placing the Minions card.

    Attributes:
        CARD (Card): Always ``Cards.MINIONS``.
    """

    CARD = Cards.MINIONS
