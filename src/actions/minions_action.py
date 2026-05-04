"""Minions strategic coordination module."""

from src.actions.generic.aerial_coordination import AerialCoordination
from src.namespaces.cards import Cards


class MinionsAction(AerialCoordination):
    CARD = Cards.MINIONS
