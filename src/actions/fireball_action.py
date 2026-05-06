"""Fireball strategic coordination module."""

from src.actions.generic.targeted_ability import TargetedAbility
from src.namespaces.cards import Cards


class FireballAction(TargetedAbility):
    CARD = Cards.FIREBALL
    RADIUS = 2.5
