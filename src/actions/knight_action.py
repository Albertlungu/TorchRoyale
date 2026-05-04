"""Knight strategic coordination module."""

from src.actions.generic.defensive_positioning import DefensivePositioning
from src.namespaces.cards import Cards


class KnightAction(DefensivePositioning):
    CARD = Cards.KNIGHT
