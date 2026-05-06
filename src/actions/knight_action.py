"""Knight strategic coordination module."""

from src.actions.generic.defensive_positioning import DefensivePositioning
from src.namespaces.cards import Cards


class KnightAction(DefensivePositioning):
    """
    Heuristic action for placing the Knight card.

    Attributes:
        CARD (Card): Always ``Cards.KNIGHT``.
    """

    CARD = Cards.KNIGHT
