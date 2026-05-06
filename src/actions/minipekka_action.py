"""Mini P.E.K.K.A strategic coordination module."""

from src.actions.generic.offensive_tactics import OffensiveTactics
from src.namespaces.cards import Cards


class MinipekkaAction(OffensiveTactics):
    """
    Heuristic action for placing the Mini P.E.K.K.A card.

    Attributes:
        CARD (Card): Always ``Cards.MINIPEKKA``.
    """

    CARD = Cards.MINIPEKKA
