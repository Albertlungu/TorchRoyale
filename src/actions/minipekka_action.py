"""Mini P.E.K.K.A play heuristics."""

from src.actions.generic.bridge_action import BridgeAction
from src.namespaces.cards import Cards


class MinipekkaAction(BridgeAction):
    CARD = Cards.MINIPEKKA
