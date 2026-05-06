from PIL import Image

from src.live.detector import LiveDetector
from src.namespaces.cards import Cards


class _FakeCardDetector:
    def __init__(self, cards) -> None:
        self.cards = cards

    def run(self, _image):
        return (
            [
                Cards.ARCHERS,
                Cards.GIANT,
                Cards.FISHERMAN,
                Cards.ARROWS,
                Cards.KNIGHT,
            ],
            [0, 2, 3],
        )


class _FakeNumberDetector:
    def run(self, _image):
        return object()


class _FakeUnitDetector:
    def __init__(self, cards) -> None:
        self.cards = cards

    def run(self, _image):
        return [], []


class _FakeScreenDetector:
    def run(self, _image):
        return "in_game"


def test_live_detector_uses_slot_aware_card_detector(monkeypatch):
    monkeypatch.setattr("src.live.detector.CardDetector", _FakeCardDetector)
    monkeypatch.setattr("src.live.detector.NumberDetector", _FakeNumberDetector)
    monkeypatch.setattr("src.live.detector.UnitDetector", _FakeUnitDetector)
    monkeypatch.setattr("src.live.detector.ScreenDetector", _FakeScreenDetector)

    detector = LiveDetector(
        [
            Cards.GIANT,
            Cards.FISHERMAN,
            Cards.ARROWS,
            Cards.KNIGHT,
            Cards.ARCHERS,
            Cards.MUSKETEER,
            Cards.FIREBALL,
            Cards.THE_LOG,
        ]
    )

    state = detector.run(Image.new("RGB", (368, 652)))

    assert [card.name for card in state.cards] == [
        "archers",
        "giant",
        "fisherman",
        "arrows",
        "knight",
    ]
    assert state.ready == [0, 2, 3]
