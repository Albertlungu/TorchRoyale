import json

import numpy as np
from PIL import Image

from src.detection.card_detector import CardDetector
from src.detection.card_detector import ClassifierCardDetector
from src.detection.card_detector import ReferenceCardDetector
from src.detection.card_detector import normalize_classifier_label
from src.namespaces.cards import Cards


class _FakeInput:
    name = "input"


class _FakeOutput:
    name = "output"


class _FakeSession:
    calls = 0
    batch_sizes = []

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def get_inputs(self):
        return [_FakeInput()]

    def get_outputs(self):
        return [_FakeOutput()]

    def run(self, _outputs, feeds):
        self.__class__.calls += 1
        batch_size = feeds["input"].shape[0]
        self.__class__.batch_sizes.append(batch_size)
        return [np.full((batch_size, 1), 10.0, dtype=np.float32)]


def test_normalize_classifier_label_maps_suite_names_to_cards():
    assert normalize_classifier_label("Hand Musketeer") == "musketeer"
    assert normalize_classifier_label("Hand Mini P.E.K.K.A") == "minipekka"
    assert normalize_classifier_label("Hand P.E.K.K.A") == "pekka"
    assert normalize_classifier_label("Hand The Log") == "the_log"
    assert normalize_classifier_label("Hand X-Bow") == "x_bow"
    assert normalize_classifier_label("Hand Evo Cannon") == "cannon"
    assert normalize_classifier_label("Hand Hero Ice Golem") == "ice_golem"
    assert normalize_classifier_label("Hand Empty") == "blank"


def test_classifier_slot_boxes_match_current_screenshot_scale():
    image = Image.new("RGB", (368, 652))

    boxes = ClassifierCardDetector._slot_boxes(image)

    assert len(boxes) == 5
    assert boxes[0] == (21, 609, 47, 642)
    assert boxes[1] == (84, 543, 145, 616)
    assert boxes[4] == (291, 543, 352, 616)


def test_card_detector_prefers_classifier_when_assets_exist(monkeypatch):
    class FakeClassifier:
        @staticmethod
        def assets_available():
            return True

        def __init__(self, cards) -> None:
            self.cards = cards

    monkeypatch.setattr(
        "src.detection.card_detector.ClassifierCardDetector",
        FakeClassifier,
    )

    detector = CardDetector([Cards.MUSKETEER])

    assert isinstance(detector._detector, FakeClassifier)


def test_card_detector_falls_back_to_reference_when_assets_missing(monkeypatch):
    monkeypatch.setattr(ClassifierCardDetector, "assets_available", lambda: False)

    detector = CardDetector([Cards.MUSKETEER])

    assert isinstance(detector._detector, ReferenceCardDetector)


def test_classifier_gatekeeper_reuses_locked_slot_predictions(tmp_path, monkeypatch):
    classes_path = tmp_path / "classes.json"
    classes_path.write_text(json.dumps(["Hand Musketeer"]), encoding="utf-8")
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"")
    _FakeSession.calls = 0
    _FakeSession.batch_sizes = []

    monkeypatch.setattr(
        "src.detection.card_detector.ort.get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )
    monkeypatch.setattr(
        "src.detection.card_detector.ort.InferenceSession",
        _FakeSession,
    )

    detector = ClassifierCardDetector(
        [Cards.MUSKETEER],
        model_path=model_path,
        classes_path=classes_path,
    )
    image = Image.new("RGB", (368, 652), (10, 20, 30))

    cards, _ready = detector.run(image)
    detector.run(image)
    changed_cards, _ready = detector.run(Image.new("RGB", (368, 652), (240, 20, 30)))

    assert cards == [Cards.MUSKETEER] * 5
    assert changed_cards == [Cards.MUSKETEER] * 5
    assert _FakeSession.calls == 2
    assert _FakeSession.batch_sizes == [5, 5]
