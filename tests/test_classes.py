from app.core.classes import CLASSES, classwise_to_counts


def test_classes_constant():
    assert CLASSES == ["empty_shackles", "slaughtered_chicken", "load"]


def test_classwise_to_counts_maps_in_values():
    cw = {"slaughtered_chicken": {"IN": 5, "OUT": 1},
          "empty_shackles": {"IN": 2, "OUT": 0},
          "load": {"IN": 1, "OUT": 0}}
    assert classwise_to_counts(cw) == {
        "empty_shackles": 2, "slaughtered_chicken": 5, "load": 1,
    }


def test_classwise_to_counts_handles_empty_and_missing():
    assert classwise_to_counts({}) == {
        "empty_shackles": 0, "slaughtered_chicken": 0, "load": 0,
    }
    assert classwise_to_counts({"load": {}}) == {
        "empty_shackles": 0, "slaughtered_chicken": 0, "load": 0,
    }
