from app.core.classes import CLASSES, classwise_to_counts


def test_classes_constant():
    assert CLASSES == ["empty_shackles", "single_legged", "slaughtered_chicken"]


def test_classwise_to_counts_maps_in_values():
    cw = {"slaughtered_chicken": {"IN": 5, "OUT": 1},
          "empty_shackles": {"IN": 2, "OUT": 0}}
    assert classwise_to_counts(cw) == {
        "empty_shackles": 2, "single_legged": 0, "slaughtered_chicken": 5,
    }


def test_classwise_to_counts_handles_empty_and_missing():
    assert classwise_to_counts({}) == {
        "empty_shackles": 0, "single_legged": 0, "slaughtered_chicken": 0,
    }
    assert classwise_to_counts({"single_legged": {}}) == {
        "empty_shackles": 0, "single_legged": 0, "slaughtered_chicken": 0,
    }
