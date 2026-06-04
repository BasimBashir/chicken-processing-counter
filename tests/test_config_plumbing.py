from app.config import Settings
from app.core.runtime_config import RuntimeConfig
from app.core.stream_registry import StreamRegistry


def test_conveyor_speed_default_is_belt_calibrated():
    assert Settings().conveyor_speed_px == 34.0


def test_zone_half_default():
    assert Settings().zone_half == 15


def test_runtime_config_exposes_new_keys():
    snap = RuntimeConfig().snapshot()
    assert "conveyor_speed_px" in snap
    assert "zone_half" in snap


def test_merge_overrides_passes_conveyor_speed():
    snap = RuntimeConfig().snapshot()
    cfg = StreamRegistry._merge_overrides(snap, {"conveyor_speed_px": 40.0})
    assert cfg["conveyor_speed_px"] == 40.0
    assert cfg["zone_half"] == snap["zone_half"]
