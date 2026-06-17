import pytest

from app.config import Settings
from app.core.runtime_config import RuntimeConfig
from app.core.stream_registry import StreamRegistry
from app.routers.streams import StreamCreate


def test_conveyor_speed_default_is_belt_calibrated():
    assert Settings().conveyor_speed_px == 34.0


def test_zone_half_default():
    assert Settings().zone_half == 18


def test_runtime_config_exposes_new_keys():
    snap = RuntimeConfig().snapshot()
    assert "conveyor_speed_px" in snap
    assert "zone_half" in snap


def test_merge_overrides_passes_conveyor_speed():
    snap = RuntimeConfig().snapshot()
    cfg = StreamRegistry._merge_overrides(snap, {"conveyor_speed_px": 40.0})
    assert cfg["conveyor_speed_px"] == 40.0
    assert cfg["zone_half"] == snap["zone_half"]


def test_streamcreate_accepts_conveyor_speed_and_zone_zero():
    """Per-stream REST path (POST /api/streams) must carry conveyor_speed_px
    and allow zone_half=0 (single-pixel tripwire)."""
    m = StreamCreate(id="x", url="rtsp://x", conveyor_speed_px=40.0, zone_half=0)
    assert m.conveyor_speed_px == 40.0
    assert m.zone_half == 0
    overrides = {k: v for k, v in
                 m.model_dump(exclude={"id", "url", "start_counting"}).items()
                 if v is not None}
    assert overrides["conveyor_speed_px"] == 40.0
    assert overrides["zone_half"] == 0


def test_streamcreate_rejects_nonpositive_conveyor_speed():
    with pytest.raises(ValueError):
        StreamCreate(id="x", url="rtsp://x", conveyor_speed_px=0)


def test_streamcreate_rejects_absurd_zone_half():
    with pytest.raises(ValueError):
        StreamCreate(id="x", url="rtsp://x", zone_half=500)
