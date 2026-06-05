import pytest

from app.core.video_processor import VideoProcessor
from app.routers.streams import StreamUpdate


def _proc():
    """A VideoProcessor constructed but NOT started (no capture thread)."""
    p = VideoProcessor(source="dummy", model=None, roi_x=100, confidence=0.25,
                       conf_empty_shackles=0.45, conveyor_speed_px=34.0,
                       zone_half=15, is_stream=True)
    p.frame_width = 1280  # pretend capture has reported its width
    return p


def test_apply_overrides_zone_and_speed_live():
    p = _proc()
    out = p.apply_overrides(zone_half=25, conveyor_speed_px=40.0)
    assert p.counter.zone_half == 25
    assert p.counter.conveyor_speed_px == 40.0
    assert out == {"zone_half": 25, "conveyor_speed_px": 40.0}


def test_apply_overrides_confidence_resyncs_infer_conf():
    p = _proc()
    p.apply_overrides(confidence=0.10)
    assert p.confidence == 0.10
    assert p._class_conf["single_legged"] == 0.10
    assert p._class_conf["slaughtered_chicken"] == 0.10
    # inference runs at the lowest class threshold (empty_shackles still 0.45)
    assert p._infer_conf == 0.10


def test_apply_overrides_empty_shackles_conf():
    p = _proc()
    p.apply_overrides(conf_empty_shackles=0.20)
    assert p._class_conf["empty_shackles"] == 0.20
    assert p._infer_conf == 0.20  # now the new minimum


def test_apply_overrides_roi_uses_live_frame_width():
    p = _proc()
    p.apply_overrides(roi_position=0.5)
    assert p.counter.roi_x == 640  # 1280 * 0.5


def test_apply_overrides_roi_noop_without_width():
    p = _proc()
    p.frame_width = 0  # capture hasn't reported width yet
    before = p.counter.roi_x
    out = p.apply_overrides(roi_position=0.5)
    assert p.counter.roi_x == before  # unchanged, but acknowledged
    assert out["roi_position"] == 0.5


def test_apply_overrides_returns_only_changed():
    p = _proc()
    out = p.apply_overrides(nms_iou=0.5)
    assert out == {"nms_iou": 0.5}
    assert p.nms_iou == 0.5


def test_streamupdate_rejects_bad_values():
    with pytest.raises(ValueError):
        StreamUpdate(confidence=1.5)
    with pytest.raises(ValueError):
        StreamUpdate(zone_half=500)
    with pytest.raises(ValueError):
        StreamUpdate(imgsz=100)
    with pytest.raises(ValueError):
        StreamUpdate(conveyor_speed_px=0)


def test_streamupdate_allows_zone_zero_and_valid():
    m = StreamUpdate(zone_half=0, conveyor_speed_px=34.0, roi_position=0.65)
    assert m.zone_half == 0
    assert m.conveyor_speed_px == 34.0
    assert m.roi_position == 0.65


def test_apply_overrides_tracker_params_live():
    p = _proc()
    out = p.apply_overrides(max_distance=120, max_disappeared=5)
    for t in p.counter.trackers.values():
        assert t.max_distance == 120
        assert t.max_disappeared == 5
    assert out == {"max_distance": 120, "max_disappeared": 5}


def test_streamupdate_rejects_unknown_field():
    with pytest.raises(ValueError):
        StreamUpdate(zonehalf=20)  # typo: should be zone_half


def test_streamupdate_rejects_nonpositive_tracker_params():
    with pytest.raises(ValueError):
        StreamUpdate(max_distance=0)
    with pytest.raises(ValueError):
        StreamUpdate(max_disappeared=0)
