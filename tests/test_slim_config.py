from app.config import Settings
from app.core.runtime_config import RuntimeConfig

KEPT = {"rtsp_url", "model_path", "upload_dir", "output_dir",
        "rtsp_streams", "max_streams", "api_key"}
GONE = {"roi_position", "confidence", "conf_empty_shackles", "nms_iou", "imgsz",
        "conveyor_speed_px", "zone_half", "sway_k", "stop_motion_thresh",
        "stop_run_frames", "stop_resume_thresh", "zone_speed_factor",
        "proc_width", "proc_height", "batch_max", "max_distance"}


def test_settings_has_only_kept_fields():
    fields = set(Settings().model_dump().keys())
    assert KEPT <= fields
    assert fields & GONE == set()


def test_runtime_config_snapshot_is_slim():
    snap = RuntimeConfig().snapshot()
    assert KEPT <= set(snap.keys())
    assert set(snap.keys()) & GONE == set()
