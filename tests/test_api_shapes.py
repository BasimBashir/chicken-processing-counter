from fastapi.testclient import TestClient
from app.main import app


def test_config_get_is_slim():
    with TestClient(app) as client:
        cfg = client.get("/api/config").json()
        assert "model_path" in cfg
        for gone in ("roi_position", "imgsz", "conveyor_speed_px", "zone_half"):
            assert gone not in cfg


def test_config_patch_rejects_removed_field_silently():
    with TestClient(app) as client:
        r = client.patch("/api/config", json={"imgsz": 640})
        assert r.status_code == 200
        assert "imgsz" not in r.json()["config"]


def test_streams_list_has_no_total_count():
    with TestClient(app) as client:
        r = client.get("/api/streams")
        assert r.status_code == 200
        assert r.json() == {"streams": []}


def test_legacy_stream_status_shape():
    with TestClient(app) as client:
        s = client.get("/api/stream/status").json()
        assert set(s.keys()) == {"is_connected", "is_counting", "counts", "fps"}
        assert "total_count" not in s


def test_video_patch_endpoint_removed():
    with TestClient(app) as client:
        # PATCH /api/video/{id} no longer exists -> no route matches that method.
        assert client.patch("/api/video/abc", json={}).status_code in (404, 405)
