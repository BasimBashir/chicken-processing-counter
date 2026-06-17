import threading
from app.config import Settings


class RuntimeConfig:
    """Thread-safe live configuration. Boots from .env via pydantic-settings;
    `model_path` / `rtsp_url` can be patched at runtime via PATCH /api/config."""

    def __init__(self) -> None:
        boot = Settings()
        object.__setattr__(self, "_lock", threading.RLock())
        object.__setattr__(self, "_data", {
            "rtsp_url":     boot.rtsp_url,
            "model_path":   boot.model_path,
            "upload_dir":   boot.upload_dir,
            "output_dir":   boot.output_dir,
            "rtsp_streams": boot.rtsp_streams,
            "max_streams":  boot.max_streams,
            "api_key":      boot.api_key,
        })

    def __getattr__(self, name: str):
        data = object.__getattribute__(self, "_data")
        if name in data:
            lock = object.__getattribute__(self, "_lock")
            with lock:
                return data[name]
        raise AttributeError(f"RuntimeConfig has no field '{name}'")

    def snapshot(self) -> dict:
        lock = object.__getattribute__(self, "_lock")
        data = object.__getattribute__(self, "_data")
        with lock:
            return dict(data)

    def update(self, patch: dict) -> dict:
        lock = object.__getattribute__(self, "_lock")
        data = object.__getattribute__(self, "_data")
        with lock:
            for key, value in patch.items():
                if key in data:
                    data[key] = value
            return dict(data)


runtime_config = RuntimeConfig()
