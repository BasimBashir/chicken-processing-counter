import threading
from ultralytics import YOLO

_cache: dict[str, YOLO] = {}
_lock = threading.Lock()


def get_model(path: str) -> YOLO:
    """Return cached YOLO model for path, loading it on first call."""
    if path in _cache:
        return _cache[path]
    with _lock:
        if path not in _cache:
            _cache[path] = YOLO(path)
        return _cache[path]


def preload_model(path: str) -> None:
    import os
    if not os.path.exists(path):
        import logging
        logging.getLogger("model_cache").warning(
            "Model file '%s' not found — skipping preload. "
            "Place best.pt in the project root before making inference requests.", path
        )
        return
    get_model(path)
