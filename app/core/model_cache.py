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
            model = YOLO(path)
            # .pt is a PyTorch module and needs explicit device placement.
            # Exported formats (.engine, .onnx) have device baked in at export time
            # and Ultralytics raises TypeError if you call .to() on them.
            if path.endswith(".pt"):
                model.to('cuda:0')
            _cache[path] = model
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
