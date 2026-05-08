from fastapi import APIRouter
from app.core.runtime_config import runtime_config

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    snap = runtime_config.snapshot()
    info: dict = {"status": "ok", "model_path": snap["model_path"]}
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["cuda_available"] = False
    return info
