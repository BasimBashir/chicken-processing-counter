from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from app.core.runtime_config import runtime_config
from app.core.model_cache import get_model

router = APIRouter(prefix="/api/config", tags=["config"])


class ConfigPatch(BaseModel):
    rtsp_url: Optional[str] = None
    model_path: Optional[str] = None
    roi_position: Optional[float] = None
    confidence: Optional[float] = None
    nms_iou: Optional[float] = None
    imgsz: Optional[int] = None
    max_distance: Optional[int] = None
    max_disappeared: Optional[int] = None

    @field_validator("roi_position")
    @classmethod
    def roi_in_range(cls, v):
        if v is not None and not (0.0 < v < 1.0):
            raise ValueError("roi_position must be between 0 and 1 exclusive")
        return v

    @field_validator("confidence")
    @classmethod
    def conf_in_range(cls, v):
        if v is not None and not (0.0 < v < 1.0):
            raise ValueError("confidence must be between 0 and 1 exclusive")
        return v

    @field_validator("nms_iou")
    @classmethod
    def iou_in_range(cls, v):
        if v is not None and not (0.0 < v < 1.0):
            raise ValueError("nms_iou must be between 0 and 1 exclusive")
        return v

    @field_validator("imgsz")
    @classmethod
    def imgsz_multiple_of_32(cls, v):
        if v is not None and v % 32 != 0:
            raise ValueError("imgsz must be a multiple of 32")
        return v

    @field_validator("max_distance", "max_disappeared")
    @classmethod
    def positive_int(cls, v):
        if v is not None and v < 1:
            raise ValueError("must be >= 1")
        return v


@router.get("")
def get_config():
    return runtime_config.snapshot()


@router.patch("")
def patch_config(patch: ConfigPatch):
    changes = {k: v for k, v in patch.model_dump().items() if v is not None}
    if not changes:
        return {"status": "no_change", "config": runtime_config.snapshot()}

    if "model_path" in changes:
        new_path = changes["model_path"]
        try:
            get_model(new_path)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Cannot load model '{new_path}': {exc}")

    updated = runtime_config.update(changes)
    return {"status": "ok", "config": updated}
