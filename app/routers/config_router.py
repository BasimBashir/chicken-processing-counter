from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.core.auth import verify_api_key
from app.core.runtime_config import runtime_config
from app.core.model_cache import get_model

router = APIRouter(prefix="/api/config", tags=["config"],
                   dependencies=[Depends(verify_api_key)])


class ConfigPatch(BaseModel):
    rtsp_url: Optional[str] = None
    model_path: Optional[str] = None


@router.get("")
def get_config():
    return runtime_config.snapshot()


@router.patch("")
def patch_config(patch: ConfigPatch):
    changes = {k: v for k, v in patch.model_dump().items() if v is not None}
    if not changes:
        return {"status": "no_change", "config": runtime_config.snapshot()}

    if "model_path" in changes:
        try:
            get_model(changes["model_path"])
        except Exception as exc:
            raise HTTPException(status_code=422,
                                detail=f"Cannot load model '{changes['model_path']}': {exc}")

    return {"status": "ok", "config": runtime_config.update(changes)}
