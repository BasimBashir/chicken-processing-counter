from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.core.auth import verify_api_key
from app.core.exporter import exporter
from app.core.runtime_config import runtime_config

router = APIRouter(prefix="/api/export", tags=["export"],
                   dependencies=[Depends(verify_api_key)])


class ExportRequest(BaseModel):
    half: bool = True
    # Build the engine at the size ObjectCounter actually infers at (ultralytics
    # default 640 = the 640x480 sub-stream's native size). Raise for a hi-res
    # main stream. Must match TRT_IMGSZ used by docker-entrypoint.sh.
    imgsz: int = 640


@router.post("/tensorrt")
def start_export(body: ExportRequest = ExportRequest()):
    snap = runtime_config.snapshot()
    started = exporter.start(
        model_path=snap["model_path"],
        imgsz=body.imgsz,
        half=body.half,
    )
    if not started:
        raise HTTPException(status_code=409, detail="Export already running")
    return {"status": "started", "model_path": snap["model_path"]}


@router.get("/tensorrt")
def get_export_status():
    s = exporter.get_status()
    result = {
        "state": s.state,
        "source_model": s.source_model,
        "output_path": s.output_path,
        "error": s.error,
    }
    if s.started_at and s.finished_at:
        result["elapsed_seconds"] = round(s.finished_at - s.started_at, 1)
    return result
