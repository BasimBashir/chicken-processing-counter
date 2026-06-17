"""Legacy single-stream API kept for backward compatibility with the bundled
HTML dashboard and any external integrations built against the old contract.
All endpoints proxy to a single registry entry with id 'default'.

For new integrations, prefer /api/streams/* (see app/routers/streams.py).
"""
import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.auth import verify_api_key
from app.core.classes import CLASSES
from app.core.runtime_config import runtime_config
from app.core.stream_registry import (
    registry,
    StreamNotFoundError,
    StreamCapacityError,
)

router = APIRouter(
    prefix="/api/stream",
    tags=["stream (legacy)"],
    dependencies=[Depends(verify_api_key)],
)

_DEFAULT_ID = "default"


class StreamStart(BaseModel):
    url: str | None = None


@router.post("/start")
def start_stream(body: StreamStart = StreamStart()):
    """Start (or replace) the legacy 'default' stream. URL from body, or
    RTSP_URL env if body omits it."""
    url = (body.url or runtime_config.snapshot().get("rtsp_url", "")).strip()
    if not url:
        raise HTTPException(status_code=400, detail="No RTSP URL provided")
    try:
        if registry.exists(_DEFAULT_ID):
            info = registry.replace_url(_DEFAULT_ID, url)
        else:
            info = registry.register(_DEFAULT_ID, url)
    except StreamCapacityError as exc:
        raise HTTPException(status_code=429, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "connected", "url": info.url}


@router.post("/stop")
def stop_stream():
    try:
        registry.unregister(_DEFAULT_ID)
    except StreamNotFoundError:
        pass
    return {"status": "disconnected"}


@router.post("/counting/start")
def start_counting():
    proc = _default_or_400()
    proc.start_counting()
    return {"status": "counting"}


@router.post("/counting/stop")
def stop_counting():
    proc = _default_or_400()
    proc.stop_counting()
    return {"status": "not_counting"}


@router.get("/feed")
def stream_feed():
    proc = _default_or_400()
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/status")
def stream_status():
    if not registry.exists(_DEFAULT_ID):
        return {
            "is_connected": False,
            "is_counting": False,
            "counts": {cls: 0 for cls in CLASSES},
            "fps": 0,
        }
    info = registry.info(_DEFAULT_ID)
    return {
        "is_connected": info.is_playing,
        "is_counting": info.is_counting,
        "counts": info.counts,
        "fps": info.fps,
        "error": info.error,
    }


def _default_or_400():
    try:
        return registry.get(_DEFAULT_ID)
    except StreamNotFoundError:
        raise HTTPException(status_code=400, detail="Stream not active")


def _mjpeg_generator():
    while registry.exists(_DEFAULT_ID):
        proc = registry.get(_DEFAULT_ID)
        frame_bytes = proc.latest_frame
        if frame_bytes:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )
        time.sleep(0.03)
