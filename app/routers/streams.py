"""Multi-stream RTSP API. Each stream runs in its own VideoProcessor thread with
its own ObjectCounter."""
import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.auth import verify_api_key
from app.core.classes import CLASSES
from app.core.stream_registry import (
    registry, StreamCapacityError, StreamExistsError, StreamNotFoundError,
)
from app.core.video_processor import VideoProcessor

router = APIRouter(prefix="/api/streams", tags=["streams"],
                   dependencies=[Depends(verify_api_key)])


class StreamCreate(BaseModel):
    id: str = Field(..., description="Unique identifier for this stream")
    url: str = Field(..., description="RTSP/HTTP URL of the source feed")
    start_counting: bool = Field(True, description="Begin counting immediately on register")


@router.get("")
def list_streams():
    return {"streams": [_info_to_dict(i) for i in registry.list()]}


@router.post("", status_code=201)
def register_stream(body: StreamCreate):
    try:
        info = registry.register(body.id, body.url, start_counting=body.start_counting)
    except StreamExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except StreamCapacityError as exc:
        raise HTTPException(status_code=429, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _info_to_dict(info)


@router.delete("/{stream_id}")
def unregister_stream(stream_id: str):
    try:
        registry.unregister(stream_id)
    except StreamNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"status": "removed", "id": stream_id}


@router.get("/{stream_id}/status")
def stream_status(stream_id: str):
    return _info_to_dict(_resolve(stream_id, info=True))


@router.get("/{stream_id}/feed")
def stream_feed(stream_id: str):
    proc = _resolve(stream_id)
    return StreamingResponse(_mjpeg_generator(proc),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@router.post("/{stream_id}/counting/start")
def start_counting(stream_id: str):
    _resolve(stream_id).start_counting()
    return {"status": "counting", "id": stream_id}


@router.post("/{stream_id}/counting/stop")
def stop_counting(stream_id: str):
    _resolve(stream_id).stop_counting()
    return {"status": "not_counting", "id": stream_id}


@router.post("/{stream_id}/counting/reset")
def reset_counts(stream_id: str):
    _resolve(stream_id).reset_counts()
    return {"status": "reset", "id": stream_id, "counts": {cls: 0 for cls in CLASSES}}


def _resolve(stream_id: str, info: bool = False):
    try:
        return registry.info(stream_id) if info else registry.get(stream_id)
    except StreamNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


def _info_to_dict(info) -> dict:
    return {
        "id": info.id,
        "url": info.url,
        "is_playing": info.is_playing,
        "is_counting": info.is_counting,
        "counts": info.counts,
        "fps": info.fps,
        "error": info.error,
    }


def _mjpeg_generator(proc: VideoProcessor):
    while proc.is_playing:
        frame_bytes = proc.latest_frame
        if frame_bytes:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        time.sleep(0.03)
