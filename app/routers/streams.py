"""Multi-stream RTSP API. Each stream is identified by a user-supplied id
and runs in its own VideoProcessor thread. Inference funnels through the
shared batched InferenceWorker so adding streams scales sub-linearly with GPU.
"""
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app.core.auth import verify_api_key
from app.core.counter import CLASSES
from app.core.stream_registry import (
    registry,
    StreamCapacityError,
    StreamExistsError,
    StreamNotFoundError,
)
from app.core.video_processor import VideoProcessor

router = APIRouter(
    prefix="/api/streams",
    tags=["streams"],
    dependencies=[Depends(verify_api_key)],
)


# ── Request / response schemas ─────────────────────────────────────────────

class StreamCreate(BaseModel):
    id: str = Field(..., description="Unique identifier for this stream")
    url: str = Field(..., description="RTSP/HTTP URL of the source feed")
    roi_position: Optional[float] = Field(None, description="ROI as fraction 0..1 of frame width")
    confidence: Optional[float] = Field(None, description="YOLO confidence threshold (0..1)")
    conf_empty_shackles: Optional[float] = Field(None, description="Confidence threshold for empty_shackles class only (overrides global confidence)")
    nms_iou: Optional[float] = Field(None, description="NMS IoU threshold (0..1)")
    imgsz: Optional[int] = Field(None, description="Inference image size (multiple of 32)")
    max_distance: Optional[int] = Field(None, description="Tracker max pixel distance")
    max_disappeared: Optional[int] = Field(None, description="Frames before lost track is dropped")
    zone_half: Optional[int] = Field(None, description="Half-width of counting zone in pixels (zone = roi_x ± zone_half)")
    appear_margin: Optional[int] = Field(None, description="Max px past zone_left where a brand-new track is still counted")
    start_counting: bool = Field(True, description="Begin counting immediately on register")

    @field_validator("roi_position", "confidence", "nms_iou", "conf_empty_shackles")
    @classmethod
    def _zero_one(cls, v):
        if v is not None and not (0.0 < v < 1.0):
            raise ValueError("must be between 0 and 1 exclusive")
        return v

    @field_validator("imgsz")
    @classmethod
    def _imgsz_mod32(cls, v):
        if v is not None and v % 32 != 0:
            raise ValueError("imgsz must be a multiple of 32")
        return v

    @field_validator("max_distance", "max_disappeared", "zone_half", "appear_margin")
    @classmethod
    def _positive(cls, v):
        if v is not None and v < 1:
            raise ValueError("must be >= 1")
        return v


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.get("")
def list_streams():
    """List all registered streams with each one's own counts and status."""
    return {"streams": [_info_to_dict(i) for i in registry.list()]}


@router.post("", status_code=201)
def register_stream(body: StreamCreate):
    """Register and start a new RTSP stream. Inference begins immediately;
    if start_counting is true (default), the counter is also armed."""
    overrides = {
        k: v for k, v in body.model_dump(exclude={"id", "url", "start_counting"}).items()
        if v is not None
    }
    try:
        info = registry.register(body.id, body.url, overrides,
                                 start_counting=body.start_counting)
    except StreamExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except StreamCapacityError as exc:
        raise HTTPException(status_code=429, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _info_to_dict(info)


@router.delete("/{stream_id}")
def unregister_stream(stream_id: str):
    """Stop a stream's capture thread and drop it from the registry."""
    try:
        registry.unregister(stream_id)
    except StreamNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"status": "removed", "id": stream_id}


@router.get("/{stream_id}/status")
def stream_status(stream_id: str):
    """Per-stream status: counts, fps, dropped_frames, error, is_counting."""
    return _info_to_dict(_resolve(stream_id, info=True))


@router.get("/{stream_id}/feed")
def stream_feed(stream_id: str):
    """MJPEG feed of the annotated frame from this stream only."""
    proc = _resolve(stream_id)
    return StreamingResponse(
        _mjpeg_generator(proc),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.post("/{stream_id}/counting/start")
def start_counting(stream_id: str):
    proc = _resolve(stream_id)
    proc.start_counting()
    return {"status": "counting", "id": stream_id}


@router.post("/{stream_id}/counting/stop")
def stop_counting(stream_id: str):
    proc = _resolve(stream_id)
    proc.stop_counting()
    return {"status": "not_counting", "id": stream_id}


@router.post("/{stream_id}/counting/reset")
def reset_counts(stream_id: str):
    """Zero out this stream's counts without disrupting capture.
    Useful for shift changes or after recalibrating ROI."""
    proc = _resolve(stream_id)
    proc.counter.reset()
    proc.counter_alt.reset()
    return {"status": "reset", "id": stream_id,
            "counts": {cls: 0 for cls in CLASSES}}


# ── Helpers ────────────────────────────────────────────────────────────────

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
        "total_count": info.total_count,
        "fps": info.fps,
        "dropped_frames": info.dropped_frames,
        "error": info.error,
    }


def _mjpeg_generator(proc: VideoProcessor):
    while proc.is_playing:
        frame_bytes = proc.latest_frame
        if frame_bytes:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )
        time.sleep(0.03)
