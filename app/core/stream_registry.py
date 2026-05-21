"""Thread-safe registry of live RTSP streams.

Each entry is a (stream_id, VideoProcessor) pair. The registry enforces a
soft cap (`max_streams`) and supports declarative startup via the
RTSP_STREAMS env var.

ID conventions:
    - "default" is reserved for the legacy /api/stream/* single-stream API.
    - User-supplied ids must be non-empty and unique.
"""
import json
import logging
import threading
from dataclasses import dataclass
from typing import Optional

import cv2

from app.core.model_cache import get_model
from app.core.runtime_config import runtime_config
from app.core.video_processor import VideoProcessor

log = logging.getLogger("stream_registry")

_OVERRIDE_KEYS = {
    "roi_position", "confidence", "nms_iou", "imgsz",
    "max_distance", "max_disappeared",
}


@dataclass
class StreamInfo:
    id: str
    url: str
    is_playing: bool
    is_counting: bool
    counts: dict
    total_count: int
    fps: float
    dropped_frames: int
    error: Optional[str]


class StreamRegistryError(Exception):
    """Base for registry errors."""


class StreamExistsError(StreamRegistryError):
    pass


class StreamNotFoundError(StreamRegistryError):
    pass


class StreamCapacityError(StreamRegistryError):
    pass


class StreamRegistry:
    def __init__(self) -> None:
        self._streams: dict[str, VideoProcessor] = {}
        self._urls: dict[str, str] = {}
        self._lock = threading.RLock()

    # ── Public API ─────────────────────────────────────────────────────────

    def register(self, stream_id: str, url: str,
                 overrides: dict | None = None,
                 start_counting: bool = False) -> StreamInfo:
        stream_id = (stream_id or "").strip()
        if not stream_id:
            raise StreamRegistryError("stream_id must be non-empty")
        if not url:
            raise StreamRegistryError("url must be non-empty")

        snap = runtime_config.snapshot()
        cfg = self._merge_overrides(snap, overrides or {})

        with self._lock:
            if stream_id in self._streams:
                raise StreamExistsError(f"Stream id '{stream_id}' already exists")
            cap = int(snap.get("max_streams", 10))
            if len(self._streams) >= cap:
                raise StreamCapacityError(
                    f"Stream cap reached ({cap}). Remove a stream or raise MAX_STREAMS."
                )

            roi_x = self._resolve_roi_x(url, cfg["roi_position"])
            processor = VideoProcessor(
                source=url,
                model=get_model(snap["model_path"]),
                roi_x=roi_x,
                confidence=cfg["confidence"],
                nms_iou=cfg["nms_iou"],
                imgsz=cfg["imgsz"],
                max_disappeared=cfg["max_disappeared"],
                max_distance=cfg["max_distance"],
                is_stream=True,
            )
            self._streams[stream_id] = processor
            self._urls[stream_id] = url

        processor.start()
        if start_counting:
            processor.start_counting()
        log.info("Registered stream '%s' -> %s (counting=%s)",
                 stream_id, url, start_counting)
        return self.info(stream_id)

    def replace_url(self, stream_id: str, url: str) -> StreamInfo:
        """Stop the existing processor and re-register at the same id with a new url.
        Used by the legacy /api/stream/start endpoint."""
        with self._lock:
            if stream_id in self._streams:
                self._streams[stream_id].stop()
                del self._streams[stream_id]
                del self._urls[stream_id]
        return self.register(stream_id, url)

    def unregister(self, stream_id: str) -> None:
        with self._lock:
            proc = self._streams.pop(stream_id, None)
            self._urls.pop(stream_id, None)
        if proc is None:
            raise StreamNotFoundError(f"No stream with id '{stream_id}'")
        proc.stop()
        log.info("Unregistered stream '%s'", stream_id)

    def get(self, stream_id: str) -> VideoProcessor:
        with self._lock:
            proc = self._streams.get(stream_id)
        if proc is None:
            raise StreamNotFoundError(f"No stream with id '{stream_id}'")
        return proc

    def exists(self, stream_id: str) -> bool:
        with self._lock:
            return stream_id in self._streams

    def list(self) -> list[StreamInfo]:
        with self._lock:
            ids = list(self._streams.keys())
        return [self.info(sid) for sid in ids]

    def info(self, stream_id: str) -> StreamInfo:
        proc = self.get(stream_id)
        status = proc.get_status()
        return StreamInfo(
            id=stream_id,
            url=self._urls.get(stream_id, ""),
            is_playing=status["is_playing"],
            is_counting=status["is_counting"],
            counts=status["counts"],
            total_count=status["total_count"],
            fps=status["fps"],
            dropped_frames=status.get("dropped_frames", 0),
            error=status.get("error"),
        )

    def stop_all(self) -> None:
        with self._lock:
            ids = list(self._streams.keys())
        for sid in ids:
            try:
                self.unregister(sid)
            except Exception as exc:
                log.warning("Error stopping stream '%s': %s", sid, exc)

    # ── Startup helpers ────────────────────────────────────────────────────

    def start_all_from_env(self) -> None:
        """Parse RTSP_STREAMS env (JSON list) and register each. Failures are
        logged but do not abort startup."""
        snap = runtime_config.snapshot()
        raw = (snap.get("rtsp_streams") or "").strip()
        if not raw:
            return
        try:
            entries = json.loads(raw)
        except json.JSONDecodeError as exc:
            log.error("RTSP_STREAMS is not valid JSON; skipping auto-start: %s", exc)
            return
        if not isinstance(entries, list):
            log.error("RTSP_STREAMS must be a JSON list of {id,url,...} entries")
            return

        for entry in entries:
            if not isinstance(entry, dict):
                log.warning("Skipping invalid RTSP_STREAMS entry: %r", entry)
                continue
            sid = entry.get("id")
            url = entry.get("url")
            if not sid or not url:
                log.warning("Skipping entry without id+url: %r", entry)
                continue
            overrides = {k: entry[k] for k in _OVERRIDE_KEYS if k in entry}
            try:
                self.register(sid, url, overrides=overrides,
                              start_counting=bool(entry.get("start_counting", True)))
            except Exception as exc:
                log.error("Failed to auto-register stream '%s' (%s): %s",
                          sid, url, exc)

    # ── Internals ──────────────────────────────────────────────────────────

    @staticmethod
    def _merge_overrides(snap: dict, overrides: dict) -> dict:
        cfg = {k: snap[k] for k in (
            "roi_position", "confidence", "nms_iou", "imgsz",
            "max_distance", "max_disappeared",
        )}
        for k, v in overrides.items():
            if k in cfg and v is not None:
                cfg[k] = v
        return cfg

    @staticmethod
    def _resolve_roi_x(url: str, roi_position: float) -> int:
        """Resolve fractional roi_position (0..1) to an absolute pixel x using
        the stream's reported frame width. Falls back to 640 if probe fails."""
        try:
            cap = cv2.VideoCapture(url)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
        except Exception:
            width = 0
        if width <= 0:
            width = 640
        return int(width * roi_position)


# Module singleton.
registry = StreamRegistry()
