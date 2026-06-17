"""Thread-safe registry of live RTSP streams. Each entry is a
(stream_id, VideoProcessor) pair with a soft cap (max_streams) and declarative
startup via RTSP_STREAMS."""
import json
import logging
import threading
from dataclasses import dataclass
from typing import Optional

from app.core.runtime_config import runtime_config
from app.core.video_processor import VideoProcessor

log = logging.getLogger("stream_registry")


@dataclass
class StreamInfo:
    id: str
    url: str
    is_playing: bool
    is_counting: bool
    counts: dict
    fps: float
    error: Optional[str]


class StreamRegistryError(Exception):
    pass


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

    def register(self, stream_id: str, url: str,
                 start_counting: bool = False) -> StreamInfo:
        stream_id = (stream_id or "").strip()
        if not stream_id:
            raise StreamRegistryError("stream_id must be non-empty")
        if not url:
            raise StreamRegistryError("url must be non-empty")

        snap = runtime_config.snapshot()
        with self._lock:
            if stream_id in self._streams:
                raise StreamExistsError(f"Stream id '{stream_id}' already exists")
            cap = int(snap.get("max_streams", 10))
            if len(self._streams) >= cap:
                raise StreamCapacityError(
                    f"Stream cap reached ({cap}). Remove a stream or raise MAX_STREAMS.")
            processor = VideoProcessor(source=url, model_path=snap["model_path"],
                                       is_stream=True)
            self._streams[stream_id] = processor
            self._urls[stream_id] = url

        processor.start()
        if start_counting:
            processor.start_counting()
        log.info("Registered stream '%s' -> %s (counting=%s)",
                 stream_id, url, start_counting)
        return self.info(stream_id)

    def replace_url(self, stream_id: str, url: str) -> StreamInfo:
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
            fps=status["fps"],
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

    def start_all_from_env(self) -> None:
        raw = (runtime_config.snapshot().get("rtsp_streams") or "").strip()
        if not raw:
            return
        try:
            entries = json.loads(raw)
        except json.JSONDecodeError as exc:
            log.error("RTSP_STREAMS is not valid JSON; skipping auto-start: %s", exc)
            return
        if not isinstance(entries, list):
            log.error("RTSP_STREAMS must be a JSON list of {id,url} entries")
            return
        for entry in entries:
            if not isinstance(entry, dict):
                log.warning("Skipping invalid RTSP_STREAMS entry: %r", entry)
                continue
            sid, url = entry.get("id"), entry.get("url")
            if not sid or not url:
                log.warning("Skipping entry without id+url: %r", entry)
                continue
            try:
                self.register(sid, url,
                              start_counting=bool(entry.get("start_counting", True)))
            except Exception as exc:
                log.error("Failed to auto-register stream '%s' (%s): %s", sid, url, exc)


registry = StreamRegistry()
