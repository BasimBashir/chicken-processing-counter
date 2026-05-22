"""Single-thread batched YOLO inference worker.

All model.forward() calls funnel through one background thread that groups
queued frames by (imgsz, conf, iou, agnostic_nms) and runs each group as
one batched forward pass. Producers (per-stream VideoProcessor threads,
image router) submit frames and block on a Future for the result.

Tunables (env via app.config.Settings):
    BATCH_MAX            max frames per forward pass            default 16
    BATCH_WINDOW_MS      max wait to fill a batch (ms)          default 25
    INFERENCE_QUEUE_MAX  hard cap on backlog before backpressure default 100
"""
import logging
import threading
import time
from collections import defaultdict
from concurrent.futures import Future
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from typing import Optional

import numpy as np

from app.core.model_cache import get_model

log = logging.getLogger("inference_worker")


@dataclass
class _Job:
    frame: np.ndarray
    conf: float
    iou: float
    imgsz: int
    agnostic_nms: bool
    future: Future = field(default_factory=Future)


class InferenceWorker:
    """Batched single-thread inference dispatcher."""

    def __init__(self, model_path: str, batch_max: int = 16,
                 batch_window_ms: int = 25, queue_max: int = 100):
        self.model_path = model_path
        self.batch_max = max(1, int(batch_max))
        self.batch_window_s = max(0.001, float(batch_window_ms) / 1000.0)
        self._queue: Queue[_Job] = Queue(maxsize=max(1, int(queue_max)))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._model = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._model = get_model(self.model_path)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="InferenceWorker", daemon=True
        )
        self._thread.start()
        log.info("InferenceWorker started (batch_max=%d, window_ms=%.0f)",
                 self.batch_max, self.batch_window_s * 1000)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def submit(self, frame: np.ndarray, conf: float, iou: float,
               imgsz: int, agnostic_nms: bool = True) -> Future:
        """Queue a frame for inference. Returns a Future resolving to
        list[dict] of detections. Raises Full if the queue is saturated
        — caller should drop the frame rather than block."""
        job = _Job(frame=frame, conf=float(conf), iou=float(iou),
                   imgsz=int(imgsz), agnostic_nms=bool(agnostic_nms))
        self._queue.put_nowait(job)
        return job.future

    def submit_sync(self, frame: np.ndarray, conf: float, iou: float,
                    imgsz: int, agnostic_nms: bool = True,
                    timeout: float = 10.0) -> list[dict]:
        """Convenience wrapper for synchronous callers (image router)."""
        return self.submit(frame, conf, iou, imgsz, agnostic_nms).result(timeout=timeout)

    # ── Worker loop ────────────────────────────────────────────────────────

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                first = self._queue.get(timeout=0.1)
            except Empty:
                continue

            batch = [first]
            deadline = time.monotonic() + self.batch_window_s

            while len(batch) < self.batch_max:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._queue.get(timeout=remaining))
                except Empty:
                    break

            # Group by inference settings so each model() call has uniform args.
            groups: dict[tuple, list[_Job]] = defaultdict(list)
            for job in batch:
                key = (job.imgsz, job.conf, job.iou, job.agnostic_nms)
                groups[key].append(job)

            for (imgsz, conf, iou, agnostic_nms), jobs in groups.items():
                self._process_group(jobs, imgsz, conf, iou, agnostic_nms)

    def _process_group(self, jobs: list[_Job], imgsz: int, conf: float,
                       iou: float, agnostic_nms: bool) -> None:
        frames = [j.frame for j in jobs]
        try:
            results = self._model(
                frames, conf=conf, iou=iou, imgsz=imgsz,
                agnostic_nms=agnostic_nms, verbose=False,
                device=0, half=True  # Force RTX 3090 optimizations: CUDA + FP16
            )
        except Exception as exc:
            log.exception("Batch inference failed (n=%d): %s", len(jobs), exc)
            for job in jobs:
                if not job.future.done():
                    job.future.set_exception(exc)
            return

        names = getattr(self._model, "names", {})
        for job, result in zip(jobs, results):
            try:
                det_info = []
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cls_idx = int(box.cls[0])
                    det_info.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "conf": float(box.conf[0]),
                        "class_name": names.get(cls_idx, str(cls_idx)),
                    })
                job.future.set_result(det_info)
            except Exception as exc:
                if not job.future.done():
                    job.future.set_exception(exc)


# ── Module singleton ───────────────────────────────────────────────────────

_worker: Optional[InferenceWorker] = None
_worker_lock = threading.Lock()


def get_worker() -> InferenceWorker:
    """Return the singleton worker. Caller must have called start_worker()
    during app startup, or this raises RuntimeError."""
    if _worker is None:
        raise RuntimeError("InferenceWorker not started; call start_worker() in app lifespan")
    return _worker


def start_worker(model_path: str, batch_max: int, batch_window_ms: int,
                 queue_max: int) -> InferenceWorker:
    global _worker
    with _worker_lock:
        if _worker is None:
            _worker = InferenceWorker(model_path, batch_max, batch_window_ms, queue_max)
            _worker.start()
    return _worker


def stop_worker() -> None:
    global _worker
    with _worker_lock:
        if _worker is not None:
            _worker.stop()
            _worker = None


class QueueFull(Exception):
    """Raised by helpers when callers should drop the frame instead of blocking."""


def try_submit(frame: np.ndarray, conf: float, iou: float, imgsz: int,
               agnostic_nms: bool = True) -> Future:
    """Submit a frame, raising QueueFull if the queue is saturated.
    Producers should catch and drop the frame so live streams don't back up."""
    try:
        return get_worker().submit(frame, conf, iou, imgsz, agnostic_nms)
    except Full:
        raise QueueFull("inference queue saturated")
