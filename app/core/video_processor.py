import cv2
import time
import logging
import threading
import subprocess
import os

from ultralytics import solutions

from app.core.classes import CLASSES, classwise_to_counts
from app.core.detector import detect_frame
from app.core.model_cache import get_model
from app.core.annotator import annotate_boxes
from app.core.shipment import (ShipmentTracker, EmptyGapFilter, LOAD_CLASS,
                               SHIPMENT_COUNT_CLASS, EMPTY_CLASS)

log = logging.getLogger("video_processor")

# Prefer TCP for RTSP and disable input buffering for low latency.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|timeout;5000000|stimeout;5000000|fflags;nobuffer",
)

# Frozen-feed backstop (stream robustness only — NOT counting-coupled): a feed
# delivering byte-identical frames at full rate gets a fixed short reconnect.
FROZEN_FRAME_LIMIT = 150
FROZEN_RECONNECT_DELAY = 2.0

# Counter hygiene for long-running streams. Ultralytics' ObjectCounter keeps two
# structures that grow with uptime and are only ever reset by a fresh process:
#   - counted_ids: a list scanned once per visible object per frame (O(n))
#   - track_history: a defaultdict(list) with one key per track id EVER seen
#     (each value holds cpu tensors; on a busy line this is the dominant RSS
#     leak — measured ~11 GB after 29 h on a single 640x480 stream).
# We swap counted_ids for an O(1) set and periodically evict finished tracks
# from track_history. Both are count-neutral (see _AppendableSet / _prune docs).
PRUNE_INTERVAL_FRAMES = 300      # ~10-15 s at 20-30 fps
HEALTH_LOG_INTERVAL_S = 30.0


class _AppendableSet(set):
    """A set that also answers to list-style .append(), so it can replace
    ObjectCounter.counted_ids (Ultralytics calls .append() on it) while giving
    O(1) membership instead of the stock list's O(n) scan. Count-neutral: track
    ids are unique and each is added at most once, so `x in set` is identical to
    `x in list` for the same contents — only the lookup cost changes."""

    append = set.add


def _rss_mib() -> float:
    """Best-effort resident-set size in MiB (Linux /proc); 0.0 elsewhere."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def reconnect_delay(attempt: int, base: float = 1.0, cap: float = 30.0) -> float:
    """Exponential backoff (seconds) for stream reconnection, capped."""
    return min(cap, base * (2 ** max(0, attempt)))


def frame_signature(frame) -> int:
    """Cheap coarse signature of a frame for frozen-stream detection."""
    return int(frame[::32, ::32].sum())


def extract_solution_boxes(counter) -> list[dict]:
    """Read the per-object tracked boxes the ObjectCounter extracted this frame
    into annotate_boxes-compatible dicts. Defensive across ultralytics versions:
    missing track_ids/confs degrade gracefully."""
    boxes = getattr(counter, "boxes", None)
    if boxes is None:
        return []
    clss = list(getattr(counter, "clss", []) or [])
    track_ids = list(getattr(counter, "track_ids", []) or [])
    confs = list(getattr(counter, "confs", []) or [])
    names = getattr(counter, "names", {}) or {}
    out = []
    for i, xyxy in enumerate(boxes):
        try:
            x1, y1, x2, y2 = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
        except Exception:
            continue
        ci = int(clss[i]) if i < len(clss) else -1
        cls_name = names.get(ci, "slaughtered_chicken") if isinstance(names, dict) else "slaughtered_chicken"
        out.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "class_name": cls_name,
            "conf": float(confs[i]) if i < len(confs) else 0.0,
            "obj_id": int(track_ids[i]) if i < len(track_ids) else None,
        })
    return out


class VideoProcessor:
    """Background video/stream processor. Counting is delegated to a per-source
    ultralytics ObjectCounter (vertical center line, left->right flow)."""

    def __init__(self, source: str, model_path: str,
                 save_raw_path: str = None, is_stream: bool = False):
        self.source = source
        self.model_path = model_path
        self.is_stream = is_stream
        self.save_raw_path = save_raw_path

        self.is_playing = False
        self.is_counting = False
        self.frame_num = 0
        self.total_frames = 0
        self.fps_source = 30.0
        self.fps_display = 0.0
        self.is_complete = False
        self.error = None

        self._counts = {c: 0 for c in CLASSES}
        # Load-to-load shipment segmentation. Reads the cumulative counts the
        # ObjectCounter already produces; never mutates them, so the lifetime
        # totals above keep climbing untouched.
        self._shipment = ShipmentTracker()
        # Drops the irrelevant empty-shackle run between a LOAD placard and the
        # delivery's first bird from the REPORTED empty count (all endpoints/UI
        # read self._counts). Derived read-only from the same cumulatives.
        self._empty_gap = EmptyGapFilter()
        self._counter = None          # solutions.ObjectCounter, built on first frame
        self._frame_dims = None       # (w, h) of the source
        self._line_points = None      # counting line [(x,0),(x,h)] for the overlay
        self._line_thickness = 4      # matches ObjectCounter's region thickness
        self._reset_requested = False

        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._thread = None
        self._stop_event = threading.Event()
        self._writer = None

        self._frames_since_prune = 0
        self._last_health_log = 0.0

    @property
    def counts(self) -> dict:
        return dict(self._counts)

    @property
    def latest_frame(self) -> bytes:
        with self._frame_lock:
            return self._latest_frame

    def start(self):
        if self.is_playing:
            return
        self._stop_event.clear()
        self.is_playing = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self.is_playing = False

    def start_counting(self):
        self.is_counting = True

    def stop_counting(self):
        self.is_counting = False

    def reset_counts(self):
        """Zero counts and rebuild the ObjectCounter (fresh tracker) on the next
        processed frame. Thread-safe — the rebuild happens in the capture loop."""
        self._counts = {c: 0 for c in CLASSES}
        # The counter is about to jump back to zero; resync the shipment baseline
        # so the derived per-shipment count can't go negative.
        self._shipment.reset()
        self._empty_gap.reset()
        self._reset_requested = True

    def get_shipments(self, since: int = 0) -> list[dict]:
        """Closed (completed) shipment records with index >= `since`."""
        return self._shipment.records_since(since)

    def _update_counts(self, classwise_count: dict) -> None:
        """Publish this frame's counts from the counter's cumulative classwise
        totals. The reported empty_shackles value is gap-adjusted: empties
        between a LOAD placard and the delivery's first bird are dropped
        (EmptyGapFilter). Shipment segmentation reads the raw cumulatives —
        a rising `load` count closes the open shipment and opens the next.
        Both are read-only on the counter; the dict swap is atomic for readers."""
        counts = classwise_to_counts(classwise_count)
        counts[EMPTY_CLASS] = self._empty_gap.update(
            load_cum=counts.get(LOAD_CLASS, 0),
            empty_cum=counts.get(EMPTY_CLASS, 0),
            chicken_cum=counts.get(SHIPMENT_COUNT_CLASS, 0),
        )
        self._counts = counts
        self._shipment.update(
            load_cum=counts.get(LOAD_CLASS, 0),
            slaughtered_cum=counts.get(SHIPMENT_COUNT_CLASS, 0),
        )

    def get_status(self) -> dict:
        return {
            "is_playing": self.is_playing,
            "is_counting": self.is_counting,
            "counts": self.counts,
            "frame_num": self.frame_num,
            "total_frames": self.total_frames,
            "fps": round(self.fps_display, 1),
            "is_complete": self.is_complete,
            "is_stream": self.is_stream,
            "error": self.error,
            "current_shipment": self._shipment.current(
                self._counts.get(SHIPMENT_COUNT_CLASS, 0)),
            "total_shipments": self._shipment.total_shipments(),
        }

    def _open_capture(self):
        cap = cv2.VideoCapture(self.source)
        if self.is_stream:
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        return cap

    def _build_counter(self, w: int, h: int):
        # Vertical line at the horizontal center — left->right flow (per test.py).
        line_points = [(w // 2, 0), (w // 2, h)]
        self._line_points = line_points
        counter = solutions.ObjectCounter(
            model=self.model_path, region=line_points, show=False, verbose=False,
        )
        # Replace the stock counted_ids list with an O(1) set (count-neutral —
        # see _AppendableSet). This also makes the track_history prune's
        # `tid in counted_ids` check O(1).
        try:
            counter.counted_ids = _AppendableSet(getattr(counter, "counted_ids", []) or [])
        except Exception:
            pass
        # Match ObjectCounter's own region thickness (line_width * 2) so the line
        # we draw is identical to test.py's plot_im line.
        lw = int(getattr(counter, "line_width", 2) or 2)
        self._line_thickness = max(2, lw * 2)
        return counter

    def _prune_counter_history(self):
        """Evict track_history entries for objects that are BOTH already counted
        AND no longer active. Provably count-neutral: a counted id can't be
        recounted (ObjectCounter short-circuits on `track_id in counted_ids`),
        and an inactive id has no live trajectory to lose — so dropping its
        history can neither double-count nor miss a count. Pure memory hygiene
        against Ultralytics' unbounded track_history (one key per track id ever
        seen). Called every PRUNE_INTERVAL_FRAMES, so it's O(keys) amortized to
        near-nothing once the working set stays bounded."""
        counter = self._counter
        history = getattr(counter, "track_history", None)
        counted = getattr(counter, "counted_ids", None)
        if not history or counted is None:
            return
        active = set(getattr(counter, "track_ids", []) or [])
        stale = [tid for tid in list(history.keys())
                 if tid not in active and tid in counted]
        for tid in stale:
            history.pop(tid, None)

    def _run(self):
        cap = self._open_capture()
        if not cap.isOpened():
            self.error = f"Could not open: {self.source}"
            self.is_playing = False
            return

        self.fps_source = cap.get(cv2.CAP_PROP_FPS) or 30
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.total_frames <= 0:
            self.is_stream = True

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        self._frame_dims = (w, h)
        self._counter = self._build_counter(w, h)
        preview_model = get_model(self.model_path)

        if self.save_raw_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self.save_raw_path, fourcc,
                                           self.fps_source, (w, h))

        fps_timer = time.time()
        fps_frame_count = 0
        frame_delay = 1.0 / self.fps_source if not self.is_stream else 0
        reconnect_attempt = 0
        last_sig = None
        frozen_count = 0

        while not self._stop_event.is_set():
            frame_start = time.time()
            ret, frame = cap.read()
            if not ret:
                if not self.is_stream:
                    self.is_complete = True
                    break
                cap.release()
                delay = reconnect_delay(reconnect_attempt)
                self.error = f"Stream lost; reconnecting in {delay:.0f}s"
                reconnect_attempt += 1
                if self._stop_event.wait(delay):
                    break
                cap = self._open_capture()
                continue

            if self.is_stream:
                sig = frame_signature(frame)
                if sig == last_sig:
                    frozen_count += 1
                    if frozen_count >= FROZEN_FRAME_LIMIT:
                        cap.release()
                        self.error = "Stream frozen; reconnecting"
                        frozen_count = 0
                        last_sig = None
                        if self._stop_event.wait(FROZEN_RECONNECT_DELAY):
                            break
                        cap = self._open_capture()
                        continue
                else:
                    frozen_count = 0
                    last_sig = sig

            if reconnect_attempt or self.error:
                reconnect_attempt = 0
                self.error = None

            self.frame_num += 1
            fps_frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 0.5:
                self.fps_display = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_timer = time.time()

            # Periodic health line: makes FPS/RSS decay and counter-state growth
            # visible in `docker logs` before it bites. track_history should stay
            # bounded thanks to the prune below; if it climbs, the prune isn't
            # keeping up and that's the signal to investigate.
            now = time.time()
            if now - self._last_health_log >= HEALTH_LOG_INTERVAL_S:
                self._last_health_log = now
                counter = self._counter
                log.info(
                    "[health] src=%s fps=%.1f rss=%.0fMiB counted_ids=%d "
                    "track_history=%d frame=%d",
                    self.source, self.fps_display, _rss_mib(),
                    len(getattr(counter, "counted_ids", ()) or ()),
                    len(getattr(counter, "track_history", {}) or {}),
                    self.frame_num,
                )

            if self._reset_requested:
                self._counter = self._build_counter(*self._frame_dims)
                self._reset_requested = False

            # ObjectCounter.process() draws its OWN boxes/labels/region line/counts
            # onto `frame` IN PLACE (ultralytics' Annotator wraps the input array).
            # Snapshot a pristine copy BEFORE processing and annotate that, so only
            # our bbox-only overlay shows — never ObjectCounter's default drawing.
            annotate_src = frame
            if self.is_counting:
                annotate_src = frame.copy()
                try:
                    results = self._counter.process(frame)
                except Exception as exc:
                    self.error = f"Counting failed: {exc}"
                    continue
                self._update_counts(
                    getattr(results, "classwise_count", {}) or {})
                boxes = extract_solution_boxes(self._counter)
                self._frames_since_prune += 1
                if self._frames_since_prune >= PRUNE_INTERVAL_FRAMES:
                    self._frames_since_prune = 0
                    self._prune_counter_history()
            else:
                try:
                    det = detect_frame(preview_model, frame)
                except Exception as exc:
                    self.error = f"Inference failed: {exc}"
                    continue
                boxes = [{"x1": d["x1"], "y1": d["y1"], "x2": d["x2"], "y2": d["y2"],
                          "class_name": d["class_name"], "conf": d["conf"]} for d in det]

            annotated = annotate_boxes(annotate_src, boxes,
                                       region_pts=self._line_points,
                                       region_thickness=self._line_thickness)

            if self._writer:
                self._writer.write(annotated)

            _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with self._frame_lock:
                self._latest_frame = jpeg.tobytes()

            if not self.is_stream and frame_delay > 0:
                wait = frame_delay - (time.time() - frame_start)
                if wait > 0:
                    time.sleep(wait)

        cap.release()
        if self._writer:
            self._writer.release()
        self.is_playing = False

    @staticmethod
    def reencode_h264(input_path: str, output_path: str):
        cmd = ["ffmpeg", "-y", "-i", input_path, "-c:v", "libx264",
               "-preset", "fast", "-crf", "23", "-movflags", "+faststart",
               "-an", output_path]
        subprocess.run(cmd, capture_output=True, check=True)
        if os.path.exists(input_path):
            os.remove(input_path)
