import cv2
import time
import threading
import subprocess
import os

from ultralytics import solutions

from app.core.classes import CLASSES, classwise_to_counts
from app.core.detector import detect_frame
from app.core.model_cache import get_model
from app.core.annotator import annotate_boxes

# Prefer TCP for RTSP and disable input buffering for low latency.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|timeout;5000000|stimeout;5000000|fflags;nobuffer",
)

# Frozen-feed backstop (stream robustness only — NOT counting-coupled): a feed
# delivering byte-identical frames at full rate gets a fixed short reconnect.
FROZEN_FRAME_LIMIT = 150
FROZEN_RECONNECT_DELAY = 2.0


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
        self._counter = None          # solutions.ObjectCounter, built on first frame
        self._frame_dims = None       # (w, h) of the source
        self._reset_requested = False

        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._thread = None
        self._stop_event = threading.Event()
        self._writer = None

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
        self._reset_requested = True

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
        return solutions.ObjectCounter(
            model=self.model_path, region=line_points, show=False, verbose=False,
        )

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

            if self._reset_requested:
                self._counter = self._build_counter(*self._frame_dims)
                self._reset_requested = False

            if self.is_counting:
                try:
                    results = self._counter.process(frame)
                except Exception as exc:
                    self.error = f"Counting failed: {exc}"
                    continue
                self._counts = classwise_to_counts(
                    getattr(results, "classwise_count", {}) or {})
                boxes = extract_solution_boxes(self._counter)
            else:
                try:
                    det = detect_frame(preview_model, frame)
                except Exception as exc:
                    self.error = f"Inference failed: {exc}"
                    continue
                boxes = [{"x1": d["x1"], "y1": d["y1"], "x2": d["x2"], "y2": d["y2"],
                          "class_name": d["class_name"], "conf": d["conf"]} for d in det]

            annotated = annotate_boxes(frame, boxes)

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
