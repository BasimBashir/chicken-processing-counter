import cv2
import time
import threading
import subprocess
import os

from app.core.counter import ChickenCounter, CLASSES
from app.core.annotator import annotate_detections
from app.core.inference_worker import try_submit, QueueFull

# Prefer TCP for RTSP (UDP drops frames silently on congested networks) and
# disable input buffering for low latency. Set before any VideoCapture opens.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;5000000|fflags;nobuffer",
)

# Frozen-frame detection. A genuinely HUNG connection is already caught by the
# stimeout above (read() returns ret=False -> reconnect-with-backoff path). This
# content check is a backstop for the rarer case of a feed that keeps delivering
# byte-identical frames at full rate (stuck encoder/grabber). Caveat: a static
# but LIVE scene (idle conveyor under fixed lighting, no sensor noise) can also
# look identical, so this uses a deliberately high threshold and a FIXED short
# reconnect delay that does NOT ramp the main backoff — that way an idle period
# can never delay the first post-idle bird by the 30s read-failure cap.
FROZEN_FRAME_LIMIT = 150          # ~6s at 25 fps before a stuck feed reconnects
FROZEN_RECONNECT_DELAY = 2.0      # fixed; independent of read-failure backoff


def reconnect_delay(attempt: int, base: float = 1.0, cap: float = 30.0) -> float:
    """Exponential backoff (seconds) for stream reconnection, capped."""
    return min(cap, base * (2 ** max(0, attempt)))


def frame_signature(frame) -> int:
    """Cheap signature of a frame for frozen-stream detection. Coarse
    subsample keeps it O(1)-ish regardless of resolution."""
    return int(frame[::32, ::32].sum())


class VideoProcessor:
    """Background video/stream processor with independent play/count controls."""

    def __init__(self, source: str, model, roi_x: int, confidence: float = 0.25,
                 nms_iou: float = 0.45, imgsz: int = 640,
                 max_disappeared: int = 15, max_distance: int = 55,
                 conf_empty_shackles: float = 0.15,
                 conveyor_speed_px: float = 34.0, zone_half: int = 15,
                 save_raw_path: str = None, is_stream: bool = False):
        self.source = source
        # `model` kept for backward-compat construction; actual inference
        # goes through the shared InferenceWorker now.
        self.model = model
        self.roi_x = roi_x
        self.confidence = confidence
        # Per-class confidence thresholds. Inference runs at the lowest of
        # these so no class is suppressed inside YOLO; post-inference
        # filtering then applies each class's own threshold.
        self._class_conf = {
            "empty_shackles":      conf_empty_shackles,
            "single_legged":       confidence,
            "slaughtered_chicken": confidence,
        }
        self._infer_conf = min(self._class_conf.values())
        self.nms_iou = nms_iou
        self.imgsz = imgsz
        self.is_stream = is_stream
        self.save_raw_path = save_raw_path
        self.dropped_frames = 0

        self.counter = ChickenCounter(roi_x=roi_x, max_disappeared=max_disappeared,
                                      max_distance=max_distance,
                                      conveyor_speed_px=conveyor_speed_px,
                                      zone_half=zone_half)

        self.is_playing = False
        self.is_counting = False
        self.frame_num = 0
        self.total_frames = 0
        self.fps_source = 30.0
        self.fps_display = 0.0
        self.is_complete = False
        self.error = None

        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._thread = None
        self._stop_event = threading.Event()
        self._writer = None

    @property
    def total_count(self) -> int:
        return self.counter.total_count

    @property
    def counts(self) -> dict:
        return dict(self.counter.counts)

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

    def get_status(self) -> dict:
        return {
            "is_playing": self.is_playing,
            "is_counting": self.is_counting,
            "counts": self.counts,
            "total_count": self.total_count,
            "frame_num": self.frame_num,
            "total_frames": self.total_frames,
            "fps": round(self.fps_display, 1),
            "is_complete": self.is_complete,
            "is_stream": self.is_stream,
            "dropped_frames": self.dropped_frames,
            "error": self.error,
        }

    def _open_capture(self):
        cap = cv2.VideoCapture(self.source)
        if self.is_stream:
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep only newest frame
            except Exception:
                pass
        return cap

    def _run(self):
        cap = self._open_capture()
        if not cap.isOpened():
            self.error = f"Could not open: {self.source}"
            self.is_playing = False
            return

        self.fps_source = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.total_frames <= 0:
            self.is_stream = True

        # ROI x based on actual video width
        self.counter.roi_x = int(width * (self.roi_x / max(width, 1))) if self.roi_x > 1 else int(width * self.roi_x)

        if self.save_raw_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                self.save_raw_path, fourcc, self.fps_source, (width, height)
            )

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
                # Live stream dropped — reconnect with backoff, keep counts.
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
                        # Fixed short delay, and do NOT touch reconnect_attempt:
                        # a static-but-live scene must not ramp the backoff.
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

            # Healthy frame — reset backoff + transient error.
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

            try:
                future = try_submit(frame, self._infer_conf, self.nms_iou,
                                    self.imgsz, agnostic_nms=True)
            except QueueFull:
                self.dropped_frames += 1
                continue
            try:
                det_info = future.result(timeout=2.0)
            except Exception as exc:
                self.error = f"Inference failed: {exc}"
                self.dropped_frames += 1
                continue

            # Apply per-class confidence thresholds. YOLO ran at the global
            # minimum; drop anything below its own class's threshold here.
            det_info = [
                d for d in det_info
                if d["conf"] >= self._class_conf.get(d["class_name"], self.confidence)
            ]

            objects_by_class: dict = {}
            if self.is_counting:
                objects_by_class = self.counter.update(det_info)
            else:
                by_class = {cls: [] for cls in CLASSES}
                for d in det_info:
                    cls = d.get("class_name", "slaughtered_chicken")
                    if cls in by_class:
                        cx = (d["x1"] + d["x2"]) // 2
                        cy = (d["y1"] + d["y2"]) // 2
                        by_class[cls].append((cx, cy, d["x1"], d["y1"], d["x2"], d["y2"]))
                for cls in CLASSES:
                    objects_by_class[cls] = dict(self.counter.trackers[cls].update(by_class[cls]))

            flash_with_frame = [
                (fx, fy, cls, self.frame_num - i)
                for i, (fx, fy, cls) in enumerate(
                    reversed(self.counter.flash_events[-12:])
                )
            ]

            annotated = annotate_detections(
                frame=frame,
                detections=det_info,
                objects_by_class=objects_by_class,
                flash_events=flash_with_frame,
                roi_x=self.counter.roi_x if self.is_counting else None,
                frame_num=self.frame_num,
                zone_half=self.counter.zone_half if self.is_counting else 0,
            )

            if self._writer:
                self._writer.write(annotated)

            _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with self._frame_lock:
                self._latest_frame = jpeg.tobytes()

            if not self.is_stream and frame_delay > 0:
                proc_time = time.time() - frame_start
                wait = frame_delay - proc_time
                if wait > 0:
                    time.sleep(wait)

        cap.release()
        if self._writer:
            self._writer.release()
        self.is_playing = False

    @staticmethod
    def reencode_h264(input_path: str, output_path: str):
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-movflags", "+faststart",
            "-an",
            output_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        if os.path.exists(input_path):
            os.remove(input_path)
