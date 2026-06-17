# ObjectCounter Counting Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the custom straddle/tripwire counting stack with `ultralytics.solutions.ObjectCounter` (exactly per `test.py`), keeping the video/stream/image API surface, reporting per-class counts (no total), and drawing bbox-only annotations in the existing style.

**Architecture:** Each capture thread (`VideoProcessor`) owns one `ObjectCounter` built on a vertical center line. When counting is armed, `counter.process(frame)` does detection + tracking + line-crossing; per-class `IN` values become `counts`. Annotated MJPEG frames are redrawn bbox-only from the solution's tracked boxes using the existing `draw_bbox`. The batched inference worker and all tuning config are deleted.

**Tech Stack:** Python 3.11, FastAPI, OpenCV, Ultralytics YOLO/solutions, Docker (GPU via `docker-compose.gpu.yml`). The model is the existing 3-class `best.pt` (`empty_shackles`, `single_legged`, `slaughtered_chicken`).

**Reference spec:** `docs/superpowers/specs/2026-06-17-objectcounter-counting-refactor-design.md`

---

## Conventions: test/run commands (local venv is broken — everything runs in Docker)

Run all commands from the **repo root** in **git-bash** (the Bash tool). The image is built once; Python source is bind-mounted so edits need no rebuild.

**`BUILD`** (run once, and again only if `requirements.txt` changes):
```bash
docker compose build chicken-counter
```

**`TEST_UNIT`** — CPU, for pure-logic tests (substitute the pytest args):
```bash
docker compose run --rm --no-deps \
  -v "$PWD/app:/app/app" -v "$PWD/tests:/app/tests" \
  --entrypoint sh chicken-counter -c "pip install -q pytest && pytest -q $ARGS"
```

**`TEST_INT`** — GPU, for tests that load `best.pt` / boot the app (TestClient, parity):
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm \
  -v "$PWD/app:/app/app" -v "$PWD/tests:/app/tests" \
  --entrypoint sh chicken-counter -c "pip install -q pytest && pytest -q $ARGS"
```

Throughout, "Run `TEST_UNIT tests/foo.py`" means: use the `TEST_UNIT` command with `$ARGS = tests/foo.py`.

---

## Task 0: Build the image (prerequisite)

**Files:** none.

- [ ] **Step 1: Build**

Run: `docker compose build chicken-counter`
Expected: build completes; image `basim123/chicken-counter:latest` exists. (First build is slow — torch/ultralytics/tensorrt.)

- [ ] **Step 2: Sanity-check ultralytics + model load in the image (GPU)**

Run:
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm --entrypoint sh chicken-counter -c \
  "python -c \"from ultralytics import solutions, YOLO; print('classes:', YOLO('best.pt').names)\""
```
Expected: prints `classes: {0: 'empty_shackles', 1: 'single_legged', 2: 'slaughtered_chicken'}` (order/index may differ; the three names must appear). **If the names differ, STOP** — the whole refactor assumes these three; reconcile with the maintainer before continuing.

---

## Task 1: `CLASSES` + `classwise_to_counts` helper (pure, TDD)

**Files:**
- Create: `app/core/classes.py`
- Test: `tests/test_classes.py`

- [ ] **Step 1: Write the failing test**

`tests/test_classes.py`:
```python
from app.core.classes import CLASSES, classwise_to_counts


def test_classes_constant():
    assert CLASSES == ["empty_shackles", "single_legged", "slaughtered_chicken"]


def test_classwise_to_counts_maps_in_values():
    cw = {"slaughtered_chicken": {"IN": 5, "OUT": 1},
          "empty_shackles": {"IN": 2, "OUT": 0}}
    assert classwise_to_counts(cw) == {
        "empty_shackles": 2, "single_legged": 0, "slaughtered_chicken": 5,
    }


def test_classwise_to_counts_handles_empty_and_missing():
    assert classwise_to_counts({}) == {
        "empty_shackles": 0, "single_legged": 0, "slaughtered_chicken": 0,
    }
    assert classwise_to_counts({"single_legged": {}}) == {
        "empty_shackles": 0, "single_legged": 0, "slaughtered_chicken": 0,
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `TEST_UNIT tests/test_classes.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.core.classes'`.

- [ ] **Step 3: Write the implementation**

`app/core/classes.py`:
```python
"""Shared detection class names for the 3-class chicken model (best.pt) and a
helper to flatten ObjectCounter's classwise output to per-class IN counts.

Kept dependency-free (no ultralytics/torch) so it imports cheaply anywhere.
"""

CLASSES = ["empty_shackles", "single_legged", "slaughtered_chicken"]


def classwise_to_counts(classwise_count: dict, classes=CLASSES) -> dict:
    """Map ObjectCounter `results.classwise_count` to a flat per-class count.

    `classwise_count` looks like ``{class_name: {"IN": n, "OUT": n}, ...}`` and
    only includes classes seen so far. The belt is one-way left->right, so each
    class's reported count is its ``IN`` value; unseen classes report 0.
    """
    out = {}
    for c in classes:
        entry = classwise_count.get(c) or {}
        out[c] = int(entry.get("IN", 0))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `TEST_UNIT tests/test_classes.py`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add app/core/classes.py tests/test_classes.py
git commit -m "feat: add CLASSES + classwise_to_counts helper"
```

> **Validation note (resolved in Task 6):** the `"IN"`/`"OUT"` key casing is confirmed against the installed ultralytics in Task 6 Step 1. If the real keys differ (e.g. lowercase), update `classwise_to_counts` and its test then.

---

## Task 2: Trim annotator to bbox-only (`annotate_boxes`) (pure, TDD)

**Files:**
- Modify: `app/core/annotator.py`
- Test: `tests/test_annotator.py`

- [ ] **Step 1: Write the failing test**

`tests/test_annotator.py`:
```python
import numpy as np
from app.core.annotator import annotate_boxes, annotate_image_detections


def test_annotate_boxes_draws_without_mutating_input():
    frame = np.zeros((200, 320, 3), dtype=np.uint8)
    boxes = [{"x1": 10, "y1": 10, "x2": 80, "y2": 120,
              "class_name": "slaughtered_chicken", "conf": 0.9, "obj_id": 3}]
    out = annotate_boxes(frame, boxes)
    assert out.shape == frame.shape
    assert out.sum() > 0          # something was drawn
    assert frame.sum() == 0       # original untouched


def test_annotate_boxes_handles_missing_optional_fields():
    frame = np.zeros((200, 320, 3), dtype=np.uint8)
    boxes = [{"x1": 5, "y1": 5, "x2": 50, "y2": 60, "class_name": "empty_shackles"}]
    out = annotate_boxes(frame, boxes)   # no conf, no obj_id
    assert out.shape == frame.shape


def test_annotate_image_detections_still_returns_counts():
    frame = np.zeros((200, 320, 3), dtype=np.uint8)
    det = [{"x1": 5, "y1": 5, "x2": 50, "y2": 60,
            "conf": 0.8, "class_name": "single_legged"}]
    out, counts = annotate_image_detections(frame, det)
    assert out.shape == frame.shape
    assert counts == {"single_legged": 1}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `TEST_UNIT tests/test_annotator.py`
Expected: FAIL — `ImportError: cannot import name 'annotate_boxes'`.

- [ ] **Step 3: Rewrite `app/core/annotator.py`**

Replace the whole file with (keeps `draw_bbox`, `draw_rounded_rect`, `annotate_image_detections`; adds `annotate_boxes`; removes `draw_roi_line`, `draw_crossing_flash`, `annotate_detections`):
```python
import cv2

COLORS = {
    "panel_bg":  (20, 20, 20),
    "accent":    (0, 200, 255),
    "white":     (255, 255, 255),
    "dim":       (160, 160, 160),
}

# Per-class colors in BGR (unchanged from the previous annotator).
CLASS_COLORS = {
    "empty_shackles":      (0, 165, 255),   # orange
    "single_legged":       (255, 200, 0),   # gold/cyan
    "slaughtered_chicken": (0, 230, 118),   # green
}

CLASS_LABELS = {
    "empty_shackles":      "E.Shackle",
    "single_legged":       "Single",
    "slaughtered_chicken": "Slaughtered",
}


def draw_rounded_rect(img, pt1, pt2, color, radius=12, thickness=-1, alpha=0.85):
    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2
    r = radius
    cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, thickness)
    cv2.ellipse(overlay, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(overlay, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(overlay, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
    cv2.ellipse(overlay, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_bbox(img, x1, y1, x2, y2, counted, conf, class_name, obj_id=None):
    """Existing corner-accented bbox + small label. Colors/size unchanged."""
    color = CLASS_COLORS.get(class_name, COLORS["accent"])
    if counted:
        b, g, r = color
        color = (min(b + 40, 255), min(g + 40, 255), min(r + 40, 255))
    corner_len = 8
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, 2, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, 2, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, 2, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, 2, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, 2, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, 2, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, 2, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, 2, cv2.LINE_AA)
    short_label = CLASS_LABELS.get(class_name, class_name)
    id_tag = f"#{obj_id} " if obj_id is not None else ""
    label = f"{id_tag}{short_label} {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1, cv2.LINE_AA)


def annotate_boxes(frame, boxes):
    """Bbox-only annotation. `boxes` is a list of dicts with keys
    x1,y1,x2,y2,class_name and optional conf,obj_id. No ROI line, no HUD."""
    annotated = frame.copy()
    for b in boxes:
        draw_bbox(annotated, int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"]),
                  counted=False, conf=float(b.get("conf", 0.0)),
                  class_name=b.get("class_name", "slaughtered_chicken"),
                  obj_id=b.get("obj_id"))
    return annotated


def annotate_image_detections(frame, det_info):
    """Annotate a single still. Returns (annotated, per-class counts)."""
    annotated = frame.copy()
    class_counts: dict[str, int] = {}
    for info in det_info:
        cls = info.get("class_name", "slaughtered_chicken")
        class_counts[cls] = class_counts.get(cls, 0) + 1
        color = CLASS_COLORS.get(cls, COLORS["accent"])
        draw_bbox(annotated, info["x1"], info["y1"], info["x2"], info["y2"],
                  counted=False, conf=info["conf"], class_name=cls)
        cx = (info["x1"] + info["x2"]) // 2
        cy = (info["y1"] + info["y2"]) // 2
        cv2.circle(annotated, (cx, cy), 4, color, -1, cv2.LINE_AA)

    total = len(det_info)
    draw_rounded_rect(annotated, (8, 8), (280, 60), COLORS["panel_bg"], radius=8, alpha=0.85)
    cv2.putText(annotated, f"{total}", (18, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, COLORS["white"], 3, cv2.LINE_AA)
    tw = cv2.getTextSize(f"{total}", cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)[0][0]
    cv2.putText(annotated, "objects detected", (18 + tw + 8, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["dim"], 1, cv2.LINE_AA)
    return annotated, class_counts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `TEST_UNIT tests/test_annotator.py`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add app/core/annotator.py tests/test_annotator.py
git commit -m "refactor: trim annotator to bbox-only (annotate_boxes)"
```

---

## Task 3: Slim the config (`config.py`, `runtime_config.py`, `config_router.py`) (TDD)

**Files:**
- Modify: `app/config.py`, `app/core/runtime_config.py`, `app/routers/config_router.py`
- Test: `tests/test_slim_config.py`

- [ ] **Step 1: Write the failing test**

`tests/test_slim_config.py`:
```python
from app.config import Settings
from app.core.runtime_config import RuntimeConfig

KEPT = {"rtsp_url", "model_path", "upload_dir", "output_dir",
        "rtsp_streams", "max_streams", "api_key"}
GONE = {"roi_position", "confidence", "conf_empty_shackles", "nms_iou", "imgsz",
        "conveyor_speed_px", "zone_half", "sway_k", "stop_motion_thresh",
        "stop_run_frames", "stop_resume_thresh", "zone_speed_factor",
        "proc_width", "proc_height", "batch_max", "max_distance"}


def test_settings_has_only_kept_fields():
    fields = set(Settings().model_dump().keys())
    assert KEPT <= fields
    assert fields & GONE == set()


def test_runtime_config_snapshot_is_slim():
    snap = RuntimeConfig().snapshot()
    assert KEPT <= set(snap.keys())
    assert set(snap.keys()) & GONE == set()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `TEST_UNIT tests/test_slim_config.py`
Expected: FAIL — `GONE` fields still present.

- [ ] **Step 3a: Replace `app/config.py`**

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Runtime configuration. Counting is fixed to test.py's ObjectCounter
    behaviour (vertical center line, model defaults), so no detection/counting
    tuning knobs are exposed here."""

    # Source / model
    rtsp_url: str = ""
    model_path: str = "best.pt"

    # Filesystem
    upload_dir: str = "app/uploads"
    output_dir: str = "app/outputs"

    # Multi-stream: JSON list, e.g. RTSP_STREAMS='[{"id":"line-1","url":"rtsp://..."}]'
    rtsp_streams: str = ""
    max_streams: int = 10

    # Auth — if empty, /api/* is open (dev mode); else require X-API-Key header.
    api_key: str = ""

    class Config:
        env_file = ".env"


settings = Settings()
```

- [ ] **Step 3b: Replace `app/core/runtime_config.py`**

```python
import threading
from app.config import Settings


class RuntimeConfig:
    """Thread-safe live configuration. Boots from .env via pydantic-settings;
    `model_path` / `rtsp_url` can be patched at runtime via PATCH /api/config."""

    def __init__(self) -> None:
        boot = Settings()
        object.__setattr__(self, "_lock", threading.RLock())
        object.__setattr__(self, "_data", {
            "rtsp_url":     boot.rtsp_url,
            "model_path":   boot.model_path,
            "upload_dir":   boot.upload_dir,
            "output_dir":   boot.output_dir,
            "rtsp_streams": boot.rtsp_streams,
            "max_streams":  boot.max_streams,
            "api_key":      boot.api_key,
        })

    def __getattr__(self, name: str):
        data = object.__getattribute__(self, "_data")
        if name in data:
            lock = object.__getattribute__(self, "_lock")
            with lock:
                return data[name]
        raise AttributeError(f"RuntimeConfig has no field '{name}'")

    def snapshot(self) -> dict:
        lock = object.__getattribute__(self, "_lock")
        data = object.__getattribute__(self, "_data")
        with lock:
            return dict(data)

    def update(self, patch: dict) -> dict:
        lock = object.__getattribute__(self, "_lock")
        data = object.__getattribute__(self, "_data")
        with lock:
            for key, value in patch.items():
                if key in data:
                    data[key] = value
            return dict(data)


runtime_config = RuntimeConfig()
```

- [ ] **Step 3c: Replace `app/routers/config_router.py`**

```python
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.core.auth import verify_api_key
from app.core.runtime_config import runtime_config
from app.core.model_cache import get_model

router = APIRouter(prefix="/api/config", tags=["config"],
                   dependencies=[Depends(verify_api_key)])


class ConfigPatch(BaseModel):
    rtsp_url: Optional[str] = None
    model_path: Optional[str] = None


@router.get("")
def get_config():
    return runtime_config.snapshot()


@router.patch("")
def patch_config(patch: ConfigPatch):
    changes = {k: v for k, v in patch.model_dump().items() if v is not None}
    if not changes:
        return {"status": "no_change", "config": runtime_config.snapshot()}

    if "model_path" in changes:
        try:
            get_model(changes["model_path"])
        except Exception as exc:
            raise HTTPException(status_code=422,
                                detail=f"Cannot load model '{changes['model_path']}': {exc}")

    return {"status": "ok", "config": runtime_config.update(changes)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `TEST_UNIT tests/test_slim_config.py`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add app/config.py app/core/runtime_config.py app/routers/config_router.py tests/test_slim_config.py
git commit -m "refactor: slim config to infra-only (drop all counting knobs)"
```

---

## Task 4: Rewrite `VideoProcessor` around `ObjectCounter`

**Files:**
- Modify (full rewrite): `app/core/video_processor.py`
- Keep test: `tests/test_rtsp_helpers.py` (must still pass — `reconnect_delay` + `frame_signature` retained)

- [ ] **Step 1: Replace `app/core/video_processor.py`**

```python
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
```

- [ ] **Step 2: Run the retained helper tests**

Run: `TEST_UNIT tests/test_rtsp_helpers.py`
Expected: PASS (3 passed) — `reconnect_delay` + `frame_signature` unchanged. (This imports `video_processor`, which imports ultralytics; runs CPU in the image.)

- [ ] **Step 3: Commit**

```bash
git add app/core/video_processor.py
git commit -m "refactor: VideoProcessor wraps ObjectCounter (per-class counts, bbox-only)"
```

> The `extract_solution_boxes` attribute names + `classwise_count` keys are confirmed live in **Task 6**.

---

## Task 5: Update registry + routers + main

**Files:**
- Modify: `app/core/stream_registry.py`, `app/routers/streams.py`, `app/routers/stream.py`,
  `app/routers/video.py`, `app/routers/image.py`, `app/routers/export_router.py`, `app/main.py`
- Test: `tests/test_api_shapes.py`

- [ ] **Step 1: Write the failing test (TestClient — GPU)**

`tests/test_api_shapes.py`:
```python
from fastapi.testclient import TestClient
from app.main import app


def test_config_get_is_slim():
    with TestClient(app) as client:
        cfg = client.get("/api/config").json()
        assert "model_path" in cfg
        for gone in ("roi_position", "imgsz", "conveyor_speed_px", "zone_half"):
            assert gone not in cfg


def test_config_patch_rejects_removed_field_silently():
    with TestClient(app) as client:
        r = client.patch("/api/config", json={"imgsz": 640})
        assert r.status_code == 200
        assert "imgsz" not in r.json()["config"]


def test_streams_list_has_no_total_count():
    with TestClient(app) as client:
        r = client.get("/api/streams")
        assert r.status_code == 200
        assert r.json() == {"streams": []}


def test_legacy_stream_status_shape():
    with TestClient(app) as client:
        s = client.get("/api/stream/status").json()
        assert set(s.keys()) == {"is_connected", "is_counting", "counts", "fps"}
        assert "total_count" not in s


def test_video_patch_endpoint_removed():
    with TestClient(app) as client:
        # PATCH /api/video/{id} no longer exists -> 405 Method Not Allowed
        assert client.patch("/api/video/abc", json={}).status_code == 405
```

- [ ] **Step 2: Run to verify it fails**

Run: `TEST_INT tests/test_api_shapes.py`
Expected: FAIL (e.g. `total_count` present / PATCH returns 200 / config still has removed keys).

- [ ] **Step 3a: Rewrite `app/core/stream_registry.py`**

```python
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
```

- [ ] **Step 3b: Rewrite `app/routers/streams.py`**

```python
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
```

- [ ] **Step 3c: Edit `app/routers/stream.py`** — two changes:

Change the import (line ~14):
```python
from app.core.classes import CLASSES
```
(was `from app.core.counter import CLASSES`)

Replace the `stream_status()` function body so neither branch returns `total_count`:
```python
@router.get("/status")
def stream_status():
    if not registry.exists(_DEFAULT_ID):
        return {"is_connected": False, "is_counting": False,
                "counts": {cls: 0 for cls in CLASSES}, "fps": 0}
    info = registry.info(_DEFAULT_ID)
    return {
        "is_connected": info.is_playing,
        "is_counting": info.is_counting,
        "counts": info.counts,
        "fps": info.fps,
        "error": info.error,
    }
```
(Note: the no-stream branch returns 4 keys — matches `test_legacy_stream_status_shape`. The active branch adds `error`.)

- [ ] **Step 3d: Rewrite `app/routers/video.py`**

```python
import os
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import StreamingResponse, FileResponse

from app.core.auth import verify_api_key
from app.core.runtime_config import runtime_config
from app.core.video_processor import VideoProcessor

router = APIRouter(prefix="/api/video", tags=["video"],
                   dependencies=[Depends(verify_api_key)])

_sessions: dict[str, VideoProcessor] = {}


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    snap = runtime_config.snapshot()
    session_id = str(uuid.uuid4())[:8]
    filepath = os.path.join(snap["upload_dir"], f"{session_id}_{file.filename}")
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    raw_output = os.path.join(snap["output_dir"], f"{session_id}_raw.mp4")
    processor = VideoProcessor(source=filepath, model_path=snap["model_path"],
                               save_raw_path=raw_output, is_stream=False)
    _sessions[session_id] = processor
    return {"session_id": session_id, "filename": file.filename}


@router.post("/{session_id}/start")
def start_video(session_id: str):
    _get_session(session_id).start()
    return {"status": "playing"}


@router.post("/{session_id}/stop")
def stop_video(session_id: str):
    _get_session(session_id).stop()
    return {"status": "stopped"}


@router.post("/{session_id}/counting/start")
def start_counting(session_id: str):
    _get_session(session_id).start_counting()
    return {"status": "counting"}


@router.post("/{session_id}/counting/stop")
def stop_counting(session_id: str):
    _get_session(session_id).stop_counting()
    return {"status": "not_counting"}


@router.get("/{session_id}/feed")
def video_feed(session_id: str):
    proc = _get_session(session_id)
    return StreamingResponse(_mjpeg_generator(proc),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@router.get("/{session_id}/status")
def video_status(session_id: str):
    return _get_session(session_id).get_status()


@router.get("/{session_id}/download")
def download_video(session_id: str):
    snap = runtime_config.snapshot()
    proc = _get_session(session_id)
    raw_path = proc.save_raw_path
    if not raw_path or not os.path.exists(raw_path):
        raise HTTPException(status_code=404, detail="Output not ready")
    output_path = os.path.join(snap["output_dir"], f"{session_id}_output.mp4")
    if not os.path.exists(output_path):
        try:
            VideoProcessor.reencode_h264(raw_path, output_path)
        except Exception:
            output_path = raw_path
    return FileResponse(output_path, media_type="video/mp4",
                        filename=f"chicken_count_{session_id}.mp4")


def _get_session(session_id: str) -> VideoProcessor:
    proc = _sessions.get(session_id)
    if proc is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return proc


def _mjpeg_generator(proc: VideoProcessor):
    import time
    while proc.is_playing or not proc.is_complete:
        frame_bytes = proc.latest_frame
        if frame_bytes:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        time.sleep(0.03)
```

- [ ] **Step 3e: Rewrite `app/routers/image.py`** (repoint off the deleted worker)

```python
import cv2
import numpy as np
from fastapi import APIRouter, Depends, UploadFile, File
from fastapi.responses import Response

from app.core.auth import verify_api_key
from app.core.runtime_config import runtime_config
from app.core.model_cache import get_model
from app.core.detector import detect_frame
from app.core.annotator import annotate_image_detections

router = APIRouter(prefix="/api/image", tags=["image"],
                   dependencies=[Depends(verify_api_key)])


@router.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return Response(content="Invalid image", status_code=400)

    snap = runtime_config.snapshot()
    det_info = detect_frame(get_model(snap["model_path"]), frame)
    annotated, class_counts = annotate_image_detections(frame, det_info)

    _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
    total = sum(class_counts.values())
    return Response(
        content=jpeg.tobytes(),
        media_type="image/jpeg",
        headers={
            "X-Total-Count": str(total),
            "X-Count-Empty-Shackles": str(class_counts.get("empty_shackles", 0)),
            "X-Count-Single-Legged": str(class_counts.get("single_legged", 0)),
            "X-Count-Slaughtered-Chicken": str(class_counts.get("slaughtered_chicken", 0)),
            "Access-Control-Expose-Headers": (
                "X-Total-Count, X-Count-Empty-Shackles, "
                "X-Count-Single-Legged, X-Count-Slaughtered-Chicken"),
        },
    )
```

- [ ] **Step 3f: Edit `app/routers/export_router.py`** — carry `imgsz` in the request:

Replace the `ExportRequest` class and the `snap["imgsz"]` usage:
```python
class ExportRequest(BaseModel):
    half: bool = True
    imgsz: int = 1280   # model training size; imgsz left runtime config


@router.post("/tensorrt")
def start_export(body: ExportRequest = ExportRequest()):
    snap = runtime_config.snapshot()
    started = exporter.start(model_path=snap["model_path"], imgsz=body.imgsz, half=body.half)
    if not started:
        raise HTTPException(status_code=409, detail="Export already running")
    return {"status": "started", "model_path": snap["model_path"]}
```
(Leave `get_export_status()` unchanged.)

- [ ] **Step 3g: Edit `app/main.py`** — drop the inference worker:

Remove the import line `from app.core.inference_worker import start_worker, stop_worker`.
In `lifespan`, remove the `start_worker(...)` call (the whole call, lines preloading aside) and the `stop_worker()` call in the `finally` block. The result:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    snap = runtime_config.snapshot()
    os.makedirs(snap["upload_dir"], exist_ok=True)
    os.makedirs(snap["output_dir"], exist_ok=True)

    preload_model(snap["model_path"])
    log_auth_state()
    registry.start_all_from_env()
    try:
        yield
    finally:
        registry.stop_all()
```

- [ ] **Step 4: Run the API-shape tests**

Run: `TEST_INT tests/test_api_shapes.py`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add app/core/stream_registry.py app/routers/streams.py app/routers/stream.py \
        app/routers/video.py app/routers/image.py app/routers/export_router.py \
        app/main.py tests/test_api_shapes.py
git commit -m "refactor: wire routers/registry to ObjectCounter processor; drop total_count + tune endpoints"
```

---

## Task 6: Live-validate the ObjectCounter coupling (GPU)

Confirms the two version-sensitive assumptions in `video_processor.py` / `classes.py`.

**Files:** none (throwaway probe), then fix-ups if needed.

- [ ] **Step 1: Probe the real attribute/key names**

Run:
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm \
  -v "$PWD/app:/app/app" --entrypoint sh chicken-counter -c 'python - <<PY
import numpy as np, cv2
from ultralytics import solutions
c = solutions.ObjectCounter(model="best.pt", region=[(320,0),(320,720)], show=False, verbose=False)
frame = (np.random.rand(720,1280,3)*255).astype("uint8")
r = c.process(frame)
print("results attrs:", [a for a in dir(r) if not a.startswith("_")])
print("classwise_count:", getattr(r, "classwise_count", "MISSING"))
print("counter has:", {k: type(getattr(c,k)).__name__ for k in ("boxes","clss","track_ids","confs","names") if hasattr(c,k)})
PY'
```
Expected: `results attrs` includes `classwise_count`, `in_count`, `out_count`, `plot_im`; `classwise_count` is a dict keyed by class name with `IN`/`OUT` sub-keys; `counter has` lists `boxes, clss, track_ids, confs, names`.

- [ ] **Step 2: Reconcile if needed**

- If `classwise_count` sub-keys are **not** `"IN"`/`"OUT"` (e.g. lowercase), update `classwise_to_counts` in `app/core/classes.py` and `tests/test_classes.py`.
- If any of `boxes/clss/track_ids/confs/names` is named differently, update `extract_solution_boxes` in `app/core/video_processor.py`.
- If nothing differs, no change.

- [ ] **Step 3: Re-run affected unit tests + commit any fix**

Run: `TEST_UNIT tests/test_classes.py`
Expected: PASS.
```bash
git add -A && git commit -m "fix: pin ObjectCounter attribute/key names to installed ultralytics" || echo "no changes needed"
```

---

## Task 7: Delete dead code + dead tests; drop scipy

**Files:**
- Delete: `app/core/counter.py`, `app/core/tracker.py`, `app/core/inference_worker.py`,
  `tests/test_counter_band.py`, `tests/test_counter_velocity.py`, `tests/test_sway_and_proc.py`,
  `tests/test_stream_overrides.py`, `tests/test_config_plumbing.py`, `tests/test_undercount_fixes.py`
- Modify: `requirements.txt` (remove `scipy`)

- [ ] **Step 1: Verify nothing live still imports the doomed modules**

Run:
```bash
grep -rEn "core\.(counter|tracker|inference_worker)|import scipy|from scipy" app/ || echo "CLEAN"
```
Expected: `CLEAN`. (If anything prints, it is a leftover reference — fix it before deleting.)

- [ ] **Step 2: Delete the files**

```bash
git rm app/core/counter.py app/core/tracker.py app/core/inference_worker.py \
       tests/test_counter_band.py tests/test_counter_velocity.py tests/test_sway_and_proc.py \
       tests/test_stream_overrides.py tests/test_config_plumbing.py tests/test_undercount_fixes.py
```

- [ ] **Step 3: Remove `scipy` from `requirements.txt`**

Delete the line `scipy` (it was only used by the deleted counter/tracker).

- [ ] **Step 4: Rebuild and run the full surviving test suite**

Run: `docker compose build chicken-counter` (requirements changed)
Then GPU full run:
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm \
  -v "$PWD/app:/app/app" -v "$PWD/tests:/app/tests" \
  --entrypoint sh chicken-counter -c "pip install -q pytest && pytest -q tests/"
```
Expected: PASS — only `test_classes`, `test_annotator`, `test_slim_config`, `test_rtsp_helpers`, `test_api_shapes` remain, all green. No import errors for removed modules.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: delete custom counter/tracker/inference-worker + dead tests; drop scipy"
```

---

## Task 8: Update the static dashboard

**Files:**
- Modify: `app/static/js/video.js`, `app/static/js/stream.js`, `app/static/video.html`, `app/static/stream.html`

No automated test — verified visually in Task 9 Step 4.

- [ ] **Step 1: Rewrite `app/static/js/video.js`** (remove tuning + total; keep per-class counts)

```javascript
const uploadZone = document.getElementById("uploadZone");
const fileInput = document.getElementById("fileInput");
const uploadCard = document.getElementById("uploadCard");
const playerSection = document.getElementById("playerSection");

const btnPlay = document.getElementById("btnPlay");
const btnStop = document.getElementById("btnStop");
const btnCountStart = document.getElementById("btnCountStart");
const btnCountStop = document.getElementById("btnCountStop");
const feedImg = document.getElementById("feedImg");
const feedPlaceholder = document.getElementById("feedPlaceholder");
const statusDot = document.getElementById("statusDot");
const statusLabel = document.getElementById("statusLabel");
const emptyCount = document.getElementById("emptyCount");
const singleCount = document.getElementById("singleCount");
const slaughteredCount = document.getElementById("slaughteredCount");
const frameNum = document.getElementById("frameNum");
const fpsVal = document.getElementById("fpsVal");
const downloadBtn = document.getElementById("downloadBtn");

let sessionId = null;
let pollInterval = null;

uploadZone.addEventListener("click", () => fileInput.click());
uploadZone.addEventListener("dragover", (e) => { e.preventDefault(); uploadZone.classList.add("dragover"); });
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));
uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("dragover");
    if (e.dataTransfer.files.length) handleUpload(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => { if (fileInput.files.length) handleUpload(fileInput.files[0]); });

async function handleUpload(file) {
    uploadZone.querySelector(".label").textContent = "Uploading...";
    const formData = new FormData();
    formData.append("file", file);
    try {
        const resp = await fetch("/api/video/upload", { method: "POST", body: formData });
        const data = await resp.json();
        sessionId = data.session_id;
        uploadCard.style.display = "none";
        playerSection.style.display = "block";
    } catch (err) {
        uploadZone.querySelector(".label").textContent = "Upload failed - try again";
        console.error(err);
    }
}

btnPlay.addEventListener("click", async () => {
    await fetch(`/api/video/${sessionId}/start`, { method: "POST" });
    feedImg.src = `/api/video/${sessionId}/feed?t=${Date.now()}`;
    feedImg.style.display = "block";
    feedPlaceholder.style.display = "none";
    btnPlay.disabled = true;
    btnStop.disabled = false;
    btnCountStart.disabled = false;
    setStatus(true, "Playing");
    startPolling();
});

btnStop.addEventListener("click", async () => {
    await fetch(`/api/video/${sessionId}/stop`, { method: "POST" });
    btnPlay.disabled = false;
    btnStop.disabled = true;
    btnCountStart.disabled = true;
    btnCountStop.disabled = true;
    setStatus(false, "Stopped");
    stopPolling();
    showDownload();
});

btnCountStart.addEventListener("click", async () => {
    await fetch(`/api/video/${sessionId}/counting/start`, { method: "POST" });
    btnCountStart.disabled = true;
    btnCountStop.disabled = false;
});

btnCountStop.addEventListener("click", async () => {
    await fetch(`/api/video/${sessionId}/counting/stop`, { method: "POST" });
    btnCountStart.disabled = false;
    btnCountStop.disabled = true;
});

function setStatus(active, text) {
    statusDot.className = `status-dot ${active ? "active" : "inactive"}`;
    statusLabel.textContent = text;
}

function startPolling() {
    pollInterval = setInterval(async () => {
        try {
            const resp = await fetch(`/api/video/${sessionId}/status`);
            const s = resp.ok ? await resp.json() : null;
            if (!s) return;
            emptyCount.textContent = s.counts?.empty_shackles ?? 0;
            singleCount.textContent = s.counts?.single_legged ?? 0;
            slaughteredCount.textContent = s.counts?.slaughtered_chicken ?? 0;
            frameNum.textContent = s.total_frames > 0 ? `${s.frame_num}/${s.total_frames}` : s.frame_num;
            fpsVal.textContent = s.fps;
            if (s.is_complete) {
                setStatus(false, "Complete");
                btnPlay.disabled = false;
                btnStop.disabled = true;
                btnCountStart.disabled = true;
                btnCountStop.disabled = true;
                stopPolling();
                showDownload();
            }
        } catch (_) {}
    }, 500);
}

function stopPolling() {
    if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }
}

function showDownload() {
    downloadBtn.href = `/api/video/${sessionId}/download`;
    downloadBtn.style.display = "inline-flex";
}
```

- [ ] **Step 2: Rewrite `app/static/js/stream.js`** (remove tuning + total; keep per-class counts)

```javascript
const rtspUrl = document.getElementById("rtspUrl");
const btnConnect = document.getElementById("btnConnect");
const btnDisconnect = document.getElementById("btnDisconnect");
const btnCountStart = document.getElementById("btnCountStart");
const btnCountStop = document.getElementById("btnCountStop");
const feedImg = document.getElementById("feedImg");
const feedPlaceholder = document.getElementById("feedPlaceholder");
const statusDot = document.getElementById("statusDot");
const statusLabel = document.getElementById("statusLabel");
const emptyCount = document.getElementById("emptyCount");
const singleCount = document.getElementById("singleCount");
const slaughteredCount = document.getElementById("slaughteredCount");
const fpsVal = document.getElementById("fpsVal");

let pollInterval = null;

fetch("/api/config").then(r => r.json()).then(cfg => {
    if (cfg.rtsp_url) rtspUrl.value = cfg.rtsp_url;
}).catch(() => {});

btnConnect.addEventListener("click", async () => {
    const url = rtspUrl.value.trim();
    const body = url ? { url } : {};
    try {
        const resp = await fetch("/api/stream/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        if (!resp.ok) {
            const err = await resp.json();
            alert(err.detail || "Failed to connect");
            return;
        }
        feedImg.src = `/api/stream/feed?t=${Date.now()}`;
        feedImg.style.display = "block";
        feedPlaceholder.style.display = "none";
        btnConnect.disabled = true;
        btnDisconnect.disabled = false;
        btnCountStart.disabled = false;
        setStatus(true, "Connected");
        startPolling();
    } catch (err) {
        console.error(err);
    }
});

btnDisconnect.addEventListener("click", async () => {
    await fetch("/api/stream/stop", { method: "POST" });
    feedImg.style.display = "none";
    feedPlaceholder.style.display = "block";
    btnConnect.disabled = false;
    btnDisconnect.disabled = true;
    btnCountStart.disabled = true;
    btnCountStop.disabled = true;
    setStatus(false, "Disconnected");
    stopPolling();
});

btnCountStart.addEventListener("click", async () => {
    await fetch("/api/stream/counting/start", { method: "POST" });
    btnCountStart.disabled = true;
    btnCountStop.disabled = false;
});

btnCountStop.addEventListener("click", async () => {
    await fetch("/api/stream/counting/stop", { method: "POST" });
    btnCountStart.disabled = false;
    btnCountStop.disabled = true;
});

function setStatus(active, text) {
    statusDot.className = `status-dot ${active ? "active" : "inactive"}`;
    statusLabel.textContent = text;
}

function startPolling() {
    pollInterval = setInterval(async () => {
        try {
            const resp = await fetch("/api/stream/status");
            const s = await resp.json();
            emptyCount.textContent = s.counts?.empty_shackles ?? 0;
            singleCount.textContent = s.counts?.single_legged ?? 0;
            slaughteredCount.textContent = s.counts?.slaughtered_chicken ?? 0;
            fpsVal.textContent = s.fps;
            if (!s.is_connected) {
                setStatus(false, s.error || "Disconnected");
                btnConnect.disabled = false;
                btnDisconnect.disabled = true;
                stopPolling();
            }
        } catch (_) {}
    }, 500);
}

function stopPolling() {
    if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }
}
```

- [ ] **Step 3: Edit `app/static/video.html` and `app/static/stream.html`**

In **each** file:
- Remove the entire tuning-controls block — every element whose id ends in `Slider` or `Value`, plus `imgszSelect` and their containing panel/`<section>`/card. (Search the file for `Slider` and `imgszSelect`; delete the wrapping control group.)
- Remove the total-count stat element `id="totalCount"` and its label/wrapper.
- **Keep** `emptyCount`, `singleCount`, `slaughteredCount`, `frameNum` (video only), `fpsVal`, and all play/connect/count buttons.

(No id referenced by the rewritten JS may be deleted: `uploadZone, fileInput, uploadCard, playerSection, btnPlay, btnStop, btnConnect, btnDisconnect, btnCountStart, btnCountStop, feedImg, feedPlaceholder, statusDot, statusLabel, emptyCount, singleCount, slaughteredCount, frameNum, fpsVal, downloadBtn, rtspUrl`.)

- [ ] **Step 4: Commit**

```bash
git add app/static/js/video.js app/static/js/stream.js app/static/video.html app/static/stream.html
git commit -m "refactor(ui): drop tuning controls + total-count; keep per-class counts"
```

---

## Task 9: Integration validation (GPU)

**Files:** `tools/parity_check.py` (throwaway; not committed unless useful).

- [ ] **Step 1: Counting parity vs test.py**

Create `tools/parity_check.py`:
```python
"""Run a clip through the new VideoProcessor and compare its per-class counts to
a direct ObjectCounter pass (test.py logic). They must match."""
import sys, time, cv2
from ultralytics import solutions
from app.core.video_processor import VideoProcessor

CLIP = sys.argv[1] if len(sys.argv) > 1 else "/data/clip.mp4"

# Direct ObjectCounter (reference)
cap = cv2.VideoCapture(CLIP)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
ref = solutions.ObjectCounter(model="best.pt", region=[(w//2,0),(w//2,h)], show=False, verbose=False)
while True:
    ok, frame = cap.read()
    if not ok: break
    r = ref.process(frame)
cap.release()
print("REFERENCE classwise_count:", r.classwise_count)

# New processor
vp = VideoProcessor(source=CLIP, model_path="best.pt", is_stream=False)
vp.start(); vp.start_counting()
while vp.is_playing and not vp.is_complete:
    time.sleep(0.2)
print("PROCESSOR counts:", vp.counts)
```
Run (mount a real dense clip as `/data/clip.mp4`):
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm \
  -v "$PWD/app:/app/app" -v "$PWD/tools:/app/tools" -v "/c/path/to/dense_clip.mp4:/data/clip.mp4" \
  --entrypoint sh chicken-counter -c "python -m tools.parity_check /data/clip.mp4"
```
Expected: `PROCESSOR counts` per-class IN equals the reference `classwise_count` IN values. **If they diverge, STOP and debug** before claiming success.

- [ ] **Step 2: Accuracy sanity (maintainer's ground truth)**

Confirm with the maintainer that the per-class counts on a representative **dense** clip match expectation / BAADER (the spec's risk item). This is a human check, not an assertion.

- [ ] **Step 3: Endpoint smoke (boot the real server)**

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
sleep 20
curl -fsS http://localhost:5581/health && echo
curl -fsS http://localhost:5581/api/config && echo
curl -fsS -X POST http://localhost:5581/api/image/detect -F "file=@/c/path/to/sample.jpg" -D - -o /dev/null
docker compose down
```
Expected: `/health` ok; `/api/config` is the slim dict; image POST returns `200` with `X-Count-*` + `X-Total-Count` headers.

- [ ] **Step 4: Visual check — bbox-only annotation**

Upload a clip via the dashboard (`http://localhost:5581/video.html`), Play + Start Counting, and confirm the feed shows **bounding boxes with the existing per-class colors and small labels**, **no** counting line, **no** in/out HUD, and the per-class count tiles increment.

- [ ] **Step 5: Final commit (if parity tool kept)**

```bash
git add -A && git commit -m "test: add ObjectCounter parity check tool" || echo "nothing to commit"
```

---

## Self-review (completed during planning)

- **Spec coverage:** core/counting (Task 4), counts contract + no total (Tasks 1,4,5), bbox-only annotation (Tasks 2,4), config slim (Task 3), routers incl. image/export/main (Task 5), deletions + scipy (Task 7), static UI (Task 8), validation incl. attribute pinning + parity (Tasks 6,9). `tools/` left in place per spec (not deleted).
- **Placeholder scan:** none — every code step is complete.
- **Type consistency:** `classwise_to_counts`, `extract_solution_boxes`, `annotate_boxes(frame, boxes)`, `VideoProcessor(source, model_path, save_raw_path, is_stream)`, `registry.register(id, url, start_counting)`, `StreamInfo` (no `total_count`), and `get_status` keys are used identically across tasks.

## Notes for the engineer

- **`tools/analyze_video*.py`** are obsolete (they calibrate removed belt-stop/zone params) but are the maintainer's files — leave them; flag for removal.
- The branch is `refactor/objectcounter-counting` (spec already committed there).
- If `docker compose build` is too slow to iterate, only Task 0 and Task 7 Step 4 need a (re)build; every other test run bind-mounts source over the existing image.
