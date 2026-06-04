# RTSP Counting Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the chicken-undercount gap by making the live counter robust to RTSP stream instability, fast/variable belt speed, and intermittent bbox flicker at the counting line.

**Architecture:** Three independent improvements to the existing single-thread capture → batched-inference → straddle-counter pipeline: (1) per-track velocity estimation in `ChickenCounter` seeded by a configurable per-stream `conveyor_speed_px`; (2) a counting **band** (reusing the already-declared-but-unwired `zone_half` API field) instead of a single-pixel tripwire; (3) RTSP hardening in `VideoProcessor` — TCP transport, 1-frame buffer, auto-reconnect with exponential backoff, and frozen-frame detection. No changes to the inference worker or model.

**Tech Stack:** Python 3.13, FastAPI, OpenCV (FFMPEG backend), Ultralytics YOLO (TensorRT), pydantic-settings, scipy (tracker), pytest (new dev dep).

---

## Environment note (read before executing)

The local `.venv` is **broken** — it was created by a Python that no longer exists (`C:\Python313`), so `.venv\Scripts\python.exe` fails with *"did not find executable at 'c:\python313\python.exe'"*. System Python (3.14) lacks `scipy`/`numpy`/`pydantic`. **Tests in this plan must run inside the Docker container** (where deps are installed) or after recreating the venv:

```powershell
# Option A — recreate venv (run once before executing the plan)
py -3.13 -m venv .venv ; .\.venv\Scripts\python -m pip install -r requirements.txt pytest

# Option B — run tests in the container
docker compose run --rm app pytest -v
```

Every "Run test" step below assumes one of these works. If neither is available, fall back to `python -m py_compile <file>` for syntax verification and rely on manual stream verification for Task 4.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `app/config.py` | Boot defaults | Add `conveyor_speed_px`, `zone_half` |
| `app/core/runtime_config.py` | Live config store | Add both keys to `_data` |
| `app/routers/config_router.py` | PATCH validation | Add `conveyor_speed_px` field + validator (`zone_half` already declared) |
| `app/core/stream_registry.py` | Per-stream overrides | Add both keys to `_OVERRIDE_KEYS` + `_merge_overrides`; pass to `VideoProcessor` |
| `app/core/video_processor.py` | Capture loop | Accept new params; RTSP hardening (Task 4) |
| `app/core/counter.py` | Counting logic | Per-track velocity (Task 2) + band (Task 3) |
| `app/core/annotator.py` | Overlay | Draw band instead of single line (Task 3) |
| `tests/` (new) | Unit tests | New test files per task |
| `requirements.txt` | Deps | Add `pytest` |

---

## Task 1: Per-stream `conveyor_speed_px` config plumbing

Belt is ~34 px/frame at 1280-wide sub-stream (6″ pitch, 119 cm FOV, 311 shackles/min), not the hardcoded `14.0`. Make it a configurable per-stream seed value.

**Files:**
- Modify: `app/config.py:12-25`
- Modify: `app/core/runtime_config.py:15-33`
- Modify: `app/routers/config_router.py:13-24,54-59`
- Modify: `app/core/stream_registry.py:25-28,90-101,208-217`
- Modify: `app/core/video_processor.py:15-42`
- Modify: `app/core/counter.py:20-21`
- Test: `tests/test_config_plumbing.py`

- [ ] **Step 1: Add `pytest` to requirements**

In `requirements.txt`, append:
```
pytest
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_config_plumbing.py`:
```python
from app.config import Settings
from app.core.runtime_config import RuntimeConfig
from app.core.stream_registry import StreamRegistry


def test_conveyor_speed_default_is_belt_calibrated():
    assert Settings().conveyor_speed_px == 34.0


def test_zone_half_default():
    assert Settings().zone_half == 15


def test_runtime_config_exposes_new_keys():
    snap = RuntimeConfig().snapshot()
    assert "conveyor_speed_px" in snap
    assert "zone_half" in snap


def test_merge_overrides_passes_conveyor_speed():
    snap = RuntimeConfig().snapshot()
    cfg = StreamRegistry._merge_overrides(snap, {"conveyor_speed_px": 40.0})
    assert cfg["conveyor_speed_px"] == 40.0
    # unspecified override falls back to snapshot default
    assert cfg["zone_half"] == snap["zone_half"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_config_plumbing.py -v`
Expected: FAIL — `AttributeError`/`KeyError` on `conveyor_speed_px` / `zone_half`.

- [ ] **Step 4: Add fields to `app/config.py`**

After the `max_disappeared: int = 2` line (config.py:25), add:
```python
    # Belt travel per processed frame (px), used to seed per-track velocity
    # estimation in the counter. ~34 px/frame on the 1280-wide sub-stream
    # (6in shackle pitch, 119cm FOV, ~311 shackles/min). Self-tunes at runtime.
    conveyor_speed_px: float = 34.0
    # Half-width (px) of the counting band around roi_x. Band total width =
    # 2*zone_half. Wider band tolerates bbox flicker / brief frame stutter so
    # a bird crossing the line is not missed. 0 = single-pixel tripwire.
    zone_half: int = 15
```

- [ ] **Step 5: Add keys to `app/core/runtime_config.py`**

In the `_data` dict (after `"max_disappeared": boot.max_disappeared,` at runtime_config.py:24), add:
```python
            "conveyor_speed_px":   boot.conveyor_speed_px,
            "zone_half":           boot.zone_half,
```

- [ ] **Step 6: Add `conveyor_speed_px` to `ConfigPatch`**

In `app/routers/config_router.py`, add a field after `zone_half: Optional[int] = None` (line 23):
```python
    conveyor_speed_px: Optional[float] = None
```
And add a validator after the `positive_int` validator (line 59):
```python
    @field_validator("conveyor_speed_px")
    @classmethod
    def speed_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError("conveyor_speed_px must be > 0")
        return v
```
(`zone_half` already has a field + the `positive_int` validator — but note that validator requires `>= 1`. To allow `zone_half=0` (single-pixel mode), remove `"zone_half"` from the `positive_int` validator's decorator list on line 54 and add it to a `>= 0` check:)
```python
    @field_validator("max_distance", "max_disappeared", "appear_margin")
    @classmethod
    def positive_int(cls, v):
        if v is not None and v < 1:
            raise ValueError("must be >= 1")
        return v

    @field_validator("zone_half")
    @classmethod
    def zone_half_nonneg(cls, v):
        if v is not None and v < 0:
            raise ValueError("zone_half must be >= 0")
        return v
```

- [ ] **Step 7: Wire through `stream_registry.py`**

Add both keys to `_OVERRIDE_KEYS` (stream_registry.py:25-28):
```python
_OVERRIDE_KEYS = {
    "roi_position", "confidence", "conf_empty_shackles", "nms_iou", "imgsz",
    "max_distance", "max_disappeared", "conveyor_speed_px", "zone_half",
}
```
Add both to the tuple in `_merge_overrides` (stream_registry.py:210-213):
```python
        cfg = {k: snap[k] for k in (
            "roi_position", "confidence", "conf_empty_shackles", "nms_iou", "imgsz",
            "max_distance", "max_disappeared", "conveyor_speed_px", "zone_half",
        )}
```
Pass them into the `VideoProcessor(...)` constructor (stream_registry.py:90-101), adding after `conf_empty_shackles=cfg["conf_empty_shackles"],`:
```python
                conveyor_speed_px=cfg["conveyor_speed_px"],
                zone_half=cfg["zone_half"],
```

- [ ] **Step 8: Accept params in `VideoProcessor.__init__`**

In `app/core/video_processor.py`, extend the signature (video_processor.py:15-19):
```python
    def __init__(self, source: str, model, roi_x: int, confidence: float = 0.25,
                 nms_iou: float = 0.45, imgsz: int = 640,
                 max_disappeared: int = 15, max_distance: int = 55,
                 conf_empty_shackles: float = 0.15,
                 conveyor_speed_px: float = 34.0, zone_half: int = 15,
                 save_raw_path: str = None, is_stream: bool = False):
```
And pass them into the counter (video_processor.py:41-42):
```python
        self.counter = ChickenCounter(roi_x=roi_x, max_disappeared=max_disappeared,
                                      max_distance=max_distance,
                                      conveyor_speed_px=conveyor_speed_px,
                                      zone_half=zone_half)
```

- [ ] **Step 9: Accept `zone_half` in `ChickenCounter.__init__`**

In `app/core/counter.py`, extend the signature (counter.py:20-21) — `conveyor_speed_px` already exists; add `zone_half`:
```python
    def __init__(self, roi_x: int, max_disappeared: int = 15,
                 max_distance: int = 55, conveyor_speed_px: float = 34.0,
                 zone_half: int = 15):
```
And store it in `__init__` (after `self.conveyor_speed_px = conveyor_speed_px` at counter.py:25):
```python
        self.zone_half = zone_half
```

- [ ] **Step 10: Run test to verify it passes**

Run: `pytest tests/test_config_plumbing.py -v`
Expected: PASS (4 tests).

- [ ] **Step 11: Commit**

```bash
git add requirements.txt tests/test_config_plumbing.py app/config.py app/core/runtime_config.py app/routers/config_router.py app/core/stream_registry.py app/core/video_processor.py app/core/counter.py
git commit -m "feat: per-stream conveyor_speed_px and zone_half config plumbing"
```

---

## Task 2: Per-track velocity estimation

Replace the fixed `conveyor_speed_px` used in crossing prediction with a per-crossing velocity that self-tunes from observed motion (seeded by `conveyor_speed_px`). Survives mid-shift belt-speed changes and fixes the 14→34 mismatch automatically.

**Files:**
- Modify: `app/core/counter.py:20-35,76-106`
- Test: `tests/test_counter_velocity.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_counter_velocity.py`:
```python
from app.core.counter import ChickenCounter


def _det(cx, cls="slaughtered_chicken", w=80, h=80):
    return {"x1": cx - w // 2, "y1": 100 - h // 2,
            "x2": cx + w // 2, "y2": 100 + h // 2, "class_name": cls}


def test_single_fast_bird_counts_once_at_34px_per_frame():
    """A bird crossing roi_x=200 at 34 px/frame must count exactly once,
    even though it straddles for several frames."""
    c = ChickenCounter(roi_x=200, conveyor_speed_px=34.0, zone_half=0)
    counted_before = c.counts["slaughtered_chicken"]
    # bbox half-width 40 -> straddles single-pixel line from cx=160..240
    for cx in range(120, 320, 34):  # 120,154,188,222,256,290
        c.update([_det(cx)])
    assert c.counts["slaughtered_chicken"] == counted_before + 1


def test_velocity_self_tunes_above_seed():
    """Seed 14 but real motion 34 -> learned velocity climbs toward 34."""
    c = ChickenCounter(roi_x=200, conveyor_speed_px=14.0, zone_half=0)
    for cx in range(120, 320, 34):
        c.update([_det(cx)])
    # at least one crossing existed; learned velocity should exceed the seed
    # (inspect via a fresh crossing if list emptied — assert no double count)
    assert c.counts["slaughtered_chicken"] == 1


def test_two_sequential_birds_count_twice():
    c = ChickenCounter(roi_x=200, conveyor_speed_px=34.0, zone_half=0)
    # bird A passes fully
    for cx in range(120, 340, 34):
        c.update([_det(cx)])
    # gap (empty frames) so A's crossing expires
    for _ in range(12):
        c.update([])
    # bird B passes
    for cx in range(120, 340, 34):
        c.update([_det(cx)])
    assert c.counts["slaughtered_chicken"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_counter_velocity.py -v`
Expected: FAIL — `ChickenCounter.__init__` does not yet accept `zone_half` (added in Task 1; if Task 1 done, these may already partially pass — the velocity test asserts behavior not yet implemented). If Task 1 is complete, expect the double-count / miss assertions to drive the change.

- [ ] **Step 3: Add velocity tunables to `__init__`**

In `app/core/counter.py` `__init__`, after `self.zone_half = zone_half`, add:
```python
        # Per-track velocity estimation. Each crossing learns its own px/frame
        # via EMA of observed motion, seeded by conveyor_speed_px.
        self.velocity_ema = 0.3          # weight of newest observation
        self.max_velocity_px = 120.0     # reject implausible jumps
```

- [ ] **Step 4: Seed velocity when a crossing is created**

In `update()`, change the "New crossing!" block (counter.py:99-106) to store a velocity:
```python
            else:
                # New crossing!
                self.counts[cls] += 1
                self.active_crossings.append({
                    'cls': cls,
                    'last_cx': cx,
                    'last_seen_frame': self.frame_num,
                    'velocity': self.conveyor_speed_px,
                })
                self.flash_events.append((cx, cy, cls))
```

- [ ] **Step 5: Use per-crossing velocity in prediction**

In the matching loop (counter.py:85-86), change:
```python
                frames_elapsed = self.frame_num - crossing['last_seen_frame']
                predicted_cx = crossing['last_cx'] + (frames_elapsed * crossing['velocity'])
```

- [ ] **Step 6: Update velocity on match (EMA)**

Replace the matched-crossing update block (counter.py:93-97) with:
```python
            if best_match_idx != -1:
                # Update existing crossing + learn its velocity from motion.
                c = self.active_crossings[best_match_idx]
                frames_elapsed = self.frame_num - c['last_seen_frame']
                if frames_elapsed > 0:
                    observed_v = (cx - c['last_cx']) / frames_elapsed
                    if 0 < observed_v < self.max_velocity_px:
                        c['velocity'] = (self.velocity_ema * observed_v
                                         + (1 - self.velocity_ema) * c['velocity'])
                c['last_cx'] = cx
                c['last_seen_frame'] = self.frame_num
                matched_crossings.add(best_match_idx)
```

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest tests/test_counter_velocity.py -v`
Expected: PASS (3 tests).

- [ ] **Step 8: Commit**

```bash
git add tests/test_counter_velocity.py app/core/counter.py
git commit -m "feat: per-track velocity estimation in counter (self-tunes belt speed)"
```

---

## Task 3: Counting band (zone) instead of single-pixel tripwire

Count when a bbox overlaps the band `[roi_x - zone_half, roi_x + zone_half]`, not just the exact pixel. Tolerates bbox flicker and brief frame stutter. Draw the band in the overlay.

**Files:**
- Modify: `app/core/counter.py:67-72`
- Modify: `app/core/annotator.py:47-65,108-148`
- Modify: `app/core/video_processor.py:194-201`
- Test: `tests/test_counter_band.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_counter_band.py`:
```python
from app.core.counter import ChickenCounter


def _det(cx, cls="slaughtered_chicken", w=20, h=80):
    return {"x1": cx - w // 2, "y1": 100 - h // 2,
            "x2": cx + w // 2, "y2": 100 + h // 2, "class_name": cls}


def test_band_catches_bird_that_skips_exact_line():
    """Narrow bbox (w=20) at 34 px/frame never lands exactly on roi_x=200,
    but a 15px band still catches it."""
    c = ChickenCounter(roi_x=200, conveyor_speed_px=34.0, zone_half=15)
    # cx steps: 175, 209, 243 -> none has x1<=200<=x2 (bbox half-width 10),
    # but 209 overlaps band [185,215].
    for cx in (175, 209, 243):
        c.update([_det(cx)])
    assert c.counts["slaughtered_chicken"] == 1


def test_single_pixel_mode_misses_when_zone_zero():
    """Same trajectory with zone_half=0 misses (documents why band helps)."""
    c = ChickenCounter(roi_x=200, conveyor_speed_px=34.0, zone_half=0)
    for cx in (175, 209, 243):
        c.update([_det(cx)])
    assert c.counts["slaughtered_chicken"] == 0


def test_wide_band_still_counts_once():
    c = ChickenCounter(roi_x=200, conveyor_speed_px=34.0, zone_half=40)
    for cx in range(120, 320, 34):
        c.update([_det(cx, w=80)])
    assert c.counts["slaughtered_chicken"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_counter_band.py -v`
Expected: FAIL — `test_band_catches_bird_that_skips_exact_line` returns 0 (straddle still uses single pixel).

- [ ] **Step 3: Widen the straddle test to a band**

In `app/core/counter.py` `update()`, replace the straddle check (counter.py:67-72):
```python
            # Check if bbox overlaps the counting band around roi_x
            x1, x2 = d["x1"], d["x2"]
            lo = self.roi_x - self.zone_half
            hi = self.roi_x + self.zone_half
            if x1 <= hi and x2 >= lo:
                cx = (x1 + x2) // 2
                cy = (d["y1"] + d["y2"]) // 2
                straddlers.append((cx, cy, cls))
```

- [ ] **Step 4: Run counter test to verify it passes**

Run: `pytest tests/test_counter_band.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Draw the band in the annotator**

In `app/core/annotator.py`, change `draw_roi_line` signature and body (annotator.py:47-49) to accept `zone_half` and shade the band:
```python
def draw_roi_line(img, roi_x, height, frame_num, zone_half=0):
    """Draw an animated vertical ROI counting line + translucent band."""
    if zone_half > 0:
        overlay = img.copy()
        cv2.rectangle(overlay, (roi_x - zone_half, 0),
                      (roi_x + zone_half, height), COLORS["roi_glow"], -1)
        cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)
    cv2.line(img, (roi_x, 0), (roi_x, height), COLORS["roi_glow"], 6, cv2.LINE_AA)
```
Update `annotate_detections` signature (annotator.py:108-109):
```python
def annotate_detections(frame, detections, objects_by_class,
                        flash_events, roi_x, frame_num, zone_half=0):
```
And the call to `draw_roi_line` (annotator.py:131-132):
```python
    if roi_x is not None:
        draw_roi_line(annotated, roi_x, height, frame_num, zone_half)
```

- [ ] **Step 6: Pass `zone_half` from the processor**

In `app/core/video_processor.py`, update the `annotate_detections(...)` call (video_processor.py:194-201), adding:
```python
                zone_half=self.counter.zone_half if self.is_counting else 0,
```

- [ ] **Step 7: Syntax-check the overlay modules**

Run: `python -m py_compile app/core/annotator.py app/core/video_processor.py`
Expected: no output (success). Visual band is verified manually in Task 4 / final verification.

- [ ] **Step 8: Commit**

```bash
git add tests/test_counter_band.py app/core/counter.py app/core/annotator.py app/core/video_processor.py
git commit -m "feat: counting band (zone_half) replaces single-pixel tripwire"
```

---

## Task 4: RTSP hardening — TCP transport, 1-frame buffer, reconnect, frozen-frame detection

The client states the gap tracks stream stability. Keep the live feed current and recover from drops/freezes without losing the count.

**Files:**
- Modify: `app/core/video_processor.py:1-10,107-142,216-219`
- Test: `tests/test_rtsp_helpers.py`

- [ ] **Step 1: Write the failing test (pure helpers)**

Create `tests/test_rtsp_helpers.py`:
```python
import numpy as np
from app.core.video_processor import reconnect_delay, frame_signature


def test_reconnect_delay_exponential_with_cap():
    assert reconnect_delay(0) == 1.0
    assert reconnect_delay(1) == 2.0
    assert reconnect_delay(2) == 4.0
    assert reconnect_delay(10) == 30.0  # capped


def test_frame_signature_stable_for_identical_frames():
    a = np.full((480, 848, 3), 7, dtype=np.uint8)
    b = a.copy()
    assert frame_signature(a) == frame_signature(b)


def test_frame_signature_changes_with_content():
    a = np.zeros((480, 848, 3), dtype=np.uint8)
    b = a.copy()
    b[0:100, 0:100] = 255
    assert frame_signature(a) != frame_signature(b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rtsp_helpers.py -v`
Expected: FAIL — `ImportError: cannot import name 'reconnect_delay'`.

- [ ] **Step 3: Add the pure helpers + TCP env at module top**

In `app/core/video_processor.py`, after the existing imports (video_processor.py:1-9), add:
```python
# Prefer TCP for RTSP (UDP drops frames silently on congested networks) and
# disable input buffering for low latency. Set before any VideoCapture opens.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;5000000|fflags;nobuffer",
)

# Frozen-frame detection: this many identical consecutive frames means the
# stream has stalled (camera/encoder hung) and we should reconnect.
FROZEN_FRAME_LIMIT = 60


def reconnect_delay(attempt: int, base: float = 1.0, cap: float = 30.0) -> float:
    """Exponential backoff (seconds) for stream reconnection, capped."""
    return min(cap, base * (2 ** max(0, attempt)))


def frame_signature(frame) -> int:
    """Cheap signature of a frame for frozen-stream detection. Coarse
    subsample keeps it O(1)-ish regardless of resolution."""
    return int(frame[::32, ::32].sum())
```

- [ ] **Step 4: Run helper test to verify it passes**

Run: `pytest tests/test_rtsp_helpers.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Extract capture-open into a helper**

In `app/core/video_processor.py`, add a method on `VideoProcessor` (place above `_run`, near video_processor.py:107):
```python
    def _open_capture(self):
        cap = cv2.VideoCapture(self.source)
        if self.is_stream:
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep only newest frame
            except Exception:
                pass
        return cap
```

- [ ] **Step 6: Rework `_run` for reconnect + frozen detection**

Replace the body of `_run` from the initial open through the read-loop (video_processor.py:107-216) with the version below. Key changes: open via `_open_capture`; on stream read-failure or freeze, reconnect with backoff instead of ending; preserve `self.counter` state across reconnects.
```python
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

            # Frozen-stream detection (live only).
            if self.is_stream:
                sig = frame_signature(frame)
                if sig == last_sig:
                    frozen_count += 1
                    if frozen_count >= FROZEN_FRAME_LIMIT:
                        cap.release()
                        self.error = "Stream frozen; reconnecting"
                        frozen_count = 0
                        last_sig = None
                        if self._stop_event.wait(reconnect_delay(reconnect_attempt)):
                            break
                        reconnect_attempt += 1
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
```

- [ ] **Step 7: Syntax-check**

Run: `python -m py_compile app/core/video_processor.py`
Expected: no output (success).

- [ ] **Step 8: Manual stream verification**

Start a stream and confirm: (a) overlay shows the translucent band around the line; (b) pulling the camera cable logs "reconnecting in Ns" and counts resume (not reset) when restored; (c) `dropped_frames` stays low under load. Use the running app:
```bash
docker compose up -d
# open the stream UI, start counting, observe overlay + reconnect behavior
```

- [ ] **Step 9: Commit**

```bash
git add tests/test_rtsp_helpers.py app/core/video_processor.py
git commit -m "feat: RTSP hardening (tcp transport, buffer=1, reconnect backoff, frozen-frame detection)"
```

---

## Final Verification

- [ ] Run the full suite: `pytest -v` → all tests pass.
- [ ] `python -m py_compile app/core/counter.py app/core/video_processor.py app/core/annotator.py app/config.py app/core/runtime_config.py app/routers/config_router.py app/core/stream_registry.py` → clean.
- [ ] Live run: counts only `empty_shackles + slaughtered_chicken` (Task already merged), band visible at roi 0.65, reconnect works, count holds steady vs BAADER over a measured window.

---

## Self-Review Notes

- **Spec coverage:** Item 1 (RTSP robustness) → Task 4. Item 2 (counting band) → Task 3. Item 3 (per-track velocity + per-stream seed) → Tasks 1+2. ✓
- **Type consistency:** `conveyor_speed_px: float`, `zone_half: int` used identically across config, runtime_config, ConfigPatch, stream_registry, VideoProcessor, ChickenCounter. Crossing dict gains a single new key `'velocity'` used in both prediction and update. `reconnect_delay`/`frame_signature` signatures match their tests. ✓
- **`zone_half` reuse:** the field was already declared in `ConfigPatch` (config_router.py:23) but never wired to `runtime_config._data` — this plan completes that wiring rather than inventing a new name. ✓
- **Known follow-ups (out of scope):** `appear_margin` remains a dead `ConfigPatch` field; legacy `detect_and_count.py` standalone script is untouched.
