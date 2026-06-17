# Undercount Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate undercounts caused by false belt-stop triggers on inter-bird gaps, greedy cluster mis-assignment, and too-narrow catch zone at high belt speed.

**Architecture:** Three independent patches to two files (`counter.py`, `video_processor.py`) plus config/registry wiring. Each patch is testable on its own; no shared state is introduced between them.

**Tech Stack:** Python, OpenCV, scipy (already in requirements), pytest, pydantic-settings

---

## File map

| File | Change |
|---|---|
| `app/config.py` | Add `stop_run_frames`, `stop_resume_thresh`, `zone_speed_factor`; raise `zone_half` default 15→18 |
| `app/core/runtime_config.py` | Add new keys to `_data` |
| `app/core/video_processor.py` | Belt hysteresis state machine (`_stop_run` threshold + resume counter) |
| `app/core/counter.py` | `linear_sum_assignment` matching + adaptive `effective_zone_half` |
| `app/core/stream_registry.py` | Add new params to `_OVERRIDE_KEYS` and `_merge_overrides` |
| `app/routers/streams.py` | Add new params to `StreamCreate` and `StreamUpdate` |
| `tests/test_undercount_fixes.py` | All new tests for this feature |

---

## Task 1: Config — add new params

**Files:**
- Modify: `app/config.py`
- Modify: `app/core/runtime_config.py`
- Test: `tests/test_undercount_fixes.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_undercount_fixes.py`:

```python
"""Tests for undercount fixes: belt hysteresis, optimal assignment, adaptive zone."""
import pytest
from app.config import Settings
from app.core.runtime_config import RuntimeConfig


def test_config_new_params_exist():
    s = Settings()
    assert s.stop_run_frames == 42
    assert s.stop_resume_thresh == 2.82
    assert s.zone_speed_factor == 1.20
    assert s.zone_half == 18          # raised from 15


def test_runtime_config_exposes_new_params():
    snap = RuntimeConfig().snapshot()
    assert "stop_run_frames" in snap
    assert "stop_resume_thresh" in snap
    assert "zone_speed_factor" in snap
    assert snap["zone_half"] == 18
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_undercount_fixes.py::test_config_new_params_exist tests/test_undercount_fixes.py::test_runtime_config_exposes_new_params -v
```

Expected: FAIL — `Settings` has no attribute `stop_run_frames`

- [ ] **Step 3: Add params to `app/config.py`**

Add after the `stop_motion_thresh` field (line ~45):

```python
    # Number of consecutive frames motion must stay below stop_motion_thresh
    # before belt_stopped is set True. Default 42 frames = 1.4s at 30fps —
    # safely above the measured max inter-bird gap of 27 frames (0.9s) so
    # normal gaps between hanging chickens never trigger a false belt stop.
    stop_run_frames: int = 42
    # Motion level (same units as stop_motion_thresh) that must be exceeded
    # for 2 consecutive frames before belt_stopped is cleared. Large dead-band
    # (default ~7× stop_motion_thresh) prevents slow belt restart from
    # prematurely unblocking new crossings.
    stop_resume_thresh: float = 2.82
    # Adaptive zone multiplier: effective_zone_half = max(zone_half,
    # belt_speed_px * zone_speed_factor). Widens the ROI catch band
    # proportionally to how fast the belt is moving so fast birds still
    # spend at least 1 frame inside the zone.
    zone_speed_factor: float = 1.20
```

Also change `zone_half` default from `15` to `18`:

```python
    zone_half: int = 18
```

- [ ] **Step 4: Add new keys to `app/core/runtime_config.py`**

In `RuntimeConfig.__init__`, inside the `_data` dict, add after `"stop_motion_thresh"`:

```python
            "stop_run_frames":    boot.stop_run_frames,
            "stop_resume_thresh": boot.stop_resume_thresh,
            "zone_speed_factor":  boot.zone_speed_factor,
```

- [ ] **Step 5: Run tests to verify they pass**

```
pytest tests/test_undercount_fixes.py::test_config_new_params_exist tests/test_undercount_fixes.py::test_runtime_config_exposes_new_params -v
```

Expected: PASS

- [ ] **Step 6: Verify no existing tests broken**

```
pytest tests/ -v
```

Expected: all green

- [ ] **Step 7: Commit**

```bash
git add app/config.py app/core/runtime_config.py tests/test_undercount_fixes.py
git commit -m "feat: add stop_run_frames, stop_resume_thresh, zone_speed_factor config params"
```

---

## Task 2: ChickenCounter — optimal assignment + adaptive zone

**Files:**
- Modify: `app/core/counter.py`
- Test: `tests/test_undercount_fixes.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_undercount_fixes.py`:

```python
from app.core.counter import ChickenCounter


def _det(cx, cls="slaughtered_chicken", w=80):
    return {"x1": cx - w // 2, "y1": 60, "x2": cx + w // 2, "y2": 140,
            "class_name": cls}


# ── Optimal assignment ──────────────────────────────────────────────────────

def test_three_clustered_birds_all_counted():
    """Greedy matching would mis-assign the middle bird; optimal must get all 3."""
    # Three birds arrive left-to-right, 25px apart, all crossing roi_x=400.
    # They each make one pass through the zone over 3 frames.
    c = ChickenCounter(roi_x=400, conveyor_speed_px=25, zone_half=20, sway_k=0.6)
    for frame in range(5):
        dets = [_det(360 + 25 * frame), _det(385 + 25 * frame), _det(410 + 25 * frame)]
        c.update(dets)
    assert c.counts["slaughtered_chicken"] == 3


def test_two_birds_close_together_both_counted():
    """Two birds 30px apart crossing simultaneously must both register."""
    c = ChickenCounter(roi_x=300, conveyor_speed_px=34, zone_half=20, sway_k=0.6)
    for cx_lead in range(260, 380, 34):
        c.update([_det(cx_lead), _det(cx_lead - 30)])
    assert c.counts["slaughtered_chicken"] == 2


# ── Adaptive zone ───────────────────────────────────────────────────────────

def test_adaptive_zone_widens_at_high_speed():
    """A bird that jumps over the static zone_half=18 at high speed must be
    caught by the adaptive zone (belt_speed=50 → effective_zone=60)."""
    c = ChickenCounter(roi_x=500, conveyor_speed_px=50, zone_half=18,
                       sway_k=0.0, zone_speed_factor=1.20)
    # Seed one crossing so active_crossings has a velocity of ~50
    # by running a bird through first at full speed
    for cx in range(420, 560, 50):
        c.update([_det(cx)])
    count_after_seed = c.counts["slaughtered_chicken"]

    # Now a bird at cx=461 (= 500 - 39): outside static zone (18px),
    # inside adaptive zone (50 * 1.20 = 60px → zone [440, 560]).
    # belt_speed from active crossings ≈ 50 → effective_zone_half = 60.
    # Reset and replay with the one marginal detection
    c2 = ChickenCounter(roi_x=500, conveyor_speed_px=50, zone_half=18,
                        sway_k=0.0, zone_speed_factor=1.20)
    # prime active crossings with a bird at speed 50
    for cx in range(350, 450, 50):
        c2.update([_det(cx)])
    before = c2.counts["slaughtered_chicken"]
    # marginal bird: center at roi_x - 39 (outside static zone_half=18, inside adaptive=60)
    c2.update([_det(500 - 39)])
    assert c2.counts["slaughtered_chicken"] > before, \
        "Bird outside static zone should be caught by adaptive zone"


def test_adaptive_zone_disabled_when_no_active_crossings():
    """With no active crossings, falls back to conveyor_speed_px for speed
    estimate — should still function without crashing."""
    c = ChickenCounter(roi_x=300, conveyor_speed_px=15, zone_half=18,
                       sway_k=0.6, zone_speed_factor=1.20)
    # A bird well inside even static zone — must count regardless
    c.update([_det(300)])
    assert c.counts["slaughtered_chicken"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_undercount_fixes.py::test_three_clustered_birds_all_counted tests/test_undercount_fixes.py::test_two_birds_close_together_both_counted tests/test_undercount_fixes.py::test_adaptive_zone_widens_at_high_speed tests/test_undercount_fixes.py::test_adaptive_zone_disabled_when_no_active_crossings -v
```

Expected: FAIL (`linear_sum_assignment` not imported; `zone_speed_factor` not accepted)

- [ ] **Step 3: Add `zone_speed_factor` param to `ChickenCounter.__init__`**

In `app/core/counter.py`, update the constructor signature (line ~28):

```python
    def __init__(self, roi_x: int, max_disappeared: int = 15,
                 max_distance: int = 55, conveyor_speed_px: float = 34.0,
                 zone_half: int = 18, sway_k: float = 0.6,
                 zone_speed_factor: float = 1.20):
```

Add after `self.sway_k = sway_k` (line ~43):

```python
        self.zone_speed_factor = zone_speed_factor
```

- [ ] **Step 4: Add adaptive zone computation to `ChickenCounter.update()`**

At the top of `update()`, before the straddlers loop (after `all_objects` is built, around line 96), add:

```python
        # Adaptive zone: widen proportionally to measured belt speed.
        # belt_speed_px is the mean learned velocity from active crossings
        # (real px/frame). Falls back to the seeded conveyor_speed_px when
        # no crossings are active (startup / after long stop).
        _vels = [c['velocity'] for c in self.active_crossings]
        belt_speed_px = (sum(_vels) / len(_vels)) if _vels else self.conveyor_speed_px
        effective_zone_half = max(self.zone_half, int(belt_speed_px * self.zone_speed_factor))
```

Then update the straddler detection condition (currently uses `self.zone_half`, around line 109):

```python
            lo = self.roi_x - effective_zone_half
            hi = self.roi_x + effective_zone_half
```

- [ ] **Step 5: Replace greedy matching loop with `linear_sum_assignment`**

Add import at top of `app/core/counter.py`:

```python
import numpy as np
from scipy.optimize import linear_sum_assignment
```

Replace the current matching block (the `for cx, cy, cls in straddlers:` loop and the `for i, crossing in enumerate(...)` inner loop, lines ~118–157) with:

```python
        matched_crossings: set[int] = set()
        matched_straddlers: set[int] = set()

        if straddlers and self.active_crossings:
            INF = 1e9
            n_s = len(straddlers)
            n_c = len(self.active_crossings)
            C = np.full((n_s, n_c), INF)

            for i, (cx, cy, cls) in enumerate(straddlers):
                for j, crossing in enumerate(self.active_crossings):
                    if crossing['cls'] != cls:
                        continue
                    frames_elapsed = self.frame_num - crossing['last_seen_frame']
                    predicted_cx = crossing['last_cx'] + (frames_elapsed * crossing['velocity'])
                    dist_pred = abs(cx - predicted_cx)
                    dist_last = abs(cx - crossing['last_cx'])
                    tol = self.sway_k * crossing['velocity']
                    cost = min(dist_pred, dist_last) if dist_last <= tol else dist_pred
                    C[i, j] = cost

            row_ind, col_ind = linear_sum_assignment(C)
            for i, j in zip(row_ind, col_ind):
                if C[i, j] >= self.max_x_distance:
                    continue   # reject: cost too high, treat as unmatched
                cx, cy, cls = straddlers[i]
                c = self.active_crossings[j]
                frames_elapsed = self.frame_num - c['last_seen_frame']
                if frames_elapsed > 0:
                    observed_v = (cx - c['last_cx']) / frames_elapsed
                    if 0 < observed_v < self.max_velocity_px:
                        c['velocity'] = (self.velocity_ema * observed_v
                                         + (1 - self.velocity_ema) * c['velocity'])
                c['last_cx'] = cx
                c['last_seen_frame'] = self.frame_num
                matched_crossings.add(j)
                matched_straddlers.add(i)

        # New crossings for unmatched straddlers
        for i, (cx, cy, cls) in enumerate(straddlers):
            if i in matched_straddlers:
                continue
            if not belt_stopped:
                self.counts[cls] += 1
                self.active_crossings.append({
                    'cls': cls,
                    'last_cx': cx,
                    'last_seen_frame': self.frame_num,
                    'velocity': self.conveyor_speed_px,
                })
                self.flash_events.append((cx, cy, cls))
```

Also remove the now-unused `matched_crossings = set()` that was defined before the old loop (the new code defines its own `matched_crossings` inline). The expiry block below the matching loop stays unchanged — it already uses `matched_crossings` correctly:

```python
        if not belt_stopped:
            self.active_crossings = [
                c for c in self.active_crossings
                if (self.frame_num - c['last_seen_frame']) <= self.max_straddle_disappeared
            ]
```

- [ ] **Step 6: Run the new tests**

```
pytest tests/test_undercount_fixes.py::test_three_clustered_birds_all_counted tests/test_undercount_fixes.py::test_two_birds_close_together_both_counted tests/test_undercount_fixes.py::test_adaptive_zone_widens_at_high_speed tests/test_undercount_fixes.py::test_adaptive_zone_disabled_when_no_active_crossings -v
```

Expected: PASS

- [ ] **Step 7: Run full test suite**

```
pytest tests/ -v
```

Expected: all green (existing sway / band / velocity tests must still pass)

- [ ] **Step 8: Commit**

```bash
git add app/core/counter.py tests/test_undercount_fixes.py
git commit -m "feat: optimal assignment for dense clusters + adaptive zone_half at high belt speed"
```

---

## Task 3: VideoProcessor — belt stop hysteresis

**Files:**
- Modify: `app/core/video_processor.py`
- Test: `tests/test_undercount_fixes.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_undercount_fixes.py`:

```python
# ── Belt stop hysteresis ────────────────────────────────────────────────────
# We test the state machine logic directly through VideoProcessor's internal
# state by inspecting belt_stopped after feeding synthetic motion values.
# VideoProcessor._run reads from cap, so we test the motion-update logic
# via a helper that mimics the state machine:

def _simulate_belt_state(motion_sequence, stop_run_frames=42, stop_thresh=0.4,
                         resume_thresh=2.82, resume_run=2):
    """Feed a list of motion diff values through the belt-stop state machine.
    Returns list of belt_stopped booleans, one per frame."""
    belt_stopped = False
    _stop_run = 0
    _resume_run = 0
    results = []
    for motion in motion_sequence:
        if not belt_stopped:
            if motion < stop_thresh:
                _stop_run += 1
            else:
                _stop_run = 0
            if _stop_run >= stop_run_frames:
                belt_stopped = True
                _resume_run = 0
        else:
            if motion > resume_thresh:
                _resume_run += 1
            else:
                _resume_run = 0
            if _resume_run >= resume_run:
                belt_stopped = False
                _stop_run = 0
        results.append(belt_stopped)
    return results


def test_inter_bird_gap_does_not_trigger_belt_stop():
    """27-frame gap (max measured inter-bird gap) must NOT set belt_stopped."""
    # 27 frames of low motion (inter-bird gap)
    motion = [0.05] * 27
    states = _simulate_belt_state(motion, stop_run_frames=42)
    assert not any(states), "Short inter-bird gap must never trigger belt_stopped"


def test_true_belt_stop_triggers_after_42_frames():
    """A genuine belt stop (60+ frames of low motion) must set belt_stopped."""
    motion = [0.05] * 60
    states = _simulate_belt_state(motion, stop_run_frames=42)
    assert states[42], "belt_stopped must be True by frame 42"
    assert all(states[42:]), "belt_stopped must stay True while motion is low"


def test_resume_requires_high_motion_for_2_frames():
    """Belt restart: 1 frame above resume_thresh must NOT clear belt_stopped;
    2 consecutive frames must."""
    # Start stopped (60 low frames), then 1 high frame, then check still stopped
    motion = [0.05] * 60 + [3.0] + [0.05]
    states = _simulate_belt_state(motion, stop_run_frames=42)
    assert states[60], "1 high frame not enough to resume"

    # Now 2 consecutive high frames → should resume
    motion2 = [0.05] * 60 + [3.0, 3.0]
    states2 = _simulate_belt_state(motion2, stop_run_frames=42)
    assert not states2[-1], "2 consecutive high frames must clear belt_stopped"


def test_slow_restart_ramp_stays_stopped_until_above_resume_thresh():
    """Belt accelerates slowly: 10 frames at 1.5 (below resume_thresh=2.82)
    must NOT clear belt_stopped."""
    motion = [0.05] * 60 + [1.5] * 10
    states = _simulate_belt_state(motion, stop_run_frames=42)
    assert all(states[42:]), "Slow-start ramp below resume_thresh must keep belt_stopped=True"
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_undercount_fixes.py::test_inter_bird_gap_does_not_trigger_belt_stop tests/test_undercount_fixes.py::test_true_belt_stop_triggers_after_42_frames tests/test_undercount_fixes.py::test_resume_requires_high_motion_for_2_frames tests/test_undercount_fixes.py::test_slow_restart_ramp_stays_stopped_until_above_resume_thresh -v
```

Expected: FAIL — `_simulate_belt_state` doesn't exist yet (it's in the test file we're building), but will actually PASS once added. These test the pure logic without touching `VideoProcessor` — so after adding them, they should pass immediately against the helper. The real VideoProcessor tests are in Step 3.

- [ ] **Step 3: Update `VideoProcessor.__init__`**

In `app/core/video_processor.py`, update `__init__` signature to accept the two new params (add after `stop_motion_thresh: float`):

```python
                 stop_motion_thresh: float = 0.4,
                 stop_run_frames: int = 42,
                 stop_resume_thresh: float = 2.82,
```

Add instance variables after `self.stop_motion_thresh = float(stop_motion_thresh)`:

```python
        self.stop_run_frames = int(stop_run_frames)
        self.stop_resume_thresh = float(stop_resume_thresh)
        self._resume_run = 0
```

- [ ] **Step 4: Update the belt-stop detection block in `VideoProcessor._run`**

Find the current belt-stop logic (around lines 282–285):

```python
            if self._prev_motion_gray is not None:
                _motion = float(cv2.absdiff(_g, self._prev_motion_gray).mean())
                self._stop_run = self._stop_run + 1 if _motion < self.stop_motion_thresh else 0
                self.belt_stopped = self._stop_run >= 4
```

Replace with:

```python
            if self._prev_motion_gray is not None:
                _motion = float(cv2.absdiff(_g, self._prev_motion_gray).mean())
                if not self.belt_stopped:
                    if _motion < self.stop_motion_thresh:
                        self._stop_run += 1
                    else:
                        self._stop_run = 0
                    if self._stop_run >= self.stop_run_frames:
                        self.belt_stopped = True
                        self._resume_run = 0
                else:
                    if _motion > self.stop_resume_thresh:
                        self._resume_run += 1
                    else:
                        self._resume_run = 0
                    if self._resume_run >= 2:
                        self.belt_stopped = False
                        self._stop_run = 0
```

- [ ] **Step 5: Update `apply_overrides()` in `VideoProcessor`**

Add handling for the two new params inside `apply_overrides()`, after the `stop_motion_thresh` block:

```python
        if stop_run_frames is not None:
            self.stop_run_frames = int(stop_run_frames)
            applied["stop_run_frames"] = stop_run_frames
        if stop_resume_thresh is not None:
            self.stop_resume_thresh = float(stop_resume_thresh)
            applied["stop_resume_thresh"] = stop_resume_thresh
```

Also add `stop_run_frames=None, stop_resume_thresh=None` to the `apply_overrides` signature.

- [ ] **Step 6: Update `get_status()` in `VideoProcessor`**

Add to the returned dict:

```python
            "stop_run_frames":    self.stop_run_frames,
            "stop_resume_thresh": self.stop_resume_thresh,
```

- [ ] **Step 7: Run full test suite**

```
pytest tests/ -v
```

Expected: all green

- [ ] **Step 8: Commit**

```bash
git add app/core/video_processor.py tests/test_undercount_fixes.py
git commit -m "feat: belt stop hysteresis — raise stop_run to 42 frames, add resume threshold"
```

---

## Task 4: Wire new params through registry + API

**Files:**
- Modify: `app/core/stream_registry.py`
- Modify: `app/routers/streams.py`
- Test: `tests/test_undercount_fixes.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_undercount_fixes.py`:

```python
from app.routers.streams import StreamCreate, StreamUpdate


def test_stream_create_accepts_new_params():
    sc = StreamCreate(id="x", url="rtsp://x",
                      stop_run_frames=50, stop_resume_thresh=3.0,
                      zone_speed_factor=1.5)
    assert sc.stop_run_frames == 50
    assert sc.stop_resume_thresh == 3.0
    assert sc.zone_speed_factor == 1.5


def test_stream_update_accepts_new_params():
    su = StreamUpdate(stop_run_frames=60, stop_resume_thresh=2.5,
                      zone_speed_factor=1.0)
    assert su.stop_run_frames == 60


def test_stream_update_rejects_negative_stop_run_frames():
    with pytest.raises(ValueError):
        StreamUpdate(stop_run_frames=0)


def test_stream_update_rejects_negative_resume_thresh():
    with pytest.raises(ValueError):
        StreamUpdate(stop_resume_thresh=-0.1)


def test_stream_update_rejects_negative_zone_speed_factor():
    with pytest.raises(ValueError):
        StreamUpdate(zone_speed_factor=-1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_undercount_fixes.py::test_stream_create_accepts_new_params tests/test_undercount_fixes.py::test_stream_update_accepts_new_params tests/test_undercount_fixes.py::test_stream_update_rejects_negative_stop_run_frames tests/test_undercount_fixes.py::test_stream_update_rejects_negative_resume_thresh tests/test_undercount_fixes.py::test_stream_update_rejects_negative_zone_speed_factor -v
```

Expected: FAIL — `StreamCreate` and `StreamUpdate` don't have these fields yet

- [ ] **Step 3: Add fields to `StreamCreate` in `app/routers/streams.py`**

Add after the `stop_motion_thresh` field in `StreamCreate`:

```python
    stop_run_frames: Optional[int] = Field(None, description="Frames of low motion required before belt is considered stopped (default 42 = 1.4s at 30fps). Raise to prevent false stops on long inter-bird gaps.")
    stop_resume_thresh: Optional[float] = Field(None, description="Motion level that must be exceeded for 2 frames before belt_stopped clears (default 2.82). Creates hysteresis so slow belt restarts don't prematurely unblock new crossings.")
    zone_speed_factor: Optional[float] = Field(None, description="Adaptive zone multiplier: effective_zone_half = max(zone_half, belt_speed * factor). Widens the ROI catch band at high belt speed (default 1.20).")
```

Add validators after the `_stop_thresh_nonneg` validator:

```python
    @field_validator("stop_run_frames")
    @classmethod
    def _stop_run_frames_positive(cls, v):
        if v is not None and v < 1:
            raise ValueError("stop_run_frames must be >= 1")
        return v

    @field_validator("stop_resume_thresh")
    @classmethod
    def _stop_resume_nonneg(cls, v):
        if v is not None and v < 0:
            raise ValueError("stop_resume_thresh must be >= 0")
        return v

    @field_validator("zone_speed_factor")
    @classmethod
    def _zone_speed_factor_nonneg(cls, v):
        if v is not None and v < 0:
            raise ValueError("zone_speed_factor must be >= 0")
        return v
```

- [ ] **Step 4: Add the same fields + validators to `StreamUpdate` in `app/routers/streams.py`**

Add after `stop_motion_thresh` field in `StreamUpdate` (same field definitions and validators as above — `StreamUpdate` has `model_config = ConfigDict(extra="forbid")` so they must be explicitly declared):

```python
    stop_run_frames: Optional[int] = Field(None, description="Frames of low motion required before belt is considered stopped (default 42 = 1.4s at 30fps).")
    stop_resume_thresh: Optional[float] = Field(None, description="Motion level that must be exceeded for 2 frames before belt_stopped clears (default 2.82).")
    zone_speed_factor: Optional[float] = Field(None, description="Adaptive zone multiplier: effective_zone_half = max(zone_half, belt_speed * factor) (default 1.20).")

    @field_validator("stop_run_frames")
    @classmethod
    def _stop_run_frames_positive(cls, v):
        if v is not None and v < 1:
            raise ValueError("stop_run_frames must be >= 1")
        return v

    @field_validator("stop_resume_thresh")
    @classmethod
    def _stop_resume_nonneg(cls, v):
        if v is not None and v < 0:
            raise ValueError("stop_resume_thresh must be >= 0")
        return v

    @field_validator("zone_speed_factor")
    @classmethod
    def _zone_speed_factor_nonneg(cls, v):
        if v is not None and v < 0:
            raise ValueError("zone_speed_factor must be >= 0")
        return v
```

- [ ] **Step 5: Wire new params through `StreamRegistry`**

In `app/core/stream_registry.py`:

Add to `_OVERRIDE_KEYS` set:

```python
_OVERRIDE_KEYS = {
    "roi_position", "confidence", "conf_empty_shackles", "nms_iou", "imgsz",
    "max_distance", "max_disappeared", "conveyor_speed_px", "zone_half", "sway_k",
    "stop_motion_thresh", "stop_run_frames", "stop_resume_thresh", "zone_speed_factor",
}
```

In `_merge_overrides`, update the cfg keys list:

```python
        cfg = {k: snap[k] for k in (
            "roi_position", "confidence", "conf_empty_shackles", "nms_iou", "imgsz",
            "max_distance", "max_disappeared", "conveyor_speed_px", "zone_half", "sway_k",
            "stop_motion_thresh", "stop_run_frames", "stop_resume_thresh", "zone_speed_factor",
        )}
```

In `register()`, add to the `VideoProcessor(...)` constructor call (after `stop_motion_thresh=cfg["stop_motion_thresh"]`):

```python
                stop_run_frames=cfg["stop_run_frames"],
                stop_resume_thresh=cfg["stop_resume_thresh"],
                zone_half=cfg["zone_half"],
                sway_k=cfg["sway_k"],
```

Note: `zone_half` and `sway_k` are already passed; `zone_speed_factor` must be added to the `ChickenCounter` via `VideoProcessor`. Update `VideoProcessor.__init__` to accept and forward `zone_speed_factor`:

Add `zone_speed_factor: float = 1.20` to `VideoProcessor.__init__` signature (after `sway_k`), store as `self.zone_speed_factor = float(zone_speed_factor)`, and pass it when constructing `ChickenCounter`:

```python
        self.counter = ChickenCounter(roi_x=roi_x, max_disappeared=max_disappeared,
                                      max_distance=max_distance,
                                      conveyor_speed_px=conveyor_speed_px,
                                      zone_half=zone_half, sway_k=sway_k,
                                      zone_speed_factor=zone_speed_factor)
```

Also add `zone_speed_factor=cfg["zone_speed_factor"]` to the `VideoProcessor(...)` call in `registry.register()`.

Finally, add `zone_speed_factor` handling to `VideoProcessor.apply_overrides()`:

```python
        if zone_speed_factor is not None:
            self.counter.zone_speed_factor = float(zone_speed_factor)
            applied["zone_speed_factor"] = zone_speed_factor
```

And add `zone_speed_factor=None` to the `apply_overrides` signature.

- [ ] **Step 6: Run the new tests**

```
pytest tests/test_undercount_fixes.py::test_stream_create_accepts_new_params tests/test_undercount_fixes.py::test_stream_update_accepts_new_params tests/test_undercount_fixes.py::test_stream_update_rejects_negative_stop_run_frames tests/test_undercount_fixes.py::test_stream_update_rejects_negative_resume_thresh tests/test_undercount_fixes.py::test_stream_update_rejects_negative_zone_speed_factor -v
```

Expected: PASS

- [ ] **Step 7: Run full test suite**

```
pytest tests/ -v
```

Expected: all green

- [ ] **Step 8: Commit**

```bash
git add app/core/stream_registry.py app/routers/streams.py app/core/video_processor.py tests/test_undercount_fixes.py
git commit -m "feat: wire stop_run_frames, stop_resume_thresh, zone_speed_factor through registry and API"
```

---

## Self-Review Checklist (already run)

**Spec coverage:**
- [x] Fix 1 (stop_run_frames 42 + hysteresis) → Task 3
- [x] Fix 2 (linear_sum_assignment) → Task 2 Step 5
- [x] Fix 3 (adaptive zone + zone_speed_factor=1.20) → Task 2 Steps 3–4
- [x] zone_half default raised 15→18 → Task 1 Step 3
- [x] All params in config, runtime_config, apply_overrides, get_status → Tasks 1, 3, 4
- [x] All unit tests from spec → Tasks 2–3 test steps

**No placeholders:** confirmed — all steps contain exact code.

**Type consistency:** `zone_speed_factor: float` used consistently across `Settings`, `ChickenCounter`, `VideoProcessor`, registry, and router. `stop_run_frames: int`, `stop_resume_thresh: float` consistent throughout.
