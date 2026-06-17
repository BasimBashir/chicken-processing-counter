# Undercount Fixes — Design Spec
**Date:** 2026-06-17
**Status:** Approved

## Problem

The straddle counter undercounts in three conditions:

1. **Dense clusters** — greedy nearest-neighbour matching assigns straddlers to crossings in arrival order; with 2–3 birds close together the cascade produces wrong assignments, leaving one bird unmatched (and potentially suppressed by `belt_stopped`).
2. **Belt restart / slow start** — `belt_stopped` uses a single motion threshold (default 0.4). On restart the belt accelerates slowly; pixel motion stays below the threshold for 5–15 frames, keeping `belt_stopped = True` and suppressing legitimate new crossings.
3. **Fast belt** — the ROI catch zone is fixed at `2 × zone_half = 30 px`. At 34 px/frame a bird spends ~1 frame in the zone; at higher speeds it can jump over entirely, especially if that frame is dropped.

## Scope

Three targeted fixes to `app/core/counter.py` and `app/core/video_processor.py`. No changes to routing, the inference worker, the tracker, or any external API. All new params are optional with backward-compatible defaults.

---

## Fix 1 — Optimal Assignment (dense cluster fix)

**File:** `app/core/counter.py`

Replace the greedy matching loop (lines 118–157) with `scipy.optimize.linear_sum_assignment`.

### Algorithm

1. Build cost matrix `C[i][j]` (n_straddlers × n_crossings) using the existing distance formula:
   - `predicted_cx = crossing['last_cx'] + frames_elapsed * crossing['velocity']`
   - `dist_pred = |cx_i - predicted_cx|`
   - `dist_last = |cx_i - crossing['last_cx']|`
   - `tol = sway_k * crossing['velocity']`
   - `cost = min(dist_pred, dist_last) if dist_last <= tol else dist_pred`
2. Set `C[i][j] = INF` if `crossing['cls'] != straddler['cls']` (class guard).
3. Call `linear_sum_assignment(C)` → row/col index pairs.
4. Reject any pair where `C[row][col] >= max_x_distance` (treat as unmatched).
5. Unmatched straddlers follow the existing new-crossing path (suppressed if `belt_stopped`).

`scipy` is already in `requirements.txt` (pulled by ultralytics). No new dependency.

---

## Fix 2 — Belt State Hysteresis (restart fix)

**File:** `app/core/video_processor.py`

Split the single `stop_motion_thresh` into two thresholds with a dead-band.

### State machine

```
belt_stopped = False
  → belt_stopped = True   when pixel_diff < stop_motion_thresh  for _stop_run >= 4 frames
  → belt_stopped = False  when pixel_diff > stop_resume_thresh  for _resume_run >= 2 frames
```

### New param

| Param | Default | Notes |
|---|---|---|
| `stop_resume_thresh` | `stop_motion_thresh * 3` (≈ 1.2) | Separate counter `_resume_run`; resets to 0 whenever diff drops below it |

### Wiring

- `VideoProcessor.__init__` accepts `stop_resume_thresh: float = None`; if `None`, defaults to `stop_motion_thresh * 3`.
- `apply_overrides()` exposes `stop_resume_thresh` for live retuning.
- `get_status()` adds `stop_resume_thresh` to the returned dict.
- `app/core/runtime_config.py` and `app/config.py` add `STOP_RESUME_THRESH` env var (optional, float, default `None` → auto-computed from `STOP_MOTION_THRESH`).

---

## Fix 3 — Adaptive Zone Width (fast belt fix)

**Files:** `app/core/video_processor.py`, `app/core/counter.py`

Use the per-frame belt motion measurement (already computed in `VideoProcessor._run`) to widen the ROI catch zone proportionally to measured speed.

### Belt speed estimate

`_motion` (mean pixel intensity diff on a 160×90 downscale) is a proxy for movement but is NOT in px/frame. The correct source is the per-crossing `velocity` field inside `ChickenCounter` — each active crossing already tracks its learned belt speed in real px/frame via EMA from observed bird motion.

Inside `ChickenCounter.update()`, compute before the straddler loop:
```python
_active_velocities = [c['velocity'] for c in self.active_crossings]
belt_speed_px = (sum(_active_velocities) / len(_active_velocities)
                 if _active_velocities else self.conveyor_speed_px)
```

This requires no new kwarg from `VideoProcessor` — the counter has everything it needs internally. Falls back to `conveyor_speed_px` when no crossings are active (e.g. at startup or after a long stop).

### Effective zone half

```python
effective_zone_half = max(self.zone_half, int(belt_speed_px * self.zone_speed_factor))
```

Use `effective_zone_half` in place of `self.zone_half` for the straddler detection condition only (not for display — the annotator still uses `self.zone_half` so the overlay line doesn't jump around).

### New param

| Param | Default | Notes |
|---|---|---|
| `zone_speed_factor` | `0.8` | Multiplied by measured belt speed to get dynamic zone half. At 34 px/frame: max(15, 27)=27. At 50 px/frame: max(15, 40)=40. |

- Added to `ChickenCounter.__init__`, `VideoProcessor.__init__`, `apply_overrides()`, config.

---

## Data flow summary

```
VideoProcessor._run (each frame)
  ├── compute _motion (existing)
  ├── update belt_stopped / _stop_run / _resume_run   [Fix 2]
  └── counter.update(det_info, belt_stopped=belt_stopped)

ChickenCounter.update
  ├── compute belt_speed_px from active crossing velocities  [Fix 3]
  ├── effective_zone_half = max(zone_half, speed*k)         [Fix 3]
  ├── build cost matrix                                      [Fix 1]
  └── linear_sum_assignment                                  [Fix 1]
```

---

## Parameters summary

| Param | Default | Exposed via |
|---|---|---|
| `stop_resume_thresh` | `stop_motion_thresh * 3` | env, `apply_overrides`, `get_status` |
| `zone_speed_factor` | `0.8` | env, `apply_overrides` |

All existing params (`stop_motion_thresh`, `zone_half`, `conveyor_speed_px`, `sway_k`) are unchanged.

---

## Testing

- Unit test `counter.py`: 3 straddlers, 3 crossings in scrambled order → all 3 matched correctly (not just first 2).
- Unit test hysteresis: mock pixel diffs that ramp slowly from 0 → 2 over 20 frames; assert `belt_stopped` stays True until diff exceeds `stop_resume_thresh` for 2 frames.
- Unit test adaptive zone: straddler at `roi_x - zone_half - 5` (outside static zone, inside adaptive zone at high speed) → counted; same at belt_stopped → not counted.
- Integration: run existing benchmark video and assert count is ≥ previous score (1463).
