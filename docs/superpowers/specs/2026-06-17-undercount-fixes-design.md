# Undercount Fixes — Design Spec
**Date:** 2026-06-17
**Status:** Approved
**Calibrated from:** `mix.mp4` (9012 frames, 30 fps, 300s)

## Problem

The straddle counter undercounts in three conditions:

1. **`_stop_run` too short (primary cause)** — the current threshold of 4 frames fires on normal
   inter-bird gaps (median gap = 1 frame, max = 27 frames / 0.9s). This sets `belt_stopped = True`
   between every pair of chickens, suppressing new crossings for the next bird that enters the zone.
   Real belt stops are ≥ 30 frames (1.0s). Fix: raise threshold to 42 frames (1.4s).

2. **Dense clusters** — greedy nearest-neighbour matching assigns straddlers to crossings in
   arrival order. With 2–3 birds close together, cascading wrong assignments leave one bird
   unmatched and suppress it if `belt_stopped` happens to be True at that moment.

3. **Fast belt / narrow zone** — the ROI catch zone is fixed at `2 × zone_half = 30 px`. At
   p90 belt speed (36 px/frame) a bird spends ~1 frame in the zone; a dropped inference frame
   causes a miss. Fix: raise zone_half to 18 px and add adaptive widening at high speed.

---

## Measured values from mix.mp4

| Measurement | Value |
|---|---|
| fps | 30 |
| Stopped segments (< 0.4 motion) | 1082 total |
| — 1-4 frames (inter-bird gaps) | 1020 (94.3%) |
| — 5-30 frames | 17 (1.6%) |
| — ≥ 30 frames = true belt stops | 45 (4.2%) |
| Max inter-bird gap | 27 frames (0.90s) |
| Min true belt stop | 30 frames (1.0s) |
| Belt speed p10 | 1 px/frame |
| Belt speed median | 15 px/frame |
| Belt speed p90 | 36 px/frame |
| Belt speed max | 78 px/frame |
| Running motion p10 (for resume thresh) | 2.82 |

---

## Scope

Three targeted fixes to `app/core/counter.py` and `app/core/video_processor.py`. No changes to
routing, the inference worker, the tracker, or any external API. All new or changed params are
backward-compatible.

---

## Fix 1 — Raise `_stop_run` threshold + belt state hysteresis

**File:** `app/core/video_processor.py`

This is the highest-impact fix. Two changes:

### 1a — Raise stop threshold from 4 → 42 frames

```python
# Before
self.belt_stopped = self._stop_run >= 4

# After
self.belt_stopped = self._stop_run >= self._stop_run_thresh   # default 42
```

New constructor param `stop_run_frames: int = 42` (= 1.4s at 30fps — safely above the measured
max inter-bird gap of 27 frames, and safely below the minimum true belt stop of 30 frames).
Exposed via `apply_overrides()`, `get_status()`, and env var `STOP_RUN_FRAMES`.

### 1b — Hysteresis on resume

Split the single `stop_motion_thresh` into two thresholds so a slow-starting belt cannot
prematurely release `belt_stopped`.

```
belt_stopped = False
  → belt_stopped = True   when pixel_diff < stop_motion_thresh  for _stop_run  >= stop_run_frames
  → belt_stopped = False  when pixel_diff > stop_resume_thresh  for _resume_run >= 2 frames
```

New param `stop_resume_thresh: float = 2.82` (= p10 of running-frame motion from mix.mp4;
~7× the stop threshold — large dead-band required because running motion jumps immediately
from ~0 to 3–7, not a slow ramp).

`_resume_run` is a new counter that resets to 0 whenever `pixel_diff <= stop_resume_thresh`.

Wiring:
- `VideoProcessor.__init__` accepts both new params.
- `apply_overrides()` exposes both.
- `get_status()` includes both current values.
- `app/config.py` adds `STOP_RUN_FRAMES` (int) and `STOP_RESUME_THRESH` (float) env vars.
- `app/core/runtime_config.py` wires them through.

---

## Fix 2 — Optimal Assignment for dense clusters

**File:** `app/core/counter.py`

Replace the greedy matching loop (lines 118–157) with `scipy.optimize.linear_sum_assignment`.

### Algorithm

1. Build cost matrix `C[i][j]` (n_straddlers × n_crossings):
   - `predicted_cx = crossing['last_cx'] + frames_elapsed * crossing['velocity']`
   - `dist_pred = |cx_i - predicted_cx|`
   - `dist_last = |cx_i - crossing['last_cx']|`
   - `tol = sway_k * crossing['velocity']`
   - `cost = min(dist_pred, dist_last) if dist_last <= tol else dist_pred`
   - `C[i][j] = INF` if `crossing['cls'] != straddler['cls']` (class guard)
2. Call `linear_sum_assignment(C)` → globally optimal row/col pairs.
3. Reject any pair where `C[row][col] >= max_x_distance` (treat as unmatched).
4. Unmatched straddlers → new-crossing path (suppressed if `belt_stopped`).

`scipy` is already in `requirements.txt` (pulled by ultralytics). No new dependency.

---

## Fix 3 — Adaptive zone width for fast belt

**File:** `app/core/counter.py`

### Belt speed estimate

Use the mean of active crossing velocities (already in px/frame via EMA) as the current belt
speed estimate. No new data from `VideoProcessor` needed.

```python
# At top of ChickenCounter.update(), before the straddler loop:
_vels = [c['velocity'] for c in self.active_crossings]
belt_speed_px = sum(_vels) / len(_vels) if _vels else self.conveyor_speed_px
effective_zone_half = max(self.zone_half, int(belt_speed_px * self.zone_speed_factor))
```

Falls back to `conveyor_speed_px` when no crossings are active.

### Calibrated values

- **`zone_half` default: 18 px** (raised from 15 — guarantees a bird at p90 speed of 36 px/frame
  is caught in at least 1 frame, since zone width = 36 px ≥ p90 speed).
- **`zone_speed_factor` default: 1.20** — at median speed 15 px/frame: `max(18, 18) = 18`;
  at p90 speed 36 px/frame: `max(18, 43) = 43`. Scales the zone to the actual belt speed.

`effective_zone_half` is used for the straddler detection condition only. The annotator and
overlay continue to use `self.zone_half` (static) so the displayed ROI band doesn't jump.

### New param

| Param | Old default | New default | Notes |
|---|---|---|---|
| `zone_half` | 15 | **18** | Raised based on p90 belt speed measurement |
| `zone_speed_factor` | — (new) | **1.20** | `effective_zone_half = max(zone_half, speed * factor)` |

`zone_speed_factor` added to `ChickenCounter.__init__`, `VideoProcessor.__init__`,
`apply_overrides()`, and config.

---

## Data flow summary

```
VideoProcessor._run (each frame)
  ├── compute _motion (existing)
  ├── update _stop_run / _resume_run → belt_stopped   [Fix 1 — 42-frame threshold + hysteresis]
  └── counter.update(det_info, belt_stopped=belt_stopped)

ChickenCounter.update
  ├── compute belt_speed_px from active crossing velocities  [Fix 3]
  ├── effective_zone_half = max(zone_half, speed * factor)  [Fix 3]
  ├── build cost matrix C[straddlers × crossings]           [Fix 2]
  └── linear_sum_assignment(C) → optimal 1-to-1 matching   [Fix 2]
```

---

## Parameters summary

| Param | Old | New default | Env var |
|---|---|---|---|
| `stop_run_frames` | hardcoded 4 | **42** | `STOP_RUN_FRAMES` |
| `stop_resume_thresh` | none | **2.82** | `STOP_RESUME_THRESH` |
| `zone_half` | 15 | **18** | `ZONE_HALF` (existing) |
| `zone_speed_factor` | none | **1.20** | `ZONE_SPEED_FACTOR` |

All other params (`stop_motion_thresh`, `conveyor_speed_px`, `sway_k`, `max_x_distance`) unchanged.

---

## Testing

- **Unit — optimal assignment:** 3 straddlers + 3 crossings in scrambled proximity order → all 3
  matched correctly (greedy would mis-assign the middle pair).
- **Unit — belt hysteresis:** simulate pixel diffs: 30 frames at 0.05 (stopped) → 42 frames
  triggers `belt_stopped=True`; then 1 frame at 1.5 (below resume) → still stopped; then 2
  frames at 3.0 (above resume) → `belt_stopped=False`.
- **Unit — inter-bird gap not triggering stop:** 10 frames at 0.05 (gap) → `belt_stopped` stays
  False (threshold is 42 frames, gap is only 10).
- **Unit — adaptive zone:** straddler at `roi_x - zone_half - 3` (outside static 18px zone, inside
  adaptive zone at belt_speed=40 px/frame) → counted; same straddler when belt_stopped → not counted.
- **Integration:** run benchmark video → count ≥ 1463 (current best).
