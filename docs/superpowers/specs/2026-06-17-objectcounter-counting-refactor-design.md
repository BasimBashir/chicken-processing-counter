# ObjectCounter Counting Refactor — Design

**Date:** 2026-06-17
**Status:** Approved (design); pending implementation plan
**Author:** pairing session

## 1. Problem

The current counting core is an elaborate custom stack — a straddle/virtual-tripwire
`ChickenCounter`, a `CentroidTracker`, belt-stop hysteresis, sway/zone adaptive tuning,
a batched `InferenceWorker`, and ~15 runtime-tunable knobs. It is fragile, hard to
calibrate (see the churn in recent commits), and the maintainer reports it produces
**wrong counts**.

A 36-line reference script (`test.py`) using `ultralytics.solutions.ObjectCounter`
produces **correct** counts on the maintainer's validation clips. It uses the **same
model** the API already loads (`best.pt`, a 3-class detector: `empty_shackles`,
`single_legged`, `slaughtered_chicken`).

This refactor replaces the custom counting brain with `ObjectCounter`, while keeping
the API surface other developers depend on (video / stream / streams / image endpoints).

## 2. Goals & non-goals

**Goals**
- Counting logic is **exactly** test.py's `ObjectCounter` usage: same model, vertical
  center counting line, `show=False`, `verbose=False`, **no** `conf`/`iou`/`imgsz`
  overrides, **no** pre-resize. "Simple but accurate."
- Keep the video, stream (legacy), streams (multi), and image endpoints working with
  response shapes as close to today as the new method allows.
- Per-class counts reported **separately**; **no summed total**.
- Annotated MJPEG frames show **bounding boxes only**, using the **existing** per-class
  colors and label size — no counting line, no in/out HUD, no count panel.
- Delete the now-dead custom counting/tracking/tuning code.

**Non-goals**
- No model retraining or change of weights.
- No new counting features beyond what `ObjectCounter` provides.
- No attempt to preserve the deleted live-tuning knobs.

## 3. Approach

**Swap the brain, keep the API skin.** Keep the routers, the multi-stream registry, the
per-source capture threads, and the MJPEG plumbing. Replace the per-source counting/
inference internals with a thin wrapper around one `ObjectCounter` instance.

Rejected alternatives: a full ground-up rewrite (needless contract churn for other devs);
keeping the old counter behind a config flag (maintainer wants it gone).

## 4. Detailed design

### 4.1 Counting core — `app/core/video_processor.py` (rewrite)

Each `VideoProcessor` owns **one** `ObjectCounter`, constructed per test.py once the
source's frame size is known:

```python
from ultralytics import solutions
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
line_points = [(w // 2, 0), (w // 2, h)]   # vertical line, left→right flow
counter = solutions.ObjectCounter(model=model_path, region=line_points,
                                  show=False, verbose=False)
```

Per frame, when counting is armed:

```python
results = counter.process(frame)   # runs detection + tracking + line-crossing count
```

- **No pre-resize** of frames; **no** `imgsz`/`conf`/`iou` passed — defaults, like test.py.
- One `ObjectCounter` **per session/stream** (each carries its own tracker state — they
  cannot be shared).
- The shared batched `InferenceWorker` is **removed**: `ObjectCounter` runs its own
  `model.track` internally, which is incompatible with funneling frames through a batch
  queue. Each capture thread calls `counter.process` directly.
- When counting is **not** armed (preview before "start counting"), run a plain detection
  pass for the overlay (`detector.detect_frame`) without touching `counter`, so counts
  stay at zero until armed. (Matches today's preview-vs-count split.)

Retained from the current `VideoProcessor`: the threading model, play/stop and
counting/stop controls, FPS measurement, file-vs-stream detection, RTSP reconnect
backoff, `save_raw_path` writer + `reencode_h264`. **Removed:** belt-stop motion
detection, frozen-frame detection, per-class confidence filtering, the inference queue /
`dropped_frames`, and every tuning attribute.

### 4.2 Counts & output contract

- **Per-class counts** come from `results.classwise_count` (per ultralytics docs:
  `{class_name: {"IN": n, "OUT": n}, ...}`). The belt is one-way left→right, so the
  reported count per class is its **`IN`** value:

  ```python
  counts = {cls: results.classwise_count.get(cls, {}).get("IN", 0) for cls in CLASSES}
  ```

  `CLASSES = ["empty_shackles", "single_legged", "slaughtered_chicken"]` moves out of the
  deleted `counter.py` into a **new leaf module `app/core/classes.py`**, imported by
  `video_processor.py`, `stream.py`, and `streams.py` (which import `CLASSES` today).

- **`total_count` is removed everywhere** it appears: `VideoProcessor.get_status`,
  `VideoProcessor.total_count` property, `StreamInfo`, `stream_registry.info`,
  legacy `/api/stream/status`, `/api/streams` list + `{id}/status` (`_info_to_dict`),
  and the streams `counting/reset` response.

- **Status dict** (`get_status`) becomes:
  ```json
  {"is_playing", "is_counting", "counts", "frame_num", "total_frames",
   "fps", "is_complete", "is_stream", "error"}
  ```
  Removed: `total_count`, `belt_stopped`, `stop_motion_thresh`, `stop_run_frames`,
  `stop_resume_thresh`, `dropped_frames`.

- **Reset** (`/api/streams/{id}/counting/reset`): re-create the stream's `ObjectCounter`
  (fresh tracker + zeroed counts) rather than calling the deleted `counter.reset()`.
  Response: `{"status": "reset", "id", "counts": {cls: 0}}`.

### 4.3 Annotation — `app/core/annotator.py` (trim)

Annotated frames are **bbox-only**, drawn with the **existing** styling so colors/label
size match today:

- After `counter.process(frame)`, read the per-object tracked data the solution extracted
  this frame — `counter.boxes` (xyxy), `counter.clss` (class indices), `counter.track_ids`,
  `counter.confs`, and `counter.names`. (**Exact attribute names pinned against the
  installed ultralytics in Docker during implementation**; fall back to a `model.track`
  pass only if these are unavailable.)
- Draw each with the existing `draw_bbox` (existing `CLASS_COLORS`, label font scale 0.32,
  `"#<id> <Class> <conf>%"`).
- **Not drawn:** counting line, in/out HUD, crossing flash, count panel.
- `annotate_detections` is replaced by a simpler `annotate_boxes(frame, boxes, classes,
  track_ids, confs, names)`. Keep `draw_bbox`, `CLASS_COLORS`, `CLASS_LABELS`, and
  `annotate_image_detections` (+ its `draw_rounded_rect`). Remove `draw_roi_line`,
  `draw_crossing_flash`, and the ROI/flash code paths.

When counting is not yet armed, annotate from the plain detection pass (no track IDs).

### 4.4 Config slimming — `config.py`, `runtime_config.py`, `config_router.py`

- **Keep:** `rtsp_url`, `model_path`, `upload_dir`, `output_dir`, `rtsp_streams`,
  `max_streams`, `api_key`.
- **Remove:** `roi_position`, `confidence`, `conf_empty_shackles`, `nms_iou`, `imgsz`,
  `max_distance`, `max_disappeared`, `conveyor_speed_px`, `zone_half`, `sway_k`,
  `stop_motion_thresh`, `stop_run_frames`, `stop_resume_thresh`, `zone_speed_factor`,
  `proc_width`, `proc_height`, `batch_max`, `batch_window_ms`, `inference_queue_max`.
- `config_router.ConfigPatch` → `{rtsp_url?, model_path?}`. `model_path` swap still
  validated via `get_model`. `GET /api/config` returns the slimmed snapshot.

### 4.5 Routers

- **`streams.py`**: `StreamCreate` → `{id, url, start_counting=True}` (all tuning fields +
  validators removed). **Remove `PATCH /api/streams/{id}`** and the `StreamUpdate` schema
  (only tuned now-deleted params). `register_stream` passes no overrides. Status/list lose
  `total_count`.
- **`video.py`**: **Remove `PATCH /api/video/{session_id}`** (imported `StreamUpdate` is
  gone). `upload` builds `VideoProcessor(source, model_path, save_raw_path, is_stream=False)`
  with no tuning args. Feed/status/download/start/stop/counting endpoints unchanged in shape
  (status minus `total_count`).
- **`stream.py`** (legacy): unchanged behavior; status loses `total_count`
  (`{is_connected, is_counting, counts, fps, error}`).
- **`stream_registry.py`**: drop `_OVERRIDE_KEYS`, `_merge_overrides`, `_resolve_roi_x`, the
  tuning passthrough, and `total_count` from `StreamInfo`. `register` constructs the slim
  `VideoProcessor`.

### 4.6 Image endpoint — `app/routers/image.py`

`ObjectCounter` is line-crossing/tracking and does not apply to a single still. Keep this
endpoint as **plain detection**:

- Repoint from the deleted worker to `detector.detect_frame(get_model(snap["model_path"]),
  frame)` (defaults; no tuning args).
- Annotate with `annotate_image_detections` (unchanged).
- Headers **unchanged**: `X-Count-Empty-Shackles`, `X-Count-Single-Legged`,
  `X-Count-Slaughtered-Chicken`, **and `X-Total-Count` kept** (it is an
  objects-in-image detection count, not a crossing total).

### 4.7 Export endpoint — `app/routers/export_router.py`

`imgsz` leaves the config, so `ExportRequest` gains `imgsz: int = 1280` (training size);
the endpoint uses `body.imgsz` instead of `snap["imgsz"]`. `exporter.py` unchanged.

### 4.8 Files

- **Create:** `app/core/classes.py` (the `CLASSES` constant).
- **Delete:** `app/core/counter.py`, `app/core/tracker.py`, `app/core/inference_worker.py`,
  `tests/test_sway_and_proc.py`.
- **Rewrite:** `app/core/video_processor.py`, `app/config.py`, `app/core/runtime_config.py`,
  `app/routers/config_router.py`, `app/routers/streams.py`, `app/routers/stream.py`,
  `app/routers/video.py`, `app/core/stream_registry.py`, `app/core/annotator.py`,
  `app/routers/image.py`, `app/routers/export_router.py`, `app/main.py` (drop
  `start_worker`/`stop_worker`).
- **Keep:** `app/core/auth.py`, `app/core/model_cache.py`, `app/core/detector.py`,
  `app/routers/health_router.py`, `app/core/exporter.py`.
- **Leave in place (not ours to delete):** `tools/analyze_video.py`,
  `tools/analyze_video2.py` — now obsolete (they calibrate removed belt-stop/zone params);
  flagged for the maintainer to remove.

### 4.9 Static dashboard — `app/static/{video,stream}.html`, `app/static/js/video.js`

Minimal edits: remove controls/sliders for deleted tuning params, remove any
`total_count` display, render the three per-class counts from `counts`. Keep the feed and
the play / start-counting controls. Secondary scope — must not 500, may look plainer.

## 5. Decisions resolved

- **total_count:** removed; per-class counts only.
- **Annotation:** bbox-only, existing colors + existing label size; no line/HUD.
- **Counting line:** vertical center per test.py (drop `roi_position` 0.65).
- **Image `X-Total-Count`:** kept.
- **PATCH tune endpoints (`/streams/{id}`, `/video/{id}`):** removed.
- **`tools/`:** left in place, flagged.

## 6. Risks & tradeoffs

- **Prior benchmark conflict:** earlier notes recorded a track-ID count badly
  undercounting on the dense chain (~409 vs ~1463 straddle, BAADER truth). `ObjectCounter`
  is line-crossing + tracking (more robust than raw ID counting) and the maintainer has
  validated it, so we proceed — but accuracy on a **dense** clip must be confirmed in
  validation (§7), not just on the sparse "stopping-running.mp4".
- **No cross-stream GPU batching** and **N model instances in VRAM** (one per stream).
  Acceptable for the small `best.pt` on the RTX 3090 at the expected stream count; revisit
  if many high-res streams run at once.
- **Native-resolution processing** (no 1280×720 pre-resize): faithful to test.py, but a 4K
  RTSP source will be heavier and the model (trained at 1280) may detect differently than
  on the old resized pipeline. Optional resize can be re-added later if needed.
- **Contract changes for other devs:** `total_count` gone; PATCH tune endpoints gone; most
  config fields gone. Communicated as the cost of matching test.py.

## 7. Validation plan (Docker / GPU — local venv is broken)

1. **Pin the ultralytics surface:** in the container, confirm `results.classwise_count`
   shape and the `counter.boxes/clss/track_ids/confs/names` attribute names; adjust the
   annotation/counts mapping to match the installed version.
2. **Counting parity:** run a representative **dense** clip through the new `VideoProcessor`
   and through `test.py`; per-class `IN` counts must match test.py's `classwise_count`.
3. **Annotation:** visually confirm bbox-only output with the existing class colors/labels
   and no line/HUD.
4. **Endpoint smoke tests:** image `/detect` (headers present); video `upload → start →
   counting/start → status/feed → download`; streams `register → status → reset → delete`;
   `GET/PATCH /api/config`; `/api/export/tensorrt`. Confirm response shapes match this spec.
5. **Unit-testable pieces (no model):** `classwise_count → counts` mapping; status-dict
   shape; `annotate_boxes` drawing given synthetic boxes.

## 8. Out of scope

Model retraining, optional re-resize pipeline, restoring any removed tuning knob, and a
richer annotated overlay (line/HUD) — all deferred unless the maintainer asks.
