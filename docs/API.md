# Chicken Counter — API Reference

HTTP API for the 3-class slaughtered-chicken counting service. Built for
left-to-right conveyor belts: objects are counted as their bounding box crosses
a vertical ROI line (a counting **band**, configurable width).

- **Base URL:** `http://<host>:5581`
- **Interactive docs (live, auto-generated):** `http://<host>:5581/docs` (Swagger UI) · `http://<host>:5581/redoc`
- **Content type:** JSON unless noted. Image/feed endpoints return binary.
- **Classes:** `empty_shackles`, `single_legged`, `slaughtered_chicken`

> **Counting scope:** the headline `total_count` is **`slaughtered_chicken` only** — it is meant to match the BAADER weight counter, which never counts empty shackles. `empty_shackles` and `single_legged` are still detected and reported as their own per-class entries in `counts`, but are **not** summed into `total_count`.

---

## 1. Running the service

### Pull the image

Published on Docker Hub as **[`basim123/chicken-counter`](https://hub.docker.com/r/basim123/chicken-counter)**. One image runs on both CPU and GPU hosts; it auto-detects CUDA at startup.

```bash
docker pull basim123/chicken-counter:latest
```

### Run — GPU host (recommended)

```bash
docker run -d --name chicken-counter --gpus all -p 5581:5581 \
  -v engine_cache:/app/engine_cache \
  -e RTSP_STREAMS='[{"id":"line-1","url":"rtsp://user:pass@cam-ip:554/Streaming/Channels/802","start_counting":true}]' \
  basim123/chicken-counter:latest
```

On first boot on a new GPU it builds a **TensorRT engine** for that card (2–6 min on an RTX 3090) and caches it to the `engine_cache` volume; later boots reuse it. Disable with `-e TRT_AUTO_BUILD=0` (falls back to the `.pt` model, also CPU-safe).

### Run — CPU host

```bash
docker run -d --name chicken-counter -p 5581:5581 \
  -e TRT_AUTO_BUILD=0 \
  basim123/chicken-counter:latest
```

### docker-compose

```bash
docker compose up -d            # CPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d   # GPU
```

### Health check

```bash
curl http://localhost:5581/health
```

---

## 2. Authentication

If the `API_KEY` env var is set on the container, **every `/api/*` request** must send it as a header:

```
X-API-Key: <your key>
```

Missing/wrong key → `401 {"detail": "Missing or invalid X-API-Key header"}`.
If `API_KEY` is empty (default/dev), endpoints are **open** (a startup warning is logged). `/health` and the static UI never require a key.

```bash
curl -H "X-API-Key: $KEY" http://localhost:5581/api/streams
```

---

## 3. Endpoint summary

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness + CUDA/GPU info |
| GET | `/api/config` | Read all runtime config values |
| PATCH | `/api/config` | Update config **defaults** (applies to *future* streams) |
| POST | `/api/image/detect` | Detect on one uploaded image (returns annotated JPEG) |
| **GET** | **`/api/streams`** | **List all streams with per-stream counts/status** |
| **POST** | **`/api/streams`** | **Register + start a stream (per-stream params)** |
| **PATCH** | **`/api/streams/{id}`** | **Live-retune a running stream (no count reset)** |
| DELETE | `/api/streams/{id}` | Stop + remove a stream |
| GET | `/api/streams/{id}/status` | One stream's counts/status |
| GET | `/api/streams/{id}/feed` | MJPEG annotated video feed |
| POST | `/api/streams/{id}/counting/start` | Arm counting |
| POST | `/api/streams/{id}/counting/stop` | Pause counting (keeps counts) |
| POST | `/api/streams/{id}/counting/reset` | Zero this stream's counts |
| POST | `/api/stream/start` | *(legacy)* start/replace the single `default` stream |
| POST | `/api/stream/stop` | *(legacy)* stop the `default` stream |
| GET | `/api/stream/status` | *(legacy)* `default` stream status |
| GET | `/api/stream/feed` | *(legacy)* `default` stream MJPEG feed |
| POST | `/api/stream/counting/start` · `/stop` | *(legacy)* `default` counting controls |
| POST | `/api/video/upload` | Upload a video file → session |
| POST | `/api/video/{sid}/start` · `/stop` | Play / pause a video session |
| POST | `/api/video/{sid}/counting/start` · `/stop` | Counting controls for a session |
| GET | `/api/video/{sid}/status` | Session status |
| GET | `/api/video/{sid}/feed` | Session MJPEG feed |
| GET | `/api/video/{sid}/download` | Download annotated (H.264) result |
| POST | `/api/export/tensorrt` | Start a TensorRT export job |
| GET | `/api/export/tensorrt` | Export job status |

> **New integrations should use `/api/streams/*`** (multi-stream). The `/api/stream/*` (singular) routes are a backward-compatible wrapper over a single stream with id `default`, used by the bundled dashboard.

---

## 4. Live parameters (the important part)

These are the parameters you can tune. There are **two layers**:

- **`PATCH /api/config`** — changes the **defaults**. Takes effect for **streams started afterwards**. Does *not* retune an already-running stream.
- **`PATCH /api/streams/{id}`** — retunes a **running** stream **immediately** (next processed frame), **without resetting counts**. Also accepts per-stream values at registration via `POST /api/streams`.

### Parameter table

| Field | Type | Default | Range / rule | What it does |
|-------|------|---------|--------------|--------------|
| `roi_position` | float | `0.65` | `0 < v < 1` | Counting-line X as a fraction of frame width. Higher = further right. Put it where birds are clearly separated. |
| `confidence` | float | `0.25` | `0 < v < 1` | Min YOLO score for `single_legged`/`slaughtered_chicken`. Lower = catch more (fewer misses) but more false positives. |
| `conf_empty_shackles` | float | `0.45` | `0 < v < 1` | Min score for the `empty_shackles` class only (tuned separately). |
| `nms_iou` | float | `0.45` | `0 < v < 1` | Agnostic-NMS overlap for merging duplicate boxes. Lower = more aggressive merge (risk: a chicken beside a shackle gets suppressed). |
| `imgsz` | int | `1280` | multiple of 32 | Inference resolution. Higher = better small-object accuracy, slower. Model trained at 1280. |
| `conveyor_speed_px` | float | `34.0` | `> 0` | Belt travel per processed frame (px). Seeds the per-track velocity estimator, which then **self-tunes** from motion. ~34 at the 1280-wide sub-stream. |
| `zone_half` | int | `15` | `0 ≤ v ≤ 200` | Half-width (px) of the counting band around the line. Wider tolerates bbox flicker / frame stutter. `0` = single-pixel tripwire. |
| `max_distance` | int | `90` | `≥ 1` | **Overlay #ID tracker only — no effect on the count.** Max px an on-screen ID may jump between frames before it's a new object. |
| `max_disappeared` | int | `2` | `≥ 1` | **Overlay #ID tracker only — no effect on the count.** Frames an on-screen ID survives without a match before being dropped. |

> `PATCH /api/streams/{id}` rejects unknown field names with `422` (so typos fail loudly). `roi_position` retunes the line using the stream's live frame width.

### PATCH `/api/streams/{id}` — live retune

**Request** (any subset of the fields above):

```bash
curl -X PATCH http://localhost:5581/api/streams/line-1 \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $KEY" \
  -d '{"zone_half": 25, "conveyor_speed_px": 40, "roi_position": 0.62}'
```

**Response `200`** — echoes only the fields actually applied:

```json
{
  "status": "updated",
  "id": "line-1",
  "applied": { "zone_half": 25, "conveyor_speed_px": 40, "roi_position": 0.62 }
}
```

**Empty body / nothing to change → `200`:**

```json
{ "status": "no_change", "id": "line-1", "applied": {} }
```

**Errors:** `404` unknown stream id · `422` invalid value or unknown field name.

### PATCH `/api/config` — change defaults

```bash
curl -X PATCH http://localhost:5581/api/config \
  -H "Content-Type: application/json" -H "X-API-Key: $KEY" \
  -d '{"zone_half": 20, "confidence": 0.3}'
```

**Response `200`:**

```json
{ "status": "ok", "config": { /* full updated config snapshot */ } }
```

`{"status": "no_change", ...}` if the body resolves to nothing. `422` on invalid values. Accepts the same detection/counting fields plus infra fields (see `GET /api/config`).

---

## 5. Streams API (detail)

### GET `/api/streams`

```json
{
  "streams": [
    {
      "id": "line-1",
      "url": "rtsp://user:pass@cam-ip:554/Streaming/Channels/802",
      "is_playing": true,
      "is_counting": true,
      "counts": { "empty_shackles": 12, "single_legged": 1, "slaughtered_chicken": 980 },
      "total_count": 980,
      "fps": 24.7,
      "dropped_frames": 0,
      "error": null
    }
  ]
}
```

Field notes:
- `total_count` = `slaughtered_chicken` only (BAADER-comparable). Read the other classes from `counts`.
- `fps` = actual processing FPS (not source FPS).
- `dropped_frames` = frames skipped due to inference-queue backpressure.
- `error` = last transient error string, e.g. `"Stream lost; reconnecting in 4s"`, or `null` when healthy. The stream **auto-reconnects** with backoff and **preserves counts** across drops.

### POST `/api/streams`

**Request** — `id` and `url` required; every parameter from §4 is an optional per-stream override:

```json
{
  "id": "line-1",
  "url": "rtsp://user:pass@cam-ip:554/Streaming/Channels/802",
  "start_counting": true,
  "roi_position": 0.65,
  "zone_half": 15,
  "conveyor_speed_px": 34
}
```

**Response `201`** — same object shape as a `streams[]` entry above.

**Errors:** `409` id already exists · `429` stream cap reached (`MAX_STREAMS`, default 10) · `400` bad url/params.

### GET `/api/streams/{id}/status`

Same object shape as a `streams[]` entry. `404` if unknown.

### GET `/api/streams/{id}/feed`

`200` MJPEG stream — `multipart/x-mixed-replace; boundary=frame`, JPEG frames with bounding boxes, the ROI band, #IDs and crossing flashes drawn on. Embed directly:

```html
<img src="http://localhost:5581/api/streams/line-1/feed">
```

### Counting controls

```bash
curl -X POST .../api/streams/line-1/counting/start   # {"status":"counting","id":"line-1"}
curl -X POST .../api/streams/line-1/counting/stop    # {"status":"not_counting","id":"line-1"}
curl -X POST .../api/streams/line-1/counting/reset    # {"status":"reset","id":"line-1","counts":{...:0}}
```

`stop` pauses counting but **keeps** the running total; `reset` zeroes it.

### DELETE `/api/streams/{id}`

```json
{ "status": "removed", "id": "line-1" }
```

---

## 6. Legacy single-stream API (`/api/stream/*`)

Wraps one stream with the fixed id `default`. Used by the bundled UI; handy for a one-camera setup. To live-retune it, PATCH `/api/streams/default`.

**POST `/api/stream/start`** — body `{"url": "rtsp://..."}` (or omit to use the `RTSP_URL` env). → `{"status":"connected","url":"..."}`.

**GET `/api/stream/status`** — note the **different shape** from the multi-stream status:

```json
{
  "is_connected": true,
  "is_counting": true,
  "counts": { "empty_shackles": 12, "single_legged": 1, "slaughtered_chicken": 980 },
  "total_count": 980,
  "fps": 24.7,
  "error": null
}
```

When no `default` stream exists, returns `is_connected:false` with zeroed counts (not a 404).

**POST `/api/stream/stop`** → `{"status":"disconnected"}`. **`/api/stream/counting/start|stop`** → `{"status":"counting"|"not_counting"}`. **GET `/api/stream/feed`** → MJPEG.

---

## 7. Single-image detection

### POST `/api/image/detect`

`multipart/form-data` with field `file` = an image.

```bash
curl -X POST http://localhost:5581/api/image/detect \
  -H "X-API-Key: $KEY" -F "file=@frame.jpg" -o annotated.jpg -D headers.txt
```

**Response:** `image/jpeg` (annotated). Counts come back in **headers** (image total is all classes detected, since there's no ROI line on a still image):

| Header | Meaning |
|--------|---------|
| `X-Total-Count` | total detections (all classes) |
| `X-Count-Empty-Shackles` | empty_shackles count |
| `X-Count-Single-Legged` | single_legged count |
| `X-Count-Slaughtered-Chicken` | slaughtered_chicken count |

`400` on an undecodable image.

---

## 8. Video file sessions (`/api/video/*`)

For analysing an uploaded clip (not live).

- **POST `/api/video/upload`** (`file` multipart) → `{"session_id":"ab12cd34","filename":"clip.mp4"}`. Uses current config defaults.
- **POST `/api/video/{sid}/start`** → `{"status":"playing"}` · **`/stop`** → `{"status":"stopped"}`
- **POST `/api/video/{sid}/counting/start`|`/stop`** → `{"status":"counting"|"not_counting"}`
- **GET `/api/video/{sid}/status`** → `{is_playing, is_counting, counts, total_count, frame_num, total_frames, fps, is_complete, is_stream, dropped_frames, error}`
- **GET `/api/video/{sid}/feed`** → MJPEG
- **GET `/api/video/{sid}/download`** → `video/mp4` (H.264, re-encoded on first request). `404` if not ready.

`404 {"detail":"Session not found"}` for an unknown `sid`.

---

## 9. Config endpoint (full)

### GET `/api/config`

Returns the **entire** runtime snapshot (defaults + any overrides). Detection/counting fields are in §4; the rest:

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `rtsp_url` | str | `""` | Default URL for the legacy single stream |
| `model_path` | str | `"best.pt"` | Active model (promoted to `best.engine` when TRT auto-build runs) |
| `rtsp_streams` | str(JSON) | `""` | Declarative auto-start stream list (read at boot) |
| `max_streams` | int | `10` | Hard cap on concurrent streams |
| `batch_max` | int | `32` | Max frames per batched forward pass |
| `batch_window_ms` | int | `10` | Max wait to fill a batch |
| `inference_queue_max` | int | `400` | Backpressure cap before frames are dropped |
| `upload_dir` / `output_dir` | str | `app/uploads` / `app/outputs` | Filesystem paths |
| `api_key` | str | `""` | Empty = auth disabled |

> Infra fields (`batch_*`, `inference_queue_max`, `max_streams`, paths, `model_path`) are **start-time** — set them via env at container launch, not live PATCH.

---

## 10. TensorRT export job (`/api/export/*`)

Manual export trigger (separate from the entrypoint's first-boot auto-build).

- **POST `/api/export/tensorrt`** — body `{"half": true}` → `{"status":"started","model_path":"best.pt"}`. `409` if an export is already `RUNNING`.
- **GET `/api/export/tensorrt`** →

```json
{
  "state": "RUNNING",
  "source_model": "best.pt",
  "output_path": null,
  "error": null,
  "elapsed_seconds": 73.4
}
```

`state` ∈ `IDLE | RUNNING | DONE | FAILED`. `output_path` set when `DONE`; `error` set when `FAILED`; `elapsed_seconds` present once finished.

---

## 11. Configuration via environment variables

Any config field is settable as an UPPERCASE env var at container launch (e.g. `ZONE_HALF=20`, `CONVEYOR_SPEED_PX=40`, `CONFIDENCE=0.3`, `ROI_POSITION=0.65`). Container-specific:

| Env | Default | Purpose |
|-----|---------|---------|
| `RTSP_STREAMS` | `""` | JSON list of `{id,url,...overrides}` auto-started at boot |
| `API_KEY` | `""` | Enable `X-API-Key` auth |
| `TRT_AUTO_BUILD` | `1` | Build a TensorRT engine for the host GPU on first boot; `0` to disable |
| `TRT_HALF` | `true` | FP16 engine (vs FP32) |

`RTSP_STREAMS` example:

```bash
-e RTSP_STREAMS='[{"id":"line-1","url":"rtsp://user:pass@cam:554/Streaming/Channels/802","start_counting":true,"zone_half":20,"conveyor_speed_px":40}]'
```

---

## 12. Quick recipes

**Add a stream, watch it, tune the band live:**

```bash
KEY=...; BASE=http://localhost:5581
# 1. register + start counting
curl -X POST $BASE/api/streams -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"id":"line-1","url":"rtsp://user:pass@cam:554/Streaming/Channels/802","start_counting":true}'
# 2. poll the count (slaughtered = total_count)
curl -H "X-API-Key: $KEY" $BASE/api/streams/line-1/status
# 3. widen the counting band on the fly (counts preserved)
curl -X PATCH $BASE/api/streams/line-1 -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"zone_half": 25}'
```

**Embed the live feed in a web app:**

```html
<img src="http://localhost:5581/api/streams/line-1/feed" alt="Line 1">
```

**Poll loop (pseudo):** `GET /api/streams/{id}/status` every ~500 ms → read `total_count` (birds), `counts.empty_shackles`, `fps`, `error`.
