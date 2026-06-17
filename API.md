# Slaughtered Chicken Counter — API Guide

REST + MJPEG API that detects and **counts chickens crossing a vertical line** on a
left‑to‑right conveyor belt, for images, uploaded videos, and live RTSP streams.

- **Framework:** FastAPI (Python 3.11) · **Vision:** Ultralytics YOLO + `solutions.ObjectCounter`
- **Model:** `best.pt` — a 3‑class detector: `empty_shackles`, `single_legged`, `slaughtered_chicken`
- **Default port:** `5581`
- **Interactive docs:** once running, open `http://<host>:5581/docs` (Swagger) or `/redoc`
- **Bundled dashboard (UI):** `http://<host>:5581/` (Image / Video / Stream pages)

---

## 1. What the API does

Counting is delegated to Ultralytics `solutions.ObjectCounter`:

- A **vertical counting line is placed at the horizontal center** of every frame
  (`x = width/2`, full height). This is fixed (not configurable) and matches the
  reference implementation `test.py` exactly.
- Each tracked object is counted **once, when it crosses the line**. Belt flow is
  **left → right**, which the counter reports as the **`IN`** direction.
- Counts are reported **per class**, never summed into a single total.
- Inference runs at the Ultralytics default **`imgsz=640`**, which matches a native
  640×480 camera sub‑stream (no frame resizing is performed).

### Annotated frames
Video/stream feeds return JPEG frames showing:
- **Bounding boxes** in per‑class colors with a small `#<track_id> <Class> <conf>%` label
- The **counting line** (purple, same as the reference)
- **No** in/out count HUD, no extra panels

Per‑class colors (BGR): `empty_shackles` = orange, `single_legged` = gold, `slaughtered_chicken` = green.

---

## 2. How to run it

### A. Local (development) — uvicorn

```bash
# from the repo root, with a Python env that has the deps installed
uvicorn app.main:app --host 0.0.0.0 --port 5581
# add --reload during development to auto-restart on code changes
```

`best.pt` must be in the working directory (project root). With a CUDA GPU the model
runs on the GPU automatically; otherwise it falls back to CPU.

### B. Docker (recommended for deployment) — CPU

```bash
docker compose up -d --build
# API on http://localhost:5581 ; health: curl http://localhost:5581/health
docker compose down
```

### C. Docker + NVIDIA GPU (+ automatic TensorRT)

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

The GPU compose file adds the NVIDIA device reservation. On first GPU boot the
container **auto‑builds a TensorRT engine** (see §3).

> First GPU boot builds the TRT engine (2–6 min on an RTX 3090) before `/health`
> goes green — the healthcheck `start_period` allows for this.

---

## 3. TensorRT (TRT)

For maximum GPU throughput the container can run a compiled **TensorRT engine**
instead of the `.pt` weights. A TRT engine is locked to a specific
`{GPU compute capability, TRT version, precision, imgsz, source .pt}`, so it is **built
on the target machine on first boot** and cached to a named Docker volume
(`engine_cache`); later boots reuse it.

### How it works
1. `docker-entrypoint.sh` runs on container start.
2. If `TRT_AUTO_BUILD=1` (default), it builds/loads an engine at **`TRT_IMGSZ` (default 640)**
   and promotes `MODEL_PATH` to the `.engine` for the app.
3. If anything fails (no GPU, no TensorRT, export error) it **falls back to `best.pt`**
   automatically — the API still works.

> **Important:** `TRT_IMGSZ` must match the size the counter infers at (**640**, the
> native 640×480 sub‑stream size). Only raise it if you switch to a higher‑resolution
> main stream. An engine built at the wrong size will mismatch the counter’s input.

### TRT environment variables
| Var | Default | Meaning |
|---|---|---|
| `TRT_AUTO_BUILD` | `1` | Build/use a TRT engine on boot. Set `0` to force plain `.pt` (also CPU‑safe). |
| `TRT_HALF` | `true` | FP16 precision (recommended on RTX cards). |
| `TRT_IMGSZ` | `640` | Engine input size. Keep `640` for the 640×480 sub‑stream. |

### Manual export (alternative to auto‑build)
You can also trigger an export through the API (see [`/api/export/tensorrt`](#export--api-export)).
The engine is written next to the source model. The build at `imgsz=640` matches the
counter; pass a different `imgsz` only for a hi‑res stream.

---

## 4. Configuration

Configuration is intentionally minimal — counting behavior is fixed to the reference
implementation, so there are **no detection/counting tuning knobs**.

Set via environment variables (or a `.env` file in the project root). Env var names are
the uppercased field names.

| Env var | Field | Default | Meaning |
|---|---|---|---|
| `MODEL_PATH` | `model_path` | `best.pt` | Path to the model weights (`.pt` or `.engine`). |
| `RTSP_URL` | `rtsp_url` | `""` | Default RTSP URL for the legacy single‑stream endpoint. |
| `RTSP_STREAMS` | `rtsp_streams` | `""` | JSON list of streams to auto‑start (see below). |
| `MAX_STREAMS` | `max_streams` | `10` | Max concurrent multi‑streams. |
| `API_KEY` | `api_key` | `""` | If set, requires `X-API-Key` on every `/api/*` call. Empty = open (dev). |
| `UPLOAD_DIR` | `upload_dir` | `app/uploads` | Where uploaded videos are stored. |
| `OUTPUT_DIR` | `output_dir` | `app/outputs` | Where annotated output videos are written. |

`RTSP_STREAMS` example (auto‑register on startup):
```json
[{"id":"line-1","url":"rtsp://user:pass@cam1:554/Streaming/Channels/102"},
 {"id":"line-2","url":"rtsp://user:pass@cam2:554/Streaming/Channels/102","start_counting":false}]
```

Two fields can be changed at runtime via `PATCH /api/config`: `rtsp_url` and `model_path`.

---

## 5. Authentication

- If `API_KEY` is **empty** (default), all endpoints are **open** (dev mode) and a
  warning is logged at startup.
- If `API_KEY` is **set**, every `/api/*` request must include the header
  `X-API-Key: <value>`. Missing/wrong key → **`401 Unauthorized`**.
- `/health`, `/docs`, `/redoc`, and the static dashboard are **never** authenticated.

```bash
curl -H "X-API-Key: my-secret" http://localhost:5581/api/config
```

---

## 6. Counting model & response semantics

### Classes & counts
The `counts` object always has these three keys; each value is the number of objects of
that class that have **crossed the line left→right (`IN`)** so far:

```json
{ "empty_shackles": 0, "single_legged": 0, "slaughtered_chicken": 0 }
```

There is **no `total_count`** field — sum the classes yourself if you need a total.

### Counting lifecycle
- A processor first **plays** (decodes + shows the live preview with boxes) and only
  starts counting once you **arm counting**. Counts accumulate from the moment counting
  is armed.
- `reset` (multi‑stream only) rebuilds the counter with a fresh tracker and zeroed counts.

---

## 7. Endpoint reference

Base URL: `http://<host>:5581`. All `/api/*` endpoints honor `X-API-Key` when `API_KEY` is set.
MJPEG feeds are `multipart/x-mixed-replace; boundary=frame` (use directly as an `<img>` `src`).

### Health
**`GET /health`** — liveness + GPU info. *(no auth)*
```json
{ "status": "ok", "model_path": "best.pt", "cuda_available": true, "gpu": "NVIDIA GeForce RTX 3090" }
```

### Config — `/api/config`
| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/config` | — | The current config snapshot (the fields in §4). |
| PATCH | `/api/config` | `{ "rtsp_url"?, "model_path"? }` | `{ "status": "ok", "config": {...} }` |

- Only `rtsp_url` and `model_path` are accepted; unknown fields are ignored.
- Changing `model_path` validates the model loads; failure → **`422`**.

```bash
curl -X PATCH http://localhost:5581/api/config \
  -H "Content-Type: application/json" -d '{"model_path":"best.pt"}'
```

### Image — `/api/image/detect`
**`POST /api/image/detect`** — detect + count objects in a single still.
- **Accepts:** `multipart/form-data` with `file=<image>` (JPG/PNG/BMP/WEBP).
- **Returns:** `image/jpeg` (annotated image) plus headers:

| Header | Meaning |
|---|---|
| `X-Total-Count` | total objects detected in the image |
| `X-Count-Empty-Shackles` | empty_shackles detected |
| `X-Count-Single-Legged` | single_legged detected |
| `X-Count-Slaughtered-Chicken` | slaughtered_chicken detected |

- Invalid image → **`400`**. (This endpoint is plain detection — no line crossing.)

```bash
curl -X POST http://localhost:5581/api/image/detect \
  -F "file=@/path/frame.jpg" -D headers.txt -o annotated.jpg
```

### Video — `/api/video` (uploaded files)
| Method | Path | Body / Params | Returns |
|---|---|---|---|
| POST | `/api/video/upload` | `multipart file=<video>` | `{ "session_id": "ab12cd34", "filename": "clip.mp4" }` |
| POST | `/api/video/{id}/start` | — | `{ "status": "playing" }` |
| POST | `/api/video/{id}/stop` | — | `{ "status": "stopped" }` |
| POST | `/api/video/{id}/counting/start` | — | `{ "status": "counting" }` |
| POST | `/api/video/{id}/counting/stop` | — | `{ "status": "not_counting" }` |
| GET | `/api/video/{id}/feed` | — | MJPEG stream (annotated) |
| GET | `/api/video/{id}/status` | — | Video status object (below) |
| GET | `/api/video/{id}/download` | — | `video/mp4` annotated output (`404` until ready) |

Unknown `session_id` → **`404`** `Session not found`.

**Video status object:**
```json
{
  "is_playing": true,
  "is_counting": true,
  "counts": { "empty_shackles": 8, "single_legged": 0, "slaughtered_chicken": 146 },
  "frame_num": 1500,
  "total_frames": 8997,
  "fps": 28.4,
  "is_complete": false,
  "is_stream": false,
  "error": null
}
```

**Typical video workflow**
```
POST /api/video/upload            -> { session_id }
POST /api/video/{id}/start        -> begin decoding/preview
POST /api/video/{id}/counting/start
GET  /api/video/{id}/feed         -> show MJPEG in an <img>
GET  /api/video/{id}/status       -> poll counts/progress (e.g. every 500 ms)
POST /api/video/{id}/counting/stop
POST /api/video/{id}/stop
GET  /api/video/{id}/download     -> save annotated .mp4
```

### Stream (legacy single stream) — `/api/stream`
Backward‑compatible single‑stream API. Proxies to one stream with id `default`.
| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/api/stream/start` | `{ "url"? }` (falls back to `RTSP_URL`) | `{ "status": "connected", "url": "..." }` |
| POST | `/api/stream/stop` | — | `{ "status": "disconnected" }` |
| POST | `/api/stream/counting/start` | — | `{ "status": "counting" }` |
| POST | `/api/stream/counting/stop` | — | `{ "status": "not_counting" }` |
| GET | `/api/stream/feed` | — | MJPEG stream |
| GET | `/api/stream/status` | — | see below |

- `start` with no URL and no `RTSP_URL` → **`400`**; capacity reached → **`429`**.
- `counting/*` and `feed` when no stream is active → **`400`**.

**Legacy stream status:**
```json
{ "is_connected": true, "is_counting": true,
  "counts": { "empty_shackles": 0, "single_legged": 0, "slaughtered_chicken": 0 },
  "fps": 25.0, "error": null }
```
(When no stream is connected, the same shape is returned with `is_connected:false` and no `error` key.)

### Streams (multi‑stream) — `/api/streams`
Run many RTSP streams concurrently, each with its own counter and id.
| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/streams` | — | `{ "streams": [ <stream info>, ... ] }` |
| POST | `/api/streams` | `{ "id", "url", "start_counting"? = true }` | `201` + `<stream info>` |
| DELETE | `/api/streams/{id}` | — | `{ "status": "removed", "id": "..." }` |
| GET | `/api/streams/{id}/status` | — | `<stream info>` |
| GET | `/api/streams/{id}/feed` | — | MJPEG stream |
| POST | `/api/streams/{id}/counting/start` | — | `{ "status": "counting", "id": "..." }` |
| POST | `/api/streams/{id}/counting/stop` | — | `{ "status": "not_counting", "id": "..." }` |
| POST | `/api/streams/{id}/counting/reset` | — | `{ "status": "reset", "id": "...", "counts": { ...zeros } }` |

- Duplicate id → **`409`**; over `MAX_STREAMS` → **`429`**; bad request → **`400`**; unknown id → **`404`**.

**Stream info object:**
```json
{
  "id": "line-1",
  "url": "rtsp://cam1:554/Streaming/Channels/102",
  "is_playing": true,
  "is_counting": true,
  "counts": { "empty_shackles": 12, "single_legged": 1, "slaughtered_chicken": 240 },
  "fps": 25.0,
  "error": null
}
```

```bash
# register and start counting
curl -X POST http://localhost:5581/api/streams \
  -H "Content-Type: application/json" \
  -d '{"id":"line-1","url":"rtsp://user:pass@cam:554/Streaming/Channels/102"}'

# poll its counts
curl http://localhost:5581/api/streams/line-1/status
```

### Export — `/api/export/tensorrt`
| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/api/export/tensorrt` | `{ "half"? = true, "imgsz"? = 640 }` | `{ "status": "started", "model_path": "best.pt" }` |
| GET | `/api/export/tensorrt` | — | export status (below) |

- Export runs in the background. A second POST while running → **`409`**.
- **Export status:** `{ "state": "IDLE|RUNNING|DONE|FAILED", "source_model": "...", "output_path": "...", "error": null, "elapsed_seconds": 142.3 }`

---

## 8. Embedding a live feed (frontend)

```html
<img id="feed" src="" alt="feed">
<script>
  // multi-stream feed; for video use /api/video/{id}/feed
  document.getElementById("feed").src = "/api/streams/line-1/feed?t=" + Date.now();

  async function poll() {
    const s = await (await fetch("/api/streams/line-1/status")).json();
    console.log(s.counts.slaughtered_chicken, "slaughtered");
  }
  setInterval(poll, 500);
</script>
```
If `API_KEY` is set, MJPEG `<img>` tags can’t send custom headers — terminate auth at a
reverse proxy, or keep the feed endpoints behind your own gateway.

---

## 9. Notes, gotchas & migration

- **No `total_count`** anywhere — counts are per class only.
- **Removed** vs older versions: the live‑tuning `PATCH /api/streams/{id}` and
  `PATCH /api/video/{id}` endpoints, and all detection/counting tuning config
  (confidence, nms_iou, imgsz, roi_position, conveyor_speed_px, zone_half, sway_k,
  belt‑stop params, proc_width/height, batch_*). Counting is fixed to the reference logic.
- **Counting line** is always the vertical center of the frame; flow is left→right (`IN`).
- **Frame resolution:** frames are processed at their **native** size (no resize). A
  640×480 sub‑stream is ideal (matches `imgsz=640`).
- **Codec:** H.264/H.265 RTSP sub‑streams are supported (decoded via FFmpeg, forced over
  TCP for reliability). Letterbox bars in a frame come from the **camera** (e.g. a 16:9
  sensor in a 4:3 sub‑stream) — set a 16:9 sub‑stream resolution to avoid them.
- **One counter per source:** each video session / stream owns its own `ObjectCounter`
  (independent tracker state). Concurrency is capped by `MAX_STREAMS`.
- **Resilience:** live streams auto‑reconnect with backoff; a frozen feed reconnects after
  a short delay. Counts persist across reconnects (until `reset`/`delete`).
