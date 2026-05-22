# Slaughtered Chicken Counting System

A production-ready detection and counting system built with **Ultralytics YOLO** and **FastAPI**. Tracks three object classes on a **left-to-right conveyor belt** using a **vertical ROI line** — each object is counted exactly once as its centroid crosses from left to right.

Built for **multi-camera deployments**: a single container handles up to 10 concurrent RTSP streams via a shared **batched-inference worker** that funnels all model forward passes through one GPU dispatcher.

---

## Classes

| Class | Color | Description |
|-------|-------|-------------|
| `empty_shackles` | Orange | Empty shackle hooks with no chicken |
| `single_legged` | Gold | Chicken hanging by a single leg |
| `slaughtered_chicken` | Green | Fully processed slaughtered chicken |

Each class has its own **independent tracker and counter** — counts never bleed between classes. **Class-agnostic NMS** ensures the model doesn't report two boxes (e.g. `empty_shackles` + `slaughtered_chicken`) for the same physical object.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Stream RTSP** | Up to 10 simultaneous RTSP cameras in one container, each with independent counts, ROI, and tracking |
| **Batched Inference** | All streams share one `InferenceWorker` that groups frames into a single forward pass — scales sub-linearly with GPU load |
| **3-Class Independent Counting** | Separate `CentroidTracker` + count per class; no cross-class ID collisions |
| **Vertical ROI Line** | Left-to-right conveyor — vertical counting line with rightward-arrow indicators |
| **Hungarian + IoU Tracker** | Globally-optimal bbox-to-track matching via `scipy.optimize.linear_sum_assignment` |
| **API Key Auth** | `X-API-Key` header gates all `/api/*` endpoints in production; auto-disabled in dev |
| **Web Dashboard** | Dark-themed UI for image detection, video processing, and live stream monitoring |
| **Live Config** | `PATCH /api/config` — tune ROI, confidence, NMS IoU, imgsz at runtime without restart |
| **TensorRT Export** | `POST /api/export/tensorrt` — background export with `IDLE → RUNNING → DONE/FAILED` state |
| **Health Check** | `GET /health` — GPU info, model path, no auth required |
| **Docker + GPU** | Single-container deployment, healthcheck baked in, NVIDIA GPU optional |
| **CLI Tool** | Standalone `detect_and_count.py` — no web server needed |

---

## Quick Start

### Option A: Pull pre-built image from Docker Hub

Published as **[`basim123/chicken-counter`](https://hub.docker.com/r/basim123/chicken-counter)**. The image is a single artifact that runs on both CPU and GPU hosts — it auto-detects CUDA at startup, falls back to CPU when no GPU is visible, and logs the selected device.

**Available tags**
| Tag | When to use |
|-----|-------------|
| `latest` | Tracks main; updates whenever a new release is pushed |
| `2.0.0` | Pinned multi-stream + batched-inference release |
| `1.0.0` | Single-stream legacy release |

**What's bundled:** Python 3.11, PyTorch with CUDA 12.6 wheels, Ultralytics, FastAPI, OpenCV, ffmpeg, the trained `best.pt` weights, and `curl` (for the Docker healthcheck).

**Pull and run — CPU**

```bash
docker pull basim123/chicken-counter:latest

docker run -d --name chicken-counter -p 5581:5581 \
  -e API_KEY=replace-with-long-random-string \
  -v $(pwd)/uploads:/app/app/uploads \
  -v $(pwd)/outputs:/app/app/outputs \
  basim123/chicken-counter:latest
```

**Pull and run — GPU** (NVIDIA + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))

```bash
docker pull basim123/chicken-counter:latest

docker run -d --name chicken-counter --gpus all -p 5581:5581 \
  -e API_KEY=replace-with-long-random-string \
  -v $(pwd)/uploads:/app/app/uploads \
  -v $(pwd)/outputs:/app/app/outputs \
  basim123/chicken-counter:latest
```

**Pull and run — GPU + TensorRT (recommended for throughput)**

Adds `TRT_AUTO_BUILD=1` to build a host-specific TensorRT engine on first boot (2–6 min on RTX 3090, longer on smaller cards), and an `engine_cache` named volume so subsequent boots reuse it. The engine is keyed by `{GPU, TRT version, precision, imgsz, best.pt hash}` — safe to deploy the same image on any NVIDIA host.

```bash
docker pull basim123/chicken-counter:latest

docker volume create engine_cache  # one-time, persists the built engine

docker run -d --name chicken-counter --gpus all -p 5581:5581 \
  -e API_KEY=replace-with-long-random-string \
  -e TRT_AUTO_BUILD=1 \
  -v $(pwd)/uploads:/app/app/uploads \
  -v $(pwd)/outputs:/app/app/outputs \
  -v engine_cache:/app/engine_cache \
  basim123/chicken-counter:latest

# First boot: watch the build progress (2-6 min on RTX 3090)
docker logs -f chicken-counter | grep -E '^\[trt\]|MODEL_PATH'
```

Behaviour mirrors the local-developer setup exactly:
- All tuning values (`imgsz=1280`, `confidence=0.25`, `roi_position=0.60`, etc.) come from the `app/config.py` defaults baked into the image — same as a local `uvicorn` run.
- FP16 engine output may shift per-frame confidences by ~0.01–0.05 vs the `.pt` baseline; totals per video typically stay within ~1%. Set `-e TRT_HALF=false` to build FP32 if you need full `.pt` parity (slower).
- Skip the volume to test, but every container restart will rebuild the engine — fine for one-shots, painful in production.

**Pull and run — multi-stream auto-start** (declare all your cameras in one go)

```bash
docker run -d --name chicken-counter --gpus all -p 5581:5581 \
  -e API_KEY=replace-with-long-random-string \
  -e MAX_STREAMS=10 \
  -e RTSP_STREAMS='[{"id":"line-1","url":"rtsp://cam1.local/stream"},{"id":"line-2","url":"rtsp://cam2.local/stream","roi_position":0.55},{"id":"line-3","url":"rtsp://cam3.local/stream"}]' \
  -v $(pwd)/uploads:/app/app/uploads \
  -v $(pwd)/outputs:/app/app/outputs \
  basim123/chicken-counter:latest
```

**Verify it's running**

```bash
# Health (public, no auth)
curl http://localhost:5581/health

# Streams registered from RTSP_STREAMS env (needs auth)
curl -H "X-API-Key: replace-with-long-random-string" \
     http://localhost:5581/api/streams

# Container health (matches the built-in /health healthcheck)
docker inspect --format='{{.State.Health.Status}}' chicken-counter
```

Open **http://localhost:5581** for the dashboard, or **http://localhost:5581/docs** for the auto-generated Swagger UI listing every endpoint.

**Environment variables you can pass to `-e`**

All defaults below come from `app/config.py` — the container reads them via pydantic-settings, exactly like a local `uvicorn app.main:app` run. Compose passes env vars through only when set, so an unset var means the Python default applies. Container and local-uvicorn behavior are identical when neither has the var set.

| Var | Default | Purpose |
|-----|---------|---------|
| `API_KEY` | _(empty → no auth, dev only)_ | Required on every `/api/*` request as `X-API-Key` header |
| `RTSP_STREAMS` | _(empty)_ | JSON list of streams to auto-register on boot |
| `MAX_STREAMS` | `10` | Soft cap on concurrent streams |
| `RTSP_URL` | _(empty)_ | Default URL for the legacy `/api/stream/*` single-stream API |
| `MODEL_PATH` | `best.pt` | Path to YOLO weights inside the container. Promoted to `best.engine` automatically when `TRT_AUTO_BUILD=1`. |
| `TRT_AUTO_BUILD` | `0` | Opt-in TensorRT engine auto-build on first boot. Disabled by default — container runs `best.pt`. Set `1` to enable. |
| `TRT_HALF` | `true` | FP16 precision when `TRT_AUTO_BUILD=1`. Set `false` for FP32 (slower, marginally more accurate). |
| `ROI_POSITION` | `0.60` | Global ROI line as fraction of frame width |
| `CONFIDENCE` | `0.25` | Global YOLO confidence threshold |
| `CONF_EMPTY_SHACKLES` | `0.45` | Per-class confidence override for `empty_shackles` |
| `NMS_IOU` | `0.45` | Global NMS IoU threshold |
| `IMGSZ` | `1280` | Global inference image size |
| `MAX_DISTANCE` | `90` | Tracker max pixel distance |
| `MAX_DISAPPEARED` | `2` | Frames before a lost track is dropped |
| `BATCH_MAX` | `32` | Max frames per batched forward pass |
| `BATCH_WINDOW_MS` | `10` | Max time worker waits to fill a batch |
| `INFERENCE_QUEUE_MAX` | `400` | Inference backlog cap before frames are dropped |

For client examples in Python, Node.js, and other languages, see [INTEGRATION.md](INTEGRATION.md). For the full per-endpoint reference, see [REST API Reference](#rest-api-reference) below.

### Option B: Build from source with Docker Compose

```bash
# CPU
docker compose up --build

# GPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

### Option C: Local Python

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1            # Windows
# source .venv/bin/activate           # Linux/macOS
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 5581
```

---

## Production GPU Deployment (TensorRT)

> **Disabled by default.** The container ships with `TRT_AUTO_BUILD=0` so every host runs `best.pt` for predictable, `.pt`-equivalent counts. Set `TRT_AUTO_BUILD=1` to opt into the auto-build pattern below — useful when you need the 2–4× throughput boost and have validated that the `.engine`-vs-`.pt` count drift (typically <1%) is acceptable for your line.

A TensorRT `.engine` typically runs **2–4× faster** than the `.pt` model on the same GPU. The catch: an engine file is locked to the host it was built on — specifically to the `{GPU compute capability, TensorRT version, CUDA/cuDNN version, precision, imgsz, source .pt}` tuple. **Shipping a pre-built engine in the image breaks portability** across VPS hardware (3090 ≠ 4090 ≠ T4 ≠ L4, even within the same vendor).

When opted in, the container handles this with a **build-on-first-boot** pattern: ship only `best.pt`, build the engine on the target machine on first launch, cache it to a named volume, and reuse it on every subsequent boot.

### How it works

```
docker compose up           docker compose up
   (first boot)                 (every boot after)
        │                            │
        ▼                            ▼
  no cached engine?              engine present?
        │                            │
   build engine                 reuse cached engine
   (2-6 min on 3090)            (instant)
        │                            │
  symlink best.engine ──────▶ symlink best.engine
        │                            │
   exec uvicorn                 exec uvicorn
```

The cache key is composed from:
```
{GPU_name}_{compute_cap}_trt{trt_version}_{fp16|fp32}_imgsz{N}_pt{sha256_8}.engine
```
e.g. `NVIDIA_GeForce_RTX_3090_sm86_trt10.4.0_fp16_imgsz640_pt9a7e21f3.engine`.

Cached engines live in a Docker named volume (`engine_cache`) so they survive container restarts and image rebuilds. Swapping in a new `best.pt` invalidates the cache automatically via the file hash.

### Deploying to a VPS — DevOps cheat sheet

The same image runs everywhere. No per-VPS image variants needed.

```bash
# On the VPS — one-time setup
#   1. NVIDIA driver (>= 545 for TRT 10.x)
#   2. nvidia-container-toolkit
#      https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Pull and start with GPU
docker pull basim123/chicken-counter:latest
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

# First boot: container shows "starting" for 2-15 min while TRT builds.
# Watch progress:
docker compose logs -f chicken-counter | grep "^\[trt\]"

# Sample log on a fresh host:
#   [trt] gpu=NVIDIA_GeForce_RTX_4090 sm89  trt=10.4.0  prec=fp16  imgsz=640  pt=9a7e21f3
#   [trt] cache key: NVIDIA_GeForce_RTX_4090_sm89_trt10.4.0_fp16_imgsz640_pt9a7e21f3.engine
#   [trt] no cached engine — building (2-6 min on RTX 3090; longer on smaller GPUs)
#   [trt] cached engine at /app/engine_cache/NVIDIA_GeForce_RTX_4090_sm89_...
#   [trt] active engine: /app/best.engine -> NVIDIA_GeForce_RTX_4090_sm89_...
#   [startup] CUDA available: True
#   [startup] GPU: NVIDIA GeForce RTX 4090
#   [startup] MODEL_PATH=best.engine

# Subsequent boots reuse the cache and start in seconds.
```

### Force a rebuild (e.g. after upgrading TRT)

The cache key includes the TRT version, so an upgrade auto-invalidates. To force-rebuild for any other reason:

```bash
docker compose down
docker volume rm slaughtered_chicken_counting_engine_cache   # name = <project>_engine_cache
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

Or trigger a re-export at runtime without rebuilding the cache key:
```bash
curl -X POST -H "X-API-Key: $KEY" http://localhost:5581/api/export/tensorrt
```

### Skip TRT and run on `.pt`

CPU-only hosts, or GPU hosts where you want to bypass the build for any reason:

```yaml
environment:
  - TRT_AUTO_BUILD=0
```

The entrypoint detects no CUDA / no TRT / missing module automatically and falls back to `.pt` without failing the container.

### What can go wrong

| Symptom | Cause | Fix |
|---------|-------|-----|
| Healthcheck shows `unhealthy` for 10+ min | Slow GPU builds engine longer than `start_period` (900s) | Raise `start_period` further, or pre-warm the volume from a faster host with matching GPU class |
| `[trt] tensorrt module unavailable` | TRT pip wheel didn't install (cold pip cache, network) | `docker compose build --no-cache` |
| `[trt] export failed: ...` | Out-of-memory on small GPU, or driver/TRT mismatch | Drop `IMGSZ` to `512`, or upgrade host driver to match TRT version |
| Engine builds every restart | Named volume not mounted (e.g. using `docker run` without `-v engine_cache:/app/engine_cache`) | Use the compose file, or pass `-v engine_cache:/app/engine_cache` explicitly |
| `TensorRT engine load failed: serialized engine was built for compute capability X.Y` | Wrong cache used (manual copy from another host) | Delete the engine_cache volume and let it rebuild |

### Windows 11 dev host (your machine)

Docker Desktop + WSL2 + the NVIDIA driver gives you GPU-in-container the same way Linux does. The build-on-first-boot pattern works identically — first boot generates `best.engine` for your 3090; subsequent boots reuse it. The named volume lives inside the WSL2 distro, so `docker volume rm` is the way to clear it.

---

## Multi-Stream Deployment

The container can run up to **10 RTSP cameras** (configurable via `MAX_STREAMS`) concurrently. Each camera is identified by a user-supplied `id` and returns its own counts, MJPEG feed, and status independently. The shared batched-inference worker keeps GPU usage flat as you add streams.

### Auto-start streams from env

Set `RTSP_STREAMS` to a JSON list — every entry is registered and starts counting on container boot:

```env
RTSP_STREAMS='[
  {"id":"line-1","url":"rtsp://cam1.local/stream"},
  {"id":"line-2","url":"rtsp://cam2.local/stream","roi_position":0.55},
  {"id":"line-3","url":"rtsp://cam3.local/stream","confidence":0.30},
  {"id":"line-4","url":"rtsp://cam4.local/stream"}
]'
```

Per-entry overrides: `roi_position`, `confidence`, `nms_iou`, `imgsz`, `max_distance`, `max_disappeared`, `start_counting`. Missing fields inherit from the global config.

### Or add streams at runtime via API

```bash
curl -X POST http://localhost:5581/api/streams \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"id":"line-5","url":"rtsp://cam5.local/stream","roi_position":0.45}'
```

### Sizing notes for VPS deployments

10 streams × 25 fps × 640×640 YOLO inference is heavy. The batched worker collects frames in a `BATCH_WINDOW_MS=25` window and runs up to `BATCH_MAX=16` per forward pass, which on an RTX 3090 sustains the full load with headroom. Smaller GPUs (T4, A4000) handle it but need wider batch windows; raise `BATCH_WINDOW_MS` to 40-50 ms if the worker is dropping frames (`dropped_frames` in `/api/streams/{id}/status`).

If you exceed the soft cap, the registry returns `429`. Raise `MAX_STREAMS` only if your GPU can absorb the additional load.

---

## Authentication

All `/api/*` endpoints require a header:

```
X-API-Key: <your-key>
```

The key comes from `API_KEY` in `.env` (or compose env). When `API_KEY` is empty, the API is open (dev mode) and a `WARNING` line is logged at startup. **Set `API_KEY` before exposing the container to the public internet.**

The static UI (`/`) and `/health` stay public so dashboards and load-balancer probes don't need a key.

---

## REST API Reference

Base URL: `http://localhost:5581`

All endpoints under `/api/*` require `X-API-Key: <key>` when `API_KEY` is set.

### Health

#### `GET /health` &nbsp;·&nbsp; public

Server status, GPU info, active model.

```bash
curl http://localhost:5581/health
```

**Response 200**
```json
{
  "status": "ok",
  "model_path": "best.pt",
  "cuda_available": true,
  "gpu": "NVIDIA GeForce RTX 3090"
}
```

---

### Multi-Stream RTSP (new) — `/api/streams/*`

Use this for any new integration. Each stream's id is your handle for all per-stream operations.

#### `GET /api/streams`

List every registered stream with its own counts and state. Use this for dashboards that show all lines at once.

```bash
curl -H "X-API-Key: $KEY" http://localhost:5581/api/streams
```

**Response 200**
```json
{
  "streams": [
    {
      "id": "line-1",
      "url": "rtsp://cam1.local/stream",
      "is_playing": true,
      "is_counting": true,
      "counts": {"empty_shackles": 4, "single_legged": 1, "slaughtered_chicken": 27},
      "total_count": 32,
      "fps": 26.4,
      "dropped_frames": 0,
      "error": null
    },
    { "id": "line-2", "...": "..." }
  ]
}
```

#### `POST /api/streams`

Register and start a new stream. Inference starts immediately; counting starts unless `start_counting:false`.

**Body**
| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `id` | string | ✓ | — | Unique handle for this stream |
| `url` | string | ✓ | — | RTSP / HTTP source URL |
| `roi_position` | float | | global | 0 < x < 1, fraction of frame width |
| `confidence` | float | | global | 0 < x < 1 |
| `nms_iou` | float | | global | 0 < x < 1 |
| `imgsz` | int | | global | multiple of 32 |
| `max_distance` | int | | global | tracker max px |
| `max_disappeared` | int | | global | frames before lost track is dropped |
| `start_counting` | bool | | `true` | arm counter on register |

```bash
curl -X POST http://localhost:5581/api/streams \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"id":"line-5","url":"rtsp://cam5.local/stream","roi_position":0.55}'
```

**Response 201** — same shape as one entry from `GET /api/streams`.  
**Errors:** `400` (validation), `409` (id already exists), `429` (cap reached).

#### `DELETE /api/streams/{stream_id}`

Stop and remove a stream.

```bash
curl -X DELETE -H "X-API-Key: $KEY" http://localhost:5581/api/streams/line-5
```
**Response 200** `{"status":"removed","id":"line-5"}`  ·  **404** if not found.

#### `GET /api/streams/{stream_id}/status`

Per-stream counts, fps, dropped_frames, error.

```bash
curl -H "X-API-Key: $KEY" http://localhost:5581/api/streams/line-1/status
```

**Response 200** — same shape as one entry from `GET /api/streams`.

#### `GET /api/streams/{stream_id}/feed`

MJPEG feed (annotated frames). Each stream returns only its own frames. Embed in `<img src="...">` or consume with any MJPEG client.

```html
<img src="http://localhost:5581/api/streams/line-1/feed?key=ABC">
```
(For browser embeds, key-as-query is easier than headers; the dashboard handles this for you. For server-to-server use the `X-API-Key` header.)

#### `POST /api/streams/{stream_id}/counting/start`

Begin counting on a registered stream.
```bash
curl -X POST -H "X-API-Key: $KEY" http://localhost:5581/api/streams/line-1/counting/start
```
**Response 200** `{"status":"counting","id":"line-1"}`

#### `POST /api/streams/{stream_id}/counting/stop`

Pause counting (capture continues, ROI crossings stop accumulating).
```bash
curl -X POST -H "X-API-Key: $KEY" http://localhost:5581/api/streams/line-1/counting/stop
```

#### `POST /api/streams/{stream_id}/counting/reset`

Zero out this stream's counts without disrupting capture. Useful for shift changes.
```bash
curl -X POST -H "X-API-Key: $KEY" http://localhost:5581/api/streams/line-1/counting/reset
```
**Response 200**
```json
{"status":"reset","id":"line-1","counts":{"empty_shackles":0,"single_legged":0,"slaughtered_chicken":0}}
```

---

### Legacy Single-Stream — `/api/stream/*`

Kept for backward compatibility with the bundled HTML dashboard. All endpoints proxy to a single registry entry with id `"default"`. **For new integrations use `/api/streams/*` above.**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/stream/start` | Connect to RTSP stream (body `{url}` or `RTSP_URL` env) |
| `POST` | `/api/stream/stop` | Disconnect the `default` stream |
| `POST` | `/api/stream/counting/start` | Enable counting |
| `POST` | `/api/stream/counting/stop` | Disable counting |
| `GET` | `/api/stream/feed` | MJPEG feed of the `default` stream |
| `GET` | `/api/stream/status` | Per-class counts + connection state |

```bash
curl -X POST http://localhost:5581/api/stream/start \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"url":"rtsp://admin:pass@192.168.1.100:554/stream"}'
```

---

### Image — `/api/image/*`

#### `POST /api/image/detect`

Upload an image; returns an annotated JPEG. Counts are returned in response headers.

```bash
curl -X POST -H "X-API-Key: $KEY" \
  -F "file=@photo.jpg" \
  -o annotated.jpg -D - \
  http://localhost:5581/api/image/detect
```

**Response headers**
- `X-Total-Count`
- `X-Count-Empty-Shackles`
- `X-Count-Single-Legged`
- `X-Count-Slaughtered-Chicken`

---

### Video — `/api/video/*`

Session-based: upload once, control playback and counting independently, download an H.264-encoded annotated copy.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/video/upload` | Upload video file, returns `{session_id, filename}` |
| `POST` | `/api/video/{id}/start` | Start processing |
| `POST` | `/api/video/{id}/stop` | Stop processing |
| `POST` | `/api/video/{id}/counting/start` | Enable ROI counting |
| `POST` | `/api/video/{id}/counting/stop` | Disable ROI counting |
| `GET` | `/api/video/{id}/feed` | MJPEG stream of annotated frames |
| `GET` | `/api/video/{id}/status` | Counts, frame, fps, completion state |
| `GET` | `/api/video/{id}/download` | Download H.264 re-encoded output |

```bash
curl -X POST -H "X-API-Key: $KEY" \
  -F "file=@video.mp4" http://localhost:5581/api/video/upload
# {"session_id":"a1b2c3d4","filename":"video.mp4"}

curl -X POST -H "X-API-Key: $KEY" http://localhost:5581/api/video/a1b2c3d4/start
curl -X POST -H "X-API-Key: $KEY" http://localhost:5581/api/video/a1b2c3d4/counting/start
curl -H "X-API-Key: $KEY" http://localhost:5581/api/video/a1b2c3d4/status
curl -H "X-API-Key: $KEY" -o output.mp4 http://localhost:5581/api/video/a1b2c3d4/download
```

**Status response**
```json
{
  "is_playing": true, "is_counting": true,
  "counts": {"empty_shackles": 12, "single_legged": 3, "slaughtered_chicken": 47},
  "total_count": 62,
  "frame_num": 450, "total_frames": 1200,
  "fps": 28.5,
  "is_complete": false, "is_stream": false,
  "dropped_frames": 0,
  "error": null
}
```

---

### Config — `/api/config`

#### `GET /api/config`
Returns all current settings (the live snapshot, not the .env values).

#### `PATCH /api/config`
Update one or more fields live, no restart.

| Field | Type | Constraints |
|-------|------|-------------|
| `rtsp_url` | string | default RTSP URL for legacy /api/stream |
| `model_path` | string | validated by attempting to load |
| `roi_position` | float | 0 < x < 1 |
| `confidence` | float | 0 < x < 1 |
| `nms_iou` | float | 0 < x < 1 |
| `imgsz` | int | multiple of 32 |
| `max_distance` | int | ≥ 1 |
| `max_disappeared` | int | ≥ 1 |

```bash
curl -X PATCH http://localhost:5581/api/config \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"roi_position": 0.4, "confidence": 0.3}'
```

Per-stream registration overrides take precedence over the global config — `PATCH /api/config` does **not** retroactively change already-registered streams.

---

### TensorRT Export — `/api/export/tensorrt`

For **production deployments**, you usually don't call this — the container already builds and caches an engine on first boot (see [Production GPU Deployment](#production-gpu-deployment-tensorrt)). Use this endpoint to re-export at runtime, e.g. after changing `imgsz` via `PATCH /api/config`, or to validate that the export path works on a new host.

#### `POST /api/export/tensorrt`
Start background FP16 engine export (body `{"half": true|false}`, default true). Returns `201` if started, `409` if one is already in flight.

#### `GET /api/export/tensorrt`
Poll export state.

```bash
curl -X POST -H "X-API-Key: $KEY" http://localhost:5581/api/export/tensorrt
curl -H "X-API-Key: $KEY" http://localhost:5581/api/export/tensorrt
# {"state":"DONE","source_model":"best.pt","output_path":"best.engine","elapsed_seconds":142.3}
```

States: `IDLE` (no export ever run), `RUNNING`, `DONE`, `FAILED` (with `error` field).

---

## Web Dashboard

| URL | Page |
|-----|------|
| `/` | Image detection — drag-drop, side-by-side, per-class counts |
| `/video.html` | Upload video, play/count controls, download H.264 output |
| `/stream.html` | Single-stream live RTSP view (uses legacy `/api/stream/*`) |
| `/docs` | Auto-generated Swagger UI for the full API |

---

## Configuration

`.env` (all fields also settable as environment variables):

```env
# Every key is OPTIONAL — unset keys fall through to the Python defaults
# in app/config.py. The example below shows the current defaults; only
# include lines you actually want to override.

# ── Detection / counting defaults ─────────────────────────────────────────
RTSP_URL=
MODEL_PATH=best.pt
ROI_POSITION=0.60
CONFIDENCE=0.25
CONF_EMPTY_SHACKLES=0.45
NMS_IOU=0.45
IMGSZ=1280
MAX_DISTANCE=90
MAX_DISAPPEARED=2

# ── Multi-stream auto-start ───────────────────────────────────────────────
RTSP_STREAMS='[{"id":"line-1","url":"rtsp://cam1.local/stream"}]'
MAX_STREAMS=10

# ── Batched inference tuning ──────────────────────────────────────────────
BATCH_MAX=32
BATCH_WINDOW_MS=10
INFERENCE_QUEUE_MAX=400

# ── Auth ──────────────────────────────────────────────────────────────────
API_KEY=replace-with-a-long-random-value
```

### Tuning the inference worker

| Var | Default | Effect when increased |
|-----|---------|-----------------------|
| `BATCH_MAX` | 32 | Larger batches → higher throughput, slightly more per-frame latency |
| `BATCH_WINDOW_MS` | 10 | Worker waits longer for batchmates → bigger batches but more MJPEG lag |
| `INFERENCE_QUEUE_MAX` | 400 | More backlog tolerance before frames get dropped under spikes |

If you see non-zero `dropped_frames` in a stream's status, either raise `BATCH_WINDOW_MS` (better packing) or use a larger GPU.

---

## CLI Usage

Standalone `detect_and_count.py` — no server required, no API key.

```bash
# Image
python detect_and_count.py photo.jpg --save out.jpg

# Video with counting
python detect_and_count.py clip.mp4 --save out.mp4 --roi 0.5

# RTSP
python detect_and_count.py "rtsp://user:pass@cam-ip:554/stream"
```

| Argument | Default | Description |
|----------|---------|-------------|
| `input` | required | Path to image, video, or RTSP URL |
| `--save` | — | Save annotated output |
| `--conf` | `0.25` | Detection confidence threshold |
| `--iou` | `0.45` | NMS IoU threshold |
| `--imgsz` | `640` | Inference image size |
| `--model` | `best.pt` | Path to weights |
| `--roi` | `0.5` | ROI line position (0..1) |
| `--max-distance` | `50` | Tracker max pixel distance |
| `--max-disappeared` | `15` | Frames before lost track is dropped |
| `--appear-margin` | `25` | Px past ROI within which a brand-new track is counted (anti-flicker) |

Press **q** to stop.

---

## Counting Logic

- The conveyor moves **left to right** — the ROI line is **vertical** at `roi_x = width × roi_position`.
- Each class has its own `CentroidTracker` — IDs never cross classes.
- **Class-agnostic NMS** ensures the model never reports two boxes (e.g. wings + body) on the same physical object.
- A count fires when `prev_cx < roi_x ≤ current_cx` (centroid crosses the line moving right).
- A brand-new track first appearing within `appear_margin` (default 25 px) past the line counts immediately — covers the case of a chicken already past the ROI when the stream starts. Tracks appearing deeper past the line are treated as flicker re-acquisitions of an already-counted chicken and skipped.
- Lost tracks live their normal `max_disappeared=15` frame lifespan, so brief detection drops don't cause duplicate counts.

---

## Project Structure

```
Slaughtered_Chicken_Counting/
├── app/
│   ├── main.py                    # FastAPI app, lifespan (model preload + worker + auto-start)
│   ├── config.py                  # Pydantic-settings boot config
│   ├── core/
│   │   ├── runtime_config.py      # Thread-safe live config
│   │   ├── auth.py                # X-API-Key dependency
│   │   ├── model_cache.py         # YOLO singleton cache
│   │   ├── inference_worker.py    # Batched single-thread inference dispatcher
│   │   ├── stream_registry.py     # Multi-stream registry (env auto-start, soft cap)
│   │   ├── detector.py            # Legacy synchronous helper (image router uses worker now)
│   │   ├── tracker.py             # Hungarian + IoU CentroidTracker
│   │   ├── counter.py             # Per-class ROI crossing logic
│   │   ├── annotator.py           # Bboxes, trails, flash, dashboard
│   │   ├── video_processor.py     # Per-stream capture thread + MJPEG buffer
│   │   └── exporter.py            # TensorRT export state machine
│   ├── routers/
│   │   ├── image.py               # POST /api/image/detect
│   │   ├── video.py               # /api/video/* session API
│   │   ├── stream.py              # /api/stream/* legacy single-stream shim
│   │   ├── streams.py             # /api/streams/* multi-stream API
│   │   ├── config_router.py       # GET + PATCH /api/config
│   │   ├── export_router.py       # /api/export/tensorrt
│   │   └── health_router.py       # GET /health (public)
│   └── static/                    # Frontend HTML / CSS / JS
├── train.py                       # YOLO26s training script
├── detect_and_count.py            # Standalone CLI
├── dataset/                       # Unzip Roboflow export here
├── best.pt                        # Trained model weights
├── .env                           # Runtime configuration
├── Dockerfile
├── docker-compose.yml             # CPU-default with healthcheck
└── docker-compose.gpu.yml         # GPU override
```

---

## Docker

The image is a **single artifact** for CPU and GPU hosts. Published as **[`basim123/chicken-counter`](https://hub.docker.com/r/basim123/chicken-counter)**.

```bash
# CPU
docker compose up --build

# GPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

The compose file includes a `healthcheck` hitting `/health` every 30s. `docker ps` will show `(healthy)` / `(unhealthy)` once the container is up.

> **GPU first-boot:** the container builds a TensorRT engine on first launch (2–15 min depending on the GPU) and caches it to the `engine_cache` named volume. Subsequent boots reuse it. See [Production GPU Deployment](#production-gpu-deployment-tensorrt) for the full pattern.

### Build and push to Docker Hub

`docker-compose.yml` declares `image: basim123/chicken-counter:latest`, so `docker compose build` tags the built image with the push-ready name. To publish a new version:

```bash
# Build (produces basim123/chicken-counter:latest)
docker compose build

# Also tag the current version
docker tag basim123/chicken-counter:latest basim123/chicken-counter:2.0.0

# Push both tags
docker login
docker push basim123/chicken-counter:2.0.0
docker push basim123/chicken-counter:latest
```

Pin to the version tag in production: `basim123/chicken-counter:2.0.0`.

---

## Video Overlay Guide

| Element | Meaning |
|---------|---------|
| Vertical animated dashed line | ROI counting line with rightward arrows (→) |
| Orange bboxes / dots | `empty_shackles` |
| Gold bboxes / dots | `single_legged` |
| Green bboxes / dots | `slaughtered_chicken` |
| Thin gradient trails | Motion path per tracked object |
| Expanding ripple ring | Object crossed the ROI line (class-colored) |
| Dashboard panel (top-left) | Per-class counts, total, FPS, progress bar |
