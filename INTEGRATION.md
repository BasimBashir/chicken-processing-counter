# Slaughtered Chicken Counter — Integration Guide

A pre-packaged FastAPI service that detects and counts three classes of poultry-line objects on a conveyor belt using YOLO. Pull the image, run it, talk to it over HTTP — no Python/ML setup required on the consuming side.

**Image:** `basim123/chicken-counter`
**Tags:** `latest`, `2.0.0` (multi-stream + batched inference), `1.0.0` (legacy single-stream)
**Base port:** `5581`
**Health probe:** `GET /health` (no auth)
**Interactive docs:** `http://<host>:5581/docs` (Swagger UI)

---

## 1. Pull and run

The image is a single artifact that auto-selects GPU or CPU at startup and logs which one it chose.

### CPU (any host with Docker)

```bash
docker pull basim123/chicken-counter:latest

docker run -d --name chicken-counter \
  -p 5581:5581 \
  -e API_KEY=replace-with-long-random-string \
  -v $(pwd)/uploads:/app/app/uploads \
  -v $(pwd)/outputs:/app/app/outputs \
  basim123/chicken-counter:latest
```

### GPU (NVIDIA + nvidia-container-toolkit on the host)

```bash
docker pull basim123/chicken-counter:latest

docker run -d --name chicken-counter \
  --gpus all \
  -p 5581:5581 \
  -e API_KEY=replace-with-long-random-string \
  -v $(pwd)/uploads:/app/app/uploads \
  -v $(pwd)/outputs:/app/app/outputs \
  basim123/chicken-counter:latest
```

### Multi-stream auto-start (production deployment)

Declare all cameras in one env var; the container registers them at startup and starts counting immediately.

```bash
docker run -d --name chicken-counter --gpus all -p 5581:5581 \
  -e API_KEY=replace-with-long-random-string \
  -e MAX_STREAMS=10 \
  -e RTSP_STREAMS='[
    {"id":"line-1","url":"rtsp://cam1.local/stream"},
    {"id":"line-2","url":"rtsp://cam2.local/stream","roi_position":0.55},
    {"id":"line-3","url":"rtsp://cam3.local/stream","confidence":0.30}
  ]' \
  -v $(pwd)/uploads:/app/app/uploads \
  -v $(pwd)/outputs:/app/app/outputs \
  basim123/chicken-counter:latest
```

### Confirm device selection

```bash
docker logs --tail=10 chicken-counter
# 2026-05-21 ... inference_worker [INFO] InferenceWorker started (batch_max=16, window_ms=25)
# 2026-05-21 ... auth [INFO] API auth enabled (X-API-Key required for /api/*)
# 2026-05-21 ... stream_registry [INFO] Registered stream 'line-1' -> rtsp://cam1.local/stream

curl http://localhost:5581/health
# {"status":"ok","model_path":"best.pt","cuda_available":true,"gpu":"NVIDIA GeForce RTX 3090"}

docker inspect --format='{{.State.Health.Status}}' chicken-counter
# healthy
```

---

## 2. Configuration (environment variables)

All optional. Set with `-e VAR=value` on `docker run`, or in a Compose `environment:` block. The detection knobs are also live-patchable via `PATCH /api/config` (see §4.7) so you rarely need to restart.

### Detection / counting defaults

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `best.pt` | YOLO weights inside the image. Swap to `best.engine` after TensorRT export |
| `ROI_POSITION` | `0.5` | Counting line as fraction of frame width (0 = left, 1 = right) |
| `CONFIDENCE` | `0.25` | Detection confidence threshold (0–1) |
| `NMS_IOU` | `0.45` | NMS IoU threshold (0–1) |
| `IMGSZ` | `640` | Inference image size; must be a multiple of 32 |
| `MAX_DISTANCE` | `50` | Max pixel distance for tracker to match a detection to an existing track |
| `MAX_DISAPPEARED` | `15` | Frames a track can be unseen before it's dropped |

### Multi-stream

| Variable | Default | Description |
|---|---|---|
| `RTSP_STREAMS` | _(empty)_ | JSON list of streams to register on boot. Each entry: `{"id":..., "url":..., ...overrides}` |
| `MAX_STREAMS` | `10` | Soft cap on concurrent streams. New registrations beyond this return `429` |
| `RTSP_URL` | _(empty)_ | Default URL for the legacy single-stream `/api/stream/*` API |

Per-stream overrides allowed inside `RTSP_STREAMS` entries: `roi_position`, `confidence`, `nms_iou`, `imgsz`, `max_distance`, `max_disappeared`, `start_counting`. Anything omitted inherits the global defaults above.

### Batched inference worker

| Variable | Default | Description |
|---|---|---|
| `BATCH_MAX` | `16` | Max frames bundled into one GPU forward pass |
| `BATCH_WINDOW_MS` | `25` | Max time the worker waits for batchmates to arrive |
| `INFERENCE_QUEUE_MAX` | `100` | Backlog cap before frames are dropped (visible as `dropped_frames` in stream status) |

### Auth

| Variable | Default | Description |
|---|---|---|
| `API_KEY` | _(empty → dev mode, no auth)_ | When set, every `/api/*` request must include `X-API-Key: <value>`. `/health` and the static UI stay public |

If `API_KEY` is empty, the container logs a `WARNING` line at startup and accepts all requests — only acceptable for local development.

### Volumes

| Host path | Container path | Purpose |
|---|---|---|
| `./uploads` | `/app/app/uploads` | Uploaded images/videos land here |
| `./outputs` | `/app/app/outputs` | H.264 re-encoded video outputs |

---

## 3. Classes returned

| Class | Description |
|---|---|
| `empty_shackles` | Empty shackle hooks |
| `single_legged` | Chicken hanging by one leg |
| `slaughtered_chicken` | Fully processed slaughtered chicken |

Each class has its own independent tracker — IDs are never shared across classes, so counts can't bleed. The model runs **class-agnostic NMS**, so it never emits two boxes for the same physical object even when ambiguous between classes.

---

## 4. HTTP API

Base URL: `http://<host>:5581`. All endpoints return JSON unless they stream media. CORS is **not** enabled by default; front the service with a reverse proxy that adds the headers if you need cross-origin browser access.

**Auth:** When `API_KEY` is set, every request to `/api/*` requires `X-API-Key: <key>`. Missing or wrong key → `401`. The `/health` endpoint and the static dashboard remain public.

```bash
KEY=replace-with-long-random-string
HOST=http://localhost:5581
```

All examples below assume `$KEY` and `$HOST` are exported.

### 4.1 Health (public)

```http
GET /health
```

```json
{"status":"ok","model_path":"best.pt","cuda_available":true,"gpu":"NVIDIA GeForce RTX 3090"}
```

`gpu` is omitted when `cuda_available` is `false`.

### 4.2 Image detection (one-shot)

```http
POST /api/image/detect
Content-Type: multipart/form-data
X-API-Key: <key>
```

Form field `file`: image (JPEG/PNG). Response body is the annotated JPEG. Counts come back as response headers:

| Header | Value |
|---|---|
| `X-Total-Count` | Total across all classes |
| `X-Count-Empty-Shackles` | int |
| `X-Count-Single-Legged` | int |
| `X-Count-Slaughtered-Chicken` | int |

```bash
curl -X POST $HOST/api/image/detect \
  -H "X-API-Key: $KEY" \
  -F "file=@photo.jpg" -o annotated.jpg -D headers.txt
```

Internally, image inference goes through the same batched worker as live streams.

### 4.3 Video session (upload + replay)

A video session is processed in the background. Play and counting are **independent controls** — start playback to preview detections, then enable counting when you want the ROI line to start tallying.

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/video/upload` | Upload, returns `session_id` |
| POST | `/api/video/{id}/start` | Start frame processing |
| POST | `/api/video/{id}/stop` | Stop processing |
| POST | `/api/video/{id}/counting/start` | Enable ROI counting |
| POST | `/api/video/{id}/counting/stop` | Disable ROI counting |
| GET  | `/api/video/{id}/feed` | MJPEG stream of annotated frames |
| GET  | `/api/video/{id}/status` | Poll counts, frame, fps, completion |
| GET  | `/api/video/{id}/download` | Download the H.264 re-encoded output |

Status payload:

```json
{
  "is_playing": true,
  "is_counting": true,
  "counts": {
    "empty_shackles": 12,
    "single_legged": 3,
    "slaughtered_chicken": 47
  },
  "total_count": 62,
  "frame_num": 450,
  "total_frames": 1200,
  "fps": 28.5,
  "is_complete": false,
  "is_stream": false,
  "dropped_frames": 0,
  "error": null
}
```

Typical flow:

```bash
SID=$(curl -s -X POST $HOST/api/video/upload \
        -H "X-API-Key: $KEY" \
        -F "file=@video.mp4" | jq -r .session_id)

curl -X POST -H "X-API-Key: $KEY" $HOST/api/video/$SID/start
curl -X POST -H "X-API-Key: $KEY" $HOST/api/video/$SID/counting/start

# Poll until is_complete: true
curl -H "X-API-Key: $KEY" $HOST/api/video/$SID/status

curl -H "X-API-Key: $KEY" -o output.mp4 $HOST/api/video/$SID/download
```

### 4.4 Multi-stream RTSP — `/api/streams/*`

Run up to `MAX_STREAMS` concurrent RTSP cameras in one container. Each is identified by a user-supplied `id`; all per-stream operations key off that id, and counts/feeds/status are always isolated to one stream.

| Method | Endpoint | Purpose |
|---|---|---|
| GET    | `/api/streams` | List all streams with each one's counts and state |
| POST   | `/api/streams` | Register and start a stream. Body fields below |
| DELETE | `/api/streams/{id}` | Stop and unregister |
| GET    | `/api/streams/{id}/status` | Per-stream counts, fps, dropped_frames, error |
| GET    | `/api/streams/{id}/feed` | MJPEG feed of this stream only |
| POST   | `/api/streams/{id}/counting/start` | Begin counting |
| POST   | `/api/streams/{id}/counting/stop` | Pause counting (capture continues) |
| POST   | `/api/streams/{id}/counting/reset` | Zero counts without disrupting capture |

**`POST /api/streams` body:**

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `id` | string | ✓ | — | Unique handle |
| `url` | string | ✓ | — | RTSP/HTTP source URL |
| `roi_position` | float | | global | `0 < x < 1` |
| `confidence` | float | | global | `0 < x < 1` |
| `nms_iou` | float | | global | `0 < x < 1` |
| `imgsz` | int | | global | multiple of 32 |
| `max_distance` | int | | global | tracker max px |
| `max_disappeared` | int | | global | frames before drop |
| `start_counting` | bool | | `true` | arm counter on register |

**Status codes:**
- `201` — registered
- `400` — invalid input
- `409` — id already exists
- `429` — `MAX_STREAMS` reached

**Examples:**

```bash
# List all streams
curl -H "X-API-Key: $KEY" $HOST/api/streams

# Register a new stream
curl -X POST $HOST/api/streams \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"id":"line-7","url":"rtsp://cam7.local/stream","roi_position":0.55}'

# Per-stream status
curl -H "X-API-Key: $KEY" $HOST/api/streams/line-7/status

# Reset counts for shift change
curl -X POST -H "X-API-Key: $KEY" $HOST/api/streams/line-7/counting/reset

# Remove the stream
curl -X DELETE -H "X-API-Key: $KEY" $HOST/api/streams/line-7
```

Per-stream status returns the same shape as the entries in `GET /api/streams`:

```json
{
  "id": "line-7",
  "url": "rtsp://cam7.local/stream",
  "is_playing": true,
  "is_counting": true,
  "counts": {"empty_shackles": 4, "single_legged": 1, "slaughtered_chicken": 27},
  "total_count": 32,
  "fps": 26.4,
  "dropped_frames": 0,
  "error": null
}
```

### 4.5 Legacy single-stream — `/api/stream/*`

Kept for backward compatibility with the bundled HTML dashboard. All endpoints proxy to a single registry entry with id `"default"`. **For new integrations, use `/api/streams/*` above.**

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/stream/start` | Body: `{"url":"rtsp://..."}`. Omit body to use `RTSP_URL` env var |
| POST | `/api/stream/stop` | Disconnect |
| POST | `/api/stream/counting/start` | Enable counting |
| POST | `/api/stream/counting/stop` | Disable counting |
| GET  | `/api/stream/feed` | MJPEG feed |
| GET  | `/api/stream/status` | Counts + connection state (legacy shape) |

```bash
curl -X POST $HOST/api/stream/start \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"url":"rtsp://admin:pass@192.168.1.100:554/stream"}'

curl -X POST -H "X-API-Key: $KEY" $HOST/api/stream/counting/start
curl -H "X-API-Key: $KEY" $HOST/api/stream/status
```

### 4.6 TensorRT engine export (GPU only)

Builds a `.engine` file from `best.pt` for ~2-3× faster inference. **Requires a GPU at runtime.** On a CPU container the endpoint reports `FAILED` with the ultralytics error; the rest of the app keeps working.

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/export/tensorrt` | Start background FP16 export |
| GET  | `/api/export/tensorrt` | Poll state: `IDLE` / `RUNNING` / `DONE` / `FAILED` |

```bash
curl -X POST -H "X-API-Key: $KEY" $HOST/api/export/tensorrt
# {"status":"started","model_path":"best.pt"}

curl -H "X-API-Key: $KEY" $HOST/api/export/tensorrt
# {"state":"DONE","output_path":"best.engine","elapsed_seconds":142.3}

# Switch to the engine live:
curl -X PATCH $HOST/api/config \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"model_path":"best.engine"}'
```

### 4.7 Runtime config

| Method | Endpoint | Purpose |
|---|---|---|
| GET   | `/api/config` | Current settings snapshot |
| PATCH | `/api/config` | Update detection knobs without a restart |

Patchable fields:

| Field | Type | Constraint |
|---|---|---|
| `rtsp_url` | string | — |
| `model_path` | string | validated by attempting a load |
| `roi_position` | float | `0 < x < 1` |
| `confidence` | float | `0 < x < 1` |
| `nms_iou` | float | `0 < x < 1` |
| `imgsz` | int | multiple of 32 |
| `max_distance` | int | `>= 1` |
| `max_disappeared` | int | `>= 1` |

`PATCH /api/config` updates **global defaults applied to future stream registrations**. Already-registered streams keep the per-stream config they were registered with — restart the stream (DELETE + POST) to pick up changes.

`API_KEY`, `RTSP_STREAMS`, `MAX_STREAMS`, and the batch worker tunables are **boot-time only** — change them via env and restart the container.

```bash
curl -X PATCH $HOST/api/config \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"roi_position": 0.4, "confidence": 0.3}'
```

---

## 5. Building on top of this service

All examples below use a single shared API key. For multi-tenant scenarios, front the service with a proxy that rewrites/issues keys.

### Python client — image detection

```python
import requests

HOST = "http://localhost:5581"
KEY  = "replace-with-long-random-string"

def count_chickens(image_path: str) -> dict:
    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{HOST}/api/image/detect",
            headers={"X-API-Key": KEY},
            files={"file": f},
            timeout=30,
        )
    resp.raise_for_status()
    return {
        "total": int(resp.headers["X-Total-Count"]),
        "empty_shackles": int(resp.headers["X-Count-Empty-Shackles"]),
        "single_legged": int(resp.headers["X-Count-Single-Legged"]),
        "slaughtered_chicken": int(resp.headers["X-Count-Slaughtered-Chicken"]),
        "annotated_jpeg": resp.content,
    }

result = count_chickens("frame.jpg")
print(result["total"], "objects detected")
with open("annotated.jpg", "wb") as f:
    f.write(result["annotated_jpeg"])
```

### Python client — multi-stream registration + polling

```python
import requests, time

HOST = "http://localhost:5581"
KEY  = "replace-with-long-random-string"
H = {"X-API-Key": KEY}

# Register four cameras
streams = [
    {"id": "line-1", "url": "rtsp://cam1.local/stream"},
    {"id": "line-2", "url": "rtsp://cam2.local/stream", "roi_position": 0.55},
    {"id": "line-3", "url": "rtsp://cam3.local/stream"},
    {"id": "line-4", "url": "rtsp://cam4.local/stream", "confidence": 0.30},
]
for s in streams:
    r = requests.post(f"{HOST}/api/streams", headers=H, json=s)
    if r.status_code == 409:
        print(f"{s['id']} already registered")
    elif r.status_code == 429:
        raise SystemExit("MAX_STREAMS cap reached")
    else:
        r.raise_for_status()

# Poll every second; per-stream counts are returned separately
while True:
    snapshot = requests.get(f"{HOST}/api/streams", headers=H).json()
    for s in snapshot["streams"]:
        print(f"{s['id']:>8} total={s['total_count']:>4} fps={s['fps']:>5.1f} "
              f"dropped={s['dropped_frames']} err={s['error'] or '-'}")
    print("---")
    time.sleep(1)
```

### Python client — video upload + polling

```python
import requests, time

HOST = "http://localhost:5581"
H = {"X-API-Key": "replace-with-long-random-string"}

with open("line.mp4", "rb") as f:
    sid = requests.post(f"{HOST}/api/video/upload",
                        headers=H, files={"file": f}).json()["session_id"]

requests.post(f"{HOST}/api/video/{sid}/start", headers=H)
requests.post(f"{HOST}/api/video/{sid}/counting/start", headers=H)

while True:
    status = requests.get(f"{HOST}/api/video/{sid}/status", headers=H).json()
    print(f"frame {status['frame_num']}/{status['total_frames']} "
          f"total={status['total_count']} fps={status['fps']:.1f}")
    if status["is_complete"]:
        break
    time.sleep(1)

with open("output.mp4", "wb") as f:
    f.write(requests.get(f"{HOST}/api/video/{sid}/download", headers=H).content)
```

### JavaScript / Node — live multi-stream status

```javascript
const HOST = "http://localhost:5581";
const KEY  = "replace-with-long-random-string";
const headers = { "X-API-Key": KEY, "Content-Type": "application/json" };

// Register a new stream
await fetch(`${HOST}/api/streams`, {
  method: "POST",
  headers,
  body: JSON.stringify({
    id: "line-1",
    url: "rtsp://admin:pass@cam.local:554/stream",
    roi_position: 0.5,
  }),
});

// Poll all streams every second; each returns its own counts
setInterval(async () => {
  const { streams } = await fetch(`${HOST}/api/streams`, {
    headers: { "X-API-Key": KEY },
  }).then(r => r.json());
  for (const s of streams) {
    console.log(`${s.id} total=${s.total_count} fps=${s.fps} dropped=${s.dropped_frames}`);
  }
}, 1000);
```

### Embedding MJPEG feeds in a web page

```html
<!-- Multi-stream feed (preferred) -->
<img src="http://localhost:5581/api/streams/line-1/feed" />
<img src="http://localhost:5581/api/streams/line-2/feed" />

<!-- Legacy single-stream feed -->
<img src="http://localhost:5581/api/stream/feed" />

<!-- Video session feed -->
<img src="http://localhost:5581/api/video/<session_id>/feed" />
```

Browsers render `multipart/x-mixed-replace` MJPEG natively from an `<img>` tag. For authenticated embeds, front the service with a reverse proxy that injects the `X-API-Key` header (browsers can't add arbitrary headers to `<img src>`).

---

## 6. Production checklist

- **Auth** — set `API_KEY` to a long random string. The container logs a warning if it boots without one.
- **Reverse proxy** — front with nginx/Caddy/Traefik for TLS termination and (for browser dashboards) MJPEG-feed auth injection.
- **Persistent volumes** — mount `uploads/` and `outputs/`; the in-image filesystem is ephemeral.
- **Model swap** — to deploy a retrained model, mount your `best.pt` into the container at `/app/best.pt` (`-v ./my-best.pt:/app/best.pt:ro`) instead of rebuilding the image.
- **GPU sizing** — one detector instance with batched inference uses ~2 GB GPU at `imgsz=640, batch=16`. A 4 GB card handles 10 streams; an 8 GB card has headroom for TensorRT export alongside.
- **Concurrency** — all model forward passes funnel through one `InferenceWorker` thread that batches frames across streams. Throughput scales sub-linearly with stream count; adding the 11th stream just enlarges the queue, not the GPU pressure.
- **Backpressure** — under load spikes, frames beyond `INFERENCE_QUEUE_MAX` are dropped and surfaced as `dropped_frames` in `/api/streams/{id}/status`. Persistent non-zero values mean the GPU is undersized for the configured stream count.
- **Healthcheck** — the image ships with a built-in healthcheck calling `/health` every 30s. Read it with `docker inspect --format='{{.State.Health.Status}}' chicken-counter`.
- **Logs** — stdout/stderr; collect via `docker logs` or your usual log shipper.

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `401 Missing or invalid X-API-Key` | Header missing or wrong | Confirm `API_KEY` env matches the value you're sending |
| `WARNING ... API_KEY is empty` in logs | Container booted without `API_KEY` | Set `-e API_KEY=...` and restart — required for production |
| `[startup] CUDA available: false` on a GPU host | Container started without `--gpus all` | Restart with `--gpus all` |
| `/api/export/tensorrt` always FAILED | Running on CPU container | TensorRT export needs a GPU |
| `429 Stream cap reached` on register | Hit `MAX_STREAMS` | Remove a stream or raise `MAX_STREAMS` (verify GPU has headroom) |
| `dropped_frames > 0` and rising | Inference queue can't keep up | Reduce stream count, raise `BATCH_WINDOW_MS`, or use a larger GPU |
| `connection refused` on first request | Uvicorn hasn't finished booting | Wait 5–10s after `docker run`; poll `/health` first |
| Detection counts feel off | Wrong `roi_position` or `confidence` | `PATCH /api/config` to tune live; check the MJPEG feed visually |
| MJPEG feed shows old frames | Browser cache on `<img>` | Add a cache-busting query string |

---

## 8. Versioning

| Tag | Notes |
|---|---|
| `latest` | Always points to the newest published build |
| `2.0.0` | Multi-stream `/api/streams/*` API, batched inference worker, API-key auth, Docker healthcheck |
| `1.0.0` | First public release, single-stream only, no auth |

Pin to a version tag in production: `basim123/chicken-counter:2.0.0`.

### Migrating from 1.0.0 → 2.0.0

- **Add `API_KEY`** to your `docker run` / compose — without it the container boots in open-access "dev mode" with a startup warning, which is fine for upgrade testing but not for production.
- **Existing `/api/stream/*` calls keep working** — they now proxy to a registry entry with id `"default"`. No client changes required, but you can migrate to `/api/streams/*` for per-camera isolation.
- **No breaking changes** to `/api/image/*`, `/api/video/*`, `/api/config`, `/api/export/*`, or `/health` beyond the new `X-API-Key` header requirement.
- **Set `RTSP_STREAMS`** if you want declarative multi-camera deployment (replaces the runtime-only `RTSP_URL` workflow).
