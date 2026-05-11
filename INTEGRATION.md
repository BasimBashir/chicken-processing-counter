# Slaughtered Chicken Counter — Integration Guide

A pre-packaged FastAPI service that detects and counts three classes of poultry-line objects on a conveyor belt using YOLO. Pull the image, run it, talk to it over HTTP — no Python/ML setup required on the consuming side.

**Image:** `basim123/chicken-counter`
**Tags:** `latest`, `1.0.0`
**Base port:** `5581`
**Health probe:** `GET /health`
**Interactive docs:** `http://<host>:5581/docs` (Swagger UI)

---

## 1. Pull and run

The image is a single artifact that auto-selects GPU or CPU at startup and logs which one it chose.

### CPU (any host with Docker)

```bash
docker pull basim123/chicken-counter:latest

docker run -d --name chicken-counter \
  -p 5581:5581 \
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
  -v $(pwd)/uploads:/app/app/uploads \
  -v $(pwd)/outputs:/app/app/outputs \
  basim123/chicken-counter:latest
```

### Confirm device selection

```bash
docker logs --tail=5 chicken-counter
# [startup] CUDA available: true
# [startup] GPU: NVIDIA GeForce RTX 3090

curl http://localhost:5581/health
# {"status":"ok","model_path":"best.pt","cuda_available":true,"gpu":"NVIDIA GeForce RTX 3090"}
```

If you don't see the `[startup]` line, the container hasn't finished booting yet — wait a few seconds.

---

## 2. Configuration (environment variables)

All optional. Set with `-e VAR=value` on `docker run`, or in a Compose `environment:` block. Most are also live-patchable at runtime via `PATCH /api/config` (see section 4) so you rarely need to restart the container.

| Variable | Default | Description |
|---|---|---|
| `RTSP_URL` | _(empty)_ | Pre-configured RTSP source for `/api/stream/start` |
| `MODEL_PATH` | `best.pt` | YOLO weights inside the image. Swap to `best.engine` after running TensorRT export |
| `ROI_POSITION` | `0.5` | Counting line, 0 = left edge, 1 = right edge of frame |
| `CONFIDENCE` | `0.25` | Detection confidence threshold (0–1) |
| `NMS_IOU` | `0.45` | NMS IoU threshold (0–1) |
| `IMGSZ` | `640` | Inference image size; must be a multiple of 32 |
| `MAX_DISTANCE` | `40` | Max pixel distance for the centroid tracker to match a new detection to an existing track |
| `MAX_DISAPPEARED` | `50` | Frames a track can be unseen before it's dropped |

Mount volumes for persistent input/output:

| Host path | Container path | Purpose |
|---|---|---|
| `./uploads` | `/app/app/uploads` | Where uploaded images/videos land |
| `./outputs` | `/app/app/outputs` | H.264 re-encoded video outputs |

---

## 3. Classes returned

| Class | Description |
|---|---|
| `empty_shackles` | Empty shackle hooks |
| `single_legged` | Chicken hanging by one leg |
| `slaughtered_chicken` | Fully processed slaughtered chicken |

Each class has its own independent tracker — IDs are never shared across classes, so counts can't bleed.

---

## 4. HTTP API

Base URL: `http://<host>:5581`. All endpoints return JSON unless they stream media. CORS is **not** enabled by default; if you call from a browser on a different origin, front the service with a reverse proxy that adds the headers.

### 4.1 Health

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
```

Form field `file`: image (JPEG/PNG). Response body is the annotated JPEG. Counts come back as response headers:

| Header | Value |
|---|---|
| `X-Total-Count` | Total across all classes |
| `X-Count-Empty-Shackles` | int |
| `X-Count-Single-Legged` | int |
| `X-Count-Slaughtered-Chicken` | int |

```bash
curl -X POST http://localhost:5581/api/image/detect \
  -F "file=@photo.jpg" -o annotated.jpg -D headers.txt
```

### 4.3 Video session

A video session is processed in the background. Play and counting are **independent controls** — start playback to preview detections, then enable counting when you want the ROI line to start tallying.

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/video/upload` | Upload, returns `session_id` |
| POST | `/api/video/{id}/start` | Start frame processing |
| POST | `/api/video/{id}/stop` | Stop processing |
| POST | `/api/video/{id}/counting/start` | Enable ROI counting |
| POST | `/api/video/{id}/counting/stop` | Disable ROI counting |
| GET  | `/api/video/{id}/feed` | MJPEG stream of annotated frames |
| GET  | `/api/video/{id}/status` | Poll for per-class counts, frame number, fps, completion |
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
  "error": null
}
```

Typical flow:

```bash
SID=$(curl -s -X POST http://localhost:5581/api/video/upload \
        -F "file=@video.mp4" | jq -r .session_id)

curl -X POST http://localhost:5581/api/video/$SID/start
curl -X POST http://localhost:5581/api/video/$SID/counting/start

# Poll until is_complete: true
curl http://localhost:5581/api/video/$SID/status

curl http://localhost:5581/api/video/$SID/download -o output.mp4
```

### 4.4 Live RTSP stream

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/stream/start` | Body: `{"url":"rtsp://..."}`. Omit body to use `RTSP_URL` env var. |
| POST | `/api/stream/stop` | Disconnect |
| POST | `/api/stream/counting/start` | Enable ROI counting |
| POST | `/api/stream/counting/stop` | Disable ROI counting |
| GET  | `/api/stream/feed` | MJPEG stream |
| GET  | `/api/stream/status` | Per-class counts and connection state |

```bash
curl -X POST http://localhost:5581/api/stream/start \
  -H "Content-Type: application/json" \
  -d '{"url":"rtsp://admin:pass@192.168.1.100:554/stream"}'

curl -X POST http://localhost:5581/api/stream/counting/start
curl http://localhost:5581/api/stream/status
```

### 4.5 Runtime config

| Method | Endpoint | Purpose |
|---|---|---|
| GET   | `/api/config` | Current settings |
| PATCH | `/api/config` | Update one or more fields without a restart |

Patchable fields (constraints in parens):

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

```bash
curl -X PATCH http://localhost:5581/api/config \
  -H "Content-Type: application/json" \
  -d '{"roi_position": 0.4, "confidence": 0.3}'

# Switch to TensorRT engine without restarting:
curl -X PATCH http://localhost:5581/api/config \
  -H "Content-Type: application/json" \
  -d '{"model_path": "best.engine"}'
```

### 4.6 TensorRT engine export (GPU only)

Builds a `.engine` file from `best.pt` for ~2-3× faster inference. **Requires a GPU at runtime** — the build itself runs CUDA kernels. On a CPU container, the endpoint reports `FAILED` with the ultralytics error; the rest of the app keeps working.

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/export/tensorrt` | Start background FP16 export |
| GET  | `/api/export/tensorrt` | Poll state: `IDLE` / `RUNNING` / `DONE` / `FAILED` |

```bash
curl -X POST http://localhost:5581/api/export/tensorrt
# {"status":"started","model_path":"best.pt"}

# Poll until DONE
curl http://localhost:5581/api/export/tensorrt
# {"state":"DONE","output_path":"best.engine","elapsed_seconds":142.3}

# Switch to the engine live:
curl -X PATCH http://localhost:5581/api/config \
  -H "Content-Type: application/json" \
  -d '{"model_path":"best.engine"}'
```

---

## 5. Building on top of this service

### Python client (image detection + per-class counts)

```python
import requests

def count_chickens(image_path: str, host: str = "http://localhost:5581") -> dict:
    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{host}/api/image/detect",
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

### Python client (video polling pattern)

```python
import requests, time

HOST = "http://localhost:5581"

with open("line.mp4", "rb") as f:
    sid = requests.post(f"{HOST}/api/video/upload", files={"file": f}).json()["session_id"]

requests.post(f"{HOST}/api/video/{sid}/start")
requests.post(f"{HOST}/api/video/{sid}/counting/start")

while True:
    status = requests.get(f"{HOST}/api/video/{sid}/status").json()
    print(f"frame {status['frame_num']}/{status['total_frames']} "
          f"total={status['total_count']} fps={status['fps']:.1f}")
    if status["is_complete"]:
        break
    time.sleep(1)

with open("output.mp4", "wb") as f:
    f.write(requests.get(f"{HOST}/api/video/{sid}/download").content)
```

### JavaScript / Node client (live stream status)

```javascript
const HOST = "http://localhost:5581";

await fetch(`${HOST}/api/stream/start`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ url: "rtsp://admin:pass@cam.local:554/stream" }),
});

await fetch(`${HOST}/api/stream/counting/start`, { method: "POST" });

setInterval(async () => {
  const status = await fetch(`${HOST}/api/stream/status`).then(r => r.json());
  console.log("counts:", status.counts, "total:", status.total_count);
}, 1000);
```

### Embedding the MJPEG feed in a web page

```html
<img src="http://localhost:5581/api/stream/feed" />
<img src="http://localhost:5581/api/video/<session_id>/feed" />
```

Browsers render `multipart/x-mixed-replace` MJPEG natively from an `<img>` tag.

---

## 6. Production checklist

- **Reverse proxy** — front with nginx/Caddy/Traefik if exposing externally. Add TLS and auth there; the service has neither built-in.
- **Persistent volumes** — always mount `uploads/` and `outputs/`; the in-image filesystem is ephemeral.
- **Model swap** — to deploy a retrained model, mount your `best.pt` into the container at `/app/best.pt` (`-v ./my-best.pt:/app/best.pt:ro`) instead of rebuilding the image.
- **GPU memory** — one detector instance uses ~1.5 GB GPU at imgsz=640. A 4 GB card runs comfortably; an 8 GB card has headroom for TensorRT export.
- **Concurrency** — the model is a thread-safe singleton, but heavy concurrent video sessions will serialize on GPU. Scale horizontally with multiple replicas behind a load balancer if you need more throughput.
- **Logs** — stdout/stderr; collect via `docker logs` or your usual log shipper.

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `[startup] CUDA available: false` on a GPU host | Container was started without `--gpus all` (or without the GPU compose override) | Restart with `--gpus all` |
| `/api/export/tensorrt` always FAILED | Running on CPU container | TensorRT export needs a GPU; use the GPU run command |
| `connection refused` on first request | Uvicorn hasn't finished booting | Wait 5–10s after `docker run`; poll `/health` first |
| Detection counts feel off | Wrong `roi_position` or `confidence` | `PATCH /api/config` to tune live; check the MJPEG feed visually |
| MJPEG feed shows old frames | Browser cache on `<img>` | Add a cache-busting query string |

---

## 8. Versioning

| Tag | Notes |
|---|---|
| `latest` | Always points to the newest published build |
| `1.0.0` | First public release with single-image GPU/CPU auto-detect |

Pin to a version tag in production: `basim123/chicken-counter:1.0.0`.
