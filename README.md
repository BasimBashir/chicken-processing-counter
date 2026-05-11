# Slaughtered Chicken Counting System

A production-ready detection and counting system built with **Ultralytics YOLO** and **FastAPI**. Tracks three object classes on a **left-to-right conveyor belt** using a **vertical ROI line** — each object is counted exactly once as its centroid crosses from left to right.

---

## Classes

| Class | Color | Description |
|-------|-------|-------------|
| `empty_shackles` | Orange | Empty shackle hooks with no chicken |
| `single_legged` | Gold | Chicken hanging by a single leg |
| `slaughtered_chicken` | Green | Fully processed slaughtered chicken |

Each class has its own **independent tracker and counter** — counts never bleed between classes.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **3-Class Independent Counting** | Separate `CentroidTracker` + count per class, no cross-class ID collisions |
| **Vertical ROI Line** | Left-to-right conveyor — vertical counting line with rightward-arrow indicators |
| **Hungarian + IoU Tracker** | Globally-optimal bbox-to-track matching via `scipy.optimize.linear_sum_assignment` |
| **Web Dashboard** | Dark-themed UI for image detection, video processing, and live stream monitoring |
| **REST API** | Full FastAPI with Swagger docs at `/docs` |
| **Live Config** | `PATCH /api/config` — tune ROI, confidence, NMS IoU, imgsz at runtime without restart |
| **Image Detection** | Upload image, get annotated JPEG with per-class count headers |
| **Video Processing** | Upload video, control playback and counting independently, download H.264 output |
| **Live RTSP Stream** | Connect to any RTSP camera, count in real time |
| **TensorRT Export** | `POST /api/export/tensorrt` — background export with IDLE→RUNNING→DONE/FAILED state |
| **Health Check** | `GET /health` — GPU info and active model path |
| **Visual Annotations** | Per-class color-coded bboxes, motion trails, crossing flash effects, dashboard panel |
| **Docker + GPU** | Single-container deployment with NVIDIA GPU support |
| **CLI Tool** | Standalone `detect_and_count.py` — no web server needed |

---

## Quick Start

### Option A: Docker (recommended)

The same image runs on both CPU and GPU hosts — it auto-selects the device at startup and logs which one it chose.

**CPU (default):**

```bash
docker compose up --build
```

**GPU (NVIDIA + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) required):**

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Open **http://localhost:5581**. The container logs a `[startup] CUDA available: ...` line so you can confirm the selected device. `GET /health` also reports it.

### Option B: Local Setup

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

Install dependencies (includes PyTorch for CUDA 12.6 via the extra index in `requirements.txt`):

```bash
pip install -r requirements.txt
```

For a different CUDA version, install PyTorch first from https://pytorch.org/get-started/locally/ before running the above.

Start the server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 5581
```

Open **http://localhost:5581**

> Port `5581` — different from related systems on `5580` to allow both to run simultaneously.

---

## Model Training

### 1. Prepare the dataset

Unzip your Roboflow YOLO export (e.g. `Birds Counting.v1i.yolo26.zip`) into the project root so it produces:

```
dataset/
    data.yaml
    train/images/   train/labels/
    valid/images/   valid/labels/
    test/images/    test/labels/    (optional)
```

The three class names in `data.yaml` must be exactly:
```
names: [empty_shackles, single_legged, slaughtered_chicken]
```

### 2. Run training

```bash
python train.py
```

Uses **YOLO26s** pretrained on COCO (transfer learning). Key defaults:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `epochs` | 50 | with `patience=5` early stop |
| `imgsz` | 512 | |
| `batch` | 16 | |
| `device` | 0 | GPU 0; change to `"cpu"` for CPU-only |
| `optimizer` | auto | |
| `amp` | True | mixed precision |
| `flipud` | 0.0 | disabled — conveyor moves horizontally |
| `mosaic` | 1.0 | enabled |
| `mixup` | 0.1 | |

Best weights are saved to `runs/chicken_counter/weights/best.pt`.

### 3. Deploy the model

```bash
# Windows
copy runs\chicken_counter\weights\best.pt best.pt

# Linux / macOS
cp runs/chicken_counter/weights/best.pt best.pt
```

Restart the server (or update `MODEL_PATH` in `.env`) to use the new weights.

### 4. Optional: Export to TensorRT

For maximum GPU inference speed, export after training:

```bash
# Via API (server must be running)
curl -X POST http://localhost:5581/api/export/tensorrt

# Check progress
curl http://localhost:5581/api/export/tensorrt
```

Or directly:

```python
from ultralytics import YOLO
model = YOLO("best.pt")
model.export(format="engine", imgsz=640, half=True)
```

Then set `MODEL_PATH=best.engine` in `.env`.

---

## Web Dashboard

### Image Detection (`/`)

Upload an image via drag-and-drop or file picker. Shows original and annotated side-by-side with per-class counts. Download the annotated result.

### Video Processing (`/video.html`)

1. Upload a video file
2. Click **Play** to start processing (detections shown immediately)
3. Click **Start Counting** to enable the vertical ROI line tracking
4. Watch per-class counts update live in the stats panel
5. Click **Stop** when done, then **Download Output (H.264)**

Play and counting are independent controls — you can observe detections before committing to counting.

### Live Stream (`/stream.html`)

1. Enter your RTSP URL (or pre-configure in `.env`)
2. Click **Connect** to start the live feed
3. Click **Start Counting** to begin ROI line tracking
4. Adjust **ROI Position** and **Confidence** via the config API in real time

### API Docs (`/docs`)

Auto-generated Swagger UI with all endpoints.

---

## REST API

Base URL: `http://localhost:5581`

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Server status, GPU info, active model path |

```bash
curl http://localhost:5581/health
# {"status":"ok","model_path":"best.pt","cuda_available":true,"gpu":"NVIDIA ..."}
```

### Image

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/image/detect` | Upload image, returns annotated JPEG |

**Response headers:**
- `X-Total-Count`
- `X-Count-Empty-Shackles`
- `X-Count-Single-Legged`
- `X-Count-Slaughtered-Chicken`

```bash
curl -X POST http://localhost:5581/api/image/detect \
  -F "file=@photo.jpg" -o annotated.jpg -D -
```

### Video

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/video/upload` | Upload video, returns `session_id` |
| `POST` | `/api/video/{id}/start` | Start processing |
| `POST` | `/api/video/{id}/stop` | Stop processing |
| `POST` | `/api/video/{id}/counting/start` | Enable ROI counting |
| `POST` | `/api/video/{id}/counting/stop` | Disable ROI counting |
| `GET` | `/api/video/{id}/feed` | MJPEG stream of annotated frames |
| `GET` | `/api/video/{id}/status` | Per-class counts, frame, fps, completion state |
| `GET` | `/api/video/{id}/download` | Download H.264 re-encoded output |

**Status response:**
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

```bash
curl -X POST http://localhost:5581/api/video/upload -F "file=@video.mp4"
# {"session_id":"a1b2c3d4","filename":"video.mp4"}

curl -X POST http://localhost:5581/api/video/a1b2c3d4/start
curl -X POST http://localhost:5581/api/video/a1b2c3d4/counting/start
curl http://localhost:5581/api/video/a1b2c3d4/status
curl http://localhost:5581/api/video/a1b2c3d4/download -o output.mp4
```

### Live Stream

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/stream/start` | Connect to RTSP stream |
| `POST` | `/api/stream/stop` | Disconnect |
| `POST` | `/api/stream/counting/start` | Enable ROI counting |
| `POST` | `/api/stream/counting/stop` | Disable ROI counting |
| `GET` | `/api/stream/feed` | MJPEG stream |
| `GET` | `/api/stream/status` | Per-class counts and connection state |

```bash
curl -X POST http://localhost:5581/api/stream/start \
  -H "Content-Type: application/json" \
  -d '{"url": "rtsp://admin:pass@192.168.1.100:554/stream"}'

curl -X POST http://localhost:5581/api/stream/counting/start
curl http://localhost:5581/api/stream/status
```

### Config

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/config` | Get all current settings |
| `PATCH` | `/api/config` | Update one or more settings live (no restart needed) |

**Patchable fields:**

| Field | Type | Constraints |
|-------|------|-------------|
| `rtsp_url` | string | — |
| `model_path` | string | validated by attempting to load |
| `roi_position` | float | 0 < x < 1 |
| `confidence` | float | 0 < x < 1 |
| `nms_iou` | float | 0 < x < 1 |
| `imgsz` | int | multiple of 32 |
| `max_distance` | int | ≥ 1 |
| `max_disappeared` | int | ≥ 1 |

```bash
# Move ROI line to 40% from left, lower confidence
curl -X PATCH http://localhost:5581/api/config \
  -H "Content-Type: application/json" \
  -d '{"roi_position": 0.4, "confidence": 0.3}'

# Switch to TensorRT engine at runtime
curl -X PATCH http://localhost:5581/api/config \
  -H "Content-Type: application/json" \
  -d '{"model_path": "best.engine"}'
```

### TensorRT Export

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/export/tensorrt` | Start background FP16 engine export |
| `GET` | `/api/export/tensorrt` | Poll export state (IDLE / RUNNING / DONE / FAILED) |

```bash
curl -X POST http://localhost:5581/api/export/tensorrt
curl http://localhost:5581/api/export/tensorrt
# {"state":"DONE","output_path":"best.engine","elapsed_seconds":142.3}
```

---

## CLI Usage

Standalone `detect_and_count.py` — no server required.

### Image detection

```bash
python detect_and_count.py path/to/image.jpg
python detect_and_count.py path/to/image.jpg --save result.jpg
```

### Video with ROI counting

```bash
python detect_and_count.py path/to/video.mp4 --save output.mp4
python detect_and_count.py path/to/video.mp4 --roi 0.5 --conf 0.3 --iou 0.45
```

### Live RTSP stream

```bash
python detect_and_count.py "rtsp://user:pass@camera-ip:554/stream"
```

Press **q** to stop.

### CLI options

| Argument | Description | Default |
|----------|-------------|---------|
| `input` | Path to image, video, or RTSP URL | required |
| `--save` | Save annotated output to this path | — |
| `--conf` | Detection confidence threshold | `0.25` |
| `--iou` | NMS IoU threshold | `0.45` |
| `--imgsz` | Inference image size (multiple of 32) | `640` |
| `--model` | Path to YOLO model weights | `best.pt` |
| `--roi` | ROI line position (0.0=left, 1.0=right) | `0.5` |
| `--max-distance` | Max pixel distance for track matching | `50` |
| `--max-disappeared` | Frames before dropping a lost track | `15` |

---

## Configuration

`.env` file (all fields also settable as environment variables):

```env
RTSP_URL=rtsp://user:pass@camera-ip:554/stream
MODEL_PATH=best.pt
ROI_POSITION=0.5
CONFIDENCE=0.25
NMS_IOU=0.45
IMGSZ=640
MAX_DISTANCE=50
MAX_DISAPPEARED=15
```

`ROI_POSITION=0.5` places the vertical counting line at the centre of the frame width.  
All fields except `RTSP_URL` and `MODEL_PATH` can also be updated live via `PATCH /api/config`.

---

## Project Structure

```
Slaughtered_Chicken_Counting/
├── app/
│   ├── main.py                  # FastAPI app with lifespan (model preload)
│   ├── config.py                # Pydantic-settings boot config
│   ├── core/
│   │   ├── runtime_config.py    # Thread-safe live config (PATCH without restart)
│   │   ├── model_cache.py       # Thread-safe YOLO singleton cache
│   │   ├── detector.py          # Ultralytics YOLO inference → list[dict]
│   │   ├── tracker.py           # Hungarian + IoU CentroidTracker
│   │   ├── counter.py           # Per-class ROI crossing logic (3 independent trackers)
│   │   ├── line_counter.py      # Trackerless fallback counter (vertical ROI)
│   │   ├── annotator.py         # Bboxes, trails, flash, dashboard, vertical ROI line
│   │   ├── video_processor.py   # Background thread + MJPEG frame buffer
│   │   └── exporter.py          # TensorRT export state machine
│   ├── routers/
│   │   ├── image.py             # POST /api/image/detect
│   │   ├── video.py             # Video session management
│   │   ├── stream.py            # RTSP stream endpoints
│   │   ├── config_router.py     # GET + PATCH /api/config
│   │   ├── export_router.py     # POST + GET /api/export/tensorrt
│   │   └── health_router.py     # GET /health
│   └── static/                  # Frontend HTML / CSS / JS
├── train.py                     # YOLO26s training script
├── detect_and_count.py          # Standalone CLI tool (image / video / RTSP)
├── dataset/                     # Unzip Roboflow export here
│   ├── data.yaml
│   ├── train/
│   └── valid/
├── best.pt                      # Trained model weights (place here after training)
├── .env                         # Runtime configuration
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Docker

The image is a **single artifact** that works on both CPU and GPU hosts. PyTorch's CUDA-12.6 wheel is bundled and falls back to CPU automatically when no GPU is visible; the startup script logs which device was selected.

### CPU (default)

```bash
docker compose up --build
```

No host prerequisites beyond Docker itself. Inference runs on CPU.

### GPU (NVIDIA)

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

The override file `docker-compose.gpu.yml` adds an NVIDIA device reservation on top of the default compose file. Requires:

- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed on the host
- Docker configured with the `nvidia` runtime

### Verifying the selected device

Check the container logs after startup:

```bash
docker compose logs --tail=5 chicken-counter
# [startup] CUDA available: true
# [startup] GPU: NVIDIA GeForce RTX 4090
```

Or hit the health endpoint:

```bash
curl http://localhost:5581/health
```

### Endpoints that require a GPU

- `POST /api/export/tensorrt` — builds a TensorRT engine; requires CUDA. On CPU runs the endpoint returns a `FAILED` status with the ultralytics error; the rest of the app is unaffected.

---

## Counting Logic

- The conveyor moves **left to right** — the ROI line is **vertical** at `roi_x = width × roi_position`.
- Each class has its own `CentroidTracker` — IDs are never shared across classes.
- **Hungarian assignment** with an IoU-aware cost matrix ensures correct matching of dense, overlapping objects.
- A count fires when `prev_cx < roi_x ≤ current_cx` (centroid crosses the line moving right).
- If an object is first detected already past the line (`cx ≥ roi_x`), it is counted immediately.
- Counted tracks are retired on their first disappearance frame to free their ID and prevent reuse.

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
