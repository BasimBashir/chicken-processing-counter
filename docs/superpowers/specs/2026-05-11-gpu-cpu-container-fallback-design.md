# GPU/CPU Auto-Detection for Chicken-Counter Container

**Date:** 2026-05-11
**Status:** Approved
**Approach:** Option B — single image, runtime auto-detect

## Goal

Ship one container image that runs on both GPU and CPU hosts. The container should use the GPU when one is available and exposed to it, and silently fall back to CPU otherwise. Users pull a single image; the choice is made at `docker compose up` time, not at build time.

## Why one image

- Distribution simplicity: one image tag, one pull, one set of release notes.
- The Python application already auto-detects device via `ultralytics.YOLO`; no app-level branching is needed.
- The CUDA-12.6 PyTorch wheel runs unchanged on CPU-only hosts. `torch.cuda.is_available()` returns False and YOLO falls back to CPU. There is no install-time failure.

Accepted trade-off: image size stays around 3 GB because the CUDA libraries ship inside the torch wheel even on CPU-only hosts. Splitting into separate GPU/CPU images would shrink the CPU variant but doubles maintenance and was explicitly rejected.

## Non-goals

- Two separate images (gpu-tagged and cpu-tagged).
- A CPU-only PyTorch wheel variant.
- A custom entrypoint script that probes the GPU and rewrites configuration. PyTorch already does the probing.

## Components

### 1. Application code — no changes

- `app/core/detector.py` loads the YOLO model via `ultralytics.YOLO(path)`. Ultralytics selects the device automatically on each predict call.
- `app/routers/health_router.py` already reports `cuda_available` and GPU name, which is sufficient for runtime visibility.
- `app/core/exporter.py` lazy-imports `ultralytics` inside the export thread; `tensorrt` and `onnxruntime` are not touched until a user hits `POST /export/tensorrt`. On CPU-only hosts that endpoint will fail cleanly with the ultralytics error message; the rest of the app is unaffected.

### 2. `requirements.txt` — no changes

The current pins are correct for the one-image model:

- `torch` / `torchvision` from `https://download.pytorch.org/whl/cu126` — CUDA wheel, runs on CPU hosts.
- `onnxruntime-gpu`, `tensorrt`, `onnx`, `onnxslim` — install on any Linux x86_64 host; only needed at export time on GPU hosts.

### 3. `Dockerfile` — startup banner

Add a one-line CMD wrapper (or short shell preamble) that prints whether CUDA is visible before uvicorn starts. This gives operators a clear log signal of which device was selected on this host. The detection is done by a tiny inline Python call to `torch.cuda.is_available()`; nothing else changes.

The base image, system packages, working directory, copies, exposed port, and uvicorn invocation all remain as they are today.

### 4. `docker-compose.yml` — strip the hard GPU requirement

Remove the block:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

This block makes `docker compose up` fail on any host without the NVIDIA Container Toolkit. With it removed, the same compose file works on CPU-only hosts, on GPU hosts that have not exposed the GPU, and on GPU hosts that have.

All other fields (image, build, ports, volumes, environment, restart) stay the same.

### 5. Add `docker-compose.gpu.yml` — opt-in GPU override

A second compose file that contains *only* the nvidia device reservation, layered on top of the base file. GPU users run:

```
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

CPU users run the unchanged default:

```
docker compose up
```

This keeps the GPU configuration discoverable and version-controlled without making it mandatory.

### 6. `README.md` — document both invocations

Add a short "Running the container" section with two sub-sections:

- **CPU (default):** `docker compose up`. Works on any host with Docker.
- **GPU (NVIDIA):** Requires NVIDIA driver + nvidia-container-toolkit on the host. Run with both compose files layered (command shown above).

Mention that `GET /health` reports the selected device, and that the `/export/tensorrt` endpoint requires GPU.

## Data flow / runtime behavior

1. `docker compose up` starts the container.
2. The Dockerfile's startup preamble logs `CUDA available: true|false` and the GPU name if present.
3. uvicorn boots; `app/main` initializes routers.
4. On first inference request, `model_cache.get_model()` instantiates `YOLO(path)`. Ultralytics selects CUDA if visible to the process, else CPU.
5. Subsequent inferences reuse the cached model on the same device.

No code path changes between GPU and CPU runs.

## Error handling

- **Host has no NVIDIA driver:** Container starts. CUDA reports unavailable. Inference runs on CPU. No errors.
- **Host has GPU but did not include the `docker-compose.gpu.yml` override:** Container starts. CUDA reports unavailable inside the container (host GPU is not exposed to it). Inference runs on CPU. This is the user's choice; it is not a failure mode.
- **`/export/tensorrt` called on a CPU run:** Ultralytics raises during `model.export(format="engine")`. The existing `TensorRTExporter._run` catches it, sets state to `FAILED`, and records the error message. The HTTP endpoint returns the failure status. No process crash.

## Testing

Manual verification on two host configurations is sufficient for this change. There is no new logic to unit-test — the changes are configuration and a banner line.

1. **CPU host (or GPU host with no override file):**
   - `docker compose up` succeeds.
   - Banner logs `CUDA available: false`.
   - `GET /health` returns `cuda_available: false`.
   - A test inference completes successfully (slower than GPU).
   - `POST /export/tensorrt` returns a `FAILED` status.

2. **GPU host with override file:**
   - `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up` succeeds.
   - Banner logs `CUDA available: true` and the GPU name.
   - `GET /health` returns `cuda_available: true` and `gpu: <name>`.
   - A test inference completes (faster than CPU).
   - `POST /export/tensorrt` succeeds (or at least progresses past the import).

## Out of scope for this spec

- Reducing image size by splitting CPU/GPU variants.
- Apple Silicon (MPS) support.
- AMD ROCm support.
- Detection of partial GPU exposure (e.g. driver present but `nvidia-container-toolkit` missing).
