# GPU/CPU Container Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a single chicken-counter container image that boots on both GPU and CPU hosts, auto-selecting the device at runtime.

**Architecture:** No application code changes. PyTorch's CUDA wheel already runs on CPU hosts; `ultralytics.YOLO` already auto-selects device. The fixes are all at the container infrastructure layer: strip the hard NVIDIA reservation from the default compose file, add an opt-in GPU override compose file, print a startup banner so operators see which device was chosen, and document both invocations in the README.

**Tech Stack:** Docker / Docker Compose, Python 3.11, PyTorch (CUDA 12.6 wheel), Ultralytics YOLO, FastAPI, uvicorn.

**Spec:** `docs/superpowers/specs/2026-05-11-gpu-cpu-container-fallback-design.md`

---

## File Map

- **Modify:** `docker-compose.yml` — remove the `deploy.resources.reservations` block so the default compose works on any host.
- **Create:** `docker-compose.gpu.yml` — minimal override that re-adds the NVIDIA device reservation. Layered on top of the default file with `-f`.
- **Create:** `docker-entrypoint.sh` — shell preamble that prints CUDA availability, then `exec`s uvicorn so signals propagate.
- **Create:** `.gitattributes` — force LF line endings on `*.sh` so Windows checkouts don't break the script when copied into the Linux image.
- **Modify:** `Dockerfile` — COPY the entrypoint, `chmod +x`, and replace the `CMD` to invoke it.
- **Modify:** `README.md` — update Quick Start Option A and the Docker section to document CPU-default vs GPU-override invocations.

---

## Verification model

This change is infrastructure-only — no new application logic, so no unit tests. Each task ends with a concrete verification command and expected output. The final task is an end-to-end manual run.

---

## Task 1: Strip hard GPU requirement from `docker-compose.yml`

**Files:**
- Modify: `docker-compose.yml` (remove lines 17–23, the `deploy.resources.reservations` block)

- [ ] **Step 1: Confirm the current state**

Run: `git status --short`
Expected: clean working tree (or only unrelated changes).

Run: `Get-Content docker-compose.yml`
Expected: file contains a `deploy:` block reserving an `nvidia` device — confirm before editing.

- [ ] **Step 2: Remove the `deploy:` block**

Edit `docker-compose.yml`. Delete these exact lines:

```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

The final file should be:

```yaml
services:
  chicken-counter:
    image: chicken-counter:latest
    build: .
    ports:
      - "5581:5581"
    volumes:
      - ./app/uploads:/app/app/uploads
      - ./app/outputs:/app/app/outputs
    environment:
      - RTSP_URL=${RTSP_URL:-}
      - MODEL_PATH=best.pt
      - ROI_POSITION=${ROI_POSITION:-0.5}
      - CONFIDENCE=${CONFIDENCE:-0.25}
      - MAX_DISTANCE=${MAX_DISTANCE:-40}
      - MAX_DISAPPEARED=${MAX_DISAPPEARED:-50}
    restart: unless-stopped
```

- [ ] **Step 3: Verify the merged config no longer references nvidia**

Run: `docker compose config`
Expected: YAML output that does **not** contain `driver: nvidia` or `capabilities`. The `restart: unless-stopped` line should still be present on the service.

- [ ] **Step 4: Commit**

```powershell
git add docker-compose.yml
git commit -m @'
Remove hard NVIDIA reservation from default compose

So `docker compose up` succeeds on CPU-only hosts. GPU hosts opt in
via the docker-compose.gpu.yml override added in a follow-up commit.
'@
```

Note: the `@'...'@` form is a PowerShell single-quoted here-string. The closing `'@` MUST be at column 0 of its own line. Backticks inside are literal (no escaping).

---

## Task 2: Add `docker-compose.gpu.yml` opt-in override

**Files:**
- Create: `docker-compose.gpu.yml`

- [ ] **Step 1: Create the override file**

Create `docker-compose.gpu.yml` with this exact content:

```yaml
services:
  chicken-counter:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

Notes for the implementer:
- `count: all` (not `count: 1`) lets a GPU host with multiple devices expose them all. PyTorch will pick device 0 by default; users with more advanced needs can set `CUDA_VISIBLE_DEVICES`.
- Indentation must use spaces (no tabs). Match the style of `docker-compose.yml`.

- [ ] **Step 2: Verify the override merges cleanly**

Run: `docker compose -f docker-compose.yml -f docker-compose.gpu.yml config`
Expected: YAML output containing both the original service definition **and** a `deploy.resources.reservations.devices` block with `driver: nvidia` and `capabilities: [gpu]`. No "yaml" or "compose" errors.

Note: this command only validates the merge — it does not require an NVIDIA driver to be present on the host.

- [ ] **Step 3: Commit**

```powershell
git add docker-compose.gpu.yml
git commit -m @'
Add docker-compose.gpu.yml override for opt-in NVIDIA GPU use

Layer on top of docker-compose.yml with -f to enable GPU passthrough:
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
'@
```

---

## Task 3: Add CUDA-status startup banner

**Files:**
- Create: `docker-entrypoint.sh`
- Create: `.gitattributes`
- Modify: `Dockerfile` (replace `CMD` on line 23; add a COPY+chmod for the entrypoint)

- [ ] **Step 1: Create `.gitattributes` to force LF line endings on shell scripts**

This repo is on Windows with `core.autocrlf` likely on. Without this, `docker-entrypoint.sh` will be checked out with CRLF, and the Linux image will fail with `/bin/sh^M: bad interpreter: No such file or directory`.

Create `.gitattributes` at the repo root with this exact content:

```
*.sh text eol=lf
docker-entrypoint.sh text eol=lf
Dockerfile text eol=lf
```

- [ ] **Step 2: Create `docker-entrypoint.sh`**

Create `docker-entrypoint.sh` at the repo root with this exact content (single trailing newline, LF line endings):

```sh
#!/bin/sh
set -e

python - <<'PY'
import torch
print(f"[startup] CUDA available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"[startup] GPU: {torch.cuda.get_device_name(0)}", flush=True)
PY

exec uvicorn app.main:app --host 0.0.0.0 --port 5581
```

Why each piece:
- `set -e` — fail fast if the python probe errors.
- `<<'PY'` heredoc with single quotes — keeps the script readable and avoids escape-soup in the Dockerfile.
- `flush=True` — guarantees the banner appears in `docker logs` even when stdout is line-buffered.
- `exec uvicorn …` — replaces the shell process so `uvicorn` receives `SIGTERM` directly on `docker stop`; without `exec`, the shell traps signals and uvicorn never shuts down cleanly.

- [ ] **Step 3: Verify the entrypoint file uses LF line endings**

Run: `(Get-Content docker-entrypoint.sh -Raw) -match "\r\n"`
Expected: `False` (no CRLF anywhere). If `True`, re-save the file as UTF-8 without BOM and with LF endings; some editors honor `.gitattributes` only on the next commit/checkout cycle.

- [ ] **Step 4: Modify the `Dockerfile`**

Replace lines 21–23 of `Dockerfile`:

```dockerfile
EXPOSE 5581

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5581"]
```

with:

```dockerfile
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

EXPOSE 5581

CMD ["/docker-entrypoint.sh"]
```

The final `Dockerfile` should be:

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY best.pt .
COPY .env .

RUN mkdir -p app/uploads app/outputs

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

EXPOSE 5581

CMD ["/docker-entrypoint.sh"]
```

- [ ] **Step 5: Build the image and verify the banner**

Run: `docker compose build`
Expected: build completes with no errors. Final image tag `chicken-counter:latest`.

Run: `docker compose up -d`
Expected: container starts, no error.

Run: `docker compose logs --tail=20 chicken-counter`
Expected: log output contains a line `[startup] CUDA available: false` (on a CPU host) or `[startup] CUDA available: true` followed by `[startup] GPU: <name>` (on a GPU host with the override file applied), followed by the usual uvicorn `Application startup complete.` line.

Run: `docker compose down`
Expected: container stops cleanly within a couple of seconds (this confirms the `exec` in the entrypoint is propagating SIGTERM correctly — without `exec`, `docker stop` would hang for the default 10-second timeout).

- [ ] **Step 6: Commit**

```powershell
git add .gitattributes docker-entrypoint.sh Dockerfile
git commit -m @'
Add startup banner reporting CUDA availability

Introduces docker-entrypoint.sh which prints whether CUDA is visible
to the container before exec-ing uvicorn. Also adds .gitattributes
to force LF endings on shell scripts so Windows checkouts work
inside the Linux image.
'@
```

---

## Task 4: Update README to document CPU-default and GPU-override invocations

**Files:**
- Modify: `README.md` (Quick Start Option A around line 42, Docker section around lines 432–449)

- [ ] **Step 1: Update Quick Start "Option A: Docker"**

Replace these lines (around lines 42–48):

```markdown
### Option A: Docker (recommended)

```bash
docker compose up --build
```

Open **http://localhost:5581**
```

with:

````markdown
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
````

- [ ] **Step 2: Update the "Docker" section near the bottom of the README**

Replace this section (around lines 432–449):

```markdown
## Docker

### Build and run

```bash
docker compose up --build
```

### GPU support

`docker-compose.yml` includes NVIDIA GPU reservation. Requires:
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Docker configured with the `nvidia` runtime

### CPU-only

Remove the `deploy.resources.reservations` block from `docker-compose.yml`.
```

with:

````markdown
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
````

- [ ] **Step 3: Verify the README renders sensibly**

Run: `Get-Content README.md | Select-String -Pattern '## Docker' -Context 0,40`
Expected: the new "Docker" section appears with both CPU and GPU subsections and the verification snippets. Spot-check no leftover lines from the old wording (no "Remove the `deploy.resources.reservations` block").

Run: `Get-Content README.md | Select-String -Pattern '### Option A' -Context 0,20`
Expected: the new Quick Start Option A appears with the CPU/GPU split.

- [ ] **Step 4: Commit**

```powershell
git add README.md
git commit -m @'
Document CPU-default + GPU-override Docker invocations

Updates Quick Start and Docker sections to reflect the new single-
image, runtime-auto-detect model. Includes the GPU override command,
how to confirm which device was selected, and which endpoints still
require a GPU.
'@
```

---

## Task 5: End-to-end manual verification

**Files:**
- No code changes. This task is verification only.

- [ ] **Step 1: Tear down any running container from earlier tasks**

Run: `docker compose down`
Expected: any running container stops; no error if nothing was running.

- [ ] **Step 2: Verify the CPU path**

Run: `docker compose up --build -d`
Expected: build succeeds, container starts, no errors.

Run: `docker compose logs --tail=10 chicken-counter`
Expected: contains `[startup] CUDA available: <true|false>` (the value depends on whether the current host has a GPU exposed to Docker — but on the default compose file with no override, the line should print without crashing the container even if the host has no GPU).

Run: `curl http://localhost:5581/health`
Expected: HTTP 200 with JSON containing `status`, `model_path`, and `cuda_available`. On a CPU-only host `cuda_available` is `false` and the `gpu` field is absent.

Run: `curl -X POST http://localhost:5581/api/export/tensorrt; sleep 2; curl http://localhost:5581/api/export/tensorrt`
Expected on a CPU host: the second `curl` returns a JSON body with `"state": "FAILED"` and an error message from ultralytics. The container is still up; the app didn't crash.

Run: `docker compose down`
Expected: clean shutdown within a few seconds.

- [ ] **Step 3 (if a GPU host is available): Verify the GPU path**

Skip this step on a CPU-only development machine. Note in the commit message or PR description that the GPU path was not exercised locally.

Run: `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d`
Expected: build succeeds, container starts.

Run: `docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs --tail=10 chicken-counter`
Expected: log contains `[startup] CUDA available: true` and `[startup] GPU: <gpu name>`.

Run: `curl http://localhost:5581/health`
Expected: JSON with `"cuda_available": true` and a `gpu` field naming the device.

Run: `docker compose -f docker-compose.yml -f docker-compose.gpu.yml down`
Expected: clean shutdown.

- [ ] **Step 4: Verify the override compose file fails gracefully on CPU-only hosts**

This is the failure-mode check — we want to confirm that *not* using the override is the correct path on a CPU host. The override compose file should fail on a CPU host (this is expected and matches the spec — it's a GPU-only opt-in).

Run (on a CPU host only, expected to fail): `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up`
Expected: Docker emits an error like `could not select device driver "nvidia" with capabilities: [[gpu]]`. The default `docker compose up` (without the override) is the right command for this host.

Skip on GPU hosts.

- [ ] **Step 5: Final commit (only if there are post-verification fixes)**

If verification surfaced no issues, no commit is needed here — the prior task commits stand. If something needed fixing, commit the fix with a message referencing this task.

---

## Spec Coverage Check

Mapping each requirement in `docs/superpowers/specs/2026-05-11-gpu-cpu-container-fallback-design.md` to a task:

| Spec section | Implemented in |
|---|---|
| 1. App code — no changes | (intentional no-op) |
| 2. `requirements.txt` — no changes | (intentional no-op) |
| 3. `Dockerfile` startup banner | Task 3 |
| 4. `docker-compose.yml` strip hard GPU requirement | Task 1 |
| 5. Add `docker-compose.gpu.yml` opt-in override | Task 2 |
| 6. README documents both invocations | Task 4 |
| Testing: CPU-host verification | Task 5 Step 2 |
| Testing: GPU-host verification | Task 5 Step 3 |
| Error handling: CPU `/export/tensorrt` returns FAILED cleanly | Task 5 Step 2 |
