#!/bin/sh
set -e

# Build-on-first-boot TensorRT engine cache.
# A TensorRT .engine is locked to {GPU compute capability, TRT version, precision,
# imgsz, source .pt}. Baking one into the image breaks portability across hosts,
# so we build it on the target machine on first launch and cache it to a named
# volume keyed by those dimensions. Subsequent boots reuse the cached engine.
#
# Disable with TRT_AUTO_BUILD=0 (falls back to .pt — works on CPU too).

export TRT_CACHE_DIR="${TRT_CACHE_DIR:-/app/engine_cache}"
export TRT_ENGINE_PATH="${TRT_ENGINE_PATH:-/app/best.engine}"
export TRT_HALF="${TRT_HALF:-true}"
export TRT_AUTO_BUILD="${TRT_AUTO_BUILD:-1}"

mkdir -p "$TRT_CACHE_DIR"
: > /tmp/trt.env

if [ "$TRT_AUTO_BUILD" = "1" ]; then
  python - <<'PY' || echo "[trt] entrypoint hook exited non-zero — continuing with .pt"
import os, sys, hashlib, shutil
from pathlib import Path

# Pull model_path from Settings so the engine and uvicorn agree on the weights.
# imgsz is NO LONGER a Settings field (counting is fixed to ObjectCounter, which
# infers at the ultralytics default of 640 — matching the 640x480 sub-stream).
# The engine is built at TRT_IMGSZ (default 640) so the .engine's fixed input
# shape matches what ObjectCounter feeds it. Override with TRT_IMGSZ if you run
# a higher-resolution main stream.
from app.config import Settings
s = Settings()

src_pt = s.model_path if s.model_path.endswith(".pt") else "best.pt"
model_pt    = Path(src_pt if Path(src_pt).is_absolute() else f"/app/{src_pt}")
engine_dir  = Path(os.environ["TRT_CACHE_DIR"])
engine_link = Path(os.environ["TRT_ENGINE_PATH"])
imgsz       = int(os.environ.get("TRT_IMGSZ", "640"))
half        = os.environ.get("TRT_HALF", "true").lower() == "true"
env_file    = Path("/tmp/trt.env")

def bail(msg):
    print(f"[trt] {msg} — falling back to .pt", flush=True)
    sys.exit(0)

if not model_pt.exists():
    bail(f"{model_pt} missing")

try:
    import torch
except Exception as e:
    bail(f"torch import failed: {e}")

if not torch.cuda.is_available():
    bail("no CUDA visible to container")

try:
    import tensorrt as trt
    trt_ver = trt.__version__
except Exception as e:
    bail(f"tensorrt module unavailable: {e}")

gpu  = torch.cuda.get_device_name(0).replace(" ", "_").replace("/", "_")
cap  = "sm" + "".join(map(str, torch.cuda.get_device_capability(0)))
prec = "fp16" if half else "fp32"
pt_h = hashlib.sha256(model_pt.read_bytes()).hexdigest()[:8]
cache_name = f"{gpu}_{cap}_trt{trt_ver}_{prec}_imgsz{imgsz}_pt{pt_h}.engine"
cached = engine_dir / cache_name

print(f"[trt] gpu={gpu} {cap}  trt={trt_ver}  prec={prec}  imgsz={imgsz}  pt={pt_h}", flush=True)
print(f"[trt] cache key: {cache_name}", flush=True)

engine_dir.mkdir(parents=True, exist_ok=True)

if not cached.exists():
    print("[trt] no cached engine — building (2-6 min on RTX 3090; longer on smaller GPUs)", flush=True)
    try:
        from ultralytics import YOLO
        out = YOLO(str(model_pt)).export(format="engine", imgsz=imgsz, half=half)
        out_path = Path(out) if out else model_pt.with_suffix(".engine")
        if not out_path.is_absolute():
            out_path = (model_pt.parent / out_path).resolve()
        shutil.copy2(out_path, cached)
        print(f"[trt] cached engine at {cached}", flush=True)
    except Exception as e:
        bail(f"export failed: {e}")
else:
    print(f"[trt] reusing cached engine {cached}", flush=True)

if engine_link.is_symlink() or engine_link.exists():
    engine_link.unlink()
try:
    engine_link.symlink_to(cached)
except OSError:
    shutil.copy2(cached, engine_link)

print(f"[trt] active engine: {engine_link} -> {cached.name}", flush=True)
env_file.write_text(f"MODEL_PATH={engine_link.name}\n")
PY

  if [ -s /tmp/trt.env ]; then
    set -a
    . /tmp/trt.env
    set +a
  fi
fi

python - <<'PY'
import os, torch
print(f"[startup] CUDA available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"[startup] GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"[startup] MODEL_PATH={os.environ.get('MODEL_PATH', 'best.pt')}", flush=True)
PY

exec uvicorn app.main:app --host 0.0.0.0 --port 5581
