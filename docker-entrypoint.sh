#!/bin/sh
set -e

python - <<'PY'
import torch
print(f"[startup] CUDA available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"[startup] GPU: {torch.cuda.get_device_name(0)}", flush=True)
PY

exec uvicorn app.main:app --host 0.0.0.0 --port 5581
