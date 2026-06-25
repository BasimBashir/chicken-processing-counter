# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Install in stages so no single image layer is a ~13 GB blob. Docker Hub's
# registry 502s when finalizing a multi-GB blob, so a monolithic
# `-r requirements.txt` layer can't be pushed. Each RUN is its own layer, so
# isolating the two heavy wheel sets (torch+CUDA ~6 GB, then TensorRT ~4 GB)
# keeps every blob pushable. The shared pip cache mount means no re-downloads.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 5 --timeout 120 \
      --extra-index-url https://download.pytorch.org/whl/cu128 \
      torch==2.11.0+cu128 torchvision==0.26.0+cu128
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 5 --timeout 120 "tensorrt>=10.0,<11"
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 5 --timeout 120 -r requirements.txt

COPY app/ app/
COPY best.pt .

RUN mkdir -p app/uploads app/outputs

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

EXPOSE 5581

CMD ["/docker-entrypoint.sh"]
