# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 5 --timeout 120 -r requirements.txt

COPY app/ app/
COPY best.pt .

RUN mkdir -p app/uploads app/outputs

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

EXPOSE 5581

CMD ["/docker-entrypoint.sh"]
