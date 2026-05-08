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

EXPOSE 5581

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5581"]
