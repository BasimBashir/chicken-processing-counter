from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Detection / counting defaults ──────────────────────────────────────
    rtsp_url: str = ""
    model_path: str = "best.pt"
    roi_position: float = 0.60
    confidence: float = 0.25
    conf_empty_shackles: float = 0.15
    nms_iou: float = 0.45
    imgsz: int = 640
    max_distance: int = 55
    max_disappeared: int = 15
    zone_half: int = 50
    appear_margin: int = 60

    # ── Filesystem ─────────────────────────────────────────────────────────
    upload_dir: str = "app/uploads"
    output_dir: str = "app/outputs"

    # ── Multi-stream ───────────────────────────────────────────────────────
    # JSON list of stream definitions, e.g.:
    #   RTSP_STREAMS='[{"id":"line-1","url":"rtsp://cam1/stream"},
    #                  {"id":"line-2","url":"rtsp://cam2/stream","roi_position":0.6}]'
    # Each entry must have id and url. Optional per-stream overrides:
    # roi_position, confidence, nms_iou, imgsz, max_distance, max_disappeared, zone_half, appear_margin.
    rtsp_streams: str = ""
    max_streams: int = 10

    # ── Batched inference worker ───────────────────────────────────────────
    batch_max: int = 16
    batch_window_ms: int = 25
    inference_queue_max: int = 100

    # ── Auth ───────────────────────────────────────────────────────────────
    # If empty, /api/* endpoints are open (dev mode). Set in production to
    # require header `X-API-Key: <value>` on every /api/* request.
    api_key: str = ""

    class Config:
        env_file = ".env"


settings = Settings()
