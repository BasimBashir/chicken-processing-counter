from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Detection / counting defaults ──────────────────────────────────────
    # Tuned for fast conveyor on 848x478 source @ 25 fps with ~10-13 bboxes
    # per frame. Key insight from sample video: with imgsz=1280 actual loop
    # FPS drops to 11-18 (below source), so inter-processed-frame motion
    # spikes well above 50 px → tracks split → counts missed. Lowering imgsz
    # to 640 (source is already 848 wide, so no detail lost) lets inference
    # keep up at source FPS, then the tracker thresholds rarely matter.
    rtsp_url: str = ""
    model_path: str = "best.pt"
    roi_position: float = 0.65
    confidence: float = 0.25 #0.30
    conf_empty_shackles: float = 0.45
    # NMS uses agnostic_nms=True (across classes) to avoid double bboxes on
    # the same object. The IoU threshold here controls how aggressive that is:
    # lower = more suppression (risk: chicken near shackle suppresses the
    # chicken). 0.70 means only boxes overlapping >70% are merged, so adjacent
    # chicken/shackle pairs (typical IoU 0.3-0.5) both survive.
    nms_iou: float = 0.45
    imgsz: int = 1280
    max_distance: int = 90
    max_disappeared: int = 2
    # Belt travel per processed frame (px), used to seed per-track velocity
    # estimation in the counter. ~34 px/frame on the 1280-wide sub-stream
    # (6in shackle pitch, 119cm FOV, ~311 shackles/min). Self-tunes at runtime.
    conveyor_speed_px: float = 34.0
    # Half-width (px) of the counting band around roi_x. Band total width =
    # 2*zone_half. Wider band tolerates bbox flicker / brief frame stutter so
    # a bird crossing the line is not missed. 0 = single-pixel tripwire.
    zone_half: int = 15


    # ── Filesystem ─────────────────────────────────────────────────────────
    upload_dir: str = "app/uploads"
    output_dir: str = "app/outputs"

    # ── Multi-stream ───────────────────────────────────────────────────────
    # JSON list of stream definitions, e.g.:
    #   RTSP_STREAMS='[{"id":"line-1","url":"rtsp://cam1/stream"},
    #                  {"id":"line-2","url":"rtsp://cam2/stream","roi_position":0.6}]'
    # Each entry must have id and url. Optional per-stream overrides:
    # roi_position, confidence, nms_iou, imgsz, max_distance, max_disappeared, zone_half, appear_margin, conveyor_speed_px.
    rtsp_streams: str = ""
    max_streams: int = 10

    # ── Batched inference worker ───────────────────────────────────────────
    # Sized for ~10 streams × 25 fps = 250 fps aggregate throughput.
    batch_max: int = 32
    batch_window_ms: int = 10
    inference_queue_max: int = 400

    # ── Auth ───────────────────────────────────────────────────────────────
    # If empty, /api/* endpoints are open (dev mode). Set in production to
    # require header `X-API-Key: <value>` on every /api/* request.
    api_key: str = ""

    class Config:
        env_file = ".env"


settings = Settings()
