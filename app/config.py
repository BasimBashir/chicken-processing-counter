from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Runtime configuration. Counting is fixed to test.py's ObjectCounter
    behaviour (vertical center line, model defaults), so no detection/counting
    tuning knobs are exposed here."""

    # Source / model
    rtsp_url: str = ""
    model_path: str = "best.pt"

    # Filesystem
    upload_dir: str = "app/uploads"
    output_dir: str = "app/outputs"

    # Multi-stream: JSON list, e.g. RTSP_STREAMS='[{"id":"line-1","url":"rtsp://..."}]'
    rtsp_streams: str = ""
    max_streams: int = 10

    # Auth — if empty, /api/* is open (dev mode); else require X-API-Key header.
    api_key: str = ""

    class Config:
        env_file = ".env"


settings = Settings()
