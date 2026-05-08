from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    rtsp_url: str = ""
    model_path: str = "best.pt"
    roi_position: float = 0.5
    confidence: float = 0.25
    nms_iou: float = 0.45
    imgsz: int = 640
    max_distance: int = 50
    max_disappeared: int = 15
    upload_dir: str = "app/uploads"
    output_dir: str = "app/outputs"

    class Config:
        env_file = ".env"


settings = Settings()
