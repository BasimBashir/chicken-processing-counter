import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.core.auth import log_auth_state
from app.core.inference_worker import start_worker, stop_worker
from app.core.model_cache import preload_model
from app.core.runtime_config import runtime_config
from app.core.stream_registry import registry
from app.routers import image, video, stream, streams
from app.routers.config_router import router as config_router
from app.routers.export_router import router as export_router
from app.routers.health_router import router as health_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    snap = runtime_config.snapshot()
    os.makedirs(snap["upload_dir"], exist_ok=True)
    os.makedirs(snap["output_dir"], exist_ok=True)

    preload_model(snap["model_path"])
    start_worker(
        model_path=snap["model_path"],
        batch_max=snap["batch_max"],
        batch_window_ms=snap["batch_window_ms"],
        queue_max=snap["inference_queue_max"],
    )
    log_auth_state()
    registry.start_all_from_env()

    try:
        yield
    finally:
        registry.stop_all()
        stop_worker()


app = FastAPI(
    title="Slaughtered Chicken Counter",
    version="2.0.0",
    description=(
        "3-class chicken counting API for left-to-right conveyor belts. "
        "Classes: empty_shackles, single_legged, slaughtered_chicken. "
        "Vertical ROI line — objects counted as they cross left to right. "
        "Multi-stream RTSP support with batched GPU inference. "
        "All inference parameters tunable at runtime via PATCH /api/config."
    ),
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(config_router)
app.include_router(export_router)
app.include_router(image.router)
app.include_router(video.router)
app.include_router(stream.router)
app.include_router(streams.router)

static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
