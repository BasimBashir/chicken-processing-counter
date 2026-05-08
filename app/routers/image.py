import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import Response

from app.config import settings
from app.core.detector import load_model, detect_frame
from app.core.annotator import annotate_image_detections

router = APIRouter(prefix="/api/image", tags=["image"])


@router.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return Response(content="Invalid image", status_code=400)

    model = load_model(settings.model_path)
    det_info = detect_frame(model, frame, settings.confidence)
    annotated, class_counts = annotate_image_detections(frame, det_info)

    _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])

    total = sum(class_counts.values())
    return Response(
        content=jpeg.tobytes(),
        media_type="image/jpeg",
        headers={
            "X-Total-Count": str(total),
            "X-Count-Empty-Shackles": str(class_counts.get("empty_shackles", 0)),
            "X-Count-Single-Legged": str(class_counts.get("single_legged", 0)),
            "X-Count-Slaughtered-Chicken": str(class_counts.get("slaughtered_chicken", 0)),
            "Access-Control-Expose-Headers": (
                "X-Total-Count, X-Count-Empty-Shackles, "
                "X-Count-Single-Legged, X-Count-Slaughtered-Chicken"
            ),
        },
    )
