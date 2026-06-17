import numpy as np
from app.core.annotator import annotate_boxes, annotate_image_detections


def test_annotate_boxes_draws_without_mutating_input():
    frame = np.zeros((200, 320, 3), dtype=np.uint8)
    boxes = [{"x1": 10, "y1": 10, "x2": 80, "y2": 120,
              "class_name": "slaughtered_chicken", "conf": 0.9, "obj_id": 3}]
    out = annotate_boxes(frame, boxes)
    assert out.shape == frame.shape
    assert out.sum() > 0          # something was drawn
    assert frame.sum() == 0       # original untouched


def test_annotate_boxes_handles_missing_optional_fields():
    frame = np.zeros((200, 320, 3), dtype=np.uint8)
    boxes = [{"x1": 5, "y1": 5, "x2": 50, "y2": 60, "class_name": "empty_shackles"}]
    out = annotate_boxes(frame, boxes)   # no conf, no obj_id
    assert out.shape == frame.shape


def test_annotate_image_detections_still_returns_counts():
    frame = np.zeros((200, 320, 3), dtype=np.uint8)
    det = [{"x1": 5, "y1": 5, "x2": 50, "y2": 60,
            "conf": 0.8, "class_name": "single_legged"}]
    out, counts = annotate_image_detections(frame, det)
    assert out.shape == frame.shape
    assert counts == {"single_legged": 1}
