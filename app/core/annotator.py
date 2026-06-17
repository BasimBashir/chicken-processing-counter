import cv2

COLORS = {
    "panel_bg":  (20, 20, 20),
    "accent":    (0, 200, 255),
    "white":     (255, 255, 255),
    "dim":       (160, 160, 160),
}

# Per-class colors in BGR (unchanged from the previous annotator).
CLASS_COLORS = {
    "empty_shackles":      (0, 165, 255),   # orange
    "single_legged":       (255, 200, 0),   # gold/cyan
    "slaughtered_chicken": (0, 230, 118),   # green
}

CLASS_LABELS = {
    "empty_shackles":      "E.Shackle",
    "single_legged":       "Single",
    "slaughtered_chicken": "Slaughtered",
}


def draw_rounded_rect(img, pt1, pt2, color, radius=12, thickness=-1, alpha=0.85):
    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2
    r = radius
    cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, thickness)
    cv2.ellipse(overlay, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(overlay, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(overlay, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
    cv2.ellipse(overlay, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_bbox(img, x1, y1, x2, y2, counted, conf, class_name, obj_id=None):
    """Existing corner-accented bbox + small label. Colors/size unchanged."""
    color = CLASS_COLORS.get(class_name, COLORS["accent"])
    if counted:
        b, g, r = color
        color = (min(b + 40, 255), min(g + 40, 255), min(r + 40, 255))
    corner_len = 8
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, 2, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, 2, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, 2, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, 2, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, 2, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, 2, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, 2, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, 2, cv2.LINE_AA)
    short_label = CLASS_LABELS.get(class_name, class_name)
    id_tag = f"#{obj_id} " if obj_id is not None else ""
    label = f"{id_tag}{short_label} {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1, cv2.LINE_AA)


def annotate_boxes(frame, boxes):
    """Bbox-only annotation. `boxes` is a list of dicts with keys
    x1,y1,x2,y2,class_name and optional conf,obj_id. No ROI line, no HUD."""
    annotated = frame.copy()
    for b in boxes:
        draw_bbox(annotated, int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"]),
                  counted=False, conf=float(b.get("conf", 0.0)),
                  class_name=b.get("class_name", "slaughtered_chicken"),
                  obj_id=b.get("obj_id"))
    return annotated


def annotate_image_detections(frame, det_info):
    """Annotate a single still. Returns (annotated, per-class counts)."""
    annotated = frame.copy()
    class_counts: dict[str, int] = {}
    for info in det_info:
        cls = info.get("class_name", "slaughtered_chicken")
        class_counts[cls] = class_counts.get(cls, 0) + 1
        color = CLASS_COLORS.get(cls, COLORS["accent"])
        draw_bbox(annotated, info["x1"], info["y1"], info["x2"], info["y2"],
                  counted=False, conf=info["conf"], class_name=cls)
        cx = (info["x1"] + info["x2"]) // 2
        cy = (info["y1"] + info["y2"]) // 2
        cv2.circle(annotated, (cx, cy), 4, color, -1, cv2.LINE_AA)

    total = len(det_info)
    draw_rounded_rect(annotated, (8, 8), (280, 60), COLORS["panel_bg"], radius=8, alpha=0.85)
    cv2.putText(annotated, f"{total}", (18, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, COLORS["white"], 3, cv2.LINE_AA)
    tw = cv2.getTextSize(f"{total}", cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)[0][0]
    cv2.putText(annotated, "objects detected", (18 + tw + 8, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["dim"], 1, cv2.LINE_AA)
    return annotated, class_counts
