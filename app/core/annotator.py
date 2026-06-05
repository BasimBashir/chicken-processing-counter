import cv2
import numpy as np

COLORS = {
    "panel_bg":     (20, 20, 20),
    "panel_border": (60, 60, 60),
    "accent":       (0, 200, 255),
    "roi_line":     (80, 80, 255),
    "roi_glow":     (60, 60, 200),
    "flash":        (0, 255, 255),
    "white":        (255, 255, 255),
    "dim":          (160, 160, 160),
    "very_dim":     (100, 100, 100),
}

# Per-class colors in BGR
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





def draw_roi_line(img, roi_x, height, frame_num, zone_half=0):
    """Draw an animated vertical ROI counting line + translucent band."""
    if zone_half > 0:
        overlay = img.copy()
        cv2.rectangle(overlay, (roi_x - zone_half, 0),
                      (roi_x + zone_half, height), COLORS["roi_glow"], -1)
        cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)
    cv2.line(img, (roi_x, 0), (roi_x, height), COLORS["roi_glow"], 6, cv2.LINE_AA)
    dash_len = 20
    gap_len = 12
    offset = (frame_num * 2) % (dash_len + gap_len)
    y = -offset
    while y < height:
        y1 = max(0, y)
        y2 = min(height, y + dash_len)
        if y2 > y1:
            cv2.line(img, (roi_x, y1), (roi_x, y2), COLORS["roi_line"], 2, cv2.LINE_AA)
        y += dash_len + gap_len
    arrow_spacing = 120
    for ay in range(arrow_spacing // 2, height, arrow_spacing):
        cv2.arrowedLine(
            img, (roi_x - 10, ay), (roi_x + 10, ay),
            COLORS["roi_line"], 2, cv2.LINE_AA, tipLength=0.5
        )


def draw_crossing_flash(img, cx, cy, intensity, class_name):
    overlay = img.copy()
    color = CLASS_COLORS.get(class_name, COLORS["accent"])
    ring_radius = int(8 + 20 * (1.0 - intensity))
    ring_thickness = max(1, int(2 * intensity))
    alpha = intensity * 0.6
    cv2.circle(overlay, (cx, cy), ring_radius, color, ring_thickness, cv2.LINE_AA)
    if intensity > 0.5:
        inner_r = int(4 * intensity)
        cv2.circle(overlay, (cx, cy), inner_r, COLORS["flash"], -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)




def draw_bbox(img, x1, y1, x2, y2, counted, conf, class_name, obj_id=None):
    color = CLASS_COLORS.get(class_name, COLORS["accent"])
    if counted:
        # slightly desaturate counted boxes
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


def annotate_detections(frame, detections, objects_by_class,
                        flash_events, roi_x, frame_num, zone_half=0):
    """Annotate frame with bboxes (class + ID + confidence), ROI line,
    and crossing flashes. Kept minimal for RTSP production speed.
    """
    annotated = frame.copy()
    height, width = annotated.shape[:2]

    # 1. Bounding boxes (with #ID label from the debug tracker)
    for info in detections:
        cls = info.get("class_name", "slaughtered_chicken")
        cx = (info["x1"] + info["x2"]) // 2
        cy = (info["y1"] + info["y2"]) // 2
        matched_id = None
        for obj_id, (ox, oy) in objects_by_class.get(cls, {}).items():
            if abs(ox - cx) < 5 and abs(oy - cy) < 5:
                matched_id = obj_id
                break
        draw_bbox(annotated, info["x1"], info["y1"],
                  info["x2"], info["y2"], counted=False, conf=info["conf"],
                  class_name=cls, obj_id=matched_id)

    # 2. Vertical ROI line
    if roi_x is not None:
        draw_roi_line(annotated, roi_x, height, frame_num, zone_half)

    # 2. Crossing flashes — (fx, fy, cls, f_start)
    active_flashes = []
    for (fx, fy, cls, f_start) in flash_events:
        age = frame_num - f_start
        if age < 12:
            intensity = 1.0 - (age / 12.0)
            draw_crossing_flash(annotated, fx, fy, intensity, cls)
            active_flashes.append((fx, fy, cls, f_start))
    flash_events.clear()
    flash_events.extend(active_flashes)

    # 3. Minimal ROI label
    if roi_x is not None:
        cv2.putText(annotated, "COUNTING LINE", (roi_x + 6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["roi_line"], 2, cv2.LINE_AA)

    return annotated


def annotate_image_detections(frame, det_info):
    """Annotate a single image. Returns annotated frame and per-class counts dict."""
    annotated = frame.copy()
    class_counts: dict[str, int] = {}

    for info in det_info:
        cls = info.get("class_name", "slaughtered_chicken")
        class_counts[cls] = class_counts.get(cls, 0) + 1
        color = CLASS_COLORS.get(cls, COLORS["accent"])
        draw_bbox(annotated, info["x1"], info["y1"],
                  info["x2"], info["y2"], counted=False, conf=info["conf"], class_name=cls)
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
