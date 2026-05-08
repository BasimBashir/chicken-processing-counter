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


def draw_trail(img, points, base_color, max_length=20):
    pts = list(points)
    n = len(pts)
    if n < 2:
        return
    overlay = img.copy()
    for i in range(1, n):
        t = i / n
        r = int(base_color[0] * (0.3 + 0.7 * t))
        g = int(base_color[1] * (0.3 + 0.7 * t))
        b = int(base_color[2] * (0.3 + 0.7 * t))
        color = (r, g, b)
        cv2.line(overlay, pts[i - 1], pts[i], color, 1, cv2.LINE_AA)
        if i == n - 1:
            cv2.circle(overlay, pts[i], 2, base_color, -1, cv2.LINE_AA)
        elif i % 3 == 0:
            cv2.circle(overlay, pts[i], 1, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)


def draw_roi_line(img, roi_x, height, frame_num):
    """Draw an animated vertical ROI counting line with rightward arrows."""
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


def draw_dashboard(img, counts, in_frame, total_tracked, frame_num,
                   total_frames, is_stream, fps_display, height):
    panel_w = 300
    panel_h = 220
    margin = 8
    draw_rounded_rect(
        img, (margin, margin), (margin + panel_w, margin + panel_h),
        COLORS["panel_bg"], radius=10, alpha=0.88
    )
    cv2.line(img, (margin + 10, margin + 2), (margin + panel_w - 10, margin + 2),
             COLORS["accent"], 2, cv2.LINE_AA)

    x0 = margin + 14
    y0 = margin + 28

    total = sum(counts.values())
    count_text = str(total)
    cv2.putText(img, count_text, (x0, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLORS["white"], 3, cv2.LINE_AA)
    tw = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0][0]
    cv2.putText(img, "total counted", (x0 + tw + 8, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLORS["dim"], 1, cv2.LINE_AA)

    cv2.line(img, (x0, y0 + 10), (x0 + panel_w - 30, y0 + 10),
             COLORS["panel_border"], 1, cv2.LINE_AA)

    class_rows = [
        ("empty_shackles",      "Empty Shackles"),
        ("single_legged",       "Single Legged"),
        ("slaughtered_chicken", "Slaughtered"),
    ]
    y = y0 + 30
    for cls, label in class_rows:
        color = CLASS_COLORS.get(cls, COLORS["accent"])
        cnt = counts.get(cls, 0)
        cv2.putText(img, f"{label}:", (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["dim"], 1, cv2.LINE_AA)
        cv2.putText(img, str(cnt), (x0 + 140, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        y += 22

    cv2.line(img, (x0, y + 2), (x0 + panel_w - 30, y + 2),
             COLORS["panel_border"], 1, cv2.LINE_AA)

    y1 = y + 18
    cv2.putText(img, f"In Frame: {in_frame}", (x0, y1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLORS["white"], 1, cv2.LINE_AA)
    cv2.putText(img, f"Tracked: {total_tracked}", (x0 + 140, y1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLORS["accent"], 1, cv2.LINE_AA)

    y2 = y1 + 20
    if is_stream:
        frame_text = f"Frame: {frame_num}"
    else:
        pct = (frame_num / total_frames * 100) if total_frames > 0 else 0
        frame_text = f"Frame: {frame_num}/{total_frames} ({pct:.0f}%)"
    cv2.putText(img, frame_text, (x0, y2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLORS["very_dim"], 1, cv2.LINE_AA)
    cv2.putText(img, f"FPS: {fps_display:.0f}", (x0 + 200, y2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLORS["very_dim"], 1, cv2.LINE_AA)

    if not is_stream and total_frames > 0:
        y3 = y2 + 16
        bar_x1 = x0
        bar_x2 = x0 + panel_w - 30
        bar_w = bar_x2 - bar_x1
        progress = frame_num / total_frames
        cv2.rectangle(img, (bar_x1, y3), (bar_x2, y3 + 4), COLORS["panel_border"], -1)
        cv2.rectangle(img, (bar_x1, y3), (bar_x1 + int(bar_w * progress), y3 + 4),
                      COLORS["accent"], -1)


def draw_bbox(img, x1, y1, x2, y2, counted, conf, class_name):
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
    label = f"{short_label} {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1, cv2.LINE_AA)


def annotate_detections(frame, detections, objects_by_class, counted_ids_by_class,
                        trails, flash_events, roi_x, frame_num, counts,
                        total_frames, is_stream, fps_display):
    """Full-frame annotation: bboxes, trails, vertical ROI line, flashes, dashboard."""
    annotated = frame.copy()
    height, width = annotated.shape[:2]

    # 1. Motion trails
    for (cls, obj_id), trail_pts in trails.items():
        base_color = CLASS_COLORS.get(cls, COLORS["accent"])
        counted = obj_id in counted_ids_by_class.get(cls, set())
        trail_color = base_color if counted else tuple(int(c * 0.6) for c in base_color)
        draw_trail(annotated, trail_pts, trail_color)

    # 2. Bounding boxes
    for info in detections:
        cls = info.get("class_name", "slaughtered_chicken")
        cx = (info["x1"] + info["x2"]) // 2
        cy = (info["y1"] + info["y2"]) // 2
        is_counted = False
        for obj_id, (ox, oy) in objects_by_class.get(cls, {}).items():
            if abs(ox - cx) < 5 and abs(oy - cy) < 5:
                is_counted = obj_id in counted_ids_by_class.get(cls, set())
                break
        draw_bbox(annotated, info["x1"], info["y1"],
                  info["x2"], info["y2"], is_counted, info["conf"], cls)

    # 3. Vertical ROI line
    if roi_x is not None:
        draw_roi_line(annotated, roi_x, height, frame_num)

    # 4. Crossing flashes — (fx, fy, cls, f_start)
    active_flashes = []
    for (fx, fy, cls, f_start) in flash_events:
        age = frame_num - f_start
        if age < 12:
            intensity = 1.0 - (age / 12.0)
            draw_crossing_flash(annotated, fx, fy, intensity, cls)
            active_flashes.append((fx, fy, cls, f_start))
    flash_events.clear()
    flash_events.extend(active_flashes)

    # 5. Centroid dots
    for cls, objects in objects_by_class.items():
        color = CLASS_COLORS.get(cls, COLORS["accent"])
        for obj_id, (cx, cy) in objects.items():
            dot_color = color if obj_id in counted_ids_by_class.get(cls, set()) else COLORS["accent"]
            cv2.circle(annotated, (int(cx), int(cy)), 3, dot_color, -1, cv2.LINE_AA)

    # 6. Dashboard
    in_frame = sum(len(objs) for objs in objects_by_class.values())
    total_tracked = in_frame
    draw_dashboard(annotated, counts, in_frame, total_tracked,
                   frame_num, total_frames, is_stream, fps_display, height)

    # 7. ROI label
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
