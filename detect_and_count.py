import platform
import cv2
import numpy as np
import pathlib
import argparse
import warnings
import time
from collections import OrderedDict, deque
from ultralytics import YOLO

warnings.filterwarnings("ignore", category=FutureWarning)
if platform.system() != "Windows":
    pathlib.WindowsPath = pathlib.PosixPath

MODEL_PATH = r"best.pt"

CLASSES = ["empty_shackles", "single_legged", "slaughtered_chicken"]

CLASS_COLORS = {
    "empty_shackles":      (0, 165, 255),
    "single_legged":       (255, 200, 0),
    "slaughtered_chicken": (0, 230, 118),
}

CLASS_LABELS = {
    "empty_shackles":      "E.Shackle",
    "single_legged":       "Single",
    "slaughtered_chicken": "Slaughtered",
}

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


# ── Tracker ────────────────────────────────────────────────────────────────────

def _compute_iou(b1, b2):
    """IoU between two (x1,y1,x2,y2) boxes."""
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter)


class CentroidTracker:
    """Hungarian-assignment tracker with IoU-aware cost matrix."""

    def __init__(self, max_disappeared=15, max_distance=50):
        self.next_id = 0
        self.objects = OrderedDict()   # id → (cx, cy)
        self.bboxes = OrderedDict()    # id → (x1, y1, x2, y2)
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def _register(self, cx, cy, x1, y1, x2, y2):
        self.objects[self.next_id] = (cx, cy)
        self.bboxes[self.next_id] = (x1, y1, x2, y2)
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def _deregister(self, obj_id):
        del self.objects[obj_id]
        del self.bboxes[obj_id]
        del self.disappeared[obj_id]

    def update(self, detections):
        """detections: list of (cx, cy, x1, y1, x2, y2)"""
        if not detections:
            for obj_id in list(self.disappeared):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)
            return dict(self.objects)

        if not self.objects:
            for det in detections:
                self._register(*det)
            return dict(self.objects)

        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            raise RuntimeError("scipy is required: pip install scipy")

        obj_ids = list(self.objects)
        obj_cents = np.array([self.objects[i] for i in obj_ids], dtype=float)
        det_cents = np.array([(d[0], d[1]) for d in detections], dtype=float)

        diff = obj_cents[:, np.newaxis, :] - det_cents[np.newaxis, :, :]
        dist_matrix = np.sqrt((diff ** 2).sum(axis=2))

        obj_boxes = [self.bboxes[i] for i in obj_ids]
        det_boxes = [(d[2], d[3], d[4], d[5]) for d in detections]
        iou_matrix = np.zeros((len(obj_ids), len(detections)), dtype=float)
        for r, ob in enumerate(obj_boxes):
            for c, db in enumerate(det_boxes):
                iou_matrix[r, c] = _compute_iou(ob, db)

        disappeared_arr = np.array(
            [self.disappeared[i] for i in obj_ids], dtype=float
        )[:, np.newaxis]
        cost_matrix = dist_matrix - iou_matrix * self.max_distance + disappeared_arr * 3.0

        match_rows, match_cols = linear_sum_assignment(cost_matrix)

        matched_rows, matched_cols = set(), set()
        for r, c in zip(match_rows, match_cols):
            if iou_matrix[r, c] == 0 and dist_matrix[r, c] > self.max_distance:
                continue
            obj_id = obj_ids[r]
            det = detections[c]
            self.objects[obj_id] = (det[0], det[1])
            self.bboxes[obj_id] = (det[2], det[3], det[4], det[5])
            self.disappeared[obj_id] = 0
            matched_rows.add(r)
            matched_cols.add(c)

        for r, obj_id in enumerate(obj_ids):
            if r not in matched_rows:
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)

        for c, det in enumerate(detections):
            if c not in matched_cols:
                self._register(*det)

        return dict(self.objects)


# ── Drawing Helpers ────────────────────────────────────────────────────────────

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


def draw_trail(img, points, base_color):
    pts = list(points)
    n = len(pts)
    if n < 2:
        return
    overlay = img.copy()
    for i in range(1, n):
        t = i / n
        color = tuple(int(c * (0.3 + 0.7 * t)) for c in base_color)
        cv2.line(overlay, pts[i - 1], pts[i], color, 1, cv2.LINE_AA)
        if i == n - 1:
            cv2.circle(overlay, pts[i], 2, base_color, -1, cv2.LINE_AA)
        elif i % 3 == 0:
            cv2.circle(overlay, pts[i], 1, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)


def draw_roi_line(img, roi_x, height, frame_num):
    cv2.line(img, (roi_x, 0), (roi_x, height), COLORS["roi_glow"], 6, cv2.LINE_AA)
    dash_len, gap_len = 20, 12
    offset = (frame_num * 2) % (dash_len + gap_len)
    y = -offset
    while y < height:
        y1 = max(0, y)
        y2 = min(height, y + dash_len)
        if y2 > y1:
            cv2.line(img, (roi_x, y1), (roi_x, y2), COLORS["roi_line"], 2, cv2.LINE_AA)
        y += dash_len + gap_len
    for ay in range(60, height, 120):
        cv2.arrowedLine(
            img, (roi_x - 10, ay), (roi_x + 10, ay),
            COLORS["roi_line"], 2, cv2.LINE_AA, tipLength=0.5,
        )


def draw_crossing_flash(img, cx, cy, intensity, class_name):
    overlay = img.copy()
    color = CLASS_COLORS.get(class_name, COLORS["accent"])
    ring_radius = int(8 + 20 * (1.0 - intensity))
    ring_thickness = max(1, int(2 * intensity))
    alpha = intensity * 0.6
    cv2.circle(overlay, (cx, cy), ring_radius, color, ring_thickness, cv2.LINE_AA)
    if intensity > 0.5:
        cv2.circle(overlay, (cx, cy), int(4 * intensity), COLORS["flash"], -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_dashboard(img, counts, in_frame, frame_num, total_frames, is_stream, fps_display):
    panel_w, panel_h = 310, 220
    margin = 8
    draw_rounded_rect(img, (margin, margin), (margin + panel_w, margin + panel_h),
                      COLORS["panel_bg"], radius=10, alpha=0.88)
    cv2.line(img, (margin + 10, margin + 2), (margin + panel_w - 10, margin + 2),
             COLORS["accent"], 2, cv2.LINE_AA)

    x0 = margin + 14
    y0 = margin + 28
    total = sum(counts.values())
    cv2.putText(img, str(total), (x0, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLORS["white"], 3, cv2.LINE_AA)
    tw = cv2.getTextSize(str(total), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0][0]
    cv2.putText(img, "total counted", (x0 + tw + 8, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLORS["dim"], 1, cv2.LINE_AA)
    cv2.line(img, (x0, y0 + 10), (x0 + panel_w - 30, y0 + 10),
             COLORS["panel_border"], 1, cv2.LINE_AA)

    y = y0 + 30
    for cls in CLASSES:
        label = CLASS_LABELS.get(cls, cls)
        color = CLASS_COLORS.get(cls, COLORS["accent"])
        cv2.putText(img, f"{label}:", (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["dim"], 1, cv2.LINE_AA)
        cv2.putText(img, str(counts.get(cls, 0)), (x0 + 140, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        y += 22

    cv2.line(img, (x0, y + 2), (x0 + panel_w - 30, y + 2),
             COLORS["panel_border"], 1, cv2.LINE_AA)
    y1 = y + 18
    cv2.putText(img, f"In Frame: {in_frame}", (x0, y1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLORS["white"], 1, cv2.LINE_AA)
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
        bar_x1, bar_x2 = x0, x0 + panel_w - 30
        progress = frame_num / total_frames
        cv2.rectangle(img, (bar_x1, y3), (bar_x2, y3 + 4), COLORS["panel_border"], -1)
        cv2.rectangle(img, (bar_x1, y3),
                      (bar_x1 + int((bar_x2 - bar_x1) * progress), y3 + 4),
                      COLORS["accent"], -1)


def draw_bbox(img, x1, y1, x2, y2, counted, conf, class_name):
    color = CLASS_COLORS.get(class_name, COLORS["accent"])
    if counted:
        color = tuple(min(c + 40, 255) for c in color)
    corner_len = 8
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    for px, py, dx, dy in [
        (x1, y1,  1,  1), (x2, y1, -1,  1),
        (x1, y2,  1, -1), (x2, y2, -1, -1),
    ]:
        cv2.line(img, (px, py), (px + dx * corner_len, py), color, 2, cv2.LINE_AA)
        cv2.line(img, (px, py), (px, py + dy * corner_len), color, 2, cv2.LINE_AA)
    short_label = CLASS_LABELS.get(class_name, class_name)
    label = f"{short_label} {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1, cv2.LINE_AA)


# ── Core Functions ─────────────────────────────────────────────────────────────

def load_model(model_path=MODEL_PATH):
    return YOLO(model_path)


def detect_and_annotate_image(model, image_path, conf_threshold=0.25, save_path=None):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image '{image_path}'")
        return None, {}

    results = model(image, conf=conf_threshold, verbose=False)
    annotated = image.copy()
    class_counts: dict[str, int] = {}
    det_list = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_idx = int(box.cls[0])
        cls = model.names.get(cls_idx, str(cls_idx))
        conf = float(box.conf[0])
        class_counts[cls] = class_counts.get(cls, 0) + 1
        det_list.append((x1, y1, x2, y2, conf, cls))
        draw_bbox(annotated, x1, y1, x2, y2, counted=False, conf=conf, class_name=cls)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(annotated, (cx, cy), 4, CLASS_COLORS.get(cls, COLORS["accent"]), -1, cv2.LINE_AA)

    total = len(det_list)
    draw_rounded_rect(annotated, (8, 8), (280, 60), COLORS["panel_bg"], radius=8, alpha=0.85)
    cv2.putText(annotated, str(total), (18, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, COLORS["white"], 3, cv2.LINE_AA)
    tw = cv2.getTextSize(str(total), cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)[0][0]
    cv2.putText(annotated, "objects detected", (18 + tw + 8, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["dim"], 1, cv2.LINE_AA)

    if save_path:
        cv2.imwrite(save_path, annotated)
        print(f"Annotated image saved to '{save_path}'")

    print(f"Detected {total} object(s) in '{image_path}'")
    for cls, cnt in class_counts.items():
        print(f"  {cls}: {cnt}")
    return annotated, class_counts


def detect_and_annotate_video(
    model, video_path, conf_threshold=0.25, nms_iou=0.45, imgsz=640,
    save_path=None, roi_position=0.5, max_disappeared=15, max_distance=50,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video '{video_path}'")
        return {}

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    is_stream = total_frames <= 0

    roi_x = int(width * roi_position)

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    trackers     = {cls: CentroidTracker(max_disappeared, max_distance) for cls in CLASSES}
    counts       = {cls: 0 for cls in CLASSES}
    counted_ids  = {cls: set() for cls in CLASSES}
    prev_cx      = {cls: {} for cls in CLASSES}
    trails: dict = {}
    flash_events = []
    frame_num    = 0
    trail_length = 18

    fps_timer, fps_display, fps_frame_count = time.time(), 0.0, 0

    print(f"Processing: {video_path}")
    print(f"Vertical ROI at x={roi_x} ({roi_position*100:.0f}% from left) — left-to-right conveyor")

    while True:
        ret, frame = cap.read()
        if not ret:
            if is_stream:
                print("Stream ended or connection lost.")
            break

        frame_num += 1
        fps_frame_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 0.5:
            fps_display = fps_frame_count / elapsed
            fps_frame_count = 0
            fps_timer = time.time()

        results = model(frame, conf=conf_threshold, iou=nms_iou, imgsz=imgsz, verbose=False)

        by_class: dict[str, list] = {cls: [] for cls in CLASSES}
        det_info = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_idx = int(box.cls[0])
            cls = model.names.get(cls_idx, str(cls_idx))
            conf = float(box.conf[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            det_info.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                              "conf": conf, "class_name": cls})
            if cls in by_class:
                by_class[cls].append((cx, cy, x1, y1, x2, y2))

        all_objects: dict[str, dict] = {}
        for cls in CLASSES:
            objects = trackers[cls].update(by_class[cls])
            all_objects[cls] = objects
            active_ids = set(objects)

            for obj_id, (cx, cy) in objects.items():
                key = (cls, obj_id)
                if key not in trails:
                    trails[key] = deque(maxlen=trail_length)
                trails[key].append((int(cx), int(cy)))

            for key in list(trails):
                if key[0] == cls and key[1] not in active_ids:
                    del trails[key]

            for obj_id, (cx, cy) in objects.items():
                if obj_id in counted_ids[cls]:
                    continue
                px = prev_cx[cls].get(obj_id)
                if px is None:
                    if cx >= roi_x:
                        counts[cls] += 1
                        counted_ids[cls].add(obj_id)
                        flash_events.append((int(cx), int(cy), cls, frame_num))
                elif px < roi_x <= cx:
                    counts[cls] += 1
                    counted_ids[cls].add(obj_id)
                    flash_events.append((int(cx), int(cy), cls, frame_num))
                prev_cx[cls][obj_id] = cx

            for old_id in list(prev_cx[cls]):
                if old_id not in active_ids:
                    del prev_cx[cls][old_id]

            for cid in list(counted_ids[cls]):
                if trackers[cls].disappeared.get(cid, 0) > 0:
                    trackers[cls]._deregister(cid)
                    counted_ids[cls].discard(cid)
                    prev_cx[cls].pop(cid, None)

        annotated = frame.copy()

        for (cls, obj_id), trail_pts in trails.items():
            draw_trail(annotated, trail_pts, CLASS_COLORS.get(cls, COLORS["accent"]))

        for info in det_info:
            cls = info["class_name"]
            cx = (info["x1"] + info["x2"]) // 2
            cy = (info["y1"] + info["y2"]) // 2
            is_counted = False
            for obj_id, (ox, oy) in all_objects.get(cls, {}).items():
                if abs(ox - cx) < 5 and abs(oy - cy) < 5:
                    is_counted = obj_id in counted_ids.get(cls, set())
                    break
            draw_bbox(annotated, info["x1"], info["y1"],
                      info["x2"], info["y2"], is_counted, info["conf"], cls)

        draw_roi_line(annotated, roi_x, height, frame_num)

        active_flashes = []
        for (fx, fy, cls, f_start) in flash_events:
            age = frame_num - f_start
            if age < 12:
                draw_crossing_flash(annotated, fx, fy, 1.0 - age / 12.0, cls)
                active_flashes.append((fx, fy, cls, f_start))
        flash_events[:] = active_flashes

        in_frame = sum(len(objs) for objs in all_objects.values())
        for cls, objects in all_objects.items():
            color = CLASS_COLORS.get(cls, COLORS["accent"])
            for obj_id, (cx, cy) in objects.items():
                dot_color = color if obj_id in counted_ids.get(cls, set()) else COLORS["accent"]
                cv2.circle(annotated, (int(cx), int(cy)), 3, dot_color, -1, cv2.LINE_AA)

        draw_dashboard(annotated, counts, in_frame, frame_num, total_frames, is_stream, fps_display)

        if writer:
            writer.write(annotated)

        cv2.imshow("Chicken Counter", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Playback stopped by user.")
            break

    cap.release()
    if writer:
        writer.release()
        print(f"Annotated video saved to '{save_path}'")
    cv2.destroyAllWindows()

    print("\nFinal counts:")
    for cls in CLASSES:
        print(f"  {cls}: {counts[cls]}")
    print(f"  TOTAL: {sum(counts.values())}")
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Slaughtered Chicken Detection and Counting — left-to-right conveyor"
    )
    parser.add_argument("input", help="Path to image, video file, or RTSP stream URL")
    parser.add_argument("--save", default=None, help="Path to save annotated output")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (default: 0.45)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (default: 640)")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to YOLO model weights")
    parser.add_argument(
        "--roi", type=float, default=0.5,
        help="Vertical ROI line position as fraction of frame width 0–1 (default: 0.5)"
    )
    parser.add_argument("--max-distance", type=int, default=50,
                        help="Max pixel distance for track matching (default: 50)")
    parser.add_argument("--max-disappeared", type=int, default=15,
                        help="Frames before a lost track is dropped (default: 15)")
    args = parser.parse_args()

    model = load_model(args.model)
    is_stream = args.input.startswith("rtsp://") or args.input.startswith("http")

    if is_stream:
        detect_and_annotate_video(
            model, args.input, conf_threshold=args.conf, nms_iou=args.iou, imgsz=args.imgsz,
            save_path=args.save, roi_position=args.roi,
            max_distance=args.max_distance, max_disappeared=args.max_disappeared,
        )
    else:
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
        ext = pathlib.Path(args.input).suffix.lower()

        if ext in image_exts:
            annotated, class_counts = detect_and_annotate_image(
                model, args.input, conf_threshold=args.conf, save_path=args.save
            )
            if annotated is not None:
                cv2.imshow("Chicken Detection - Image", annotated)
                print("Press any key to close...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        elif ext in video_exts:
            detect_and_annotate_video(
                model, args.input, conf_threshold=args.conf, nms_iou=args.iou, imgsz=args.imgsz,
                save_path=args.save, roi_position=args.roi,
                max_distance=args.max_distance, max_disappeared=args.max_disappeared,
            )
        else:
            print(f"Error: Unsupported file extension '{ext}'")
            print(f"Supported images: {image_exts}")
            print(f"Supported videos: {video_exts}")
