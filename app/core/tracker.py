import numpy as np
from collections import OrderedDict
from scipy.optimize import linear_sum_assignment


def _bbox_iou(a, b):
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    ix1 = max(xa1, xb1)
    iy1 = max(ya1, yb1)
    ix2 = min(xa2, xb2)
    iy2 = min(ya2, yb2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class CentroidTracker:
    """Bbox-aware tracker using Hungarian assignment + IoU cost.

    Keeps overlapping bboxes as separate IDs so each one fires its own
    crossing event independently.
    """

    def __init__(self, max_disappeared=15, max_distance=50, iou_threshold=0.1):
        self.next_id = 0
        self.objects = OrderedDict()
        self.bboxes = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.iou_threshold = iou_threshold

    def update(self, detections):
        """detections: iterable of (cx, cy, x1, y1, x2, y2) tuples."""
        norm = []
        for d in detections:
            d = tuple(d)
            if len(d) == 2:
                cx, cy = d
                norm.append((float(cx), float(cy), float(cx), float(cy), float(cx), float(cy)))
            else:
                cx, cy, x1, y1, x2, y2 = d
                norm.append((float(cx), float(cy), x1, y1, x2, y2))

        if len(norm) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)
            return self.objects

        if len(self.objects) == 0:
            # Register rightmost first so leftmost (newest chicken on a
            # left→right conveyor) ends up with the highest ID.
            for d in sorted(norm, key=lambda x: -x[0]):
                self._register(d)
            return self.objects

        obj_ids = list(self.objects.keys())
        obj_centroids = np.array([self.objects[i] for i in obj_ids], dtype=float)
        obj_bboxes = [self.bboxes[i] for i in obj_ids]
        det_centroids = np.array([(d[0], d[1]) for d in norm], dtype=float)
        det_bboxes = [(d[2], d[3], d[4], d[5]) for d in norm]

        diff = obj_centroids[:, np.newaxis, :] - det_centroids[np.newaxis, :, :]
        dist_matrix = np.sqrt((diff ** 2).sum(axis=2))

        iou_matrix = np.zeros((len(obj_ids), len(norm)), dtype=float)
        for i, ob in enumerate(obj_bboxes):
            for j, db in enumerate(det_bboxes):
                iou_matrix[i, j] = _bbox_iou(ob, db)

        disappeared_arr = np.array(
            [self.disappeared[i] for i in obj_ids], dtype=float
        )[:, np.newaxis]

        cost_matrix = (dist_matrix - iou_matrix * self.max_distance
                       + disappeared_arr * 3.0)

        match_rows, match_cols = linear_sum_assignment(cost_matrix)

        used_rows = set()
        used_cols = set()
        for row, col in zip(match_rows, match_cols):
            if (iou_matrix[row, col] < self.iou_threshold and
                    dist_matrix[row, col] > self.max_distance):
                continue
            obj_id = obj_ids[row]
            self.objects[obj_id] = (float(det_centroids[col][0]), float(det_centroids[col][1]))
            self.bboxes[obj_id] = det_bboxes[col]
            self.disappeared[obj_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        for row in range(len(obj_centroids)):
            if row not in used_rows:
                obj_id = obj_ids[row]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)

        # Register unmatched detections rightmost-first so the leftmost
        # incoming chicken always picks up the highest new ID this frame.
        unmatched_cols = [c for c in range(len(norm)) if c not in used_cols]
        unmatched_cols.sort(key=lambda c: -norm[c][0])
        for col in unmatched_cols:
            self._register(norm[col])

        return self.objects

    def _register(self, det):
        cx, cy, x1, y1, x2, y2 = det
        self.objects[self.next_id] = (float(cx), float(cy))
        self.bboxes[self.next_id] = (x1, y1, x2, y2)
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def _deregister(self, obj_id):
        self.objects.pop(obj_id, None)
        self.bboxes.pop(obj_id, None)
        self.disappeared.pop(obj_id, None)

    def reset(self):
        self.next_id = 0
        self.objects = OrderedDict()
        self.bboxes = OrderedDict()
        self.disappeared = OrderedDict()
