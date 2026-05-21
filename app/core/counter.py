from collections import deque
from app.core.tracker import CentroidTracker

CLASSES = ["empty_shackles", "single_legged", "slaughtered_chicken"]


def _det_to_tuple(d):
    cx = (d["x1"] + d["x2"]) // 2
    cy = (d["y1"] + d["y2"]) // 2
    return (cx, cy, d["x1"], d["y1"], d["x2"], d["y2"])


class ChickenCounter:
    """Per-class centroid tracker with vertical ROI line crossing logic.

    Conveyor moves left-to-right. ROI line is vertical at roi_x.
    Each class uses an independent tracker to avoid cross-class ID collisions.

    Counting rules (mirroring WingCounter but on the x-axis):
    - New track first appears within `appear_margin` px to the right of roi_x:
      count immediately (object entered frame already just past the line).
      Detections deeper past the line are assumed to be flickered re-acquisitions
      of an already-counted track and are NOT counted.
    - Track crosses roi_x left→right (prev_cx < roi_x <= cx): count once.
    """

    def __init__(self, roi_x: int, max_disappeared: int = 15,
                 max_distance: int = 50, trail_length: int = 18,
                 appear_margin: int = 25):
        self.roi_x = roi_x
        self.trail_length = trail_length
        self.appear_margin = appear_margin
        self.trackers = {cls: CentroidTracker(max_disappeared, max_distance) for cls in CLASSES}
        self.counts = {cls: 0 for cls in CLASSES}
        self.counted_ids = {cls: set() for cls in CLASSES}
        self.last_cx = {cls: {} for cls in CLASSES}
        self.trails: dict = {}
        self.flash_events: list = []

    @property
    def total_count(self) -> int:
        return sum(self.counts.values())

    def update(self, det_info: list[dict]) -> dict:
        """Process detections for one frame.
        Returns {class_name: {obj_id: (cx, cy)}} for all active tracked objects.
        """
        by_class: dict[str, list] = {cls: [] for cls in CLASSES}
        for d in det_info:
            cls = d.get("class_name", "slaughtered_chicken")
            if cls in by_class:
                by_class[cls].append(_det_to_tuple(d))

        all_objects: dict[str, dict] = {}

        for cls in CLASSES:
            objects = self.trackers[cls].update(by_class[cls])
            all_objects[cls] = dict(objects)
            active_ids = set(objects.keys())

            # Update trails
            for obj_id, (cx, cy) in objects.items():
                key = (cls, obj_id)
                if key not in self.trails:
                    self.trails[key] = deque(maxlen=self.trail_length)
                self.trails[key].append((int(cx), int(cy)))

            for key in list(self.trails.keys()):
                if key[0] == cls and key[1] not in active_ids:
                    del self.trails[key]

            # Check x-crossings (left-to-right)
            for obj_id, (cx, cy) in objects.items():
                if obj_id in self.counted_ids[cls]:
                    continue
                prev_cx = self.last_cx[cls].get(obj_id)
                self.last_cx[cls][obj_id] = cx

                if prev_cx is None:
                    # First appearance just past the line: count once. A track
                    # appearing deep past the line is almost certainly a flicker
                    # re-acquisition of an already-counted chicken — skip it.
                    if self.roi_x <= cx <= self.roi_x + self.appear_margin:
                        self.counts[cls] += 1
                        self.counted_ids[cls].add(obj_id)
                        self.flash_events.append((int(cx), int(cy), cls))
                elif prev_cx < self.roi_x <= cx:
                    # Crossed the line this frame
                    self.counts[cls] += 1
                    self.counted_ids[cls].add(obj_id)
                    self.flash_events.append((int(cx), int(cy), cls))

            for old_id in list(self.last_cx[cls].keys()):
                if old_id not in active_ids:
                    del self.last_cx[cls][old_id]

        return all_objects

    def reset(self):
        for cls in CLASSES:
            self.trackers[cls].reset()
            self.counts[cls] = 0
            self.counted_ids[cls] = set()
            self.last_cx[cls] = {}
        self.trails = {}
        self.flash_events = []
