from app.core.tracker import CentroidTracker

CLASSES = ["empty_shackles", "single_legged", "slaughtered_chicken"]

# Classes that count toward the headline total compared against the BAADER
# weight counter. Currently empty_shackles + slaughtered_chicken; add
# "single_legged" here when that class is brought online. Classes left out
# are still detected, tracked, and shown in the per-class breakdown for
# diagnostics — they just don't inflate the total.
COUNTED_CLASSES = ["empty_shackles", "slaughtered_chicken"]


def _det_to_tuple(d):
    cx = (d["x1"] + d["x2"]) // 2
    cy = (d["y1"] + d["y2"]) // 2
    return (cx, cy, d["x1"], d["y1"], d["x2"], d["y2"])


class ChickenCounter:
    """Virtual Tripwire counter based on X-axis progression.

    Counting rule (independent of any tracker):
      A detection is counted on the frame its bbox crosses
      `roi_x`.
    """

    def __init__(self, roi_x: int, max_disappeared: int = 15,
                 max_distance: int = 55, conveyor_speed_px: float = 14.0):
        self.roi_x = roi_x
        
        # Straddle Tracker parameters
        self.conveyor_speed_px = conveyor_speed_px
        self.max_x_distance = 40
        self.max_straddle_disappeared = 10

        # Tracker kept solely for the overlay's #ID labels — no count side-effects.
        self.trackers = {cls: CentroidTracker(max_disappeared, max_distance) for cls in CLASSES}

        self.counts = {cls: 0 for cls in CLASSES}
        # active_crossings stores dicts: {'cls': cls, 'last_cx': cx, 'last_seen_frame': frame_num}
        self.active_crossings: list[dict] = []
        self.frame_num = 0

        self.flash_events: list = []

    @property
    def total_count(self) -> int:
        return sum(self.counts[c] for c in COUNTED_CLASSES)

    def update(self, det_info: list[dict]) -> dict:
        """Process detections for one frame. Returns {class_name: {obj_id: (cx, cy)}}
        from the debug tracker for annotating IDs on video.
        """
        self.frame_num += 1

        # 1) Run the debug tracker per class.
        by_class: dict[str, list] = {cls: [] for cls in CLASSES}
        for d in det_info:
            cls = d.get("class_name", "slaughtered_chicken")
            if cls in by_class:
                by_class[cls].append(_det_to_tuple(d))

        all_objects: dict[str, dict] = {}
        for cls in CLASSES:
            all_objects[cls] = dict(self.trackers[cls].update(by_class[cls]))

        # 2) Straddle Tracker Logic (Virtual Tripwire)
        straddlers = []
        for d in det_info:
            cls = d.get("class_name", "slaughtered_chicken")
            if cls not in self.counts:
                continue
            
            # Check if bbox mathematically straddles the roi line
            x1, x2 = d["x1"], d["x2"]
            if x1 <= self.roi_x <= x2:
                cx = (x1 + x2) // 2
                cy = (d["y1"] + d["y2"]) // 2
                straddlers.append((cx, cy, cls))

        matched_crossings = set()

        for cx, cy, cls in straddlers:
            best_match_idx = -1
            best_dist = float('inf')

            # Try to match to an active crossing
            for i, crossing in enumerate(self.active_crossings):
                if crossing['cls'] != cls or i in matched_crossings:
                    continue
                
                frames_elapsed = self.frame_num - crossing['last_seen_frame']
                predicted_cx = crossing['last_cx'] + (frames_elapsed * self.conveyor_speed_px)
                
                dist = abs(cx - predicted_cx)
                if dist < self.max_x_distance and dist < best_dist:
                    best_match_idx = i
                    best_dist = dist
            
            if best_match_idx != -1:
                # Update existing crossing
                self.active_crossings[best_match_idx]['last_cx'] = cx
                self.active_crossings[best_match_idx]['last_seen_frame'] = self.frame_num
                matched_crossings.add(best_match_idx)
            else:
                # New crossing!
                self.counts[cls] += 1
                self.active_crossings.append({
                    'cls': cls,
                    'last_cx': cx,
                    'last_seen_frame': self.frame_num
                })
                self.flash_events.append((cx, cy, cls))

        # Expire old crossings
        self.active_crossings = [
            c for c in self.active_crossings
            if (self.frame_num - c['last_seen_frame']) <= self.max_straddle_disappeared
        ]

        return all_objects

    def reset(self):
        """Reset counts/state."""
        for cls in CLASSES:
            self.counts[cls] = 0
        self.active_crossings.clear()
        self.frame_num = 0
        self.flash_events = []
