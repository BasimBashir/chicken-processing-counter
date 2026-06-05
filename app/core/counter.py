from app.core.tracker import CentroidTracker

CLASSES = ["empty_shackles", "single_legged", "slaughtered_chicken"]

# Classes summed into the BAADER-comparable bird total (`total_count`).
# BAADER is a weight sensor, so it never counts an empty shackle — including
# empty_shackles here would push our total ABOVE BAADER. So the comparable
# total is slaughtered_chicken only. empty_shackles (and single_legged) are
# still detected, tracked, and reported as their OWN separate per-class counts
# in `self.counts` — they're just not summed into the bird total.
COUNTED_CLASSES = ["slaughtered_chicken"]


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
                 max_distance: int = 55, conveyor_speed_px: float = 34.0,
                 zone_half: int = 15):
        self.roi_x = roi_x

        # Straddle Tracker parameters
        self.conveyor_speed_px = conveyor_speed_px
        self.zone_half = zone_half
        # Per-frame match gate. NOTE: a crossing's seed velocity (conveyor_speed_px)
        # must be within max_x_distance of the real belt speed or the FIRST match
        # never forms and the EMA can't bootstrap -> double counts. With the
        # calibrated seed (~34) this holds; a badly misconfigured seed would not.
        self.max_x_distance = 40
        self.max_straddle_disappeared = 10

        # Per-track velocity estimation. Each crossing learns its own px/frame
        # via EMA of observed motion, seeded by conveyor_speed_px.
        self.velocity_ema = 0.3          # weight of newest observation
        # Reject implausible jumps. ~5x the nominal 34 px/frame at 25 fps;
        # revisit if camera fps changes (this bound scales with frame rate).
        self.max_velocity_px = 120.0

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
            
            # Count when the bbox OVERLAPS the band [roi_x-zone_half, roi_x+zone_half].
            # At zone_half=0 this reduces to the original single-pixel tripwire
            # (x1 <= roi_x <= x2). Effective catch window = bbox_width + 2*zone_half,
            # so a wider band tolerates flicker/stutter without the bird being
            # jumped over (center-point logic would narrow the window instead).
            x1, x2 = d["x1"], d["x2"]
            lo = self.roi_x - self.zone_half
            hi = self.roi_x + self.zone_half
            if x1 <= hi and x2 >= lo:
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
                predicted_cx = crossing['last_cx'] + (frames_elapsed * crossing['velocity'])
                
                dist = abs(cx - predicted_cx)
                if dist < self.max_x_distance and dist < best_dist:
                    best_match_idx = i
                    best_dist = dist
            
            if best_match_idx != -1:
                # Update existing crossing + learn its velocity from motion.
                c = self.active_crossings[best_match_idx]
                frames_elapsed = self.frame_num - c['last_seen_frame']
                if frames_elapsed > 0:
                    observed_v = (cx - c['last_cx']) / frames_elapsed
                    # Forward-motion only: the belt runs left->right, so a
                    # zero/backward observed_v (jitter, or a stopped belt) is
                    # ignored and the last good velocity is held. Caveat: a real
                    # belt STOP that outlasts max_straddle_disappeared expires the
                    # crossing, so a stop-then-restart can re-count that bird.
                    if 0 < observed_v < self.max_velocity_px:
                        c['velocity'] = (self.velocity_ema * observed_v
                                         + (1 - self.velocity_ema) * c['velocity'])
                c['last_cx'] = cx
                c['last_seen_frame'] = self.frame_num
                matched_crossings.add(best_match_idx)
            else:
                # New crossing!
                self.counts[cls] += 1
                self.active_crossings.append({
                    'cls': cls,
                    'last_cx': cx,
                    'last_seen_frame': self.frame_num,
                    'velocity': self.conveyor_speed_px,
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
