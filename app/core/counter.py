import numpy as np
from scipy.optimize import linear_sum_assignment

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
                 zone_half: int = 18, sway_k: float = 0.6,
                 zone_speed_factor: float = 1.20):
        self.roi_x = roi_x

        # Straddle Tracker parameters
        self.conveyor_speed_px = conveyor_speed_px
        self.zone_half = zone_half
        # Sway tolerance, RELATIVE to the crossing's learned belt speed: a
        # straddler also matches a crossing if it is within `sway_k * velocity`
        # of that crossing's LAST position (not just its forward prediction).
        # This absorbs carcass sway / belt slow-downs / full stops without
        # spawning a duplicate, and because it scales with the belt speed it is
        # resolution- AND speed-independent (k~0.6 from on-site tuning). Set 0
        # to disable (pure forward-prediction matching).
        self.sway_k = sway_k
        self.zone_speed_factor = zone_speed_factor
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

    def update(self, det_info: list[dict], belt_stopped: bool = False) -> dict:
        """Process detections for one frame. Returns {class_name: {obj_id: (cx, cy)}}
        from the debug tracker for annotating IDs on video.

        `belt_stopped`: when the conveyor isn't moving, no bird can newly cross
        the line, so we (a) do NOT create new crossings and (b) do NOT expire
        existing ones. This stops a parked bird whose detection FLICKERS from
        being re-counted each time its box reappears (the crossing stays alive
        and re-matches it via the sway tolerance). Physically safe: nothing is
        crossing a stopped line, so no real count can be missed.
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

        # Adaptive zone: widen proportionally to measured belt speed.
        # When zone_half=0 the caller has explicitly requested single-pixel
        # tripwire mode; respect that and skip the adaptive expansion.
        _vels = [c['velocity'] for c in self.active_crossings]
        belt_speed_px = (sum(_vels) / len(_vels)) if _vels else self.conveyor_speed_px
        if self.zone_half > 0:
            effective_zone_half = max(self.zone_half, int(belt_speed_px * self.zone_speed_factor))
        else:
            effective_zone_half = 0

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
            lo = self.roi_x - effective_zone_half
            hi = self.roi_x + effective_zone_half
            if x1 <= hi and x2 >= lo:
                cx = (x1 + x2) // 2
                cy = (d["y1"] + d["y2"]) // 2
                straddlers.append((cx, cy, cls))

        matched_straddlers: set[int] = set()

        if straddlers and self.active_crossings:
            INF = 1e9
            n_s = len(straddlers)
            n_c = len(self.active_crossings)
            C = np.full((n_s, n_c), INF)

            for i, (cx, cy, cls) in enumerate(straddlers):
                for j, crossing in enumerate(self.active_crossings):
                    if crossing['cls'] != cls:
                        continue
                    frames_elapsed = self.frame_num - crossing['last_seen_frame']
                    predicted_cx = crossing['last_cx'] + (frames_elapsed * crossing['velocity'])
                    dist_pred = abs(cx - predicted_cx)
                    dist_last = abs(cx - crossing['last_cx'])
                    tol = self.sway_k * crossing['velocity']
                    cost = min(dist_pred, dist_last) if dist_last <= tol else dist_pred
                    C[i, j] = cost

            row_ind, col_ind = linear_sum_assignment(C)
            for i, j in zip(row_ind, col_ind):
                if C[i, j] >= self.max_x_distance:
                    continue
                cx, cy, cls = straddlers[i]
                c = self.active_crossings[j]
                frames_elapsed = self.frame_num - c['last_seen_frame']
                if frames_elapsed > 0:
                    observed_v = (cx - c['last_cx']) / frames_elapsed
                    if 0 < observed_v < self.max_velocity_px:
                        c['velocity'] = (self.velocity_ema * observed_v
                                         + (1 - self.velocity_ema) * c['velocity'])
                c['last_cx'] = cx
                c['last_seen_frame'] = self.frame_num
                matched_straddlers.add(i)

        # New crossings for unmatched straddlers
        for i, (cx, cy, cls) in enumerate(straddlers):
            if i in matched_straddlers:
                continue
            if not belt_stopped:
                self.counts[cls] += 1
                self.active_crossings.append({
                    'cls': cls,
                    'last_cx': cx,
                    'last_seen_frame': self.frame_num,
                    'velocity': self.conveyor_speed_px,
                })
                self.flash_events.append((cx, cy, cls))

        # Expire old crossings — but NOT while the belt is stopped, so a parked
        # bird's crossing survives a long detection flicker and re-matches it
        # instead of being re-counted.
        if not belt_stopped:
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
