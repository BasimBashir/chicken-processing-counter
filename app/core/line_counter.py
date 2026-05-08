"""Trackerless line counter for one-way left-to-right conveyors.

Each frame: any detection whose bbox straddles the vertical ROI line is a
candidate. A candidate counts iff no detection was already counted near the
same y-centre within the last `dedup_frames` frames. No IDs, no Hungarian.

Suited to: one-way conveyor (left→right), ROI near exit, objects can be
back-to-back as long as they're not in the exact same lane simultaneously.
"""
from collections import deque

CLASSES = ["empty_shackles", "single_legged", "slaughtered_chicken"]


class LineCounter:
    def __init__(self, roi_x: int, y_dedup_window: int = 30,
                 dedup_frames: int = 8, trail_length: int = 18):
        self.roi_x = roi_x
        self.y_dedup_window = y_dedup_window
        self.dedup_frames = dedup_frames
        self.trail_length = trail_length
        self.counts = {cls: 0 for cls in CLASSES}
        self.frame_num = 0
        # recent (frame_num, y_center, class) of counted crossings
        self.recent: deque = deque()
        self.counted_ids: set = set()
        self.trails: dict = {}
        self.flash_events: list = []

    @property
    def total_count(self) -> int:
        return sum(self.counts.values())

    def update(self, det_info: list[dict]) -> dict:
        self.frame_num += 1

        cutoff = self.frame_num - self.dedup_frames
        while self.recent and self.recent[0][0] < cutoff:
            self.recent.popleft()

        objects = {}
        for i, d in enumerate(det_info):
            cx = (d["x1"] + d["x2"]) // 2
            cy = (d["y1"] + d["y2"]) // 2
            cls = d.get("class_name", "slaughtered_chicken")
            objects[self.frame_num * 10000 + i] = (cx, cy)

            crosses_line = d["x1"] <= self.roi_x <= d["x2"]
            if not crosses_line:
                continue

            recently_counted = any(
                abs(ry - cy) < self.y_dedup_window and rc == cls
                for _, ry, rc in self.recent
            )
            if recently_counted:
                continue

            if cls in self.counts:
                self.counts[cls] += 1
            self.recent.append((self.frame_num, cy, cls))
            self.flash_events.append((cx, cy, cls))

        return objects

    def reset(self):
        self.counts = {cls: 0 for cls in CLASSES}
        self.frame_num = 0
        self.recent.clear()
        self.counted_ids.clear()
        self.trails.clear()
        self.flash_events.clear()
