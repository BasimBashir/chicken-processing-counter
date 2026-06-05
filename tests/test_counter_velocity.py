from app.core.counter import ChickenCounter


def _det(cx, cls="slaughtered_chicken", w=80, h=80):
    return {"x1": cx - w // 2, "y1": 100 - h // 2,
            "x2": cx + w // 2, "y2": 100 + h // 2, "class_name": cls}


def test_single_fast_bird_counts_once_at_34px_per_frame():
    c = ChickenCounter(roi_x=200, conveyor_speed_px=34.0, zone_half=0)
    for cx in range(120, 320, 34):  # 120,154,188,222,256,290
        c.update([_det(cx)])
    assert c.counts["slaughtered_chicken"] == 1


def test_velocity_self_tunes_above_seed():
    c = ChickenCounter(roi_x=200, conveyor_speed_px=14.0, zone_half=0)
    for cx in range(120, 320, 34):
        c.update([_det(cx)])
    assert c.counts["slaughtered_chicken"] == 1


def test_velocity_prevents_double_count_after_frame_gap():
    """Discriminating test: FAILS under the old fixed-speed prediction,
    PASSES with per-track learned velocity.

    Bird travels 34 px/frame but the seed is only 14. Four consecutive
    straddles let the EMA raise the crossing's velocity. Then two frames are
    dropped (update([]) x2); the bird reappears 3 counter-frames later having
    moved 3*34=102 px.

      OLD: pred = last_cx + 3*14 = +42, dist=|102-42|=60 > 40 -> no match
           -> records a 2nd crossing (count=2, WRONG)
      NEW: pred = last_cx + 3*~24 = +72, dist=|102-72|=30 < 40 -> match
           -> count stays 1 (CORRECT)
    """
    c = ChickenCounter(roi_x=250, conveyor_speed_px=14.0, zone_half=0)
    bird_w = 240  # wide zone [130..370] so all sample cx straddle roi_x=250

    def det(cx):
        return {"x1": cx - bird_w // 2, "y1": 60,
                "x2": cx + bird_w // 2, "y2": 140,
                "class_name": "slaughtered_chicken"}

    for cx in (130, 164, 198, 232):   # 4 consecutive frames in zone -> EMA learns
        c.update([det(cx)])
    c.update([])                       # dropped frame 1
    c.update([])                       # dropped frame 2
    c.update([det(334)])               # 3 counter-frames after last match (232+102)
    assert c.counts["slaughtered_chicken"] == 1


def test_two_sequential_birds_count_twice():
    c = ChickenCounter(roi_x=200, conveyor_speed_px=34.0, zone_half=0)
    for cx in range(120, 340, 34):
        c.update([_det(cx)])
    for _ in range(12):
        c.update([])
    for cx in range(120, 340, 34):
        c.update([_det(cx)])
    assert c.counts["slaughtered_chicken"] == 2
