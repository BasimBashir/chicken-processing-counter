from app.core.counter import ChickenCounter


def _det(cx, cls="slaughtered_chicken", w=80, h=80):
    return {"x1": cx - w // 2, "y1": 100 - h // 2,
            "x2": cx + w // 2, "y2": 100 + h // 2, "class_name": cls}


def test_single_fast_bird_counts_once_at_34px_per_frame():
    c = ChickenCounter(roi_x=200, conveyor_speed_px=34.0, zone_half=15)
    for cx in range(120, 320, 34):  # 120,154,188,222,256,290 — cx=188 lands in [185,215]
        c.update([_det(cx)])
    assert c.counts["slaughtered_chicken"] == 1


def test_velocity_self_tunes_above_seed():
    c = ChickenCounter(roi_x=200, conveyor_speed_px=14.0, zone_half=15)
    for cx in range(120, 320, 34):
        c.update([_det(cx)])
    assert c.counts["slaughtered_chicken"] == 1


def test_velocity_prevents_double_count_after_frame_gap():
    """Discriminating test: FAILS under the old fixed-speed prediction,
    PASSES with per-track learned velocity.

    Bird travels 34 px/frame but the seed is only 6. Four consecutive
    centre-hits let the EMA raise the crossing's velocity to ~24 px/frame.
    Then one frame is dropped (update([]) x1); the bird reappears 2
    counter-frames later having moved 2*34=68 px to cx=280.

      OLD: pred = 212 + 2*6 = 224,  dist=|280-224|=56 > 40 -> no match
           -> records a 2nd crossing (count=2, WRONG)
      NEW: pred = 212 + 2*~24 = ~260, dist=|280-260|=20 < 40 -> match
           -> count stays 1 (CORRECT)

    Uses zone_half=90 so the wide band [110,290] keeps the bird centre
    in-zone for all sample frames.
    """
    def det(cx):
        return {"x1": cx - 10, "y1": 60,
                "x2": cx + 10, "y2": 140,
                "class_name": "slaughtered_chicken"}

    c = ChickenCounter(roi_x=200, conveyor_speed_px=6.0, zone_half=90)
    for cx in (110, 144, 178, 212):   # 4 consecutive frames in zone -> EMA learns
        c.update([det(cx)])
    c.update([])                       # dropped frame
    c.update([det(280)])               # 2 counter-frames after last match (212+2*34)
    assert c.counts["slaughtered_chicken"] == 1


def test_two_sequential_birds_count_twice():
    c = ChickenCounter(roi_x=200, conveyor_speed_px=34.0, zone_half=15)
    for cx in range(120, 340, 34):
        c.update([_det(cx)])
    for _ in range(12):
        c.update([])
    for cx in range(120, 340, 34):
        c.update([_det(cx)])
    assert c.counts["slaughtered_chicken"] == 2
