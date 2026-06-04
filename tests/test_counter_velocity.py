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


def test_two_sequential_birds_count_twice():
    c = ChickenCounter(roi_x=200, conveyor_speed_px=34.0, zone_half=0)
    for cx in range(120, 340, 34):
        c.update([_det(cx)])
    for _ in range(12):
        c.update([])
    for cx in range(120, 340, 34):
        c.update([_det(cx)])
    assert c.counts["slaughtered_chicken"] == 2
