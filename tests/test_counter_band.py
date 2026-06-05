from app.core.counter import ChickenCounter


def _det(cx, cls="slaughtered_chicken", w=20, h=80):
    return {"x1": cx - w // 2, "y1": 100 - h // 2,
            "x2": cx + w // 2, "y2": 100 + h // 2, "class_name": cls}


def test_band_catches_bird_that_skips_exact_line():
    """Narrow bbox (w=20) at 34 px/frame never lands exactly on roi_x=200,
    but a 15px band still catches it."""
    c = ChickenCounter(roi_x=200, conveyor_speed_px=34.0, zone_half=15)
    for cx in (175, 209, 243):  # none has x1<=200<=x2 (half-width 10); 209 overlaps [185,215]
        c.update([_det(cx)])
    assert c.counts["slaughtered_chicken"] == 1


def test_single_pixel_mode_misses_when_zone_zero():
    """Same trajectory with zone_half=0 misses (documents why band helps)."""
    c = ChickenCounter(roi_x=200, conveyor_speed_px=34.0, zone_half=0)
    for cx in (175, 209, 243):
        c.update([_det(cx)])
    assert c.counts["slaughtered_chicken"] == 0


def test_wide_band_still_counts_once():
    c = ChickenCounter(roi_x=200, conveyor_speed_px=34.0, zone_half=40)
    for cx in range(120, 320, 34):
        c.update([_det(cx, w=80)])
    assert c.counts["slaughtered_chicken"] == 1
