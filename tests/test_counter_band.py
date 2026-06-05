from app.core.counter import ChickenCounter


def _det(cx, cls="slaughtered_chicken", w=20, h=80):
    return {"x1": cx - w // 2, "y1": 100 - h // 2,
            "x2": cx + w // 2, "y2": 100 + h // 2, "class_name": cls}


# Trajectory (180, 222) with a narrow bbox (w=20) genuinely skips roi_x=200:
#   180 -> bbox [170,190] (entirely below 200)
#   222 -> bbox [212,232] (entirely above 200)
# So at zone_half=0 (single-pixel x1<=200<=x2) neither frame straddles -> miss.
# At zone_half=15 the band is [185,215]: 190>=185 and 212<=215 -> both overlap.
_SKIP_TRAJECTORY = (180, 222)


def test_band_catches_bird_that_skips_exact_line():
    """A narrow bbox skipping the exact line between frames is still caught
    by the band."""
    c = ChickenCounter(roi_x=200, conveyor_speed_px=34.0, zone_half=15)
    for cx in _SKIP_TRAJECTORY:
        c.update([_det(cx)])
    assert c.counts["slaughtered_chicken"] == 1


def test_single_pixel_mode_misses_when_zone_zero():
    """Same trajectory with zone_half=0 misses — documents why the band helps."""
    c = ChickenCounter(roi_x=200, conveyor_speed_px=34.0, zone_half=0)
    for cx in _SKIP_TRAJECTORY:
        c.update([_det(cx)])
    assert c.counts["slaughtered_chicken"] == 0


def test_zone_zero_reduces_to_original_tripwire():
    """At zone_half=0 a bbox that DOES contain roi_x still counts (band reduces
    to the original single-pixel straddle x1 <= roi_x <= x2)."""
    c = ChickenCounter(roi_x=200, conveyor_speed_px=34.0, zone_half=0)
    for cx in range(120, 320, 34):  # 188 -> [148,228] and 222 -> [182,262] contain 200
        c.update([_det(cx, w=80)])
    assert c.counts["slaughtered_chicken"] == 1


def test_wide_band_still_counts_once():
    c = ChickenCounter(roi_x=200, conveyor_speed_px=34.0, zone_half=40)
    for cx in range(120, 320, 34):
        c.update([_det(cx, w=80)])
    assert c.counts["slaughtered_chicken"] == 1
