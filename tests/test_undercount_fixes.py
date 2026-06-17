"""Tests for undercount fixes: belt hysteresis, optimal assignment, adaptive zone."""
import pytest
from app.config import Settings
from app.core.runtime_config import RuntimeConfig


def test_config_new_params_exist():
    s = Settings()
    assert s.stop_run_frames == 42
    assert s.stop_resume_thresh == 2.82
    assert s.zone_speed_factor == 1.20
    assert s.zone_half == 18          # raised from 15


def test_runtime_config_exposes_new_params():
    snap = RuntimeConfig().snapshot()
    assert "stop_run_frames" in snap
    assert "stop_resume_thresh" in snap
    assert "zone_speed_factor" in snap
    assert snap["zone_half"] == 18


from app.core.counter import ChickenCounter


def _det(cx, cls="slaughtered_chicken", w=80):
    return {"x1": cx - w // 2, "y1": 60, "x2": cx + w // 2, "y2": 140,
            "class_name": cls}


# ── Optimal assignment ──────────────────────────────────────────────────────

def test_three_clustered_birds_all_counted():
    """Greedy matching would mis-assign the middle bird; optimal must get all 3."""
    c = ChickenCounter(roi_x=400, conveyor_speed_px=25, zone_half=20, sway_k=0.6)
    for frame in range(5):
        dets = [_det(360 + 25 * frame), _det(385 + 25 * frame), _det(410 + 25 * frame)]
        c.update(dets)
    assert c.counts["slaughtered_chicken"] == 3


def test_two_birds_close_together_both_counted():
    """Two birds 30px apart crossing simultaneously must both register."""
    c = ChickenCounter(roi_x=300, conveyor_speed_px=34, zone_half=20, sway_k=0.6)
    for cx_lead in range(260, 380, 34):
        c.update([_det(cx_lead), _det(cx_lead - 30)])
    assert c.counts["slaughtered_chicken"] == 2


# ── Adaptive zone ───────────────────────────────────────────────────────────

def test_adaptive_zone_disabled_when_no_active_crossings():
    """With no active crossings, falls back to conveyor_speed_px for speed
    estimate — should still function without crashing."""
    c = ChickenCounter(roi_x=300, conveyor_speed_px=15, zone_half=18,
                       sway_k=0.6, zone_speed_factor=1.20)
    # A bird well inside even static zone — must count regardless
    c.update([_det(300)])
    assert c.counts["slaughtered_chicken"] == 1


def test_adaptive_zone_widens_at_high_speed():
    """Bird outside static zone_half=18 must be caught when adaptive zone is on,
    and missed when zone_speed_factor=0 disables it."""
    # Adaptive ON: conveyor_speed_px=50, factor=1.20 → effective_zone_half=60
    # zone = [500-60, 500+60] = [440, 560]
    # Bird at cx=440 (x1=400, x2=480): outside static [482,518], inside adaptive [440,560]
    c_on = ChickenCounter(roi_x=500, conveyor_speed_px=50, zone_half=18,
                          sway_k=0.0, zone_speed_factor=1.20)
    c_on.update([_det(440)])
    assert c_on.counts["slaughtered_chicken"] == 1, "Adaptive zone must catch marginal bird"

    # Adaptive OFF: same geometry but factor=0 → effective_zone_half=max(18,0)=18
    # zone = [482, 518]; bird at cx=440 is outside → missed
    c_off = ChickenCounter(roi_x=500, conveyor_speed_px=50, zone_half=18,
                           sway_k=0.0, zone_speed_factor=0.0)
    c_off.update([_det(440)])
    assert c_off.counts["slaughtered_chicken"] == 0, "Static zone must miss marginal bird"
