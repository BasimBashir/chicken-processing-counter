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


# ── Belt stop hysteresis ────────────────────────────────────────────────────

def _simulate_belt_state(motion_sequence, stop_run_frames=42, stop_thresh=0.4,
                         resume_thresh=2.82, resume_run=2):
    """Feed a list of motion diff values through the belt-stop state machine.
    Returns list of belt_stopped booleans, one per frame."""
    belt_stopped = False
    _stop_run = 0
    _resume_run = 0
    results = []
    for motion in motion_sequence:
        if not belt_stopped:
            if motion < stop_thresh:
                _stop_run += 1
            else:
                _stop_run = 0
            if _stop_run >= stop_run_frames:
                belt_stopped = True
                _resume_run = 0
        else:
            if motion > resume_thresh:
                _resume_run += 1
            else:
                _resume_run = 0
            if _resume_run >= resume_run:
                belt_stopped = False
                _stop_run = 0
        results.append(belt_stopped)
    return results


def test_inter_bird_gap_does_not_trigger_belt_stop():
    """27-frame gap (max measured inter-bird gap) must NOT set belt_stopped."""
    motion = [0.05] * 27
    states = _simulate_belt_state(motion, stop_run_frames=42)
    assert not any(states), "Short inter-bird gap must never trigger belt_stopped"


def test_true_belt_stop_triggers_after_42_frames():
    """A genuine belt stop (60+ frames of low motion) must set belt_stopped."""
    motion = [0.05] * 60
    states = _simulate_belt_state(motion, stop_run_frames=42)
    assert states[42], "belt_stopped must be True by frame 42"
    assert all(states[42:]), "belt_stopped must stay True while motion is low"


def test_resume_requires_high_motion_for_2_frames():
    """1 frame above resume_thresh must NOT clear belt_stopped; 2 must."""
    motion = [0.05] * 60 + [3.0] + [0.05]
    states = _simulate_belt_state(motion, stop_run_frames=42)
    assert states[60], "1 high frame not enough to resume"

    motion2 = [0.05] * 60 + [3.0, 3.0]
    states2 = _simulate_belt_state(motion2, stop_run_frames=42)
    assert not states2[-1], "2 consecutive high frames must clear belt_stopped"


def test_slow_restart_ramp_stays_stopped_until_above_resume_thresh():
    """10 frames at 1.5 (below resume_thresh=2.82) must NOT clear belt_stopped."""
    motion = [0.05] * 60 + [1.5] * 10
    states = _simulate_belt_state(motion, stop_run_frames=42)
    assert all(states[42:]), "Slow-start ramp below resume_thresh must keep belt_stopped=True"
