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
