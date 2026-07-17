"""Tests for container-safe timestamps. The canonical value is epoch; the ISO
string renders in an explicit TZ so it is correct regardless of the container's
default (UTC) clock."""
import importlib
import os

from app.core import timeutil


def test_epoch_is_float_seconds():
    assert isinstance(timeutil.epoch(), float)


def test_iso_local_none_passthrough():
    assert timeutil.iso_local(None) is None


def test_iso_local_renders_in_configured_tz(monkeypatch):
    # 2021-01-01T00:00:00 UTC == 1609459200 epoch.
    monkeypatch.setenv("TZ", "UTC")
    assert timeutil.iso_local(1609459200).startswith("2021-01-01T00:00:00")

    # Same instant, +05:00 zone -> 05:00 local with an explicit offset. Proves
    # the string follows TZ, not the host/container default clock.
    monkeypatch.setenv("TZ", "Asia/Karachi")
    out = timeutil.iso_local(1609459200)
    assert out == "2021-01-01T05:00:00+05:00"


def test_garbage_tz_falls_back_without_crashing(monkeypatch):
    monkeypatch.setenv("TZ", "Not/AZone")
    # Must not raise; returns some ISO 8601 string in the fallback zone.
    assert "2021-01-01T" in timeutil.iso_local(1609459200)
