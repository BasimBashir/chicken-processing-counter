"""Timezone-safe timestamps for shipment events.

Containers default to UTC unless `TZ` is set, so a naive ``datetime.now()`` is a
frequent source of wrong wall-clock values inside Docker. We avoid that two ways:

  * The canonical value we store and return is the Unix epoch (``time.time()``) —
    an absolute instant that is identical in every timezone, so it can never be
    "wrong" regardless of how the container clock/zone is configured.
  * For human-readable output we format that instant in an EXPLICIT timezone,
    resolved from the ``TZ`` env var (falling back to the host/container local
    zone, then UTC). To make local-time strings match the plant's wall clock,
    run the container with ``-e TZ=Area/City`` (e.g. ``TZ=Asia/Karachi``); the
    bundled ``tzdata`` package means the zone resolves even on a slim image or
    on a Windows dev host with no system zone database.

Output format is ISO 8601 with an explicit UTC offset, e.g.
``2026-06-25T14:03:21+05:00`` — unambiguous and trivially parseable downstream.
"""
import os
import time
from datetime import datetime, timezone

try:  # zoneinfo is stdlib on 3.9+; the `tzdata` pip package supplies the DB.
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - only on a broken/ancient interpreter
    ZoneInfo = None


def _resolve_tz():
    """The timezone to render local strings in: explicit ``TZ`` env if set and
    resolvable, else the host/container local zone, else UTC."""
    name = (os.environ.get("TZ") or "").strip()
    if name and ZoneInfo is not None:
        try:
            return ZoneInfo(name)
        except Exception:
            pass  # unknown/garbage TZ -> fall through to local, never crash
    local = datetime.now().astimezone().tzinfo
    return local or timezone.utc


def epoch() -> float:
    """Absolute instant as Unix seconds (3-decimal). Timezone-independent — this
    is the canonical, never-wrong value the shipment records are keyed on."""
    return round(time.time(), 3)


def iso_local(ts: float | None) -> str | None:
    """Format an epoch instant as ISO 8601 with an explicit UTC offset, in the
    configured timezone. Returns None for a None input (open shipment end)."""
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=_resolve_tz()).isoformat(timespec="seconds")
