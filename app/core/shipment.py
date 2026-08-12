"""Load-to-load shipment segmentation.

A *shipment* is the span of belt between two consecutive ``load`` markers. The
plant hangs a fluorescent "LOAD" placard on an empty shackle ahead of each
delivery, so every time a ``load`` object crosses the counting line we close the
shipment that was in progress and open the next one.

Counting stays **dormant until the first ``load`` crosses the ROI line** — birds
seen before any marker (the "pre-load" span) are discarded, so a shipment's tally
only ever covers the interval *between two markers*. The first marker arms and
opens shipment 1; the second closes shipment 1 and opens shipment 2; and so on.

This rides entirely on the cumulative per-class IN counts the ObjectCounter
already produces — it only ever READS them. It never resets or mutates the
counter, so the lifetime cumulative counts keep climbing exactly as before; the
per-shipment number is *derived* (``slaughtered_now - slaughtered_at_boundary``),
not a competing counter.

Pure-stdlib and dependency-light (only the sibling ``timeutil``) so it imports
cheaply and is unit-testable without torch/opencv.
"""
import threading
from collections import deque

from app.core.timeutil import epoch, iso_local

# Which detected class marks a shipment boundary, and which class is tallied
# per shipment. Both MUST match the trained model's class names. If the new
# placard class is labelled something other than "load" in your dataset, change
# LOAD_CLASS here to match.
LOAD_CLASS = "load"
SHIPMENT_COUNT_CLASS = "slaughtered_chicken"
EMPTY_CLASS = "empty_shackles"

# How many closed shipments to retain in memory for the /shipments cursor. A
# day is ~10-12 shipments, so this holds well over a month; the downstream UI
# polls and aggregates, this is just a replay buffer so a boundary is never
# missed between polls.
DEFAULT_HISTORY = 500


def _record(index: int, started_at: float, ended_at: float | None,
            slaughtered_count: int) -> dict:
    """One shipment as returned by the API: epoch (canonical) + ISO-8601 local
    (human, container-safe) timestamps. ``ended_at`` is None while open."""
    return {
        "index": index,
        "started_at": started_at,
        "started_at_iso": iso_local(started_at),
        "ended_at": ended_at,
        "ended_at_iso": iso_local(ended_at),
        "slaughtered_count": slaughtered_count,
    }


class EmptyGapFilter:
    """Drops the irrelevant run of empty shackles between a LOAD placard and the
    first bird of that delivery from the reported ``empty_shackles`` count.

    From the frame a ``load`` crossing is observed until the first
    ``slaughtered_chicken`` crossing after it, empty-shackle crossings accrue to
    an internal ``discarded`` total instead of the visible count; ``update``
    returns the adjusted cumulative (``empty_cum - discarded``). Before the
    first load, and once a delivery's first bird has crossed, empties count
    normally. Same-frame ties are attributed by the state BEFORE the frame's
    transitions, which matches the one-way belt order: an empty seen alongside
    the placard was downstream of it (still the previous span — counted), one
    seen alongside the gap-ending bird was still in the gap (dropped).

    Like ShipmentTracker this only ever READS the counter's cumulative totals;
    it never resets or mutates the counter. Thread-safe: fed from the capture
    loop, reset from API request threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        """Zero all state. Called when the underlying counter is rebuilt so a
        stale gap or stale discards can't eat the fresh counter's empties."""
        with self._lock:
            self._in_gap = False
            self._discarded = 0
            self._last_load = 0
            self._last_empty = 0
            self._last_chicken = 0

    def update(self, load_cum: int, empty_cum: int, chicken_cum: int) -> int:
        """Feed the latest cumulative counts; returns the adjusted cumulative
        empty count. A falling cumulative means the counter was rebuilt
        underneath us (reset is handled separately): resync to the new values
        with the gap closed and discards cleared."""
        with self._lock:
            if (load_cum < self._last_load or empty_cum < self._last_empty
                    or chicken_cum < self._last_chicken):
                self._in_gap = False
                self._discarded = 0
            else:
                # Attribute this frame's empties by the PRIOR state, then apply
                # transitions — chicken closes the gap before a load reopens it,
                # so a same-frame bird+placard starts the next delivery's gap.
                if self._in_gap:
                    self._discarded += empty_cum - self._last_empty
                if chicken_cum > self._last_chicken:
                    self._in_gap = False
                if load_cum > self._last_load:
                    self._in_gap = True
            self._last_load = load_cum
            self._last_empty = empty_cum
            self._last_chicken = chicken_cum
            return max(0, empty_cum - self._discarded)


class ShipmentTracker:
    """Derives per-shipment counts from cumulative counter totals. Thread-safe:
    fed from the capture loop, read from API request threads."""

    def __init__(self, history: int = DEFAULT_HISTORY) -> None:
        self._lock = threading.Lock()
        self._history = history
        self.reset()

    def reset(self) -> None:
        """Zero all shipment state. Called when the underlying counter is rebuilt
        (counting reset) so the derived count can't go negative against a counter
        that just jumped back to zero. Counting stays dormant until the next load
        marker arms it again."""
        with self._lock:
            self._closed: deque = deque(maxlen=self._history)
            self._armed = False        # no shipment until a load marker crosses
            self._index = 0            # 0 == pre-load: no active shipment yet
            self._started_at = None    # set when the first load arms shipment 1
            self._baseline = 0         # cumulative slaughtered when this shipment opened
            self._last_load = 0        # cumulative load count last observed
            self._total_closed = 0

    def update(self, load_cum: int, slaughtered_cum: int,
               now: float | None = None) -> int:
        """Feed the latest cumulative counts. The FIRST rise in the cumulative
        ``load`` count arms the tracker — it discards the pre-load birds and opens
        shipment 1 without emitting a record. Each SUBSEQUENT rise closes the open
        shipment (snapshotting its slaughtered tally) and opens the next. Returns
        the number of load crossings seen this call (normally 0 or 1). Read-only
        on the counter."""
        now = epoch() if now is None else now
        with self._lock:
            crossings = load_cum - self._last_load
            if crossings <= 0:
                # No new load, or the counter was rebuilt to a lower value
                # (reset handled separately): just resync and do nothing.
                self._last_load = load_cum
                return 0
            for _ in range(crossings):
                if not self._armed:
                    # First load ever: a shipment only exists from here on. The
                    # pre-load birds are discarded (baseline = current cumulative)
                    # so "this shipment" starts at zero. No record is emitted.
                    self._armed = True
                    self._index = 1
                    self._started_at = now
                    self._baseline = slaughtered_cum
                else:
                    self._closed.append(_record(
                        self._index, self._started_at, now,
                        max(0, slaughtered_cum - self._baseline),
                    ))
                    self._total_closed += 1
                    self._index += 1
                    self._started_at = now
                    self._baseline = slaughtered_cum
            self._last_load = load_cum
            return crossings

    def current(self, slaughtered_cum: int) -> dict:
        """The open shipment: its index, start, and live slaughtered tally. Before
        the first load marker arms the tracker there is no active shipment, so it
        reports index 0 with a zero count and no start."""
        with self._lock:
            if not self._armed:
                return _record(0, None, None, 0)
            return _record(self._index, self._started_at, None,
                           max(0, slaughtered_cum - self._baseline))

    def total_shipments(self) -> int:
        """Count of CLOSED (completed) shipments since the last reset."""
        with self._lock:
            return self._total_closed

    def records_since(self, since: int = 0) -> list[dict]:
        """Closed shipment records with ``index >= since`` (oldest first). The
        downstream poller tracks the highest index it has seen and asks for the
        next one to pull incrementally without missing a boundary."""
        with self._lock:
            return [r for r in self._closed if r["index"] >= since]
