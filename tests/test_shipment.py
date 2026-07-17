"""Unit tests for load-to-load shipment segmentation. Pure logic — no torch/cv2,
so these run anywhere (including the broken local venv via system Python).

Key rule under test: counting is dormant until the first `load` crosses the line;
the pre-load span is discarded, and a shipment's tally only covers the interval
between two consecutive load markers."""
from app.core.shipment import ShipmentTracker


def test_no_shipment_until_first_load():
    t = ShipmentTracker()
    # Birds cross but NO load yet -> nothing is attributed to any shipment.
    t.update(load_cum=0, slaughtered_cum=137, now=100.0)
    cur = t.current(137)
    assert cur["index"] == 0
    assert cur["started_at"] is None
    assert cur["slaughtered_count"] == 0
    assert t.total_shipments() == 0
    assert t.records_since(0) == []


def test_first_load_arms_and_discards_preload():
    t = ShipmentTracker()
    t.update(load_cum=0, slaughtered_cum=137, now=100.0)     # pre-load birds
    n = t.update(load_cum=1, slaughtered_cum=137, now=200.0)  # first LOAD crosses
    assert n == 1
    assert t.total_shipments() == 0          # nothing completed by the first marker
    assert t.records_since(0) == []          # pre-load span is never emitted
    cur = t.current(137)
    assert cur["index"] == 1                 # shipment 1 is now open
    assert cur["started_at"] == 200.0
    assert cur["slaughtered_count"] == 0     # the 137 pre-load birds are discarded
    # Birds AFTER the first load accrue to shipment 1.
    assert t.current(140)["slaughtered_count"] == 3


def test_shipment_closes_on_second_load_with_between_count():
    t = ShipmentTracker()
    t.update(load_cum=1, slaughtered_cum=137, now=10.0)   # arm (discard 137)
    t.update(load_cum=1, slaughtered_cum=140, now=11.0)   # +3 within shipment 1
    t.update(load_cum=2, slaughtered_cum=140, now=12.0)   # second LOAD closes shipment 1
    closed = t.records_since(0)
    assert len(closed) == 1
    assert closed[0]["index"] == 1
    assert closed[0]["slaughtered_count"] == 3            # only birds between the markers
    assert closed[0]["started_at"] == 10.0
    assert closed[0]["ended_at"] == 12.0
    assert t.total_shipments() == 1
    # Shipment 2 now open and zeroed.
    assert t.current(140)["index"] == 2
    assert t.current(145)["slaughtered_count"] == 5


def test_records_since_cursor():
    t = ShipmentTracker()
    # Loads observed at cumulative slaughtered 10/20/30/40 -> shipments 1,2,3
    # close (10 birds each); shipment 4 is left open.
    for i, s in enumerate([10, 20, 30, 40], start=1):
        t.update(load_cum=i, slaughtered_cum=s, now=float(i))
    assert [r["index"] for r in t.records_since(0)] == [1, 2, 3]
    assert [r["slaughtered_count"] for r in t.records_since(0)] == [10, 10, 10]
    assert [r["index"] for r in t.records_since(2)] == [2, 3]
    assert t.records_since(99) == []
    assert t.current(40)["index"] == 4


def test_multiple_crossings_in_one_update():
    t = ShipmentTracker()
    # Three loads observed between polls; slaughtered constant at 5.
    n = t.update(load_cum=3, slaughtered_cum=5, now=1.0)
    assert n == 3
    # First arms (discard 5); the next two close empty shipments 1 and 2.
    assert [r["slaughtered_count"] for r in t.records_since(0)] == [0, 0]
    assert t.total_shipments() == 2
    assert t.current(5)["index"] == 3


def test_reset_returns_to_dormant_state():
    t = ShipmentTracker()
    t.update(load_cum=2, slaughtered_cum=20, now=1.0)   # arm + close shipment 1
    assert t.total_shipments() == 1
    t.reset()
    assert t.total_shipments() == 0
    assert t.records_since(0) == []
    cur = t.current(0)
    assert cur["index"] == 0
    assert cur["started_at"] is None
    assert cur["slaughtered_count"] == 0
    # After reset, counting is dormant again until the next load arms it.
    assert t.current(99)["slaughtered_count"] == 0


def test_counter_rebuilt_lower_resyncs_without_phantom_loads():
    t = ShipmentTracker()
    t.update(load_cum=2, slaughtered_cum=20, now=1.0)   # arm + 1 closed shipment
    # Counter rebuilt -> load_cum drops; must not fire phantom loads.
    assert t.update(load_cum=0, slaughtered_cum=0, now=2.0) == 0
    # Next genuine load increments from the resynced value.
    assert t.update(load_cum=1, slaughtered_cum=4, now=3.0) == 1
