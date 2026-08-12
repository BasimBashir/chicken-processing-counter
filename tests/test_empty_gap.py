"""Unit tests for EmptyGapFilter — suppression of the irrelevant empty-shackle
run between a LOAD placard and the first bird of that delivery. Pure logic (no
torch/cv2), so these run anywhere, same as test_shipment.py.

Rule under test: from a `load` crossing until the first `slaughtered_chicken`
crossing, empty-shackle crossings are discarded from the reported count; before
the first load and after that first chicken, empties count normally. Same-frame
ties are attributed by the state BEFORE the frame's transitions (belt order:
downstream objects crossed first)."""
from app.core.shipment import EmptyGapFilter


def test_counts_normally_before_first_load():
    f = EmptyGapFilter()
    assert f.update(load_cum=0, empty_cum=5, chicken_cum=2) == 5
    assert f.update(load_cum=0, empty_cum=9, chicken_cum=4) == 9


def test_suppresses_empties_after_load_until_first_chicken():
    f = EmptyGapFilter()
    assert f.update(load_cum=0, empty_cum=5, chicken_cum=2) == 5
    assert f.update(load_cum=1, empty_cum=5, chicken_cum=2) == 5   # placard crosses
    assert f.update(load_cum=1, empty_cum=8, chicken_cum=2) == 5   # 3 gap empties dropped
    assert f.update(load_cum=1, empty_cum=8, chicken_cum=3) == 5   # first bird ends gap
    assert f.update(load_cum=1, empty_cum=9, chicken_cum=4) == 6   # empties count again


def test_empty_in_same_frame_as_load_counts():
    # An empty crossing alongside the placard was downstream of it (tail of the
    # previous span) -> attributed by prior no-gap state -> counts.
    f = EmptyGapFilter()
    assert f.update(load_cum=0, empty_cum=5, chicken_cum=2) == 5
    assert f.update(load_cum=1, empty_cum=6, chicken_cum=2) == 6
    assert f.update(load_cum=1, empty_cum=7, chicken_cum=2) == 6   # now in gap


def test_empty_in_same_frame_as_first_chicken_is_suppressed():
    # An empty crossing alongside the gap-ending bird was still in the gap
    # (crossed just before it) -> attributed by prior in-gap state -> dropped.
    f = EmptyGapFilter()
    assert f.update(load_cum=1, empty_cum=5, chicken_cum=2) == 5   # gap opens
    assert f.update(load_cum=1, empty_cum=6, chicken_cum=3) == 5   # dropped + gap ends
    assert f.update(load_cum=1, empty_cum=7, chicken_cum=3) == 6


def test_gap_persists_across_birdless_delivery():
    f = EmptyGapFilter()
    assert f.update(load_cum=1, empty_cum=0, chicken_cum=0) == 0   # gap opens
    assert f.update(load_cum=1, empty_cum=4, chicken_cum=0) == 0   # gap empties
    assert f.update(load_cum=2, empty_cum=4, chicken_cum=0) == 0   # 2nd placard, no birds yet
    assert f.update(load_cum=2, empty_cum=6, chicken_cum=0) == 0   # still suppressed
    assert f.update(load_cum=2, empty_cum=6, chicken_cum=1) == 0   # first bird ends gap
    assert f.update(load_cum=2, empty_cum=8, chicken_cum=5) == 2


def test_every_delivery_gets_its_own_gap():
    f = EmptyGapFilter()
    # Delivery 1: placard, 2 gap empties, birds, then 1 real empty.
    f.update(load_cum=1, empty_cum=0, chicken_cum=0)
    f.update(load_cum=1, empty_cum=2, chicken_cum=0)
    f.update(load_cum=1, empty_cum=2, chicken_cum=10)
    assert f.update(load_cum=1, empty_cum=3, chicken_cum=10) == 1
    # Delivery 2: placard reopens a gap, 3 gap empties dropped, birds, 1 real.
    f.update(load_cum=2, empty_cum=3, chicken_cum=10)
    assert f.update(load_cum=2, empty_cum=6, chicken_cum=10) == 1
    f.update(load_cum=2, empty_cum=6, chicken_cum=11)
    assert f.update(load_cum=2, empty_cum=7, chicken_cum=12) == 2


def test_same_frame_chicken_and_load_opens_fresh_gap():
    # Last bird of the old delivery and the next placard in one frame: the
    # chicken closes the old gap state, the placard opens the next delivery's.
    f = EmptyGapFilter()
    f.update(load_cum=1, empty_cum=0, chicken_cum=0)               # gap 1 opens
    assert f.update(load_cum=2, empty_cum=1, chicken_cum=1) == 0   # empty was in gap 1
    assert f.update(load_cum=2, empty_cum=3, chicken_cum=1) == 0   # gap 2 active
    assert f.update(load_cum=2, empty_cum=3, chicken_cum=2) == 0   # bird ends gap 2
    assert f.update(load_cum=2, empty_cum=4, chicken_cum=2) == 1


def test_multiple_gap_empties_in_one_update_all_dropped():
    f = EmptyGapFilter()
    f.update(load_cum=1, empty_cum=10, chicken_cum=0)
    assert f.update(load_cum=1, empty_cum=14, chicken_cum=0) == 10  # burst of 4 dropped
    assert f.update(load_cum=1, empty_cum=14, chicken_cum=1) == 10
    assert f.update(load_cum=1, empty_cum=16, chicken_cum=2) == 12


def test_reset_returns_to_normal_counting():
    f = EmptyGapFilter()
    f.update(load_cum=1, empty_cum=0, chicken_cum=0)
    f.update(load_cum=1, empty_cum=5, chicken_cum=0)               # 5 dropped, in gap
    f.reset()
    # Fresh counter after reset: no gap, no carried discards.
    assert f.update(load_cum=0, empty_cum=3, chicken_cum=0) == 3


def test_counter_rebuilt_lower_resyncs_without_carried_discards():
    # Cumulative values falling means the counter was rebuilt underneath us:
    # resync to the new values, clear the gap, and drop stale discards so the
    # fresh counter's empties are never eaten by history.
    f = EmptyGapFilter()
    f.update(load_cum=1, empty_cum=2, chicken_cum=0)
    f.update(load_cum=1, empty_cum=6, chicken_cum=0)               # 4 dropped, in gap
    assert f.update(load_cum=0, empty_cum=0, chicken_cum=0) == 0   # rebuilt
    assert f.update(load_cum=0, empty_cum=2, chicken_cum=0) == 2   # counts normally
