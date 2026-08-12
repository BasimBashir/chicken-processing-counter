"""Wiring tests: VideoProcessor's reported counts carry the gap-adjusted
``empty_shackles`` value on the one dict every endpoint and the UI read
(``VideoProcessor.counts`` -> get_status -> /api/stream, /api/streams/{id},
/api/video). Needs ultralytics importable (module-level import), so these run
in the Docker image alongside test_api_shapes.py, not the bare host."""
from app.core.video_processor import VideoProcessor


def _cw(empty=0, chicken=0, load=0):
    """classwise_count as ObjectCounter reports it (belt is one-way -> IN)."""
    return {"empty_shackles": {"IN": empty, "OUT": 0},
            "slaughtered_chicken": {"IN": chicken, "OUT": 0},
            "load": {"IN": load, "OUT": 0}}


def test_reported_empty_count_is_gap_adjusted():
    p = VideoProcessor("dummy-source", "dummy.pt")
    p._update_counts(_cw(empty=5, chicken=2))
    assert p.counts["empty_shackles"] == 5          # pre-load: counts normally
    p._update_counts(_cw(empty=5, chicken=2, load=1))   # placard crosses
    p._update_counts(_cw(empty=8, chicken=2, load=1))   # 3 gap empties
    assert p.counts["empty_shackles"] == 5          # ...dropped
    assert p.counts["slaughtered_chicken"] == 2     # untouched
    assert p.counts["load"] == 1                    # untouched
    p._update_counts(_cw(empty=8, chicken=3, load=1))   # first bird ends gap
    p._update_counts(_cw(empty=9, chicken=4, load=1))
    assert p.counts["empty_shackles"] == 6


def test_shipment_tally_unaffected_by_empty_suppression():
    p = VideoProcessor("dummy-source", "dummy.pt")
    p._update_counts(_cw(empty=0, chicken=10, load=1))  # arms shipment 1
    p._update_counts(_cw(empty=4, chicken=15, load=1))  # gap empties dropped
    status = p.get_status()
    assert status["current_shipment"]["index"] == 1
    assert status["current_shipment"]["slaughtered_count"] == 5
    assert status["counts"]["empty_shackles"] == 0


def test_reset_counts_clears_gap_state():
    p = VideoProcessor("dummy-source", "dummy.pt")
    p._update_counts(_cw(empty=2, load=1))          # placard seen, gap open
    p._update_counts(_cw(empty=6, load=1))          # 4 gap empties dropped
    p.reset_counts()
    assert p.counts["empty_shackles"] == 0
    # Same raw values arriving again must not be eaten by stale gap state.
    p._update_counts(_cw(empty=6, load=1))
    assert p.counts["empty_shackles"] == 6
