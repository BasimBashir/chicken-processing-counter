import numpy as np
from app.core.video_processor import reconnect_delay, frame_signature


def test_reconnect_delay_exponential_with_cap():
    assert reconnect_delay(0) == 1.0
    assert reconnect_delay(1) == 2.0
    assert reconnect_delay(2) == 4.0
    assert reconnect_delay(10) == 30.0  # capped


def test_frame_signature_stable_for_identical_frames():
    a = np.full((480, 848, 3), 7, dtype=np.uint8)
    b = a.copy()
    assert frame_signature(a) == frame_signature(b)


def test_frame_signature_changes_with_content():
    a = np.zeros((480, 848, 3), dtype=np.uint8)
    b = a.copy()
    b[0:100, 0:100] = 255
    assert frame_signature(a) != frame_signature(b)
