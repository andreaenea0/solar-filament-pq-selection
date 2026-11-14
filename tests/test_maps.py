"""Check ensemble parity and the existing probability-map file format."""

import numpy as np
import pytest

from filseg.maps import matched_family_maps


def test_matching_preserves_channel_order_and_avoids_extra_peak_smoothing():
    a = np.array([[255, 11]], dtype=np.uint8)
    b = np.array([[240, 10]], dtype=np.uint8)
    c = np.array([[11, 255]], dtype=np.uint8)
    d = np.array([[7, 250]], dtype=np.uint8)

    oof = matched_family_maps([[a], [c, d]], (1, 2))
    matched = matched_family_maps([[a, b], [c, d, a]], (1, 2))
    full = matched_family_maps([[a, b], [c, d, a]], (2, 3))

    np.testing.assert_array_equal(matched, oof)
    np.testing.assert_array_equal(matched, [[[255, 11]], [[9, 252]]])
    assert matched.dtype == np.uint8
    assert full[0, 0, 0] < matched[0, 0, 0]


def test_matching_requires_available_compatible_members():
    p = np.zeros((2, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="requested number"):
        matched_family_maps([[p]], (2,))
    with pytest.raises(ValueError, match="same shape"):
        matched_family_maps([[p], [p.T]], (1, 1))
    with pytest.raises(ValueError, match="uint8"):
        matched_family_maps([[p.astype(float)]], (1,))
