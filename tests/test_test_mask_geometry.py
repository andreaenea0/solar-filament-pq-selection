"""Check comparable geometry bins and matched measurement cuts."""

import pytest

from tools.figures.filament_physics import Measurement
from tools.figures.test_mask_geometry import comparison


def test_comparison_normalizes_radial_shares_and_keeps_eligible_size():
    rows = [
        ("2011a", Measurement(.5, 5., 30., 100, 12.)),
        ("2011b", Measurement(.95, 15., None, 50, None)),
    ]
    training = [
        dict(section="radial", lower="0", upper=".8", filaments="3"),
        dict(section="radial", lower=".8", upper="1", filaments="1"),
        dict(section="size_by_year", year="2011", eligible="1", area_msh_median="7", area_px_median="80"),
    ]
    inner, outer, size = list(comparison(rows, training))
    assert inner["train_share"] == .75 and inner["test_share"] == .5
    assert outer["test_n"] == 1
    assert size["test_n"] == 1
    assert size["test_median"] == pytest.approx(12)
