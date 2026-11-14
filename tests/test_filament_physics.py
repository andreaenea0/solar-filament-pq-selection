"""Check deprojection and the normalization of the report summaries."""

import numpy as np
import pytest

from filseg.disk import Disk
from filseg.metric import encode
from tools.figures.filament_physics import cycle_year, measure_mask, summarize, surface_points


def test_deprojection_preserves_central_tilt_and_recovers_b0():
    b0 = np.radians(6)
    np.testing.assert_allclose(surface_points(0, 0, b0), [0, np.sin(b0), np.cos(b0)])
    disk = Disk(50, 50, 45, 0, 360)
    horizontal = np.zeros((101, 101), dtype=bool)
    horizontal[49:52, 42:59] = True
    h = measure_mask(encode(horizontal), disk, b0)
    v = measure_mask(encode(horizontal.T), disk, b0)
    assert h.latitude == pytest.approx(6)
    assert (h.tilt, v.tilt) == pytest.approx((0, 90), abs=0.1)
    assert h.area_px == horizontal.sum()
    projected_msh = h.area_px * 1e6 / (2 * np.pi * disk.radius ** 2)
    assert projected_msh < h.area_msh < 1.01 * projected_msh


def test_physics_summary_exposes_radial_and_year_counts():
    stem = "20191201000000Bh"
    disk = Disk(50, 50, 45, 0, 360)
    mask = np.zeros((101, 101), dtype=bool)
    mask[49:52, 45:56] = True
    rle = encode(mask)
    rows = list(summarize({stem: {"one": [rle], "empty": []}}, {stem: disk}, {stem: 0.}))
    assert rows[0]["filaments"] == 1 and rows[0]["records"] == 2
    assert rows[0]["density_per_record_per_disk"] == pytest.approx(12.5)
    assert rows[-1]["year"] == "2019" and rows[-1]["cycle"] == 25
    assert rows[-1]["eligible"] == 1
    assert cycle_year("20191130000000Bh") == ("2019", 24)
