"""Check the radial attribution of matches, misses and false positives."""

import numpy as np
import pytest

from filseg.disk import Disk
from filseg.metric import encode
from tools.figures.filament_physics import RADIAL_BINS
from tools.figures.radius_agreement import image_terms, radius_rows

DISK = Disk(1000, 1000, 900, 0, 360)


def box(top, bottom, left, right):
    mask = np.zeros((2048, 2048), dtype=bool)
    mask[top:bottom, left:right] = True
    return encode(mask)


def test_models_and_annotator_pairs_use_the_same_terms():
    centre, limb = box(990, 1010, 950, 1050), box(990, 1010, 1830, 1880)
    records = {"010101-a": [centre, limb], "020202-b": [centre]}
    terms = image_terms(records, [centre, box(300, 320, 990, 1010)], DISK, 0.)
    inner, far, edge = (np.searchsorted(RADIAL_BINS, rho, side="right") - 1 for rho in (0, .77, .95))
    assert terms[inner].tolist() == [2, 2, 0, 0]       # the central mask matches both records
    assert terms[far].tolist() == [0, 0, 2, 0]         # the stray mask is a false positive for both
    assert terms[edge].tolist() == [0, 0, 0, 1]        # only the first annotator drew the limb filament
    human = image_terms(records, None, DISK, 0.)
    assert human[inner].tolist() == [2, 2, 0, 0]       # both ordered pairs agree on the central mask
    assert human[edge].tolist() == [0, 0, 1, 1]        # one miss and one false positive at the limb
    assert image_terms({"010101-a": [centre]}, None, DISK, 0.).sum() == 0

    stems = ["20110109104734Ch", "20110114105034Ch"]
    rows = list(radius_rows("m", stems, np.stack([terms, terms]), replicates=20))
    assert [(r["lower"], r["upper"]) for r in rows] == [(0, .2), (.6, .8), (.9, 1.), (0, None)]
    assert rows[0]["pq"] == 1 and rows[1]["pq"] == 0
    assert rows[-1]["pq"] == pytest.approx(2 / (2 + .5 * 2 + .5 * 1))
    assert rows[-1]["pq_lo"] == pytest.approx(rows[-1]["pq"]) == pytest.approx(rows[-1]["pq_hi"])
