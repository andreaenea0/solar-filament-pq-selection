"""Check the thickness edits and the per-annotator accounting."""

import numpy as np
import pytest
from pycocotools import mask as mu

from filseg.metric import encode
from tools.experiments.annotator_style import EDITS, annotator, edit_masks, ratio_row, style_terms


def bar(top, bottom, left=10, right=50, shape=(64, 64)):
    mask = np.zeros(shape, dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def test_dilation_keeps_masks_disjoint_and_first_mask_wins():
    first, second = bar(10, 14), bar(15, 19)
    grown = [mu.decode(rle).astype(bool) for rle in edit_masks([encode(first), encode(second)], "dilate")]
    assert not (grown[0] & grown[1]).any()
    assert grown[0][14, 10:50].all() and not grown[1][14].any()
    assert grown[0].sum() == first.sum() + 2 * 40 + 2 * 4
    assert grown[1][19, 10:50].all() and grown[1][15:19, 9].all()


def test_erosion_shrinks_and_drops_vanished_masks():
    thick, thin = bar(10, 16), bar(30, 32)
    eroded = edit_masks([encode(thick), encode(thin)], "erode")
    assert len(eroded) == 1
    np.testing.assert_array_equal(mu.decode(eroded[0]).astype(bool), bar(11, 15, 11, 49))


def test_no_edit_returns_the_same_masks_and_unknown_edit_fails():
    masks = [encode(bar(10, 14))]
    assert edit_masks(masks, "none") == masks
    assert edit_masks([], "dilate") == []
    with pytest.raises(ValueError):
        edit_masks(masks, "shift")


def test_thick_and_thin_annotators_prefer_opposite_edits():
    stem = "20110109104734Ch"
    host = (2048, 2048)
    prediction = bar(20, 26, shape=host)
    gt = {stem: {f"010101-{stem}": [encode(bar(19, 27, 9, 51, host))],
                 f"020202-{stem}": [encode(bar(21, 25, 11, 49, host))]}}
    submission = {stem: [encode(prediction)["counts"].decode("ascii")]}
    stems, codes, records, terms = style_terms(gt, submission)
    assert stems == [stem] and codes == ["010101", "020202"] and records.tolist() == [1, 1]
    assert annotator(f"010101-{stem}") == "010101"
    sum_iou = terms[0, :, :, 0]
    none, dilate, erode = (EDITS.index(name) for name in ("none", "dilate", "erode"))
    assert sum_iou[0, dilate] > sum_iou[0, none] > sum_iou[0, erode]
    assert sum_iou[1, erode] > sum_iou[1, none] > sum_iou[1, dilate]
    assert terms[0, :, :, 1].tolist() == [[1, 1, 0], [1, 0, 1]]  # one pixel is enough to lose a hit
    row = ratio_row(stems, terms[:, 0], replicates=20)
    assert row["dilate_ratio"] > 1 > row["erode_ratio"]
    assert row["dilate_lo"] == pytest.approx(row["dilate_ratio"]) == pytest.approx(row["dilate_hi"])
