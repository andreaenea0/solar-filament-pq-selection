"""Check the host-notebook distributions on hand-built drawings."""

import numpy as np
import pytest

from filseg.metric import encode
from tools.figures.notebook_distributions import HUMAN, SHARED, TERMS, image_scores, tables, terms


def box(top, bottom, left, right):
    mask = np.zeros((2048, 2048), dtype=bool)
    mask[top:bottom, left:right] = True
    return encode(mask)


SHORT, LONG = box(100, 120, 100, 200), box(300, 320, 100, 300)
MASKS = [SHORT,                     # exact match
         box(300, 320, 100, 190),   # two fragments of the long filament
         box(300, 320, 210, 300),
         box(900, 920, 900, 990)]   # overlaps nothing


def test_pairs_and_partner_counts_follow_the_notebook():
    pairs, gt_degree, pred_degree = image_scores({"010101-s": [SHORT, LONG]}, MASKS)
    assert sorted(pairs) == pytest.approx([.45, .45, 1])
    assert (gt_degree.tolist(), pred_degree.tolist()) == ([1, 2], [1, 1, 1, 0])
    t = dict(zip(TERMS, terms(pairs, gt_degree, pred_degree)))
    assert (t["pairs"], t["below"], t["gt_missed"], t["gt_split"]) == (3, 2, 0, 1)
    assert (t["pred_spurious"], t["pred_merged"]) == (1, 0)
    assert t["dice"] == pytest.approx(1 + 2 * 2 * .45 / 1.45)
    merged, _, pred_degree = image_scores({"010101-s": [box(300, 320, 100, 190), box(300, 320, 210, 300)]}, [LONG])
    assert pred_degree.tolist() == [2] and len(merged) == 2


def test_annotator_pairs_are_symmetric_and_tables_cover_every_source(tmp_path):
    two = {"010101-s": [SHORT, LONG], "020202-s": [SHORT]}
    pairs, gt_degree, pred_degree = image_scores(two, None)
    assert pairs.tolist() == [1, 1] and sorted(gt_degree) == [0, 1, 1] and sorted(pred_degree) == [0, 1, 1]
    gt = {"20110109104734Ch": two, "20110114105034Ch": {"010101-s": [SHORT]}}
    counts = [mask["counts"].decode("ascii") for mask in MASKS]
    distributions, summaries = tables(gt, {"m": {"20110109104734Ch": counts}}, replicates=20, detail_dir=tmp_path)
    split = "0.45,0.62069"                                      # one line per pair, both records of the image
    assert (tmp_path / "pairs_m_shared.csv").read_text().split() == ["iou,dice", "1,1", split, split, "1,1"]
    human = (tmp_path / f"m2n_{HUMAN}.csv").read_text().split()
    assert sorted(human[1:]) == ["gt,0", "gt,1", "gt,1", "pred,0", "pred,1", "pred,1"]
    value = {(row["source"], row["metric"]): row["value"] for row in summaries}
    assert {source for source, _ in value} == {"m", "m" + SHARED, HUMAN}
    assert (value["m", "images"], value["m" + SHARED, "images"], value[HUMAN, "images"]) == (2, 1, 1)
    assert value["m", "gt_missed"] == pytest.approx(1 / 4)       # the second image has no mask
    assert value["m" + SHARED, "gt_split"] == pytest.approx(1 / 3)
    assert value[HUMAN, "gt_missed"] == value[HUMAN, "pred_spurious"] == pytest.approx(1 / 3)
    assert value[HUMAN, "mean_iou"] == 1
    for source in ("m", HUMAN):
        for section in ("iou", "dice", "gt_degree", "pred_degree"):
            share = [r["share"] for r in distributions if (r["source"], r["section"]) == (source, section)]
            assert sum(share) == pytest.approx(1)
    last = [r for r in distributions if (r["source"], r["section"]) == ("m", "iou")][-1]
    assert (last["upper"], last["count"]) == (1, 2)             # an IoU of exactly one falls in the last bin
