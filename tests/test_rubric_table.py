"""Check the rubric terms on a hand-built image with two annotators."""

import numpy as np
import pytest

from filseg.metric import encode
from tools.figures.rubric_table import METRICS, TERMS, image_terms, ratio, rubric_rows

FRAME = (2048, 2048)


def box(top, bottom, left, right):
    mask = np.zeros(FRAME, dtype=bool)
    mask[top:bottom, left:right] = True
    return encode(mask)


def scene():
    """First annotator: one exact filament and one long filament; second annotator: nothing."""
    drawings = {"010101-s": [box(100, 120, 100, 200), box(300, 320, 100, 300)], "020202-s": []}
    predictions = [box(100, 120, 100, 200),   # exact match
                   box(300, 320, 100, 190),   # two fragments of the long filament,
                   box(300, 320, 210, 300),   # neither reaches IoU 0.5
                   box(900, 920, 900, 990)]   # isolated false positive
    return drawings, predictions


def test_counts_relations_and_dice():
    terms, pairs = image_terms(*scene())
    t = dict(zip(TERMS, terms))
    assert (t["tp"], t["fp"], t["fn"]) == (1, 3 + 4, 1)
    assert t["iou_sum"] == pytest.approx(1)
    assert (t["gt"], t["gt_hit"], t["gt_many"], t["gt_split"]) == (2, 2, 1, 1)
    assert (t["pred"], t["pred_hit"], t["pred_many"], t["pred_merge"]) == (8, 3, 0, 0)
    assert (t["overlaps"], t["drawn"], t["images"]) == (3, 1, 1)
    assert pairs == [(1.0, pytest.approx(1))]
    assert t["miou_pair"] == pytest.approx(1)
    shared = 2000 + 1800 + 1800
    first = 2 * shared / (2000 + 4000 + 2000 + 1800 + 1800 + 1800)
    assert t["image_dice"] == pytest.approx((first + 0) / 2)
    assert (t["record_dice"], t["records"]) == (pytest.approx(first), 2)
    assert t["iou_record"] == pytest.approx(shared / (6000 + 7400 - shared))
    assert 0 < t["miou_record"] < 1


def test_merged_prediction_and_empty_cases():
    drawings = {"010101-s": [box(100, 120, 100, 190), box(100, 120, 195, 300)]}
    terms, pairs = image_terms(drawings, [box(100, 120, 100, 300)])
    t = dict(zip(TERMS, terms))
    assert (t["pred_many"], t["pred_merge"], t["gt_many"]) == (1, 1, 0)
    assert (t["tp"], pairs) == (1, [(pytest.approx(105 / 200), pytest.approx(pairs[0][1]))])
    empty, _ = image_terms({"010101-s": []}, [])
    assert dict(zip(TERMS, empty))["image_dice"] == 1
    missed, _ = image_terms(drawings, [])
    t = dict(zip(TERMS, missed))
    assert (t["fn"], t["image_dice"], t["drawn"]) == (2, 0, 1)
    assert 0 < t["miou_record"] < .2  # the plus-one smoothing keeps a missed record above zero


def test_metrics_are_ratios_of_terms_and_rows_carry_intervals():
    terms, pairs = image_terms(*scene())
    assert ratio(terms, "pq") == pytest.approx(1 / (1 + .5 * 7 + .5 * 1))
    assert ratio(terms, "precision") == pytest.approx(1 / 8)
    assert ratio(terms, "recall") == pytest.approx(.5)
    assert ratio(terms, "one_to_many") == pytest.approx(.5)
    assert ratio(np.zeros(len(TERMS)), "sq") == 0
    stems = ["20110109104734Ch", "20110114105034Ch"]
    rows = list(rubric_rows("m", stems, np.stack([terms, terms]), [pairs, pairs], replicates=20))
    by_metric = {row["metric"]: row for row in rows}
    assert set(METRICS) <= set(by_metric) and (by_metric["images"]["value"], by_metric["records"]["value"]) == (2, 4)
    assert by_metric["pq"]["lo"] == pytest.approx(by_metric["pq"]["value"]) == pytest.approx(by_metric["pq"]["hi"])
    assert by_metric["median_matched_iou"]["value"] == 1
