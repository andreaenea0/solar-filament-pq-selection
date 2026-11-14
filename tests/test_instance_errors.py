"""Check the exact attribution, clustered bootstrap and held-out filter choice."""

import numpy as np

from filseg.disk import Disk
from filseg.metric import encode
from tools.figures.instance_errors import RECIPES, ambiguity_counts, day_bootstrap, image_terms, nested_filter, terms_from_hits


def test_strata_conserve_strict_hits_and_unmatched_counts():
    shape = (64, 64)
    target = np.zeros(shape, dtype=np.uint8)
    target[24:30, 20:40] = 1
    half = target.copy()
    half[:, 30:] = 0
    bins, rules = image_terms("20120703070814Th", {"01-test": [encode(target)]}, [encode(half)],
                             Disk(31.5, 31.5, 29, 0, 360), 0)
    np.testing.assert_array_equal(rules[0], [0, 0, 1, 1])
    for dimension in {d for d, _ in bins}:
        np.testing.assert_array_equal(sum(v for (d, _), v in bins.items() if d == dimension), rules[0])
    iou = np.array([[.9, .2], [.7, .5]])
    np.testing.assert_allclose(terms_from_hits(iou), [1.6, 2, 1, 0])
    np.testing.assert_array_equal(terms_from_hits(iou[:, 1:]), [0, 0, 1, 2])


def test_day_bootstrap_keeps_two_observations_of_one_day_together():
    days, draws = day_bootstrap(["20200101000000Th", "20200101000001Bh", "20200102000000Th"],
                                np.array([[1, 0], [2, 0], [0, 1]]), 50)
    np.testing.assert_array_equal(days, [[3, 0], [0, 1]])
    assert np.all(draws[:, 0] % 3 == 0)
    np.testing.assert_array_equal(draws[:, 0] / 3 + draws[:, 1], 2)


def test_filter_is_selected_without_using_outer_fold():
    stems = [str(i) for i in range(5)]
    folds = {stem: int(stem) for stem in stems}
    scores = np.tile([1, 1, 5, 0], (5, len(RECIPES), 1)).astype(float)
    scores[0, 1] = [1, 1, 0, 0]
    scores[1:, 1] = [1, 1, 6, 0]
    selected, rows = nested_filter(stems, folds, scores)
    assert rows[0]["recipe"] == "unchanged"
    assert all(row["recipe"] == RECIPES[1][0] and row["delta"] < 0 for row in rows[1:])
    np.testing.assert_array_equal(selected[:, 2], [5, 6, 6, 6, 6])


def test_other_drawings_can_support_unmatched_predictions_and_targets():
    a, b, c = [encode(row.reshape(1, 3)) for row in np.eye(3, dtype=np.uint8)]
    np.testing.assert_array_equal(ambiguity_counts([[a, b], [c, b]], [a]), [1, 1, 3, 2])
