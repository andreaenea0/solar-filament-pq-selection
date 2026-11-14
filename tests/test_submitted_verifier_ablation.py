"""The submitted removal control must preserve pairing and exclude its held-out fold."""
import numpy as np
import pytest
from pycocotools import mask as mu

from tools.experiments.submitted_verifier_ablation import (
    RECIPES, fold_penalties, score_masks, summarize,
)


def encode(mask):
    return mu.encode(np.asfortranarray(mask.astype(np.uint8)))


def test_exact_host_terms_include_two_strict_gt_hits():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 1
    rle = encode(mask)
    np.testing.assert_array_equal(score_masks({"drawing": [rle, rle]}, [rle]), [2, 2, 0, 0])
    with pytest.raises(ValueError, match="disjoint"):
        score_masks({"drawing": [rle]}, [rle, rle])


def test_penalty_choice_excludes_the_held_fold():
    folds = np.repeat([0, 1, 2], 2)
    grid = np.zeros((6, 2, 4))
    grid[:, :, 1] = grid[:, :, 3] = 1
    grid[:, :, 0] = [.2, .8]
    grid[:2, :, 0] = [.9, .1]
    selected, choices = fold_penalties(folds, grid, (.24, .26))
    perturbed = grid.copy()
    perturbed[:2, :, 0] = [.1, .9]
    _, altered = fold_penalties(folds, perturbed, (.24, .26))
    assert choices[0] == altered[0] == 1
    np.testing.assert_array_equal(selected[:2], grid[:2, 1])


def example_rows():
    rows = []
    for fold, stem in enumerate(("20000101000000Ah", "20000108000000Ah", "20000201000000Ah")):
        for recipe, iou in zip(RECIPES, (.9, .7, .75)):
            rows.append(dict(stem=stem, fold=fold, records=1, recipe=recipe,
                             sum_iou=iou, tp=1, fp=0, fn=0))
    return rows


def test_public_terms_are_order_independent_and_pooled():
    rows = example_rows()
    summary, folds, blocks = summarize(rows, replicates=100)
    assert summary[0]["delta"] == pytest.approx(.2)
    assert summary[1]["delta"] == pytest.approx(.15)
    assert [row["positive_folds"] for row in summary] == [3, 3]
    assert len(folds) == len(blocks) == 6
    assert summarize(rows[::-1], replicates=100) == (summary, folds, blocks)


def test_incomplete_or_incompatible_comparisons_fail():
    rows = example_rows()
    with pytest.raises(ValueError, match="three paired"):
        summarize(rows[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        summarize(rows + [rows[0]])
    rows[1] = dict(rows[1], fn=1)
    with pytest.raises(ValueError, match="ground-truth"):
        summarize(rows)
