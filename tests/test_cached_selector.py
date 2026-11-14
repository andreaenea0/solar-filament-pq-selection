"""Check cache alignment, exact scoring, and held-out selector mechanics."""

import numpy as np
import pytest

from filseg.metric import encode, score_image
from tools.experiments.cached_selector import Terms, aligned_scores, nested_grid
from tools.experiments.cross_tree_union import ConflictGraph, bit_indices, fit_affine, union_masks
from tools.experiments.selector_ablation import single_level_terms
from tools.experiments.selector_shift import residual_offsets
from tools.experiments.style_gain import annotator_groups, factor_targets


def test_exact_pq_terms_match_host_with_multiple_hits():
    """A prediction can hit multiple overlapping GT masks in one drawing."""
    mask = np.zeros((8, 8), dtype=bool)
    mask[1:4, 1:4] = True
    smaller = mask.copy()
    smaller[1, 1:4] = False
    second = np.zeros_like(mask)
    second[6:, 6:] = True
    predictions = [encode(mask), encode(second)]
    records = {"0100": [encode(mask), encode(smaller)], "0101": [encode(smaller)]}
    terms = Terms.build(records, predictions)
    for selected in ([], [0], [1], [0, 1]):
        pred = [predictions[i] for i in selected]
        drawings = list(records.values())
        expected = [score_image(group, pred) for group in (drawings, [drawings[1]] * 2, [drawings[0]] * 2)]
        np.testing.assert_allclose(terms.selected(selected),
                                   [(t.sum_iou, t.tp + .5 * (t.fp + t.fn)) for t in expected])


def test_conflict_solver_matches_brute_force():
    """Compare exact objectives on random small graphs with an exhaustive solution."""
    rng = np.random.default_rng(1)
    for _ in range(12):
        upper = np.triu(rng.random((8, 8)) < .3, 1)
        conflicts = upper | upper.T
        weights = rng.normal(size=8)
        chosen = ConflictGraph(conflicts).select(weights)
        assert not conflicts[np.ix_(chosen, chosen)].any()
        feasible = []
        for bits in range(1 << len(weights)):
            nodes = list(bit_indices(bits))
            if not conflicts[np.ix_(nodes, nodes)].any():
                feasible.append(weights[nodes].sum())
        assert weights[chosen].sum() == pytest.approx(max(feasible))


def test_conflict_solver_large_component():
    """Check the exact integer solver on a path larger than the recursion threshold."""
    conflicts = np.eye(31, k=1, dtype=bool)
    chosen = ConflictGraph(conflicts | conflicts.T).select(np.ones(31))
    assert chosen == list(range(0, 31, 2))


def test_union_deduplicates_coco_counts_across_sources():
    mask = encode(np.eye(4, dtype=bool))
    text = {**mask, "counts": mask["counts"].decode("ascii")}
    masks, mapping = union_masks(({"rle": [mask]}, {"rle": [text, encode(np.ones((4, 4)))]}))
    assert len(masks) == 2
    np.testing.assert_array_equal(mapping[0], [0])
    np.testing.assert_array_equal(mapping[1], [0, 1])


def test_aligned_scores_only_fills_absent_empty_trees():
    candidates = {"empty": {"rle": []}, "present": {"rle": [None]}}
    assert aligned_scores(candidates, {"present": [.25]})["empty"].size == 0
    with pytest.raises(ValueError, match="present"):
        aligned_scores(candidates, {})


def test_nested_penalty_uses_inner_predictions_with_outer_group_excluded():
    groups = np.array([0, 1, 2])

    def evaluate(excluded, group):
        assert group in excluded
        terms = np.ones((1, 1, 2, 3, 2))
        if len(excluded) == 2:
            outer, = excluded - {group}
            terms[:, :, 1 - outer % 2, :, 0] = .2
        else:
            terms[:, :, 1 - group % 2, :, 0] = .8
            terms[:, :, group % 2, :, 0] = .4
        return terms

    result, choices = nested_grid(groups, evaluate, (.2, .3))
    assert choices == {"0": [.2], "1": [.3], "2": [.2]}
    np.testing.assert_array_equal(result[:, 0, 0, 0], [.4, .4, .4])


def test_gain_factorization_and_group_partition():
    np.testing.assert_allclose(factor_targets([0., .4, .7], [0., .5, 1.]), [0., .8, .7])
    with pytest.raises(ValueError, match="targets"):
        factor_targets([.3], [.2])
    with pytest.raises(ValueError, match="multiple annotator groups"):
        annotator_groups({"image": {"0100": [], "0201": []}}, ["image"])


def test_gain_calibrations():
    slope, offset = fit_affine(np.array([.2, .4]), np.array([.1, .4]), np.array([1., 2.]))
    assert (slope, offset) == pytest.approx((1.5, -.2))
    offsets = residual_offsets(np.array([0, 2]), np.array([.3, -.5]), np.array([100., 10.]), 3)
    np.testing.assert_allclose(offsets, [.05, 0., -5 / 210])


def test_single_level_is_chosen_on_other_folds():
    """Each held image receives the level with higher PQ on the other fold."""
    trees = {s: {"level": [0, 1]} for s in ("left", "right")}
    scores = {s: np.array([.7, .7]) for s in trees}
    terms = [Terms(np.zeros((3, 2)), np.tile([[a, 1.], [b, 1.]], (3, 1, 1)))
             for a, b in ((.9, .1), (.2, .8))]
    selected, choices = single_level_terms(terms, trees, list(trees), scores, .25, np.array([0, 1]))
    assert choices == {"0": 1, "1": 0}
    np.testing.assert_allclose(selected[:, 0, 0], [.1, .2])
