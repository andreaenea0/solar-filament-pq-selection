"""Screen finalist selectors for held-out PQ and label-free mask diversity."""

import argparse
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from filseg.metric import IOU_T, iou_matrix
from tools.cv_sparse import SCENARIOS, pq, read_pickle
from tools.experiments.cached_selector import (
    SOURCES, Terms, aligned_scores, comparisons, load_blend, load_training, select_tree, write_result,
)
from tools.experiments.cross_tree_union import ConflictGraph, fit_affine, merge_gains, union_masks
from tools.experiments.selector_shift import score_test

ALPHAS = (0., .1, .2, .35, .5, .65, .8, 1.)
PENALTIES = (.20, .22, .24, .25, .26, .27, .28, .29, .30, .32, .34)
UNION_PENALTIES = (.26, .27, .28, .29, .30)
TARGET_PQ = .460


@dataclass(frozen=True)
class Recipe:
    source: str
    alpha: float | None
    penalty: float


BASE_A = Recipe("pubown", .35, .27)
BASE_B = Recipe("own", .5, .25)


def gains(alpha, lgb, verifier):
    """Blend the cached image-only scores."""
    return alpha * lgb + (1. - alpha) * verifier


def mask_overlap(counts):
    """Return strict-match Dice over both submissions' mask counts."""
    matched, count_a, count_other = counts
    return 2 * matched / (count_a + count_other) if count_a + count_other else 1.


def choose(recipes, scores, test_counts):
    """Choose the lowest overlap above the CV gate, breaking ties by PQ."""
    eligible = [i for i, score in enumerate(scores) if score >= TARGET_PQ]
    if not eligible:
        return recipes.index(BASE_A)
    return min(eligible, key=lambda i: (mask_overlap(test_counts[i]), -scores[i], i))


def tree_train(gt, trees, scores, recipes):
    """Score fixed blends and penalties against the archived training drawings."""
    rows = []
    for stem, tree in trees.items():
        terms = Terms.build(gt[stem], tree["rle"])
        _, lgb, verifier = scores
        blended = {alpha: gains(alpha, lgb[stem], verifier[stem]) for alpha in ALPHAS}
        rows.append([terms.selected(select_tree(tree, blended[r.alpha], r.penalty)) for r in recipes])
    return np.asarray(rows)


def tree_test(trees, scores, recipes, baseline):
    """Count strict matches to A once per candidate row, then for each selection."""
    counts = np.zeros((len(recipes), 3), dtype=int)
    for stem, tree in trees.items():
        matches = (iou_matrix(baseline[stem], tree["rle"]) > IOU_T).any(axis=0)
        _, lgb, verifier = scores
        blended = {alpha: gains(alpha, lgb[stem], verifier[stem]) for alpha in ALPHAS}
        for i, recipe in enumerate(recipes):
            chosen = select_tree(tree, blended[recipe.alpha], recipe.penalty)
            counts[i] += (matches[chosen].sum(), len(baseline[stem]), len(chosen))
    return counts


def prepare_union(trees, stems):
    """Build exact cross-tree conflicts and keep source-to-union row maps."""
    for stem in stems:
        masks, mapping = union_masks([trees[source][stem] for source in SOURCES])
        conflicts = iou_matrix(masks, masks) > 0
        np.fill_diagonal(conflicts, False)
        yield stem, masks, mapping, ConflictGraph(conflicts)


def union_grids(gt, folds, training, test, oof, test_scores, baseline):
    """Provide fold-excluded training terms and label-free test match counts."""
    rows = {}
    for source in SOURCES:
        trees = training[source]
        sizes = [len(tree["rle"]) for tree in trees.values()]
        rows[source] = (np.repeat([folds[s] for s in trees], sizes),
                        np.concatenate([oof[source][0][s] for s in trees]),
                        np.concatenate([tree["g"] for tree in trees.values()]),
                        np.repeat([len(gt[s]) for s in trees], sizes))

    @lru_cache(maxsize=None)
    def fit(excluded):
        coefficients = []
        for source in SOURCES:
            groups, scores, targets, weights = rows[source]
            train = ~np.isin(groups, list(excluded))
            coefficients.append(fit_affine(scores[train], targets[train], weights[train]))
        return coefficients

    prepared = [(s, Terms.build(gt[s], masks), mapping, graph)
                for s, masks, mapping, graph in prepare_union(training, gt)]
    deployment = [(s, (iou_matrix(baseline[s], masks) > IOU_T).any(axis=0), mapping, graph)
                  for s, masks, mapping, graph in prepare_union(test, baseline)]

    def select(stem, mapping, graph, scores, coeff):
        adjusted = [np.clip(slope * scores[source][0][stem] + offset, 0., 1.)
                    for source, (slope, offset) in zip(SOURCES, coeff)]
        merged = merge_gains(len(graph.neighbors), mapping, adjusted)
        return [graph.select(merged - lam) for lam in UNION_PENALTIES]

    def train_grid(excluded, group):
        coeff = fit(excluded)
        return np.asarray([[terms.selected(chosen) for chosen in select(s, mapping, graph, oof, coeff)]
                           for s, terms, mapping, graph in prepared if folds[s] == group])

    def test_grid(excluded):
        coeff = fit(excluded)
        counts = np.zeros((len(UNION_PENALTIES), 3), dtype=int)
        for s, matches, mapping, graph in deployment:
            for i, chosen in enumerate(select(s, mapping, graph, test_scores, coeff)):
                counts[i] += (matches[chosen].sum(), len(baseline[s]), len(chosen))
        return counts

    return train_grid, test_grid


def pair_value(stems, terms, image_counts=(90, 180)):
    """Estimate max-of-two PQ from shared day resamples of two sizes."""
    days, day_index = np.unique([s[:8] for s in stems], return_inverse=True)
    pooled = np.zeros((len(days), *terms.shape[1:]))
    np.add.at(pooled, day_index, terms)
    rng = np.random.default_rng(2026)
    result = {}
    for n in image_counts:
        count = max(1, round(n * len(days) / len(stems)))
        draws = rng.multinomial(count, np.full(len(days), 1 / len(days)), size=4000)
        scores = pq(np.tensordot(draws, pooled, axes=1))
        value = scores.max(axis=1)
        lift = value - scores[:, 0]
        result[str(n)] = {scenario: dict(expected_max=float(value[:, i].mean()),
                                         advantage=float(lift[:, i].mean()),
                                         hedge_wins=float((scores[:, 1, i] > scores[:, 0, i]).mean()))
                          for i, scenario in enumerate(SCENARIOS)}
    return result


def recipe_summary(recipe, score, counts):
    """Return PQ and geometry metrics for one frozen recipe."""
    return dict(source=recipe.source, alpha=recipe.alpha, penalty=recipe.penalty, pq=float(score),
                strict_dice=mask_overlap(counts), test_counts=np.asarray(counts).tolist())


def main():
    """Screen fixed candidates and evaluate the hedge policy on outer folds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=Path("weights"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--public-oof-verifier", type=Path)
    parser.add_argument("--public-test-verifier", type=Path)
    args = parser.parse_args()
    if bool(args.public_oof_verifier) != bool(args.public_test_verifier):
        parser.error("Supply both current public ensemble verifier archives together.")
    print("Loading candidate trees and OOF scores.", flush=True)
    gt, folds, training = load_training(args.cache)
    gt = dict(sorted(gt.items()))
    training = {source: {s: trees[s] for s in gt} for source, trees in training.items()}
    oof = {source: load_blend(args.cache, source, training[source]) for source in SOURCES}
    test, test_scores = {}, {}
    for source in SOURCES:
        print(f"Scoring cached test trees for {source}.", flush=True)
        test[source], test_scores[source] = score_test(args.cache, args.weights, source)
    if test["pubown"].keys() != test["own"].keys():
        raise ValueError("Test archives must share the same stems.")
    if args.public_oof_verifier:
        for trees, bundle, path in ((training, oof, args.public_oof_verifier),
                                    (test, test_scores, args.public_test_verifier)):
            source = "pubown"
            _, lgb, _ = bundle[source]
            verifier = aligned_scores(trees[source], read_pickle(path))
            bundle[source] = ({s: gains(.35, lgb[s], verifier[s]) for s in trees[source]}, lgb, verifier)
    baseline = {s: [tree["rle"][i] for i in select_tree(tree, test_scores["pubown"][0][s], .27)]
                for s, tree in test["pubown"].items()}
    recipes, fixed_train, fixed_test = [], [], []
    for source in SOURCES:
        family = [Recipe(source, a, lam) for a in ALPHAS for lam in PENALTIES]
        recipes.extend(family)
        fixed_train.append(tree_train(gt, training[source], oof[source], family))
        fixed_test.append(tree_test(test[source], test_scores[source], family, baseline))
        print(f"Scored {len(family)} fixed {source} recipes.", flush=True)
    train = np.concatenate(fixed_train, axis=1)
    test_counts = np.concatenate(fixed_test, axis=0)
    print("Building the cross-tree conflict graphs.", flush=True)
    train_union, test_union = union_grids(gt, folds, training, test, oof, test_scores, baseline)
    recipes.extend(Recipe("union", None, lam) for lam in UNION_PENALTIES)
    stems = list(gt)
    group_ids = np.array([folds[s] for s in stems])
    groups = sorted(set(group_ids))
    held = np.zeros_like(train[:, 0])
    cv_union = np.zeros((len(stems), len(UNION_PENALTIES), *held.shape[1:]))
    folds_chosen = {}
    for outer in groups:
        other = group_ids != outer
        union_inner = sum(train_union(frozenset((outer, inner)), inner).sum(axis=0)
                          for inner in groups if inner != outer)
        inner_pq = pq(np.concatenate((train[other].sum(axis=0), union_inner), axis=0)[:, 0])
        union_held = train_union(frozenset((outer,)), outer)
        cv_union[~other] = union_held
        counts = np.concatenate((test_counts, test_union(frozenset((outer,)))), axis=0)
        chosen = choose(recipes, inner_pq, counts)
        held[~other] = np.concatenate((train[~other], union_held), axis=1)[:, chosen]
        folds_chosen[str(outer)] = recipe_summary(recipes[chosen], inner_pq[chosen], counts[chosen])
        print(f"Outer fold {outer} selected {recipes[chosen]}", flush=True)
    terms = np.concatenate((train, cv_union), axis=1)
    scores = pq(terms.sum(axis=0)[:, 0])
    counts = np.concatenate((test_counts, test_union(frozenset())), axis=0)
    chosen = choose(recipes, scores, counts)
    a, b = terms[:, recipes.index(BASE_A)], terms[:, recipes.index(BASE_B)]
    eligible = [i for i, score in enumerate(scores) if score >= TARGET_PQ]
    frontier = [i for i in eligible if not any(
        scores[j] > scores[i] and mask_overlap(counts[j]) <= mask_overlap(counts[i])
        or scores[j] >= scores[i] and mask_overlap(counts[j]) < mask_overlap(counts[i])
        for j in eligible)]
    result = dict(public_archive="three_verifiers" if args.public_oof_verifier else "previous_ens4",
                  target_pq=TARGET_PQ, test_images=len(baseline),
                  baseline_a=recipe_summary(BASE_A, scores[recipes.index(BASE_A)], counts[recipes.index(BASE_A)]),
                  baseline_b=recipe_summary(BASE_B, scores[recipes.index(BASE_B)], counts[recipes.index(BASE_B)]),
                  fixed_choice=recipe_summary(recipes[chosen], scores[chosen], counts[chosen]),
                  fixed_frontier=[recipe_summary(recipes[i], scores[i], counts[i]) for i in frontier],
                  outer_choices=folds_chosen,
                  cv=comparisons(stems, {"public-ensemble": a, "three-verifiers": b, "nested_hedge": held}, "public-ensemble"),
                  pair_ab=pair_value(stems, np.stack((a, b), axis=1)),
                  pair_nested=pair_value(stems, np.stack((a, held), axis=1)),
                  pair_fixed_exploratory=pair_value(stems, terms[:, [recipes.index(BASE_A), chosen]]),
                  grid=[recipe_summary(r, score, c) for r, score, c in zip(recipes, scores, counts)])
    write_result(args.out, result)


if __name__ == "__main__":
    main()
