"""Compare label-free test score shifts and nested OOF offset tuning."""

import argparse
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from filseg.select import EXTRA, FEATS, feature_frame
from tools.cv_sparse import read_pickle
from tools.experiments.cached_selector import (
    PENALTIES, SOURCES, Terms, aligned_scores, comparisons, load_blend, load_training,
    nested_grid, select_tree, write_result,
)

AREA_EDGES = (200, 500, 1500, 4000)
AREA_BINS = len(AREA_EDGES) + 1
MIN_STRATUM = 30
MODELS = {
    "pubown": ("public_ensemble/lgb_pubown_nometa.txt", "ens4", .35, .27),
    "own": ("own_ver3_a05_l025_nometa_lgb.txt", "own3", .5, .25),
}


def strata(tree):
    """Encode threshold level crossed with fixed area bins in pixels."""
    area = [row["area"] for row in tree["feats"]]
    return AREA_BINS * np.asarray(tree["level"], dtype=int) + np.digitize(area, AREA_EDGES)


def score_test(cache, weights, source):
    """Score cached test trees with the no-metadata booster and cached verifier."""
    model_path, verifier_name, alpha, _ = MODELS[source]
    trees = read_pickle(cache / f"cands_test_{source}.pkl")
    verifier = aligned_scores(trees, read_pickle(cache / f"verif_test_{verifier_name}.pkl"))
    model = lgb.Booster(model_file=str(weights / model_path))
    columns = model.feature_name()
    if set(columns) != set(FEATS + EXTRA) or len(columns) != len(FEATS + EXTRA):
        raise ValueError(f"{source}: expected the no-metadata feature schema.")
    features = [feature_frame(tree)[columns] for tree in trees.values()]
    all_scores = model.predict(pd.concat([frame for frame in features if len(frame)], ignore_index=True))
    boundaries = np.cumsum([len(frame) for frame in features])
    lgb_scores = aligned_scores(trees, dict(zip(trees, np.split(all_scores, boundaries[:-1]))))
    blend = {s: alpha * lgb_scores[s] + (1 - alpha) * verifier[s] for s in trees}
    return trees, (blend, lgb_scores, verifier)


def score_rows(trees, scores):
    """Flatten strata and scores in the tree archive's image order."""
    bins = np.concatenate([strata(tree) for tree in trees.values()])
    rows = np.stack([np.concatenate([score[s] for s in trees]) for score in scores], axis=1)
    return bins, rows


def matched_shifts(train, test):
    """Measure within-stratum median shifts and their OOF-weighted average."""
    train_bins, train_scores = train
    test_bins, test_scores = test
    matched = []
    for key in np.intersect1d(train_bins, test_bins):
        a, b = train_scores[train_bins == key], test_scores[test_bins == key]
        if min(len(a), len(b)) >= MIN_STRATUM:
            level, area_bin = divmod(int(key), AREA_BINS)
            matched.append(dict(level=level, area_bin=area_bin, oof_n=len(a), test_n=len(b),
                                oof_median=np.median(a, axis=0).tolist(),
                                test_median=np.median(b, axis=0).tolist()))
    if not matched:
        raise ValueError("No adequately populated matched candidate strata.")
    a = np.array([row["oof_median"] for row in matched])
    b = np.array([row["test_median"] for row in matched])
    weight = [row["oof_n"] for row in matched]
    return dict(columns=["blend", "lgb", "verifier"], area_edges=AREA_EDGES, min_count=MIN_STRATUM,
                oof_n=len(train_scores), test_n=len(test_scores), oof_matched_n=sum(weight),
                test_matched_n=sum(row["test_n"] for row in matched),
                oof_median=np.median(train_scores, axis=0).tolist(),
                test_median=np.median(test_scores, axis=0).tolist(),
                matched_delta=np.average(b - a, weights=weight, axis=0).tolist(), strata=matched)


def residual_offsets(bins, residuals, weights, size):
    """Shrink weighted residuals toward zero by 200 pseudo-drawings per bin."""
    count = np.bincount(bins, weights=weights, minlength=size)
    total = np.bincount(bins, weights=weights * residuals, minlength=size)
    return np.clip(total / (count + 200.), -.05, .05)


def evaluate_offsets(gt, folds, trees, blend, released_penalty):
    """Compare stratum residuals and raw blends with separate inner penalty tuning."""
    stems = sorted(gt)
    groups = np.array([folds[s] for s in stems])
    terms = [Terms.build(gt[s], trees[s]["rle"]) for s in stems]
    bins = {s: strata(trees[s]) for s in stems}
    sizes = [len(bins[s]) for s in stems]
    row_group = np.repeat(groups, sizes)
    flat_bins = np.concatenate([bins[s] for s in stems])
    residual = np.concatenate([trees[s]["g"] - blend[s] for s in stems])
    weights = np.repeat([len(gt[s]) for s in stems], sizes)
    size = int(flat_bins.max()) + 1

    @lru_cache(maxsize=None)
    def fit(excluded):
        train = ~np.isin(row_group, list(excluded))
        return residual_offsets(flat_bins[train], residual[train], weights[train], size)

    def pick(i, score, lam):
        return terms[i].selected(select_tree(trees[stems[i]], score, lam))

    raw_grid = np.stack([[pick(i, blend[s], lam) for lam in PENALTIES] for i, s in enumerate(stems)])

    def evaluate(excluded, group):
        offsets = fit(excluded)
        indices = np.flatnonzero(groups == group)
        corrected = [blend[stems[i]] + offsets[bins[stems[i]]] for i in indices]
        grid = np.array([[pick(i, score, lam) for lam in PENALTIES] for i, score in zip(indices, corrected)])
        return np.stack((raw_grid[indices], grid), axis=1)

    selected, penalties = nested_grid(groups, evaluate)
    recipes = dict(zip(("raw_tuned", "offset_tuned"), np.moveaxis(selected, 1, 0)))
    recipes["released"] = raw_grid[:, PENALTIES.index(released_penalty)]
    return dict(penalties=penalties, vs_released=comparisons(stems, recipes, "released"),
                vs_raw_tuned=comparisons(stems, recipes, "raw_tuned"),
                offsets={str(group): fit(frozenset((group,))).tolist() for group in sorted(set(groups))})


def main():
    """Run the score shift audit for both finalist trees."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=Path("weights"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    gt, folds, training = load_training(args.cache)
    result = {}
    for source in SOURCES:
        print(f"Auditing {source}", flush=True)
        trees = training[source]
        oof = load_blend(args.cache, source, trees)
        test, test_scores = score_test(args.cache, args.weights, source)
        _, _, _, lam = MODELS[source]
        result[source] = dict(shift=matched_shifts(score_rows(trees, oof), score_rows(test, test_scores)),
                              nested=evaluate_offsets(gt, folds, trees, oof[0], lam),
                              cached_test_picks=sum(len(select_tree(tree, test_scores[0][s], lam))
                                                    for s, tree in test.items()))
    write_result(args.out, result)


if __name__ == "__main__":
    main()
