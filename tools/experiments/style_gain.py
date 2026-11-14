"""Compare direct and factored expected gain on unseen annotator groups."""

import argparse
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np

from filseg.select import EXTRA, FEATS, feature_frame
from tools.experiments.cached_selector import (
    PENALTIES, Terms, comparisons, load_training, nested_grid, select_tree, write_result,
)

NAMES = ("direct_all", "direct_multi", "factored_multi")
PARAMS = dict(objective="regression", learning_rate=.03, num_leaves=31, min_data_in_leaf=40,
              feature_fraction=.8, bagging_fraction=.8, bagging_freq=1, lambda_l2=1., verbosity=-1,
              num_threads=2, deterministic=True, force_col_wise=True, seed=23)


def annotator_groups(gt, stems):
    """Require the archive's annotator groups to partition entire observations."""
    groups = [{record[:2] for record in gt[s]} for s in stems]
    if any(len(group) != 1 for group in groups):
        raise ValueError("An image belongs to multiple annotator groups.")
    return np.array([next(iter(group)) for group in groups])


def factor_targets(gain, hit_rate):
    """Separate the per-drawing hit frequency from mean IoU conditional on a hit."""
    gain, hit_rate = np.asarray(gain), np.asarray(hit_rate)
    if (gain.shape != hit_rate.shape or not np.isfinite(gain).all() or not np.isfinite(hit_rate).all()
            or np.any((hit_rate < 0) | (hit_rate > 1) | (gain < .5 * hit_rate - 1e-7)
                      | (gain > hit_rate + 1e-7))):
        raise ValueError("Expected compatible expected-gain and hit-rate targets.")
    conditional = np.zeros_like(gain, dtype=np.result_type(gain, np.float32))
    return np.divide(gain, hit_rate, out=conditional, where=hit_rate > 0)


def main():
    """Fit matching direct controls and a two-part model under nested group holdout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    gt, _, sources = load_training(args.cache)
    stems = sorted(gt)
    trees = [sources["pubown"][s] for s in stems]
    groups = annotator_groups(gt, stems)
    terms = [Terms.build(gt[s], tree["rle"]) for s, tree in zip(stems, trees)]
    sizes = np.array([len(tree["rle"]) for tree in trees])
    records = np.repeat([len(gt[s]) for s in stems], sizes)
    row_groups = np.repeat(groups, sizes)
    columns = FEATS + EXTRA
    x = np.concatenate([feature_frame(tree)[columns].to_numpy(dtype=np.float32) for tree in trees])
    gain = np.concatenate([tree["g"] for tree in trees])
    hits = np.concatenate([tree["hr"] for tree in trees])
    cond = factor_targets(gain, hits)

    def train(rows, target, weight):
        """Use the same feature set, regressor and budget for each target."""
        if not rows.any():
            raise ValueError("No eligible candidate rows for a held-out split.")
        return lgb.train(PARAMS, lgb.Dataset(x[rows], label=target[rows], weight=weight[rows],
                                             feature_name=columns), num_boost_round=600)

    @lru_cache(maxsize=None)
    def fit(excluded):
        available = ~np.isin(row_groups, list(excluded))
        multi = available & (records > 1)
        models = (train(available, gain, records), train(multi, gain, records),
                  train(multi, hits, records), train(multi & (hits > 0), cond, records * hits))
        print(f"Fitted models excluding groups {', '.join(sorted(excluded))}", flush=True)
        return models

    def evaluate(excluded, group):
        direct_all, direct_multi, hit_model, conditional = fit(excluded)
        held = x[row_groups == group]
        if len(held):
            scores = [direct_all.predict(held), direct_multi.predict(held),
                      np.clip(hit_model.predict(held), 0., 1.) * np.clip(conditional.predict(held), .5, 1.)]
        else:
            scores = [np.empty(0) for _ in NAMES]
        rows, start = [], 0
        for i in np.flatnonzero(groups == group):
            end = start + sizes[i]
            rows.append([[terms[i].selected(select_tree(trees[i], score[start:end], lam)) for lam in PENALTIES]
                         for score in scores])
            start = end
        return np.array(rows)

    selected, penalties = nested_grid(groups, evaluate)
    recipes = dict(zip(NAMES, np.moveaxis(selected, 1, 0)))
    multi = np.array([len(gt[s]) > 1 for s in stems])
    result = dict(images=len(stems), multi_images=int(multi.sum()), penalties=penalties,
                  vs_direct_all=comparisons(stems, recipes, "direct_all"),
                  vs_direct_multi=comparisons(stems, recipes, "direct_multi"),
                  multi_vs_direct_multi=comparisons(np.array(stems)[multi],
                                                    {k: v[multi] for k, v in recipes.items()}, "direct_multi"))
    write_result(args.out, result)


if __name__ == "__main__":
    main()
