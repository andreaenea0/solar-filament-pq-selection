"""Reconstruct A's historical single-verifier ablation or replay its public terms."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from filseg.metric import score_image
from filseg.rle import counts_to_rle
from tools.cv_map_parity import load_submissions
from tools.cv_sparse import pq, read_pickle
from tools.experiments.cached_selector import Terms, aligned_scores, load_training, select_tree
from tools.figures.instance_errors import day_bootstrap


def merge_exports(paths):
    """Merge held-out exports, rejecting repeated images rather than overwriting."""
    scores = {}
    for path in paths:
        part = read_pickle(path)
        if scores.keys() & part.keys():
            raise ValueError("Verifier fold exports contain repeated images.")
        scores.update(part)
    return scores


def summarize(frame):
    """Recompute paired pooled PQ, day intervals and fold points from public terms."""
    tables = {name: group.set_index("stem").sort_index() for name, group in frame.groupby("recipe")}
    if set(tables) != {"public-one-verifier", "public-ensemble"} or not tables["public-one-verifier"].index.equals(tables["public-ensemble"].index):
        raise ValueError("Expected aligned one- and three-verifier public-ensemble per-image terms.")
    if any(table.index.has_duplicates for table in tables.values()) or not tables["public-one-verifier"].fold.equals(tables["public-ensemble"].fold):
        raise ValueError("Expected unique images with matching folds in both recipes.")
    stems = tables["public-one-verifier"].index.tolist()
    values = np.stack([tables[name][["numerator", "denominator"]].to_numpy() for name in ("public-ensemble", "public-one-verifier")], axis=1)
    _, draws = day_bootstrap(stems, values, 2000)
    delta = pq(draws)[:, 1] - pq(draws)[:, 0]
    scores = pq(values.sum(axis=0))
    rows = [dict(cohort="all", a3=scores[0], a1=scores[1], delta=scores[1] - scores[0],
                 lo=np.percentile(delta, 2.5), hi=np.percentile(delta, 97.5))]
    for fold in sorted(tables["public-one-verifier"].fold.unique()):
        selected = tables["public-one-verifier"].fold.to_numpy() == fold
        scores = pq(values[selected].sum(axis=0))
        rows.append(dict(cohort=f"fold{fold}", a3=scores[0], a1=scores[1], delta=scores[1] - scores[0]))
    return pd.DataFrame(rows)


def main():
    """Use trusted training caches and raw fold exports, or only the derived CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terms", type=Path, help="Replay a previously written per-image terms CSV.")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--verifier-folds", type=Path, nargs="+", help="Five disjoint seed-0 OOF score exports.")
    parser.add_argument("--reference", type=Path, default=Path("report/data/oof_public_ensemble.csv"))
    parser.add_argument("--out", type=Path, required=True, help="Summary CSV; companion _per_image.csv is also written.")
    args = parser.parse_args()
    if args.terms:
        frame = pd.read_csv(args.terms)
    else:
        if args.cache is None or not args.verifier_folds:
            parser.error("Supply --terms or both --cache and --verifier-folds.")
        gt, folds, trees = load_training(args.cache)
        candidates = trees["pubown"]
        verifier = aligned_scores(candidates, merge_exports(args.verifier_folds))
        lgb = aligned_scores(candidates, read_pickle(args.cache / "cands_train_pubown_cv_nometa.pkl")["oof_pred"])
        reference = load_submissions([args.reference], gt)[0]
        rows = []
        for stem, tree in sorted(candidates.items()):
            gains = .35 * lgb[stem] + .65 * verifier[stem]
            selected = select_tree(tree, gains, .27)
            a1 = Terms.build(gt[stem], tree["rle"]).selected(selected)[0]
            a3 = score_image(list(gt[stem].values()), [counts_to_rle(s) for s in reference.get(stem, ())])
            for recipe, values in (("public-one-verifier", a1), ("public-ensemble", (a3.sum_iou, a3.tp + .5 * (a3.fp + a3.fn)))):
                rows.append(dict(stem=stem, fold=folds[stem], recipe=recipe,
                                 numerator=values[0], denominator=values[1]))
        frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out.with_name(args.out.stem + "_per_image.csv"), index=False)
    result = summarize(frame)
    result.to_csv(args.out, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
