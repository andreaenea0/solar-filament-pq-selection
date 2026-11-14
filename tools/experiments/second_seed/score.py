"""Exact host terms and paired day uncertainty for the original four-fold seed study."""

import argparse
import csv
from pathlib import Path
import pickle

import numpy as np

from filseg.metric import score_image
from filseg.select import dp_select
from tools.figures.instance_errors import day_bootstrap, metric_values
from training.prepare import read_folds

NAMES = ("s0", "s1", "mean2")
PENALTIES = (.21, .23, .25, .27)
COLUMNS = ("sum_iou", "tp", "fp", "fn")


def selection_terms(tree, scores, records, penalty):
    """Score selected masks directly, including multiple strict hits in a drawing."""
    scores = np.asarray(scores)
    if scores.shape != (len(tree["rle"]),) or not np.isfinite(scores).all():
        raise ValueError("Candidate scores must be finite and aligned with masks.")
    chosen = dp_select(tree["parent"], tree["children"], tree["level"], scores - penalty)
    total = score_image(list(records.values()), [tree["rle"][i] for i in chosen])
    return total.sum_iou, total.tp, total.fp, total.fn


def summary_rows(stems, terms):
    """Compare mean-map PQ with the arithmetic mean of the two single-seed PQs."""
    _, draws = day_bootstrap(stems, terms, 2000)
    sampled = metric_values(draws)[..., 0]
    point = metric_values(terms.sum(0))[..., 0]
    for i, penalty in enumerate(PENALTIES):
        delta = sampled[:, 2, i] - (sampled[:, 0, i] + sampled[:, 1, i]) / 2
        lo, hi = np.percentile(delta, [2.5, 97.5])
        yield dict(penalty=penalty, pq_seed0=point[0, i], pq_seed1=point[1, i], pq_mean_of_two=point[2, i],
                   delta_vs_single_avg=point[2, i] - (point[0, i] + point[1, i]) / 2, lo=lo, hi=hi)


def write_rows(path, rows):
    """Preserve full-precision sufficient statistics for an independent CSV replay."""
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_terms(path):
    """Require complete, unique paired rows and consistent assignments to all four folds."""
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    lookup = {(r["stem"], r["recipe"], float(r["penalty"])): r for r in rows}
    stems = sorted({r["stem"] for r in rows})
    expected = {(s, name, penalty) for s in stems for name in NAMES for penalty in PENALTIES}
    if not stems or len(lookup) != len(rows) or set(lookup) != expected:
        raise ValueError("Expected unique complete paired rows for all recipes and penalties.")
    by_stem = {s: {int(r["fold"]) for r in rows if r["stem"] == s} for s in stems}
    if any(len(v) != 1 for v in by_stem.values()) or set().union(*by_stem.values()) != set(range(4)):
        raise ValueError("Each observation must belong to one of the four held-out folds.")
    folds = {s: next(iter(v)) for s, v in by_stem.items()}
    terms = np.array([[[[float(lookup[s, name, penalty][c]) for c in COLUMNS]
                       for penalty in PENALTIES] for name in NAMES] for s in stems])
    if not np.isfinite(terms).all() or (terms < 0).any():
        raise ValueError("Host sufficient statistics must be finite and nonnegative.")
    return stems, terms, folds


def fold_rows(stems, terms, folds):
    """Pool complete host terms within each original held-out fold."""
    for fold in range(4):
        held = np.array([folds[s] == fold for s in stems])
        point = metric_values(terms[held].sum(0))[..., 0]
        for col, name in enumerate(NAMES):
            for i, penalty in enumerate(PENALTIES):
                yield dict(fold=fold, recipe=name, penalty=penalty, images=int(held.sum()), pq=point[col, i])


def score_work(work, cache, out):
    """Read trusted original experiment exports; do not retrain or read test annotations."""
    def read(path):
        with path.open("rb") as stream:
            return pickle.load(stream)
    trees = dict(s0=read(cache / "cands_train_own.pkl"), s1=read(work / "cands_s1.pkl"),
                 mean2=read(work / "cands_mean2.pkl"))
    scores, gt = read(work / "lgb_pred.pkl"), read(cache / "gt_rles_coco.pkl")
    folds = read_folds(cache / "folds.csv")
    stems = sorted(trees["mean2"])
    if set(stems) != {stem for stem, fold in folds.items() if fold < 4}:
        raise ValueError("Expected exactly the four original held-out folds 0 through 3.")
    if any(scores[name].keys() != set(stems) or set(stems) - trees[name].keys() for name in NAMES):
        raise ValueError("Every recipe must cover exactly the common held-out observations.")
    terms = np.empty((len(stems), len(NAMES), len(PENALTIES), 4))
    rows = []
    for row, stem in enumerate(stems):
        for col, name in enumerate(NAMES):
            for i, penalty in enumerate(PENALTIES):
                value = selection_terms(trees[name][stem], scores[name][stem], gt[stem], penalty)
                terms[row, col, i] = value
                rows.append(dict(recipe=name, penalty=penalty, stem=stem, fold=folds[stem], **dict(zip(COLUMNS, value))))
    write_rows(out, rows)
    return stems, terms, folds


def main():
    """Score recovered masks or reproduce published uncertainty directly from the terms CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--work", type=Path, help="Original mean/s1 trees and lgb_pred.pkl.")
    source.add_argument("--terms", type=Path, help="Published full-precision per-image terms CSV; no private inputs needed.")
    parser.add_argument("--cache", type=Path, help="Trusted seed-0 candidates and official training GT.")
    parser.add_argument("--out", type=Path, help="Per-image terms output when --work is used.")
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.terms:
        stems, terms, folds = read_terms(args.terms)
    else:
        if not args.cache or not args.out:
            parser.error("--work also requires --cache and --out.")
        stems, terms, folds = score_work(args.work, args.cache, args.out)
    write_rows(args.summary, summary_rows(stems, terms))
    write_rows(args.summary.with_name(args.summary.stem + "_folds.csv"), fold_rows(stems, terms, folds))
    print(f"Scored {len(stems)} observations across four folds; paired day draws: 2000, seed 23.")


if __name__ == "__main__":
    main()
