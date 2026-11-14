r"""Compare fixed out-of-fold submission recipes under different label densities.

Usage (from the repo root, with trusted training pickles):
    python tools/cv_sparse.py GT.pkl --records rule RULE_RECORDS.pkl \
        --dp blend CANDS.pkl BLEND_OOF.pkl 0.27 \
        --dp own CANDS_OWN.pkl OWN_CV.pkl 0.27

GT is {stem: {record_id: [COCO RLE]}} (filseg.data.gt_rles_by_image).
--records accepts {record_id: filseg.metric.Totals} (model-sync tools/per_record.py).
--dp takes a candidate pickle (tools/build_cands.py) and candidate-aligned OOF
scores {stem: array}, raw or under 'oof_pred' (tools/experiments/cv_select.py). The penalty
is fixed. To evaluate fold-specific tuning, supply the resulting record totals
via --records. Use --stems with the existing stems.txt format for a fold subset.
Repeat either recipe option for each finalist. Use recipes fixed in advance or
chosen independently within outer folds. Ranking here is for submission choice.
The first --records recipe (or first --dp without --records) is the CI reference.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filseg.metric import score_record
from filseg.select import dp_select


SCENARIOS = ("full", "sparse", "dense")


def read_pickle(path):
    """Read a trusted local pickle."""
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def record_weights(counts):
    """Keep image weight fixed while favoring its sparsest or densest records."""
    counts = np.asarray(counts)
    weights = np.ones((len(SCENARIOS), len(counts)))
    for row, count in enumerate((counts.min(), counts.max()), start=1):
        eligible = counts == count
        weights[row] = eligible * (len(counts) / eligible.sum())
    return weights


def dp_record_totals(gt, candidates, predictions, penalty):
    """Apply the existing antichain selector and score each real record."""
    if not np.isfinite(penalty):
        raise ValueError("DP penalty must be finite.")
    scores = predictions.get("oof_pred", predictions)
    totals = {}
    for stem, records in gt.items():
        if stem not in candidates:
            raise ValueError(f"Missing candidates for {stem}.")
        c = candidates[stem]
        values = np.asarray(scores.get(stem, []), dtype=float)
        if values.shape != (len(c["rle"]),) or not np.isfinite(values).all():
            raise ValueError(f"Expected one finite score per candidate for {stem}.")
        chosen = dp_select(c["parent"], c["children"], c["level"], values - penalty)
        pred = [c["rle"][index] for index in chosen]
        totals.update({record_id: score_record(masks, pred) for record_id, masks in records.items()})
    return totals


def pooled_by_day(gt, recipes):
    """Collect weighted PQ numerator and denominator by day, recipe and scenario."""
    by_day = {}
    for stem, records in gt.items():
        day = by_day.setdefault(stem[:8], np.zeros((len(recipes), len(SCENARIOS), 2)))
        weights = record_weights([len(masks) for masks in records.values()])
        for i, (name, totals) in enumerate(recipes.items()):
            for j, record_id in enumerate(records):
                if record_id not in totals:
                    raise ValueError(f"{name} is missing record {record_id}.")
                t = totals[record_id]
                day[i] += weights[:, j, None] * (t.sum_iou, t.tp + 0.5 * (t.fp + t.fn))
    return np.stack(list(by_day.values()))


def pq(pooled):
    """Divide pooled IoU by the pooled PQ denominator, as in Totals.pq."""
    numerator, denominator = pooled[..., 0], pooled[..., 1]
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def report(gt, recipes, bootstrap, seed):
    """Print pooled PQ, paired intervals and the two smallest worst regrets."""
    days = pooled_by_day(gt, recipes)
    scores = pq(days.sum(axis=0))
    informative = sum(len({len(masks) for masks in records.values()}) > 1 for records in gt.values())
    print(f"{len(gt)} images, {len(days)} observing days, {informative} images with unequal record counts")
    print("Sparse and dense use the lowest and highest label counts per image. Ties share the weight.")
    intervals = None
    if bootstrap and len(days) > 1 and len(recipes) > 1:
        rng = np.random.default_rng(seed)
        n = len(days)
        draws = rng.multinomial(n, np.full(n, 1 / n), size=bootstrap)
        sampled = pq(np.tensordot(draws, days, axes=1))
        intervals = np.quantile(sampled - sampled[:, :1], (0.025, 0.975), axis=0)

    names = list(recipes)
    if intervals is not None:
        print(f"95% paired day-bootstrap intervals for PQ differences vs {names[0]}")
    for i, name in enumerate(names):
        for j, scenario in enumerate(SCENARIOS):
            line = f"{name:16s} {scenario:6s} PQ {scores[i, j]:.4f}"
            if intervals is not None and i:
                lo, hi = intervals[:, i, j]
                line += f"  delta {scores[i, j] - scores[0, j]:+.4f} [{lo:+.4f}, {hi:+.4f}]"
            print(line)

    regret = (scores.max(axis=0) - scores).max(axis=1)
    ranked = sorted(range(len(names)), key=lambda i: (regret[i], -scores[i, 0], names[i]))
    print("Ranked by worst PQ gap to the best recipe in each scenario (ties by full PQ):")
    for i in ranked:
        print(f"  {names[i]} {regret[i]:.4f}")
    print("Shortlist:", ", ".join(names[i] for i in ranked[:2]))


def main():
    """Read the existing pickle formats and compare recipes on common images."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gt", type=Path)
    parser.add_argument("--records", action="append", nargs=2, default=[], metavar=("NAME", "PICKLE"))
    parser.add_argument("--dp", action="append", nargs=4, default=[], metavar=("NAME", "CANDS", "SCORES", "PENALTY"))
    parser.add_argument("--stems", type=Path, help="Whitespace-separated subset of image stems.")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Paired day draws, or zero to disable.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    names = [row[0] for row in args.records + args.dp]
    if not names or len(names) != len(set(names)):
        parser.error("Supply at least one recipe and use unique recipe names.")
    if args.bootstrap < 0 or args.seed < 0:
        parser.error("Bootstrap count and seed must be nonnegative.")
    gt = read_pickle(args.gt)
    stems = set(args.stems.read_text().split()) if args.stems else set(gt)
    if not stems or stems - gt.keys():
        parser.error("The image subset must be nonempty and present in the ground truth.")
    gt = {stem: gt[stem] for stem in sorted(stems)}
    record_ids = [record_id for records in gt.values() for record_id in records]
    if any(not records for records in gt.values()) or len(record_ids) != len(set(record_ids)):
        parser.error("Each image must have at least one record and all record IDs must be unique.")

    try:
        recipes = {name: read_pickle(path) for name, path in args.records}
        for name, candidates, predictions, penalty in args.dp:
            recipes[name] = dp_record_totals(gt, read_pickle(candidates), read_pickle(predictions), float(penalty))
        report(gt, recipes, args.bootstrap, args.seed)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
