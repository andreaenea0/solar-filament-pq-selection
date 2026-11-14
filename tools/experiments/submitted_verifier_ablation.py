"""Measure the selected single verifier against fixed and fold-tuned LGB controls.

Use published per-image host terms to repeat paired uncertainty, or rebuild
from official training labels and the trusted frozen OOF candidate cache.
This is an ablation of the existing final, not a new model or test score.
"""
import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
from filseg.metric import iou_matrix, score_image
from tools.figures.instance_errors import day_bootstrap, metric_values
from tools.timing_replay import read_rows, write_csv

RECIPES = ("submitted", "LGB-only", "LGB-only-tuned")


def score_masks(records, masks):
    """Use the exact host scorer, including multiple strict GT hits."""
    if np.any(np.triu(iou_matrix(masks, masks), 1) > 0):
        raise ValueError("predictions must be pixel-disjoint")
    total = score_image(list(records.values()), masks)
    return np.array([total.sum_iou, total.tp, total.fp, total.fn])


def fold_penalties(folds, grid, penalties):
    """Choose each held-out fold's control penalty using the other folds."""
    folds, grid = np.asarray(folds), np.asarray(grid)
    if grid.shape != (len(folds), len(penalties), 4) or not np.isfinite(grid).all():
        raise ValueError("expected finite image/penalty/host-term arrays")
    if len(set(folds)) < 2:
        raise ValueError("at least two folds are needed")
    choices = {}
    selected = np.empty((len(folds), 4))
    for fold in sorted(set(folds)):
        held = folds == fold
        best = int(np.argmax(metric_values(grid[~held].sum(0))[:, 0]))
        choices[int(fold)] = best
        selected[held] = grid[held, best]
    return selected, choices


def rebuild(cache, gt_path, prediction_path):
    """Remove the verifier on identical candidates and frozen no-metadata OOF gains."""
    from filseg.data import gt_rles_by_image, load_records
    from filseg.rle import counts_to_rle
    from filseg.select import dp_select
    from tools.cv_sparse import read_pickle
    from tools.experiments.cached_selector import PENALTIES, aligned_scores
    from training.prepare import read_folds

    gt = gt_rles_by_image(load_records(gt_path))
    trees = read_pickle(cache / "cands_train_own.pkl")
    score_path = cache / "cands_train_own_cv_nometa.pkl"
    scores = aligned_scores(trees, read_pickle(score_path)["oof_pred"])
    folds = read_folds(cache / "folds.csv")
    stems = sorted(gt)
    if len(stems) != 707 or set(gt) != set(trees) or set(gt) != set(folds):
        raise ValueError("the complete 707-image training cohort must align")
    predictions = {stem: [] for stem in stems}
    for key, value in read_rows(prediction_path, set(stems)).items():
        predictions[key.rpartition("_")[0]].append(counts_to_rle(value))
    grid = np.zeros((len(stems), len(PENALTIES), 4))
    counts = np.zeros(grid.shape[:2], dtype=int)
    baseline = []
    for i, stem in enumerate(stems):
        baseline.append(score_masks(gt[stem], predictions[stem]))
        tree = trees[stem]
        for j, penalty in enumerate(PENALTIES):
            chosen = dp_select(tree["parent"], tree["children"], tree["level"], scores[stem] - penalty)
            masks = [tree["rle"][index] for index in chosen]
            grid[i, j] = score_masks(gt[stem], masks)
            counts[i, j] = len(masks)
    group = np.array([folds[stem] for stem in stems])
    tuned, choices = fold_penalties(group, grid, PENALTIES)
    fixed = PENALTIES.index(.25)
    rows = []
    for i, stem in enumerate(stems):
        choice = choices[folds[stem]]
        terms = (baseline[i], grid[i, fixed], tuned[i])
        sizes = (len(predictions[stem]), counts[i, fixed], counts[i, choice])
        penalties = (.25, .25, PENALTIES[choice])
        for recipe, values, size, penalty in zip(RECIPES, terms, sizes, penalties):
            rows.append(dict(stem=stem, fold=folds[stem], recipe=recipe,
                             sum_iou=values[0], tp=int(values[1]), fp=int(values[2]),
                             fn=int(values[3]), masks=int(size), records=len(gt[stem]),
                             penalty=penalty))
    paths = (gt_path, prediction_path, cache / "cands_train_own.pkl", score_path, cache / "folds.csv")
    inputs = []
    for path in paths:
        with path.open("rb") as stream:
            inputs.append(dict(file=path.name, sha256=hashlib.file_digest(stream, "sha256").hexdigest()))
    return rows, [dict(fold=fold, penalty=PENALTIES[index]) for fold, index in choices.items()], inputs


def summarize(rows, replicates=2000):
    """Pool host counts and pair the same UTC observing-day bootstrap draws."""
    table, groups, record_counts = {}, {}, {}
    for row in rows:
        key = row["stem"], row["recipe"]
        if key in table or row["recipe"] not in RECIPES:
            raise ValueError("duplicate image/recipe or unexpected recipe")
        fold, records = int(row["fold"]), int(row["records"])
        if row["stem"] in groups and (groups[row["stem"]] != fold or record_counts[row["stem"]] != records):
            raise ValueError("fold and drawing count must align across recipes")
        groups[row["stem"]], record_counts[row["stem"]] = fold, records
        table[key] = [float(row[name]) for name in ("sum_iou", "tp", "fp", "fn")]
    stems = sorted(groups)
    if not stems or set(table) != {(stem, recipe) for stem in stems for recipe in RECIPES}:
        raise ValueError("every observation needs all three paired recipes")
    terms = np.array([[table[stem, recipe] for recipe in RECIPES] for stem in stems])
    if not np.isfinite(terms).all() or np.any(terms < 0):
        raise ValueError("host terms must be finite and nonnegative")
    if not np.all(terms[:, :, 1] + terms[:, :, 3] == (terms[:, :1, 1] + terms[:, :1, 3])):
        raise ValueError("ground-truth counts differ between recipes")
    values = metric_values(terms.sum(0))
    _, draws = day_bootstrap(stems, terms, replicates)
    draws = metric_values(draws)
    summary, fold_rows = [], []
    for index, control in enumerate(RECIPES[1:], 1):
        positive = 0
        for fold in sorted(set(groups.values())):
            held = np.array([groups[stem] == fold for stem in stems])
            points = metric_values(terms[held].sum(0))[:, 0]
            delta = float(points[0] - points[index])
            positive += delta > 0
            fold_rows.append(dict(control=control, fold=fold, images=int(held.sum()),
                                  submitted_pq=points[0], control_pq=points[index], delta=delta))
        lo, hi = np.percentile(draws[:, 0, 0] - draws[:, index, 0], [2.5, 97.5])
        summary.append(dict(control=control, images=len(stems), records=sum(record_counts.values()),
                            submitted_pq=values[0, 0], control_pq=values[index, 0],
                            delta=values[0, 0] - values[index, 0], lo=lo, hi=hi,
                            positive_folds=int(positive), replicates=replicates))
    from tools.experiments.bootstrap_sensitivity import paired_interval
    blocks = [dict(control=control, **paired_interval(
        stems, terms[:, [index, 0]], width, draws=replicates))
        for index, control in enumerate(RECIPES[1:], 1) for width in (1, 7, 28)]
    return summary, fold_rows, blocks


def main():
    """Repeat the public paired summary or rebuild its full-pipeline mask terms."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terms", type=Path, help="Published per-image CSV; no private cache needed.")
    parser.add_argument("--cache", type=Path, help="Trusted frozen OOF candidate and gain cache.")
    parser.add_argument("--gt", type=Path, help="Official competition training annotation JSON.")
    parser.add_argument("--predictions", type=Path,
                        default=Path("report/data/oof_submitted.csv"))
    parser.add_argument("--out", type=Path, required=True, help="New evidence directory.")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must be new")
    if args.terms:
        if args.cache or args.gt:
            parser.error("use --terms or the rebuild inputs")
        with args.terms.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        choices, inputs = [], []
    else:
        if not args.cache or not args.gt:
            parser.error("rebuilding requires --cache and --gt")
        rows, choices, inputs = rebuild(args.cache, args.gt, args.predictions)
    summary, folds, blocks = summarize(rows)
    args.out.mkdir(parents=True)
    write_csv(args.out / "per_image.csv", rows)
    write_csv(args.out / "summary.csv", summary)
    write_csv(args.out / "folds.csv", folds)
    write_csv(args.out / "block_sensitivity.csv", blocks)
    if choices:
        write_csv(args.out / "penalty_choices.csv", choices)
        write_csv(args.out / "inputs.csv", inputs)
    for row in summary:
        print(f"{row['control']}: delta {row['delta']:+.8f}, CI [{row['lo']:+.8f}, {row['hi']:+.8f}], "
              f"{row['positive_folds']} positive folds")


if __name__ == "__main__":
    main()
