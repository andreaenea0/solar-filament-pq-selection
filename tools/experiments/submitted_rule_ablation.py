"""Compare the frozen submitted decision stage with the unchanged own-map keep rule.

This removes the entire learned decision stage on identical held-out maps.
It does not tune the rule or estimate a new deployed model's performance.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from filseg.data import gt_rles_by_image, load_records
from filseg.maps import read_map
from filseg.rle import counts_to_rle
from filseg.rule import keep_rule
from tools.experiments.submitted_verifier_ablation import score_masks
from tools.experiments.bootstrap_sensitivity import paired_interval
from tools.figures.instance_errors import metric_values
from tools.timing_replay import read_rows, write_csv
from training.prepare import read_folds

RECIPES = ("fixed-own-map-rule", "submitted")
FIELDS = ("sum_iou", "tp", "fp", "fn")


def rebuild(maps, gt_path, folds_path, reference):
    """Use the released default rule and exact scorer on all official train images."""
    gt = gt_rles_by_image(load_records(gt_path))
    folds = read_folds(folds_path)
    paths = {p.stem: p for p in maps.glob("*.npz")}
    if len(gt) != 707 or set(gt) != set(paths) or set(gt) != set(folds):
        raise ValueError("Expected identical complete 707-image map/GT/fold cohorts.")
    predictions = {stem: [] for stem in gt}
    for key, value in read_rows(reference, set(gt)).items():
        predictions[key.rpartition("_")[0]].append(counts_to_rle(value))
    rows = []
    for i, stem in enumerate(sorted(gt), 1):
        array = read_map(paths[stem])
        if array.shape != (2048, 2048) or array.dtype != np.uint8:
            raise ValueError(f"Expected a native uint8 OOF map: {stem}")
        masks = (keep_rule(array.astype(np.float32) / 255), predictions[stem])
        for recipe, selected in zip(RECIPES, masks, strict=True):
            values = score_masks(gt[stem], selected)
            rows.append(dict(stem=stem, fold=folds[stem], recipe=recipe,
                             records=len(gt[stem]), masks=len(selected),
                             **dict(zip(FIELDS, values, strict=True))))
        if i % 100 == 0 or i == len(gt):
            print(f"Scored {i}/{len(gt)} paired observations.", flush=True)
    inputs = []
    for path in (gt_path, folds_path, reference, *[paths[s] for s in sorted(paths)]):
        with path.open("rb") as stream:
            inputs.append(dict(file=path.name,
                               sha256=hashlib.file_digest(stream, "sha256").hexdigest()))
    return rows, inputs


def summarize(rows, replicates=2000):
    """Pool host terms, then pair days and folds without refitting either recipe."""
    table, groups, records = {}, {}, {}
    for row in rows:
        key = row["stem"], row["recipe"]
        if key in table or key[1] not in RECIPES:
            raise ValueError("Duplicate observation/recipe or unknown recipe.")
        fold, count = int(row["fold"]), int(row["records"])
        if key[0] in groups and (groups[key[0]] != fold or records[key[0]] != count):
            raise ValueError("Paired folds and drawing counts must agree.")
        groups[key[0]], records[key[0]] = fold, count
        table[key] = [float(row[field]) for field in FIELDS]
    stems = sorted(groups)
    if not stems or set(table) != {(s, r) for s in stems for r in RECIPES}:
        raise ValueError("Every observation must have both recipes.")
    terms = np.array([[table[stem, recipe] for recipe in RECIPES] for stem in stems])
    if not np.isfinite(terms).all() or np.any(terms < 0):
        raise ValueError("Host terms must be finite and nonnegative.")
    if not np.all(terms[:, 0, 1] + terms[:, 0, 3] == terms[:, 1, 1] + terms[:, 1, 3]):
        raise ValueError("Paired ground-truth counts must agree.")
    points = metric_values(terms.sum(0))[:, 0]
    folds = []
    for fold in sorted(set(groups.values())):
        selected = np.array([groups[s] == fold for s in stems])
        q = metric_values(terms[selected].sum(0))[:, 0]
        folds.append(dict(fold=fold, images=int(selected.sum()), rule_pq=q[0],
                          submitted_pq=q[1], delta=q[1]-q[0]))
    summary = [dict(images=len(stems), records=sum(records.values()),
                    rule_pq=points[0], submitted_pq=points[1],
                    positive_folds=sum(row["delta"] > 0 for row in folds),
                    **paired_interval(stems, terms, width, replicates)) for width in (1, 7, 28)]
    return summary, folds


def main():
    """Rebuild from native maps, or repeat statistics from published host terms."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path)
    parser.add_argument("--gt", type=Path, help="Official competition training JSON only.")
    parser.add_argument("--folds", type=Path)
    parser.add_argument("--reference", type=Path,
                        default=Path("report/data/oof_submitted.csv"))
    parser.add_argument("--terms", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    if args.terms:
        import csv
        with args.terms.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        inputs = []
    else:
        if not all((args.maps, args.gt, args.folds)):
            parser.error("Use --terms or provide --maps, --gt and --folds.")
        rows, inputs = rebuild(args.maps, args.gt, args.folds, args.reference)
    summary, folds = summarize(rows)
    write_csv(args.out / "submitted_rule_per_image.csv", rows)
    write_csv(args.out / "submitted_rule_summary.csv", summary)
    write_csv(args.out / "submitted_rule_folds.csv", folds)
    if inputs:
        write_csv(args.out / "submitted_rule_inputs.csv", inputs)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
