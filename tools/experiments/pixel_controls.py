"""Recheck historical fold-0 pixel controls on native maps with the fixed keep rule.

No model is fitted or selected. Intervals condition on these frozen single-seed
maps; they are not five-fold evidence for changing the final submission.
"""
import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np

from filseg.data import gt_rles_by_image, load_records
from filseg.maps import read_map
from filseg.rule import LOW, HIGH, MIN_AREA, MIN_MEAN, components, kept_masks
from tools.experiments.submitted_verifier_ablation import score_masks
from tools.experiments.bootstrap_sensitivity import paired_interval
from tools.figures.instance_errors import metric_values
from tools.timing_replay import write_csv
from training.prepare import read_folds

FIELDS = ("sum_iou", "tp", "fp", "fn")


def fixed_rule(byte_map):
    """Preserve the original fold-0 script's float32 byte-scale arithmetic."""
    return kept_masks(*components(byte_map, LOW * 255), HIGH * 255,
                      MIN_AREA, MIN_MEAN * 255)


def rebuild(gt_path, folds_path, maps, means):
    """Score aligned official fold-0 maps; reject incomplete or foreign cohorts."""
    gt = gt_rles_by_image(load_records(gt_path))
    folds = read_folds(folds_path)
    stems = sorted(stem for stem, fold in folds.items() if fold == 0)
    if len(gt) != 707 or set(gt) != set(folds) or len(stems) != 142:
        raise ValueError("Expected the complete official training split and released fold 0.")
    paths = {name: {p.stem: p for p in directory.glob("*.npz")} for name, directory in maps.items()}
    if not paths or any(set(group) not in (set(stems), set(gt)) for group in paths.values()):
        raise ValueError("Each map directory must contain fold 0 or the complete training cohort.")
    if set(paths) & set(means) or any(not members or not set(members) <= set(paths) for members in means.values()):
        raise ValueError("Mean recipes must have unique names and existing map members.")
    rows, inputs = [], []
    for i, stem in enumerate(stems, 1):
        arrays = {}
        for name, group in paths.items():
            raw = read_map(group[stem])
            if raw.shape != (2048, 2048) or raw.dtype != np.uint8:
                raise ValueError(f"Expected a native uint8 probability map: {name}/{stem}")
            arrays[name] = raw.astype(np.float32)
            inputs.append(dict(recipe=name, file=group[stem].name,
                               sha256=hashlib.sha256(group[stem].read_bytes()).hexdigest()))
        for name, members in means.items():
            arrays[name] = sum(arrays[member] for member in members) / len(members)
        for name, array in arrays.items():
            masks = fixed_rule(array)
            values = score_masks(gt[stem], masks)
            rows.append(dict(recipe=name, stem=stem, fold=0, records=len(gt[stem]), masks=len(masks),
                             **dict(zip(FIELDS, values, strict=True))))
        if i % 25 == 0 or i == len(stems):
            print(f"Scored {i}/{len(stems)} observations, {len(arrays)} recipes.", flush=True)
    for path in (gt_path, folds_path):
        inputs.append(dict(recipe="cohort", file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return rows, inputs


def summarize(rows, control, replicates=2000):
    """Pair complete observing days after pooling the host's sufficient statistics."""
    table, records = {}, {}
    for row in rows:
        key = row["stem"], row["recipe"]
        if key in table or int(row["fold"]) != 0:
            raise ValueError("Expected unique observation/recipe rows on fold 0.")
        count = int(row["records"])
        if key[0] in records and records[key[0]] != count:
            raise ValueError("Paired drawing counts differ.")
        records[key[0]] = count
        table[key] = [float(row[field]) for field in FIELDS]
    stems = sorted(records)
    recipes = sorted({recipe for _, recipe in table})
    if control not in recipes or set(table) != {(stem, recipe) for stem in stems for recipe in recipes}:
        raise ValueError("Every recipe needs the same observations, including the control.")
    terms = np.array([[table[stem, recipe] for recipe in recipes] for stem in stems])
    if not stems or not np.isfinite(terms).all() or np.any(terms < 0):
        raise ValueError("Expected nonempty finite nonnegative host terms.")
    if not np.all(terms[:, :, 1] + terms[:, :, 3] == terms[:, :1, 1] + terms[:, :1, 3]):
        raise ValueError("Paired ground-truth counts differ.")
    points = metric_values(terms.sum(0))[:, 0]
    reference = recipes.index(control)
    return [dict(control=control, recipe=recipe, fold=0, images=len(stems),
                 records=sum(records.values()), control_pq=points[reference], pq=points[i],
                 **paired_interval(stems, terms[:, [reference, i]], 1, replicates))
            for i, recipe in enumerate(recipes)]


def named(values, convert):
    """Read explicit NAME=VALUE recipe arguments without silently overwriting names."""
    result = {}
    for value in values:
        name, separator, payload = value.partition("=")
        if not separator or not name or name in result:
            raise ValueError("Use unique NAME=VALUE arguments.")
        result[name] = convert(payload)
    return result


def main():
    """Rebuild from official training maps or replay the published per-image terms."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, help="Official competition training JSON only.")
    parser.add_argument("--folds", type=Path)
    parser.add_argument("--map", action="append", default=[], metavar="NAME=DIRECTORY")
    parser.add_argument("--mean", action="append", default=[], metavar="NAME=MEMBER,MEMBER")
    parser.add_argument("--terms", type=Path)
    parser.add_argument("--control", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.terms and not (args.gt and args.folds and args.map):
        parser.error("Use --terms or provide --gt, --folds and --map.")
    args.out.mkdir(parents=True, exist_ok=False)
    if args.terms:
        with args.terms.open(encoding="utf-8-sig", newline="") as stream:
            rows, inputs = list(csv.DictReader(stream)), []
    else:
        rows, inputs = rebuild(args.gt, args.folds, named(args.map, Path),
                               named(args.mean, lambda value: value.split(",")))
    summary = summarize(rows, args.control)
    write_csv(args.out / "pixel_control_per_image.csv", rows)
    write_csv(args.out / "pixel_control_summary.csv", summary)
    if inputs:
        write_csv(args.out / "pixel_control_inputs.csv", inputs)
    for row in summary:
        print(row, flush=True)


if __name__ == "__main__":
    main()
