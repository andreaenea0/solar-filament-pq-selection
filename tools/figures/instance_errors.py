"""Audit held-out submissions with host PQ, day bootstrap and fold-held-out filters."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from filseg.data import gt_rles_by_image, load_records
from filseg.disk import Disk
from filseg.metric import IOU_T, iou_matrix, score_record
from filseg.rle import counts_to_rle
from tools.cv_map_parity import load_submissions
from tools.figures.filament_physics import RADIAL_BINS, measure_mask, solar_b0

METRICS = ("pq", "sq", "rq", "fp", "fn")
RADIUS = (*RADIAL_BINS, np.inf)
SIZE = (0, 200, 500, 1500, 4000, np.inf)
TILT = (0, 15, 30, 45, 60, 75, np.inf)
# The first recipe is preferred on ties. Infinity removes every mask beyond the radius.
RECIPES = [("unchanged", np.inf, 0)] + [
    (f"r{radius:g}_veto" if np.isinf(area) else f"r{radius:g}_min{area:g}", radius, area)
    for radius in (.8, .85, .9, .95, 1.) for area in (150, 300, 600, 1200, np.inf)
] + [(f"global_min{area}", 0, area) for area in (50, 100, 200, 300, 500, 800)]


def metric_values(terms):
    """Compute pooled PQ, SQ, RQ and raw FP/FN from additive host terms."""
    iou, tp, fp, fn = np.moveaxis(terms, -1, 0)
    denominator = tp + 0.5 * (fp + fn)

    def divide(num, den):
        return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)

    return np.stack((divide(iou, denominator), divide(iou, tp), divide(tp, denominator), fp, fn), axis=-1)


def terms_from_hits(iou):
    """Count all strict host hits and unmatched rows/columns after filtering."""
    hit = iou > IOU_T
    return np.array([iou[hit].sum(), hit.sum(), (~hit.any(axis=0)).sum(), (~hit.any(axis=1)).sum()])


def bin_label(value, edges, last=None):
    """Label half-open bins, with an explicit undefined geometry group."""
    if value is None:
        return "undefined"
    i = np.searchsorted(edges, value, side="right") - 1
    if np.isinf(edges[i + 1]):
        return last or f">={edges[i]:g}"
    return f"{edges[i]:g}–{edges[i + 1]:g}"


def geometry_keys(measurement):
    """Assign each instance to exactly one bucket per geometry dimension."""
    return (
        ("radius", bin_label(measurement.rho, RADIUS)),
        ("size_px", bin_label(measurement.area_px, SIZE)),
        ("tilt_deg", bin_label(measurement.tilt, TILT, "75–90")),
    )


def image_terms(stem, records, predictions, disk, b0):
    """Attribute exact TP/FP/FN events and compute each fixed filter's totals."""
    if np.any(np.triu(iou_matrix(predictions, predictions), 1) > 0):
        raise ValueError(f"{stem} has overlapping predictions.")
    pred_measurements = [measure_mask(rle, disk, b0) for rle in predictions]
    pred_keys = [geometry_keys(m) for m in pred_measurements]
    rho = np.array([m.rho for m in pred_measurements])
    area = np.array([m.area_px for m in pred_measurements])
    keep = [(rho < r) | (area >= a) for _, r, a in RECIPES]
    by_bin = defaultdict(lambda: np.zeros(4))
    recipe_terms = np.zeros((len(RECIPES), 4))
    for record_id, gt in records.items():
        gt_keys = [geometry_keys(measure_mask(rle, disk, b0)) for rle in gt]
        iou = iou_matrix(gt, predictions)
        hit = iou > IOU_T
        exact = score_record(gt, predictions)
        total = np.array([exact.sum_iou, exact.tp, exact.fp, exact.fn])
        by_bin["overall", "all"] += total
        for key in (("site", stem[14]), ("year", stem[:4]), ("annotator_group", record_id[:2]),
                    ("drawings_per_image", str(len(records)))):
            by_bin[key] += total
        # A matched pair is bucketed by its GT mask. Unmatched masks use their own geometry.
        for row, col in zip(*np.nonzero(hit)):
            for key in gt_keys[row]:
                by_bin[key][:2] += (iou[row, col], 1)
        for row in np.flatnonzero(~hit.any(1)):
            for key in gt_keys[row]:
                by_bin[key][3] += 1
        for col in np.flatnonzero(~hit.any(0)):
            for key in pred_keys[col]:
                by_bin[key][2] += 1
        for row, retained in enumerate(keep):
            recipe_terms[row] += terms_from_hits(iou[:, retained])
    np.testing.assert_allclose(recipe_terms[0], by_bin["overall", "all"], rtol=1e-12)
    return by_bin, recipe_terms


def day_bootstrap(stems, terms, n_bootstrap):
    """Draw common observing days for all columns and sum complete days."""
    days = sorted({stem[:8] for stem in stems})
    index = {day: i for i, day in enumerate(days)}
    by_day = np.zeros((len(days), *terms.shape[1:]))
    np.add.at(by_day, [index[stem[:8]] for stem in stems], terms)
    rng = np.random.default_rng(23)
    weights = rng.multinomial(len(days), np.full(len(days), 1 / len(days)), size=n_bootstrap)
    return by_day, np.tensordot(weights, by_day, axes=1)


def stratum_rows(stems, image_bins, n_bootstrap):
    """Pool strata, check the partitions and report point estimates with intervals."""
    keys = sorted(set().union(*(bins.keys() for bins in image_bins)))
    terms = np.array([[bins.get(key, (0, 0, 0, 0)) for key in keys] for bins in image_bins])
    overall = terms[:, keys.index(("overall", "all"))].sum(axis=0)
    for dimension in {d for d, _ in keys}:
        np.testing.assert_allclose(terms[:, [d == dimension for d, _ in keys]].sum(axis=(0, 1)),
                                   overall, rtol=1e-12)
    denominator_all = overall[1] + 0.5 * (overall[2] + overall[3])
    days, draws = day_bootstrap(stems, terms, n_bootstrap)
    bounds = np.percentile(metric_values(draws), [2.5, 97.5], axis=0)
    for i, (dimension, group) in enumerate(keys):
        iou, tp, fp, fn = terms[:, i].sum(axis=0)
        loss = tp - iou + 0.5 * (fp + fn)
        row = dict(dimension=dimension, group=group, days=int(np.count_nonzero(days[:, i, 1:].sum(axis=1))),
                   tp=int(tp), iou_sum=iou, loss_points=loss / denominator_all)
        for j, (key, value) in enumerate(zip(METRICS, metric_values(terms[:, i].sum(axis=0)))):
            row.update({key: value, f"{key}_lo": bounds[0, i, j], f"{key}_hi": bounds[1, i, j]})
        yield row


def nested_filter(stems, folds, totals):
    """Choose each outer fold's filter using only the other four OOF folds."""
    fold_id = np.array([folds[stem] for stem in stems])
    if set(fold_id) != set(range(5)):
        raise ValueError("Expected all five folds numbered 0 through 4.")
    selected = np.zeros_like(totals[:, 0])
    rows = []
    for outer in range(5):
        held_out = fold_id == outer
        inner_pq = metric_values(totals[~held_out].sum(axis=0))[:, 0]
        choice = int(inner_pq.argmax())
        selected[held_out] = totals[held_out, choice]
        base = metric_values(totals[held_out, 0].sum(axis=0))[0]
        chosen = metric_values(selected[held_out].sum(axis=0))[0]
        rows.append(dict(fold=outer, recipe=RECIPES[choice][0], images=int(held_out.sum()),
                         inner_delta=inner_pq[choice] - inner_pq[0], baseline_pq=base,
                         filtered_pq=chosen, delta=chosen - base))
    return selected, rows


def filter_rows(stems, folds, totals, n_bootstrap):
    """Report outer-fold choices, pooled paired uncertainty and fixed-filter scores."""
    selected, rows = nested_filter(stems, folds, totals)
    paired = np.stack((totals[:, 0], selected), axis=1)
    _, draws = day_bootstrap(stems, paired, n_bootstrap)
    base, chosen = metric_values(paired.sum(axis=0))[:, 0]
    pq = metric_values(draws)[..., 0]
    lo, hi = np.percentile(pq[:, 1] - pq[:, 0], [2.5, 97.5])
    rows.append(dict(fold="nested", recipe="outer selections", images=len(stems), baseline_pq=base,
                     filtered_pq=chosen, delta=chosen - base, delta_lo=lo, delta_hi=hi))
    for (name, _, _), pq in zip(RECIPES, metric_values(totals.sum(axis=0))[:, 0]):
        rows.append(dict(fold="fixed descriptive", recipe=name, images=len(stems), baseline_pq=base,
                         filtered_pq=pq, delta=pq - base))
    return rows


def ambiguity_counts(drawings, predictions):
    """Count unmatched masks and how many have support from a different drawing."""
    hits = [iou_matrix(gt, predictions) > IOU_T for gt in drawings]
    pred_hits = [hit.any(axis=0) for hit in hits]
    any_pred_hit = np.logical_or.reduce(pred_hits)
    terms = np.zeros(4, dtype=int)
    for i, gt in enumerate(drawings):
        fp = ~pred_hits[i]
        fn = ~hits[i].any(axis=1)
        other_gt = [rle for j, other in enumerate(drawings) if i != j for rle in other]
        corroborated = (iou_matrix(gt, other_gt) > IOU_T).any(axis=1)
        terms += (fp.sum(), (fp & any_pred_hit).sum(), fn.sum(), (fn & corroborated).sum())
    return terms


def annotation_support_rows(stems, gt, submission, n_bootstrap):
    """Estimate ambiguity shares on multi-drawing images with paired day draws."""
    multi = [stem for stem in stems if len(gt[stem]) >= 2]
    if not multi:
        return []
    terms = np.array([ambiguity_counts(list(gt[stem].values()), [counts_to_rle(s) for s in submission.get(stem, [])])
                      for stem in multi])
    days, draws = day_bootstrap(multi, terms, n_bootstrap)
    totals = terms.sum(axis=0)
    rows = []
    for event, i in (("fp", 0), ("fn", 2)):
        shares = np.divide(draws[:, i + 1], draws[:, i], out=np.zeros(len(draws)), where=draws[:, i] > 0)
        lo, hi = np.percentile(shares, [2.5, 97.5])
        rows.append(dict(event=event, images=len(multi), days=len(days), count=totals[i], supported=totals[i + 1],
                         share=totals[i + 1] / max(totals[i], 1), share_lo=lo, share_hi=hi))
    return rows


def write_table(path, rows):
    """Write a compact aggregate CSV with stable column order."""
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: f"{v:.8g}" if isinstance(v, (float, np.floating)) else v for k, v in row.items()}
                         for row in rows)


def main():
    """Read only official training labels and score frozen held-out predictions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True, help="Official training annotation JSON.")
    parser.add_argument("--pred", type=Path, required=True, help="Held-out submission CSV on training images.")
    parser.add_argument("--disks", type=Path, required=True, help="Training geometry CSV from disk_check.")
    parser.add_argument("--folds", type=Path, required=True, help="Official OOF stem and fold columns.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/report"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap < 2:
        parser.error("Use at least two bootstrap draws.")

    gt = gt_rles_by_image(load_records(args.gt))
    stems = sorted(gt)
    submission = load_submissions([args.pred], gt)[0]
    with args.disks.open(encoding="utf-8", newline="") as f:
        disks = {r["stem"]: Disk(float(r["x"]), float(r["y"]), float(r["radius"]), float(r["rms"]), int(r["rays"]))
                 for r in csv.DictReader(f)}
    with args.folds.open(encoding="utf-8-sig", newline="") as f:
        folds = {r["stem"]: int(r["fold"]) for r in csv.DictReader(f)}
    if not gt or gt.keys() - disks.keys() or gt.keys() - folds.keys():
        parser.error("Every training image needs a disk fit and a fold.")
    b0 = solar_b0(stems)
    image_bins, filter_totals = [], []
    for i, stem in enumerate(stems, 1):
        predictions = [counts_to_rle(s) for s in submission.get(stem, [])]
        bins, totals = image_terms(stem, gt[stem], predictions, disks[stem], b0[stem])
        image_bins.append(bins)
        filter_totals.append(totals)
        if i % 100 == 0 or i == len(stems):
            print(f"Scored {i}/{len(stems)} images", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_table(args.out_dir / "three_verifiers_strata.csv", list(stratum_rows(stems, image_bins, args.bootstrap)))
    write_table(args.out_dir / "three_verifiers_filters.csv", filter_rows(stems, folds, np.array(filter_totals), args.bootstrap))
    support = annotation_support_rows(stems, gt, submission, args.bootstrap)
    if support:
        write_table(args.out_dir / "three_verifiers_annotation_support.csv", support)


if __name__ == "__main__":
    main()
