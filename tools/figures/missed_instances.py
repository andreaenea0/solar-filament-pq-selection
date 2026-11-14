"""Measure frozen three-verifiers misses and optionally score additions from its own OOF trees."""

import argparse
import csv
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mu
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.spatial import cKDTree

from filseg.data import gt_rles_by_image, load_records
from filseg.metric import IOU_T, Totals, iou_matrix, score_image
from filseg.rle import counts_to_rle
from filseg.select import dp_select
from tools.cv_map_parity import load_submissions
from tools.figures.instance_errors import write_table
from tools.figures.overlays import split_anchors

COHORTS = ("corroborated", "uncorroborated", "single drawing")
RING_PX = 6
CONNECTIVITY = np.ones((3, 3), dtype=bool)


def crop_mask(rle, pad=0):
    """Decode the mask within its bounding box and an optional clipped halo."""
    x, y, w, h = mu.toBbox(rle).astype(int)
    height, width = rle["size"]
    top, left = max(0, y - pad), max(0, x - pad)
    region = np.s_[top:min(y + h + pad, height), left:min(x + w + pad, width)]
    return mu.decode(rle)[region].astype(bool), region


def boundary(mask, region):
    """Return full-image pixel coordinates on the inner mask boundary."""
    edge = mask & ~binary_erosion(mask, structure=CONNECTIVITY)
    return np.argwhere(edge) + [axis.start for axis in region]


def miss_cohorts(drawings, predictions):
    """Return prediction IoUs and per-record cohorts for the exact host FNs."""
    for i, gt in enumerate(drawings):
        iou = iou_matrix(gt, predictions)
        others = [mask for j, record in enumerate(drawings) if i != j for mask in record]
        supported = (iou_matrix(gt, others) > IOU_T).any(axis=1)
        group = np.full(len(gt), COHORTS[2] if len(drawings) == 1 else COHORTS[1], dtype=object)
        group[supported] = COHORTS[0]
        group[(iou > IOU_T).any(axis=1)] = None
        yield iou, group


def measure(rle, image, disk, pred_tree, best_iou):
    """Measure area, centroid radius, local darkness and nearest-prediction distance."""
    mask, region = crop_mask(rle, RING_PX)
    y, x = np.nonzero(mask)
    cx, cy, radius = disk
    rho = np.hypot(x.mean() + region[1].start - cx, y.mean() + region[0].start - cy) / radius
    yy, xx = np.ogrid[region]
    on_disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    ring = binary_dilation(mask, structure=CONNECTIVITY, iterations=RING_PX) & ~mask & on_disk
    inside = mask & on_disk
    contrast = np.nan
    if ring.any() and inside.any():
        background = np.median(image[region][ring])
        contrast = (background - np.median(image[region][inside])) / max(background, 1)
    distance = np.nan
    if pred_tree is not None:
        distance = 0.0 if best_iou > 0 else pred_tree.query(boundary(mask, region))[0].min()
    return len(x), rho, contrast, distance, best_iou


def lower_tree(drawings, predictions, candidate, cohorts):
    """Measure low-level coverage and compute exact host increments for eligible additions."""
    rles = candidate["rle"]
    levels = np.asarray(candidate["level"])
    if len(levels) != len(rles):
        raise ValueError("Candidate levels and masks must align.")
    overlap = iou_matrix(predictions, rles)
    if not (overlap == 1).any(axis=1).all():
        raise ValueError("The frozen prediction masks are missing from this candidate tree.")
    low = np.isin(levels, (0, 1))
    free = low & ~(overlap > 0).any(axis=0)
    eligible = np.zeros(len(rles), dtype=bool)
    numerator, denominator = np.zeros((2, len(rles)))
    coverage = []
    for gt, group in zip(drawings, cohorts):
        iou = iou_matrix(gt, rles)
        hits = iou > IOU_T
        coverage.append(np.stack((hits[:, low].any(1), hits[:, free].any(1)), axis=1))
        eligible |= hits[group == COHORTS[0]].any(axis=0)
        numerator += np.where(hits, iou, 0).sum(axis=0)
        denominator += 0.5 * np.maximum(1, hits.sum(axis=0))
    return coverage, (candidate, free & eligible, numerator, denominator)


def oracle_additions(baseline, trees):
    """Maximize pooled host PQ while keeping frozen masks and adding only eligible masks."""
    base_denominator = baseline.tp + 0.5 * (baseline.fp + baseline.fn)
    pq = baseline.pq
    while True:
        added = {}
        numerator, denominator = baseline.sum_iou, base_denominator
        for stem, (candidate, eligible, gain, cost) in trees.items():
            indices = dp_select(candidate["parent"], candidate["children"], candidate["level"],
                                np.where(eligible, gain - pq * cost, -np.inf))
            if indices:
                added[stem] = [candidate["rle"][i] for i in indices]
                numerator += gain[indices].sum()
                denominator += cost[indices].sum()
        updated = numerator / denominator if denominator else 0.0
        if updated - pq <= 1e-12:
            return updated, added
        pq = updated


def summary_rows(values, splits, coverage):
    """Summarize cohorts without serializing any per-image competition data."""
    for cohort in COHORTS:
        if not values[cohort]:
            continue
        arr = np.asarray(values[cohort])
        row = dict(cohort=cohort, fn=len(arr), split_like=splits[cohort], zero_overlap=int((arr[:, 4] == 0).sum()),
                   iou_03_to_05=int(((arr[:, 4] >= .3) & (arr[:, 4] <= .5)).sum()))
        for name, column in zip(("area_px", "r_over_R", "contrast", "distance_px", "best_iou"), arr.T):
            defined = column[np.isfinite(column)]
            row[f"{name}_n"] = len(defined)
            if len(defined):
                for label, value in zip(("p25", "median", "p75"), np.percentile(defined, [25, 50, 75])):
                    row[f"{name}_{label}"] = value
        if cohort in coverage:
            low, free = np.sum(coverage[cohort], axis=0)
            row.update(low_tree_hit=int(low), low_tree_disjoint_hit=int(free))
        yield row


def main():
    """Run the audit on official training images and frozen held-out predictions."""
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("gt", "pred", "disks", "images", "out-dir"):
        parser.add_argument(f"--{arg}", required=True, type=Path)
    parser.add_argument("--candidates", type=Path, help="Trusted own-model OOF candidate pickle.")
    args = parser.parse_args()
    gt = gt_rles_by_image(load_records(args.gt))
    raw_predictions = load_submissions([args.pred], gt)[0]
    predictions = {stem: [counts_to_rle(s) for s in raw_predictions.get(stem, [])] for stem in gt}
    with args.disks.open(encoding="utf-8-sig", newline="") as f:
        disks = {r["stem"]: tuple(float(r[k]) for k in ("x", "y", "radius")) for r in csv.DictReader(f)}
    images = {p.stem: p for p in args.images.iterdir() if p.suffix.lower() in (".jpeg", ".jpg")}
    if not gt or gt.keys() - disks.keys() or gt.keys() - images.keys():
        parser.error("Every training image needs a JPEG and disk fit.")
    candidates = None
    if args.candidates:
        with args.candidates.open("rb") as f:
            candidates = pickle.load(f)
        if candidates.keys() != gt.keys():
            parser.error("Candidate trees must cover exactly the training stems.")

    values, splits, coverage = defaultdict(list), defaultdict(int), defaultdict(list)
    baseline, trees = Totals(), {}
    for i, stem in enumerate(sorted(gt), 1):
        drawings, pred = list(gt[stem].values()), predictions[stem]
        if np.triu(iou_matrix(pred, pred), 1).any():
            raise ValueError(f"{stem} has overlapping frozen predictions.")
        baseline += score_image(drawings, pred)
        ious, cohorts = zip(*miss_cohorts(drawings, pred))
        if candidates is not None:
            low_hits, trees[stem] = lower_tree(drawings, pred, candidates[stem], cohorts)
        if any(cohort is not None for group in cohorts for cohort in group):
            with Image.open(images[stem]) as jpg:
                image = np.asarray(jpg.convert("L"))
            pred_tree = cKDTree(np.concatenate([boundary(*crop_mask(rle)) for rle in pred])) if pred else None
            for j, (record, iou, group) in enumerate(zip(drawings, ious, cohorts)):
                misses = np.array([cohort is not None for cohort in group])
                split_indices = {k for k, _ in split_anchors(record, pred, misses)}
                for k in np.flatnonzero(misses):
                    cohort = group[k]
                    values[cohort].append(measure(record[k], image, disks[stem], pred_tree, iou[k].max(initial=0)))
                    splits[cohort] += int(k in split_indices)
                    if candidates is not None:
                        coverage[cohort].append(low_hits[j][k])
        if i % 100 == 0 or i == len(gt):
            print(f"Measured {i}/{len(gt)} images", flush=True)
    if sum(map(len, values.values())) != baseline.fn:
        raise ValueError("Cohort counts do not sum to the exact host FNs.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_table(args.out_dir / "missed_instances.csv", list(summary_rows(values, splits, coverage)))
    print(f"Frozen PQ {baseline.pq:.8f}, {baseline.tp} TP / {baseline.fp} FP / {baseline.fn} FN")
    if candidates is not None:
        pq, additions = oracle_additions(baseline, trees)
        exact = Totals()
        for stem, drawings in gt.items():
            masks = predictions[stem] + additions.get(stem, [])
            if np.triu(iou_matrix(masks, masks), 1).any():
                raise ValueError(f"{stem} has overlapping oracle predictions.")
            exact += score_image(list(drawings.values()), masks)
        np.testing.assert_allclose(pq, exact.pq, atol=1e-12, rtol=0)
        write_table(args.out_dir / "three_verifiers_miss_oracle.csv", [
            dict(baseline_pq=baseline.pq, oracle_pq=exact.pq, delta=exact.pq - baseline.pq,
                 added_masks=sum(map(len, additions.values())), tp=exact.tp, fp=exact.fp, fn=exact.fn)])
    else:
        print("Low-level coverage and add-only oracle unavailable without the own-model OOF trees.")


if __name__ == "__main__":
    main()
