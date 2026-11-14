"""Compare frozen release outputs and stress the held-out three-verifiers score."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from pycocotools import mask as mu
from scipy.optimize import linear_sum_assignment

from filseg.data import gt_rles_by_image, load_records
from filseg.metric import IOU_T, iou_matrix
from filseg.rle import counts_to_rle
from tools.cv_map_parity import load_submissions
from tools.figures.instance_errors import day_bootstrap, metric_values, terms_from_hits
from training.prepare import read_folds


def read_masks(path, stems):
    """Include images with no rows and convert masks to COCO RLE."""
    raw = load_submissions([path], stems)[0]
    return {stem: [counts_to_rle(counts) for counts in raw.get(stem, [])] for stem in stems}


def areas(masks):
    """Return mask areas in CSV order."""
    return np.asarray(mu.area(masks)) if masks else np.empty(0, dtype=int)


def quantiles(values):
    """Return the 10th, 25th, 50th, 75th and 90th percentiles."""
    return np.percentile(values, [10, 25, 50, 75, 90]).tolist() if len(values) else []


def describe(predictions):
    """Summarize count and area without reference to ground truth."""
    counts = [len(rles) for rles in predictions.values()]
    return dict(images=len(counts), masks=sum(counts), zero=counts.count(0), one=counts.count(1),
                mean_count=float(np.mean(counts)), count_q=quantiles(counts),
                area_q=quantiles(np.concatenate([areas(rles) for rles in predictions.values()])))


def portfolio_overlap(a, b):
    """Compare spatial agreement of two disjoint submissions on the same images."""
    totals = Counter()
    hits = []
    for stem, left in a.items():
        right = b[stem]
        iou = iou_matrix(left, right)
        hit = iou > IOU_T
        if (hit.sum(axis=0) > 1).any() or (hit.sum(axis=1) > 1).any():
            raise ValueError(f"{stem}: expected one-to-one strict matches between disjoint archives.")
        paired = iou[hit]
        hits.extend(paired)
        totals.update(dict(matched=len(paired), pixel_identical=int((paired == 1).sum()),
                           a_only=int((~hit.any(1)).sum()), b_only=int((~hit.any(0)).sum()),
                           a_disjoint=int((~(iou > 0).any(1)).sum()), b_disjoint=int((~(iou > 0).any(0)).sum()),
                           same_count=int(len(left) == len(right))))
    totals["images"] = len(a)
    return {**totals, "matched_iou_q": quantiles(hits)}


def mix_weights(source, target, key):
    """Weight each source image to match the target marginal group shares."""
    source_counts, target_counts = (Counter(map(key, stems)) for stems in (source, target))
    if target_counts.keys() - source_counts.keys():
        raise ValueError("Target contains groups absent from the held-out set.")
    ratios = {group: target_counts[group] * len(source) / (count * len(target))
              for group, count in source_counts.items()}
    tv = sum(abs(source_counts[g] / len(source) - target_counts[g] / len(target)) for g in source_counts) / 2
    return np.array([ratios[key(stem)] for stem in source]), tv, source_counts, target_counts


def rate_cut(area_rows, target_masks_per_image):
    """Choose a lower area cut giving the closest smaller count on tuning images."""
    all_areas = np.concatenate(area_rows)
    target = target_masks_per_image * len(area_rows)
    if len(all_areas) <= target:
        return 0
    values, counts = np.unique(all_areas, return_counts=True)
    retained = np.r_[np.cumsum(counts[::-1])[::-1], 0]
    cut = int(np.abs(retained - target).argmin())
    return int(values[cut]) if cut < len(values) else int(values[-1]) + 1


def pq_comparison(stems, base, variant):
    """Compare pooled PQ with a paired observing-day bootstrap interval."""
    paired = np.stack((base, variant), axis=1)
    _, draws = day_bootstrap(stems, paired, 2000)
    base_pq, pq = metric_values(paired.sum(axis=0))[:, 0]
    samples = metric_values(draws)[:, :, 0]
    return dict(pq=pq, delta=pq - base_pq,
                interval=np.percentile(samples[:, 1] - samples[:, 0], [2.5, 97.5]).tolist())


def one_to_one_terms(iou):
    """Score the best one-to-one assignment restricted to strict matches."""
    row, col = linear_sum_assignment(np.where(iou > IOU_T, iou, 0.), maximize=True)
    hits = iou[row, col]
    hits = hits[hits > IOU_T]
    return np.array((hits.sum(), len(hits), iou.shape[1] - len(hits), iou.shape[0] - len(hits)))


def oof_stress(gt, predictions, test_stems, folds, target_rate):
    """Use only other folds to choose a count cut and score fixed shift scenarios."""
    stems = sorted(gt)
    by_area = [areas(predictions[stem]) for stem in stems]
    fold_id = np.array([folds[stem] for stem in stems])
    cuts = [rate_cut([a for a, f in zip(by_area, fold_id) if f != k], target_rate) for k in range(5)]
    scenarios = np.zeros((len(stems), 5, 4))
    kept = multi_hit = 0
    for i, (stem, mask_areas) in enumerate(zip(stems, by_area)):
        drawings = list(gt[stem].values())
        order = np.argsort([len(rles) for rles in drawings], kind="stable")
        retain = mask_areas >= cuts[fold_id[i]]
        kept += int(retain.sum())
        for j, drawing in enumerate(drawings):
            iou = iou_matrix(drawing, predictions[stem])
            terms = terms_from_hits(iou)
            scenarios[i, 0] += terms
            scenarios[i, 1] += terms_from_hits(iou[:, retain])
            if j == order[0]:
                scenarios[i, 2] += terms
            if j == order[-1]:
                scenarios[i, 3] += terms
            matching = one_to_one_terms(iou)
            scenarios[i, 4] += matching
            multi_hit += int(matching[1] != terms[1])

    base, rate, sparse, dense, matching = np.moveaxis(scenarios, 1, 0)
    result = dict(baseline=float(metric_values(base.sum(axis=0))[0]),
                  matched_rate={**pq_comparison(stems, base, rate), "cuts": cuts, "masks": kept},
                  sparse=pq_comparison(stems, base, sparse), dense=pq_comparison(stems, base, dense),
                  one_to_one={**pq_comparison(stems, base, matching), "multi_hit_records": multi_hit})
    for label, key in (("year", lambda s: s[:4]), ("site", lambda s: s[14])):
        weights, tv, train_counts, test_counts = mix_weights(stems, test_stems, key)
        result[label] = {**pq_comparison(stems, base, base * weights[:, None]), "total_variation": tv,
                         "oof_images": train_counts, "test_images": test_counts}
    return result


def main():
    """Audit the archived test and held-out submissions without test labels."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("gt", "folds", "test-images", "a-test", "b-test", "b-oof", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    gt = gt_rles_by_image(load_records(args.gt))
    folds = read_folds(args.folds)
    if folds.keys() != gt.keys():
        parser.error("The official folds must cover exactly the training stems.")
    test = sorted(p.stem for p in args.test_images.iterdir() if p.suffix.lower() in (".jpeg", ".jpg"))
    if not test or len(set(test)) != len(test):
        parser.error("Provide the official JPEGs with unique test stems.")
    a, b, oof = (read_masks(path, stems) for path, stems in
                 ((args.a_test, test), (args.b_test, test), (args.b_oof, sorted(gt))))
    result = dict(a_test=describe(a), b_test=describe(b), b_oof=describe(oof), overlap=portfolio_overlap(a, b))
    result["oof_stress"] = oof_stress(gt, oof, test, folds, result["b_test"]["mean_count"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote aggregate results to {args.out}")


if __name__ == "__main__":
    main()
