"""Score frozen held-out submissions on the report rubric: PQ terms, relations, Dice and Multiscale IoU."""

import argparse
import csv
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

from filseg.data import gt_rles_by_image, load_records
from filseg.metric import IOU_T, iou_matrix
from filseg.rle import counts_to_rle
from tools.cv_map_parity import load_submissions
from tools.figures.instance_errors import day_bootstrap, write_table
from tools.figures.multiscale_iou import multiscale_iou, rle_edge_pixels

TERMS = ("iou_sum", "tp", "fp", "fn", "gt", "gt_hit", "gt_many", "pred", "pred_hit", "pred_many", "gt_split",
         "pred_merge", "image_dice", "images", "record_dice", "records", "miou_pair", "miou_record", "iou_record",
         "drawn", "overlap_iou", "overlap_dice", "overlaps")
# Each metric is a ratio of sums of additive terms, so one day bootstrap serves all of them.
METRICS = {
    "pq": (("iou_sum",), ("tp", .5, "fp", .5, "fn")),
    "sq": (("iou_sum",), ("tp",)),
    "rq": (("tp",), ("tp", .5, "fp", .5, "fn")),
    "precision": (("tp",), ("tp", "fp")),
    "recall": (("tp",), ("tp", "fn")),
    "gt_hit_rate": (("gt_hit",), ("gt",)),
    "pred_hit_rate": (("pred_hit",), ("pred",)),
    "one_to_many": (("gt_many",), ("gt",)),
    "many_to_one": (("pred_many",), ("pred",)),
    "gt_split_half": (("gt_split",), ("gt",)),
    "pred_merge_half": (("pred_merge",), ("pred",)),
    "dice_image": (("image_dice",), ("images",)),
    "dice_record": (("record_dice",), ("records",)),
    "miou_matched": (("miou_pair",), ("tp",)),
    "miou_record": (("miou_record",), ("drawn",)),
    "iou_record": (("iou_record",), ("drawn",)),
    "overlap_iou_mean": (("overlap_iou",), ("overlaps",)),
    "overlap_dice_mean": (("overlap_dice",), ("overlaps",)),
}
_SHARED = {}


def image_terms(drawings, predictions):
    """Return additive rubric terms and matched-pair IoU and MIoU for one image.

    `drawings` maps each annotator record to its ground-truth RLEs. Relations follow the host
    notebook (two masks are related when they overlap at all); a hit for PQ needs IoU > 0.5.
    """
    t = dict.fromkeys(TERMS, 0.)
    pairs, dice = [], []
    shape = tuple(predictions[0]["size"]) if predictions else None
    pred_area = mu.area(predictions).astype(float) if predictions else np.zeros(0)
    pred_union = mu.merge(predictions) if predictions else None
    pred_edges = [rle_edge_pixels(rle) for rle in predictions]
    union_edges = rle_edge_pixels(pred_union) if predictions else np.zeros((0, 2), dtype=int)
    for truth in drawings.values():
        iou = iou_matrix(truth, predictions)
        hit, related = iou > IOU_T, iou > 0
        t["iou_sum"] += iou[hit].sum()
        t["tp"] += hit.sum()
        t["fp"] += (~hit.any(axis=0)).sum()
        t["fn"] += (~hit.any(axis=1)).sum()
        t["gt"] += len(truth)
        t["pred"] += len(predictions)
        t["gt_hit"] += related.any(axis=1).sum()
        t["gt_many"] += (related.sum(axis=1) >= 2).sum()
        t["pred_hit"] += related.any(axis=0).sum()
        t["pred_many"] += (related.sum(axis=0) >= 2).sum()
        t["overlap_iou"] += iou[related].sum()
        t["overlap_dice"] += (2 * iou[related] / (1 + iou[related])).sum()
        t["overlaps"] += related.sum()
        truth_area = mu.area(truth).astype(float) if truth else np.zeros(0)
        if truth and predictions:
            shared = iou * (truth_area[:, None] + pred_area[None, :]) / (1 + iou)
            t["gt_split"] += ((shared >= .5 * pred_area[None, :]).sum(axis=1) >= 2).sum()
            t["pred_merge"] += ((shared >= .5 * truth_area[:, None]).sum(axis=0) >= 2).sum()
        truth_union = mu.merge(truth) if truth else None
        both = float(mu.area(mu.merge([truth_union, pred_union], intersect=True))) if truth and predictions else 0.
        total = sum(float(mu.area(union)) for union in (truth_union, pred_union) if union is not None)
        if truth:
            frame = tuple(truth[0]["size"]) if shape is None else shape
            truth_edges = [rle_edge_pixels(rle) for rle in truth]
            for i, j in zip(*np.nonzero(hit)):
                value = multiscale_iou(truth_edges[i], pred_edges[j], frame)
                pairs.append((float(iou[i, j]), value))
                t["miou_pair"] += value
            t["miou_record"] += multiscale_iou(rle_edge_pixels(truth_union), union_edges, frame)
            t["iou_record"] += both / (total - both)
            t["drawn"] += 1
        dice.append(1. if total == 0 else 2 * both / total)
    t["image_dice"], t["images"] = float(np.mean(dice)), 1
    t["record_dice"], t["records"] = float(np.sum(dice)), len(dice)
    return np.array([t[name] for name in TERMS]), pairs


def combine(terms, spec):
    """Evaluate a weighted sum of named terms on the last axis."""
    index = {name: i for i, name in enumerate(TERMS)}
    total, weight = 0., 1.
    for item in spec:
        if isinstance(item, str):
            total = total + weight * terms[..., index[item]]
            weight = 1.
        else:
            weight = item
    return total


def ratio(terms, name):
    """Evaluate one rubric metric from additive terms."""
    top, bottom = (combine(terms, spec) for spec in METRICS[name])
    return np.divide(top, bottom, out=np.zeros_like(np.asarray(top, dtype=float)), where=np.asarray(bottom) > 0)


def rubric_rows(model, stems, terms, pairs, replicates):
    """Report each metric with a paired day-bootstrap interval, then the sample sizes."""
    _, draws = day_bootstrap(stems, terms, replicates)
    totals = terms.sum(axis=0)
    for name in METRICS:
        lo, hi = np.percentile(ratio(draws, name), [2.5, 97.5])
        yield dict(model=model, metric=name, value=float(ratio(totals, name)), lo=lo, hi=hi)
    by_day = defaultdict(list)
    for stem, image_pairs in zip(stems, pairs):
        by_day[stem[:8]].extend(iou for iou, _ in image_pairs)
    days = [np.array(values) for _, values in sorted(by_day.items())]
    rng = np.random.default_rng(23)
    medians = [np.median(np.concatenate([days[i] for i in rng.integers(len(days), size=len(days))]))
               for _ in range(replicates)]
    lo, hi = np.percentile(medians, [2.5, 97.5])
    yield dict(model=model, metric="median_matched_iou", value=float(np.median(np.concatenate(days))), lo=lo, hi=hi)
    index = {name: i for i, name in enumerate(TERMS)}
    for name, term in (("images", "images"), ("records", "records"), ("drawings", "gt"), ("mask_record_pairs", "pred")):
        yield dict(model=model, metric=name, value=int(totals[index[term]]))


def _score(job):
    model, stem = job
    masks = [counts_to_rle(counts) for counts in _SHARED["submissions"][model].get(stem, [])]
    return image_terms(_SHARED["gt"][stem], masks)


def initialize_worker(gt, submissions):
    """Install explicit scoring inputs in forked or spawned worker processes."""
    _SHARED.clear()
    _SHARED.update(gt=gt, submissions=submissions)


def write_details(directory, model, stems, terms, pairs):
    """Write the per-pair and per-image tables that the report figures read."""
    folder = directory / model
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "matched_iou.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("iou", "miou"))
        writer.writerows((f"{iou:.6f}", f"{value:.6f}") for image_pairs in pairs for iou, value in image_pairs)
    with (folder / "dice_per_image.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("image_id", "dice"))
        writer.writerows((stem, f"{row[TERMS.index('image_dice')]:.6f}") for stem, row in zip(stems, terms))


def main():
    """Read the official training labels and score each named held-out submission."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True, help="Official training annotation JSON.")
    parser.add_argument("--pred", nargs="+", required=True, metavar="NAME=CSV",
                        help="Held-out submission CSVs on training images, each with a short model name.")
    parser.add_argument("--out", type=Path, required=True, help="Long-format rubric CSV.")
    parser.add_argument("--detail-dir", type=Path, help="Also write NAME/matched_iou.csv and NAME/dice_per_image.csv.")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    named = dict(item.split("=", 1) for item in args.pred)
    gt = gt_rles_by_image(load_records(args.gt))
    stems = sorted(gt)
    submissions = dict(zip(named, load_submissions([Path(path) for path in named.values()], gt)))
    rows = []
    with Pool(args.workers, initializer=initialize_worker, initargs=(gt, submissions)) as pool:
        for model in named:
            scored = pool.map(_score, [(model, stem) for stem in stems], chunksize=8)
            terms = np.array([t for t, _ in scored])
            pairs = [p for _, p in scored]
            rows.extend(rubric_rows(model, stems, terms, pairs, args.bootstrap))
            if args.detail_dir:
                write_details(args.detail_dir, model, stems, terms, pairs)
            print(f"{model}: PQ {ratio(terms.sum(axis=0), 'pq'):.4f}, "
                  f"MIoU matched {ratio(terms.sum(axis=0), 'miou_matched'):.4f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_table(args.out, rows)


if __name__ == "__main__":
    main()
