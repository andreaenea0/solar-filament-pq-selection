"""Compare held-out PQ with annotator-to-annotator PQ across the solar disk."""

import argparse
import csv
from pathlib import Path

import numpy as np

from filseg.data import gt_rles_by_image, load_records
from filseg.disk import Disk
from filseg.metric import IOU_T, iou_matrix
from filseg.rle import counts_to_rle
from tools.cv_map_parity import load_submissions
from tools.figures.filament_physics import RADIAL_BINS, measure_mask, solar_b0
from tools.figures.instance_errors import METRICS, day_bootstrap, metric_values, write_table

HUMAN = "annotator_pairs"


def radial_bin(rles, disk, b0):
    """Return the radial bin of each mask centroid; the last bin collects centroids off the disk."""
    rho = [measure_mask(rle, disk, b0).rho for rle in rles]
    return np.searchsorted(RADIAL_BINS, rho, side="right") - 1


def pair_terms(truth, truth_bin, predictions, pred_bin):
    """Return IoU sum, TP, FP and FN per radial bin for one drawing scored against one annotator record.

    Matches and misses fall in the bin of the annotated mask, false positives in their own.
    """
    terms = np.zeros((len(RADIAL_BINS), 4))
    iou = iou_matrix(truth, predictions)
    hit = iou > IOU_T
    for row, col in zip(*np.nonzero(hit)):
        terms[truth_bin[row], :2] += (iou[row, col], 1)
    np.add.at(terms[:, 2], pred_bin[~hit.any(axis=0)], 1)
    np.add.at(terms[:, 3], truth_bin[~hit.any(axis=1)], 1)
    return terms


def image_terms(records, predictions, disk, b0):
    """Score a submission against every record, or every record against every other one when it is None."""
    truths = [(rles, radial_bin(rles, disk, b0)) for rles in records.values()]
    if predictions is None:
        pairs = [(a, b) for i, a in enumerate(truths) for j, b in enumerate(truths) if i != j]
    else:
        drawn = (predictions, radial_bin(predictions, disk, b0))
        pairs = [(truth, drawn) for truth in truths]
    return sum((pair_terms(*truth, *drawn) for truth, drawn in pairs), np.zeros((len(RADIAL_BINS), 4)))


def radius_rows(source, stems, terms, replicates):
    """Report pooled PQ, SQ and RQ of each radial bin and of the whole disk with day-bootstrap intervals."""
    terms = np.concatenate((terms, terms.sum(axis=1, keepdims=True)), axis=1)
    _, draws = day_bootstrap(stems, terms, replicates)
    bounds = np.percentile(metric_values(draws), [2.5, 97.5], axis=0)
    totals = terms.sum(axis=0)
    edges = [*zip(RADIAL_BINS, RADIAL_BINS[1:]), (RADIAL_BINS[-1], None), (0, None)]
    for i, (lower, upper) in enumerate(edges):
        if not totals[i, 1:].any():
            continue
        row = dict(source=source, lower=lower, upper=upper, tp=int(totals[i, 1]))
        for j, (key, value) in enumerate(zip(METRICS, metric_values(totals[i]))):
            row.update({key: value, f"{key}_lo": bounds[0, i, j], f"{key}_hi": bounds[1, i, j]})
        yield row


def main():
    """Score each named held-out submission and the annotators against each other, by radius."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True, help="Official training annotation JSON.")
    parser.add_argument("--disks", type=Path, required=True, help="CSV from tools.figures.disk_check on training JPEGs.")
    parser.add_argument("--pred", nargs="+", required=True, metavar="NAME=CSV",
                        help="Held-out submission CSVs on training images, each with a short model name.")
    parser.add_argument("--out", type=Path, required=True, help="CSV of PQ by radial bin; a blank upper edge is open.")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    named = dict(item.split("=", 1) for item in args.pred)
    gt = gt_rles_by_image(load_records(args.gt))
    stems = sorted(gt)
    with args.disks.open(encoding="utf-8", newline="") as stream:
        disks = {r["stem"]: Disk(float(r["x"]), float(r["y"]), float(r["radius"]), float(r["rms"]), int(r["rays"]))
                 for r in csv.DictReader(stream)}
    if gt.keys() - disks.keys():
        parser.error(f"Missing disk fits for {len(gt.keys() - disks.keys())} annotated images.")
    b0 = solar_b0(stems)
    sources = dict(zip(named, load_submissions([Path(path) for path in named.values()], gt)))
    rows = []
    for name, submission in {**sources, HUMAN: None}.items():
        terms = np.array([image_terms(gt[stem], None if submission is None else
                                      [counts_to_rle(counts) for counts in submission.get(stem, [])],
                                      disks[stem], b0[stem]) for stem in stems])
        rows.extend(radius_rows(name, stems, terms, args.bootstrap))
        print(f"{name}: PQ {rows[-1]['pq']:.4f} on the whole disk", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_table(args.out, [{k: "" if v is None else v for k, v in row.items()} for row in rows])


if __name__ == "__main__":
    main()
