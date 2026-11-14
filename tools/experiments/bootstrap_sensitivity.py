"""Check paired PQ uncertainty under longer occupied calendar blocks, with fitted predictions fixed."""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from filseg.data import gt_rles_by_image, load_records
from filseg.metric import score_image
from filseg.rle import counts_to_rle
from tools.cv_map_parity import load_submissions
from tools.figures.instance_errors import metric_values


def paired_interval(stems, terms, block_days, draws=2000):
    """Resample common calendar blocks; days are UTC filename dates, used only for analysis."""
    if block_days < 1 or draws < 2:
        raise ValueError("Use positive block length and at least two draws.")
    terms = np.asarray(terms, dtype=float)
    if not stems or terms.shape != (len(stems), 2, 4) or not np.isfinite(terms).all():
        raise ValueError("Use nonempty, finite per-image terms for two paired recipes.")
    epoch = datetime(1970, 1, 1)
    keys = [(datetime.strptime(stem[:8], "%Y%m%d") - epoch).days // block_days for stem in stems]
    unique = sorted(set(keys))
    index = {key: i for i, key in enumerate(unique)}
    grouped = np.zeros((len(unique), *terms.shape[1:]))
    np.add.at(grouped, [index[key] for key in keys], terms)
    rng = np.random.default_rng(23)
    weights = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=draws)
    scores = metric_values(np.tensordot(weights, grouped, axes=1))[..., 0]
    difference = scores[:, 1] - scores[:, 0]
    point = metric_values(terms.sum(0))[:, 0]
    lo, hi = np.percentile(difference, [2.5, 97.5])
    return dict(block_days=block_days, occupied_blocks=len(unique), draws=draws,
                delta=float(point[1] - point[0]), lo=float(lo), hi=float(hi))


def main():
    """Score two frozen training OOF CSVs and report all predeclared block lengths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True, help="Official training annotations only.")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--alternative", type=Path, required=True)
    parser.add_argument("--block-days", type=int, nargs="+", default=(1, 7, 28))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    gt = gt_rles_by_image(load_records(args.gt))
    stems = sorted(gt)
    predictions = load_submissions([args.reference, args.alternative], gt)
    terms = np.zeros((len(stems), 2, 4))
    for row, stem in enumerate(stems):
        for column, prediction in enumerate(predictions):
            total = score_image(list(gt[stem].values()), [counts_to_rle(c) for c in prediction.get(stem, [])])
            terms[row, column] = total.sum_iou, total.tp, total.fp, total.fn
    rows = [paired_interval(stems, terms, width) for width in args.block_days]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
