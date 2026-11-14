"""Exact held-out PQ of named frozen recipes over the selection-penalty grid."""

import argparse
from pathlib import Path

import numpy as np

from .common import build_terms, evaluate, load_inputs, pq, resample, write_rows


def sweep(name, trees, scores, gt, penalties, draws=2000, unit="day"):
    """Reuse exact host contributions and common resamples over every penalty."""
    stems = sorted(gt)
    terms = build_terms(trees, gt)
    for penalty in penalties:
        values, counts = evaluate(trees, scores, terms, stems, penalty)
        lo, hi = np.percentile(pq(resample(stems, values, draws, unit)), [2.5, 97.5])
        yield dict(model=name, **{"lambda": float(penalty)}, pq=float(pq(values.sum(0))),
                   pq_lo=float(lo), pq_hi=float(hi), masks=int(counts.sum()))


def main():
    """Write the current 0.15–0.35 grid with 2,000 complete-day draws by default."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True, help="Trusted official training RLE pickle.")
    parser.add_argument("--model", nargs=3, action="append", required=True,
                        metavar=("NAME", "CANDIDATES", "SCORES"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--penalties", type=float, nargs="+", default=np.round(np.arange(.15, .3501, .01), 2))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--resampling", choices=("day", "image"), default="day")
    args = parser.parse_args()
    if not np.isfinite(args.penalties).all() or args.bootstrap < 2:
        parser.error("Use finite penalties and at least two bootstrap draws.")
    rows = []
    for name, candidates, scores in args.model:
        trees, gains, gt = load_inputs(candidates, scores, args.gt)
        result = list(sweep(name, trees, gains, gt, args.penalties, args.bootstrap, args.resampling))
        rows.extend(result)
        print(name, "best", max(result, key=lambda row: row["pq"]), flush=True)
    write_rows(args.out, rows)


if __name__ == "__main__":
    main()
