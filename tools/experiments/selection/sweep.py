"""Portable single-recipe sweep with the historical image-bootstrap grid.

Use lambda_sweep_finals for the current day-bootstrap evidence.
"""

import argparse
import numpy as np

from .common import add_inputs, load_inputs, write_rows
from .lambda_sweep_finals import sweep


def main():
    """Keep the old grid explicit without reading a historical machine's paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_inputs(parser)
    parser.add_argument("--penalties", type=float, nargs="+", default=np.round(np.arange(.1, .42, .02), 2))
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--resampling", choices=("image", "day"), default="image")
    args = parser.parse_args()
    trees, scores, gt = load_inputs(args.candidates, args.scores, args.gt)
    rows = list(sweep(args.name, trees, scores, gt, args.penalties, args.bootstrap, args.resampling))
    write_rows(args.out, rows)
    print(max(rows, key=lambda row: row["pq"]), flush=True)


if __name__ == "__main__":
    main()
