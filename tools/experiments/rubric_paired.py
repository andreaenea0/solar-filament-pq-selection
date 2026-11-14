"""Paired day-bootstrap differences between frozen selections on every rubric metric.

usage: python -m tools.experiments.rubric_paired TRAIN_JSON OUT.csv REF=oof.csv NAME=oof.csv ...
"""

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from filseg.data import gt_rles_by_image, load_records
from tools.cv_map_parity import load_submissions
from tools.figures.instance_errors import day_bootstrap
from tools.figures import rubric_table as rt


def paired_rows(stems, terms, reference):
    """Subtract metrics computed from the same complete-day bootstrap draws."""
    draws, totals = {}, {}
    for name, values in terms.items():
        _, draws[name] = day_bootstrap(stems, values, 2000)
        totals[name] = values.sum(axis=0)
    for name in terms:
        if name == reference:
            continue
        for metric in rt.METRICS:
            difference = rt.ratio(draws[name], metric) - rt.ratio(draws[reference], metric)
            lo, hi = np.percentile(difference, [2.5, 97.5])
            yield dict(model=name, reference=reference, metric=metric,
                       delta=float(rt.ratio(totals[name], metric) - rt.ratio(totals[reference], metric)),
                       lo=float(lo), hi=float(hi))


def main():
    """Score official training OOF CSVs; each worker receives its own explicit inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gt", type=Path, help="Official training annotation JSON.")
    parser.add_argument("out", type=Path)
    parser.add_argument("pred", nargs="+", metavar="NAME=CSV")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    named = dict(item.split("=", 1) for item in args.pred)
    if len(named) < 2 or args.workers < 1:
        parser.error("Use at least two named recipes and a positive worker count.")
    gt = gt_rles_by_image(load_records(args.gt))
    stems = sorted(gt)
    submissions = dict(zip(named, load_submissions([Path(p) for p in named.values()], gt)))
    terms = {}
    with Pool(args.workers, initializer=rt.initialize_worker, initargs=(gt, submissions)) as pool:
        for name in named:
            terms[name] = np.array([t for t, _ in pool.map(rt._score, [(name, s) for s in stems], chunksize=8)])
    result = pd.DataFrame(paired_rows(stems, terms, next(iter(named))))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(result.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
