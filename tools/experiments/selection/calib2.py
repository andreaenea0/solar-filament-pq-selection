"""Drawing-weighted gain bins and local calibration, with labelled resampling units."""

import argparse

import numpy as np

from .common import add_inputs, load_inputs, resample, write_rows


def main():
    """Reproduce the historical local-window calculation from explicit frozen inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_inputs(parser)
    parser.add_argument("--penalty", type=float, default=.25)
    parser.add_argument("--targets", type=float, nargs="+", default=(.2316, .2278))
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--resampling", choices=("image", "day"), default="image")
    args = parser.parse_args()
    from sklearn.isotonic import IsotonicRegression

    trees, scores, _ = load_inputs(args.candidates, args.scores, args.gt)
    stems = [s for s in sorted(trees) if len(scores[s])]
    g = np.concatenate([trees[s]["g"] for s in stems])
    h = np.concatenate([scores[s] for s in stems])
    weights = np.concatenate([np.full(len(scores[s]), trees[s]["R"]) for s in stems])
    local = []
    for stem in stems:
        inside = np.abs(scores[stem] - args.penalty) < .03
        local.append(np.array([np.asarray(trees[stem]["g"])[inside].sum(), scores[stem][inside].sum(),
                               inside.sum()]) * trees[stem]["R"])
    local = np.asarray(local)
    total = local.sum(0)
    if total[2] == 0:
        parser.error("No candidate falls inside the requested calibration window.")
    sampled = resample(stems, local, args.bootstrap, args.resampling)
    estimates = np.divide(sampled[:, 0], sampled[:, 2], out=np.full(len(sampled), np.nan), where=sampled[:, 2] > 0)
    lo, hi = np.nanpercentile(estimates, [2.5, 97.5])
    isotonic = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(h, g, sample_weight=weights)
    summary = dict(model=args.name, penalty=args.penalty, window=.03, mean_gain=total[0] / total[2],
                   mean_score=total[1] / total[2], lo=lo, hi=hi,
                   isotonic_gain=float(isotonic.predict([args.penalty])[0]),
                   resampling=args.resampling, draws=args.bootstrap, empty_draws=int(np.isnan(estimates).sum()))
    edges = np.arange(0, .80001, .04)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = (h >= lo) & (h < hi)
        bins.append(dict(final=args.name, lo=round(float(lo), 2), hi=round(float(hi), 2), n=int(inside.sum()),
                         mean_score=float(np.average(h[inside], weights=weights[inside])) if inside.any() else "",
                         mean_gain=float(np.average(g[inside], weights=weights[inside])) if inside.any() else ""))
    grid = np.linspace(.1, .5, 4001)
    calibrated = isotonic.predict(grid)
    targets = []
    for target in args.targets:
        position = int(np.searchsorted(calibrated, target))
        targets.append(dict(model=args.name, target=target, raw_score=float(grid[position]) if position < len(grid) else ""))
    write_rows(args.out, bins)
    write_rows(args.out.with_name(args.out.stem + "_local.csv"), [summary])
    write_rows(args.out.with_name(args.out.stem + "_targets.csv"), targets)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
