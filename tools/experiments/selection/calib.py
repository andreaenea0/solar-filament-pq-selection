"""Fold-excluded gain calibration and empirical PQ/2 fixed-point controls on frozen OOF scores."""

import argparse
from importlib.metadata import version
from pathlib import Path

import numpy as np

from training.prepare import read_folds
from .common import add_inputs, build_terms, evaluate, load_inputs, pq, write_rows

PENALTIES = np.round(np.arange(.15, .37, .02), 3)
EDGES = (-1, .05, .1, .15, .2, .23, .26, .3, .35, .4, .5, .6, .7, 2)


def fixed_point(score, initial=.2, max_iterations=30, tolerance=1e-5):
    """Reproduce the empirical iteration; flag cycles instead of assuming convergence.

    With imperfect learned gains this is not the oracle Dinkelbach algorithm.
    Preserve the original stopping penalty when the update is below tolerance.
    """
    penalty = initial
    for iteration in range(1, max_iterations + 1):
        update = float(score(penalty)) / 2
        residual = update - penalty
        if abs(residual) < tolerance:
            return dict(penalty=penalty, iterations=iteration, residual=residual, converged=True)
        penalty = update
    residual = float(score(penalty)) / 2 - penalty
    return dict(penalty=penalty, iterations=max_iterations, residual=residual,
                converged=abs(residual) < tolerance)


def calibration_bins(name, trees, scores):
    """Use drawing-count weights for the historical best-strict-IoU target."""
    stems = [s for s in sorted(trees) if len(scores[s])]
    gains = np.concatenate([trees[s]["g"] for s in stems])
    predictions = np.concatenate([scores[s] for s in stems])
    weights = np.concatenate([np.full(len(scores[s]), trees[s]["R"]) for s in stems])
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        inside = (predictions >= lo) & (predictions < hi)
        yield dict(model=name, lo=lo, hi=hi, n=int(inside.sum()),
                   mean_score=float(np.average(predictions[inside], weights=weights[inside])) if inside.any() else "",
                   mean_gain=float(np.average(gains[inside], weights=weights[inside])) if inside.any() else "")


def compare(name, trees, scores, gt, folds, release_penalty, mode):
    """Tune outside each fold while keeping the upstream OOF fits fixed."""
    from sklearn.isotonic import IsotonicRegression

    stems = sorted(gt)
    terms = build_terms(trees, gt, mode)

    def value(members, predictions, penalty):
        return float(pq(evaluate(trees, predictions, terms, members, penalty)[0].sum(0)))

    raw = fixed_point(lambda penalty: value(stems, scores, penalty))
    summary = [dict(model=name, score_mode=mode, method="fixed_release", penalty=release_penalty,
                    pq=value(stems, scores, release_penalty), converged=""),
               dict(model=name, score_mode=mode, method="raw_fp_all_oof", penalty=raw["penalty"],
                    pq=value(stems, scores, raw["penalty"]), converged=raw["converged"])]
    totals = {method: np.zeros(2) for method in ("tuned", "raw_fp", "iso_fp", "iso_tuned")}
    converged = {method: True for method in totals}
    details = []
    for outer in range(5):
        training = [s for s in stems if folds[s] != outer]
        held = [s for s in stems if folds[s] == outer]
        nonempty = [s for s in training if len(scores[s])]
        calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(
            np.concatenate([scores[s] for s in nonempty]),
            np.concatenate([trees[s]["g"] for s in nonempty]),
            sample_weight=np.concatenate([np.full(len(scores[s]), trees[s]["R"]) for s in nonempty]))
        calibrated = {s: calibrator.predict(scores[s]) if len(scores[s]) else scores[s] for s in stems}
        raw_fp = fixed_point(lambda penalty: value(training, scores, penalty))
        iso_fp = fixed_point(lambda penalty: value(training, calibrated, penalty))
        choices = {
            "tuned": (scores, max(PENALTIES, key=lambda penalty: value(training, scores, penalty)), None),
            "raw_fp": (scores, raw_fp["penalty"], raw_fp),
            "iso_fp": (calibrated, iso_fp["penalty"], iso_fp),
            "iso_tuned": (calibrated, max(PENALTIES, key=lambda penalty: value(training, calibrated, penalty)), None),
        }
        for method, (predictions, penalty, info) in choices.items():
            selected = evaluate(trees, predictions, terms, held, penalty)[0].sum(0)
            totals[method] += selected
            converged[method] &= info is None or info["converged"]
            details.append(dict(model=name, score_mode=mode, fold=outer, method=method,
                                penalty=float(penalty), pq=float(pq(selected)),
                                training_pq=value(training, predictions, penalty),
                                iterations=info["iterations"] if info else "",
                                residual=info["residual"] if info else "",
                                converged=info["converged"] if info else ""))
    summary.extend(dict(model=name, score_mode=mode, method=method, penalty="", pq=float(pq(total)),
                        converged=converged[method] if "fp" in method else "") for method, total in totals.items())
    return summary, details


def main():
    """Write summary, per-fold choices and calibration bins from explicit trusted paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_inputs(parser)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--penalty", type=float, default=.25, help="Released fixed penalty for this recipe.")
    parser.add_argument("--score-mode", choices=("host", "one-hit", "both"), default="both")
    args = parser.parse_args()
    trees, scores, gt = load_inputs(args.candidates, args.scores, args.gt)
    folds = read_folds(args.folds)
    if folds.keys() != gt.keys():
        parser.error("Fold assignments and training annotations must share all stems.")
    summaries, details = [], []
    precision = ",".join(sorted({str(v.dtype) for v in scores.values() if len(v)}))
    for mode in ("host", "one-hit") if args.score_mode == "both" else (args.score_mode,):
        rows, choices = compare(args.name, trees, scores, gt, folds, args.penalty, mode)
        for row in rows + choices:
            row.update(input_dtype=precision, sklearn_version=version("scikit-learn"))
        summaries.extend(rows)
        details.extend(choices)
        for row in rows:
            print(row, flush=True)
    write_rows(args.out, summaries)
    write_rows(args.out.with_name(args.out.stem + "_folds.csv"), details)
    write_rows(args.out.with_name(args.out.stem + "_bins.csv"), calibration_bins(args.name, trees, scores))


if __name__ == "__main__":
    main()
