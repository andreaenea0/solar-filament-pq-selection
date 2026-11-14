"""Compare fixed selector controls and held-out single-level selection."""

import argparse
from pathlib import Path

import numpy as np

from tools.cv_sparse import pq, read_pickle
from tools.experiments.cached_selector import (
    SOURCES, Terms, aligned_scores, comparisons, load_blend, load_training, select_tree, write_result,
)
from tools.experiments.selector_shift import MODELS
from tools.timing_replay import write_csv


def single_level_terms(terms_by_image, trees, stems, scores, penalty, folds):
    """Choose one tree level on other folds, using the same gains and penalty."""
    levels = sorted({int(level) for tree in trees.values() for level in tree["level"]})
    grid = []
    for stem, terms in zip(stems, terms_by_image):
        tree_level = np.asarray(trees[stem]["level"])
        eligible = scores[stem] > penalty
        grid.append([terms.selected(np.flatnonzero((tree_level == level) & eligible)) for level in levels])
    grid = np.asarray(grid)
    selected = np.empty_like(grid[:, 0])
    choices = {}
    for fold in sorted(set(folds)):
        held = folds == fold
        best = pq(grid[~held].sum(axis=0)[:, 0]).argmax()
        selected[held] = grid[held, best]
        choices[str(fold)] = levels[best]
    return selected, choices


def audit(source, gt, folds, trees, scores, current_public_verifier):
    """Score selectors on the same archived predictions and true drawing terms."""
    blend, lgb, verifier = scores
    _, _, alpha, penalty = MODELS[source]
    if source == "pubown" and current_public_verifier:
        verifier = aligned_scores(trees, read_pickle(current_public_verifier))
        blend = {s: alpha * lgb[s] + (1. - alpha) * verifier[s] for s in trees}
    stems = sorted(trees)
    terms = [Terms.build(gt[s], trees[s]["rle"]) for s in stems]

    def evaluate(gains, lam):
        return np.asarray([t.selected(select_tree(trees[s], gains[s], lam)) for s, t in zip(stems, terms)])

    alternate_alpha = .5 if source == "pubown" else .35
    alternate = {s: alternate_alpha * lgb[s] + (1. - alternate_alpha) * verifier[s] for s in stems}
    controls = {
        "release": evaluate(blend, penalty),
        "lgb_only": evaluate(lgb, penalty),
        "verifier_only": evaluate(verifier, penalty),
        "alternate_alpha": evaluate(alternate, penalty),
        "penalty_0": evaluate(blend, 0.),
        "penalty_24": evaluate(blend, .24),
        "penalty_27": evaluate(blend, .27),
        "penalty_30": evaluate(blend, .30),
    }
    controls["single_level"], choices = single_level_terms(
        terms, trees, stems, blend, penalty, np.array([folds[s] for s in stems]))
    return dict(default_alpha=alpha, default_penalty=penalty, alternate_alpha=alternate_alpha,
                single_level_choices=choices, paired=comparisons(stems, controls, "release"))


def summary_rows(result, images):
    """Flatten only full-cohort host results, distinguishing the historical four-verifier ensemble."""
    rows = []
    for source in SOURCES:
        block = result[source]
        recipe = "three-verifiers" if source == "own" else "public-ensemble" if result["public_archive"] == "three_verifiers" else "public-four-verifiers"
        for control, scenarios in block["paired"].items():
            values = scenarios["full"]
            penalty = (float(control.split("_")[1]) / 100 if control.startswith("penalty_")
                       else block["default_penalty"])
            alpha = block["alternate_alpha"] if control == "alternate_alpha" else block["default_alpha"]
            rows.append(dict(recipe=recipe, control=control, images=images,
                             pq=values["pq"], delta=values["delta"], lo=values["ci"][0],
                             hi=values["ci"][1], alpha=alpha, penalty=penalty, draws=2000))
    return rows


def main():
    """Run the ablation for both cached candidate sources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--public-oof-verifier", type=Path)
    parser.add_argument("--csv", type=Path, help="Optional flat host-control table, identifying the historical four-verifier ensemble.")
    args = parser.parse_args()
    gt, folds, trees = load_training(args.cache)
    result = {"public_archive": "three_verifiers" if args.public_oof_verifier else "previous_ens4"}
    for source in SOURCES:
        scores = load_blend(args.cache, source, trees[source])
        result[source] = audit(source, gt, folds, trees[source], scores, args.public_oof_verifier)
        print(f"Scored ablations for {source}.", flush=True)
    write_result(args.out, result)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(args.csv, summary_rows(result, len(gt)))


if __name__ == "__main__":
    main()
