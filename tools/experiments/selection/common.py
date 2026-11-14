"""Shared trusted inputs and exact host terms for frozen-selector analyses."""

import csv
from pathlib import Path
import pickle

import numpy as np

from filseg.select import dp_select
from tools.experiments.cached_selector import Terms, aligned_scores
from tools.figures.instance_errors import day_bootstrap


def read_pickle(path):
    """Read a trusted pipeline intermediate, never an untrusted upload."""
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def load_inputs(candidates, scores, gt_path):
    """Require common training observations, finite aligned scores and record counts."""
    trees, gt = read_pickle(candidates), read_pickle(gt_path)
    if not gt or trees.keys() != gt.keys():
        raise ValueError("Training trees and annotations must have identical nonempty stem sets.")
    raw = read_pickle(scores)
    checked = aligned_scores(trees, raw)
    # Isotonic regression preserves X's floating precision. A cast changes
    # its near-equal-score handling, so validation must not rewrite dtype.
    gains = {stem: np.asarray(raw[stem]) if stem in raw else checked[stem] for stem in trees}
    for stem, tree in trees.items():
        if tree["R"] != len(gt[stem]) or not tree["R"]:
            raise ValueError(f"{stem}: candidate record count differs from training annotations.")
    return trees, gains, gt


def build_terms(trees, gt, mode="host"):
    """Precompute additive numerator/denominator terms; label the historical one-hit shortcut."""
    if mode not in ("host", "one-hit"):
        raise ValueError("Scoring mode must be host or one-hit.")
    terms = {}
    for stem, tree in trees.items():
        if mode == "host":
            exact = Terms.build(gt[stem], tree["rle"])
            terms[stem] = exact.base[0], exact.contribution[0]
        else:
            count = sum(map(len, gt[stem].values()))
            g = np.asarray(tree["g"], dtype=float)
            if g.shape != (len(tree["rle"]),) or not np.isfinite(g).all():
                raise ValueError(f"{stem}: invalid one-hit candidate targets.")
            terms[stem] = np.array([0., count / 2]), np.column_stack((tree["R"] * g,
                                                np.full(len(g), tree["R"] / 2)))
    return terms


def evaluate(trees, scores, terms, stems, penalty):
    """Run the unchanged antichain and pool exact contributions for its disjoint masks."""
    rows, counts = [], []
    for stem in stems:
        tree = trees[stem]
        chosen = dp_select(tree["parent"], tree["children"], tree["level"], scores[stem] - penalty)
        base, increment = terms[stem]
        rows.append(base + increment[chosen].sum(axis=0))
        counts.append(len(chosen))
    return np.asarray(rows), np.asarray(counts)


def pq(terms):
    """Form PQ after pooling terms; preserve the host's zero-denominator convention."""
    top, bottom = np.moveaxis(np.asarray(terms), -1, 0)
    return np.divide(top, bottom, out=np.zeros_like(top, dtype=float), where=bottom > 0)


def resample(stems, terms, draws, unit):
    """Use common complete days (seed 23), or the historical image draws (seed 0)."""
    if unit == "day":
        return day_bootstrap(stems, terms, draws)[1]
    if unit == "image":
        indices = np.random.default_rng(0).integers(len(stems), size=(draws, len(stems)))
        return np.asarray(terms)[indices].sum(axis=1)
    raise ValueError("Resampling unit must be day or image.")


def write_rows(path, rows):
    """Write a complete analysis table with explicit column names and portable newlines."""
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def add_inputs(parser):
    """Share explicit paths for a single frozen recipe; no machine-specific defaults."""
    parser.add_argument("--name", required=True, help="Recipe name, including its verifier vintage.")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True, help="Trusted official training RLE pickle.")
    parser.add_argument("--out", type=Path, required=True)
