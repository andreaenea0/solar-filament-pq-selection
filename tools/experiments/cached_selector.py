"""Shared inputs, exact additive PQ terms, and nested selector evaluation."""

import json
from dataclasses import dataclass

import numpy as np

from filseg.metric import IOU_T, iou_matrix
from filseg.select import dp_select
from tools.cv_sparse import SCENARIOS, pq, read_pickle, record_weights
from tools.figures.instance_errors import day_bootstrap
from training.prepare import read_folds

PENALTIES = (.18, .20, .22, .24, .25, .26, .27, .28, .30, .32, .34)
SOURCES = ("pubown", "own")


def aligned_scores(candidates, scores):
    """Fill only absent empty trees and verify finite candidate-aligned scores."""
    if scores.keys() - candidates.keys():
        raise ValueError("Unexpected stems in score archive.")
    aligned = {}
    for stem, tree in candidates.items():
        values = np.asarray(scores.get(stem, []), dtype=float)
        if values.shape != (len(tree["rle"]),) or not np.isfinite(values).all():
            raise ValueError(f"{stem}: scores must match the candidate rows.")
        aligned[stem] = values
    return aligned


def load_training(cache):
    """Read the trusted v2 training cache with exact stem coverage."""
    gt = read_pickle(cache / "gt_rles_coco.pkl")
    folds = read_folds(cache / "folds.csv")
    trees = {source: read_pickle(cache / f"cands_train_{source}.pkl") for source in SOURCES}
    if not gt or folds.keys() != gt.keys() or any(c.keys() != gt.keys() for c in trees.values()):
        raise ValueError("The trees, training records and official folds must share all stems.")
    for source in SOURCES:
        for stem, c in trees[source].items():
            if c["R"] != len(gt[stem]) or np.shape(c["g"]) != (len(c["rle"]),):
                raise ValueError(f"{source}/{stem}: invalid training targets.")
    return gt, folds, trees


def load_blend(cache, source, trees):
    """Read the no-metadata blend and verify it against its components."""
    lgb = aligned_scores(trees, read_pickle(cache / f"cands_train_{source}_cv_nometa.pkl")["oof_pred"])
    verifier_name = "ens4" if source == "pubown" else "own3"
    verifier = aligned_scores(trees, read_pickle(cache / f"verif_oof_{verifier_name}.pkl"))
    blend = aligned_scores(trees, read_pickle(cache / f"blend_oof_{source}_nometa.pkl"))
    alpha = .35 if source == "pubown" else .5
    for stem in trees:
        np.testing.assert_allclose(blend[stem], alpha * lgb[stem] + (1 - alpha) * verifier[stem],
                                   rtol=0, atol=1e-7, err_msg=f"{source}/{stem}: incompatible score archives")
    return blend, lgb, verifier


@dataclass
class Terms:
    """Baseline and additive PQ contributions for disjoint predictions."""

    base: np.ndarray
    contribution: np.ndarray

    @classmethod
    def build(cls, records, masks):
        """Build exact numerator and denominator increments under strict host hits."""
        drawings = list(records.values())
        counts = np.array([len(drawing) for drawing in drawings])
        weights = record_weights(counts)
        base = np.zeros((len(SCENARIOS), 2))
        base[:, 1] = weights @ (0.5 * counts)
        contribution = np.zeros((len(SCENARIOS), len(masks), 2))
        for drawing, weight in zip(drawings, weights.T):
            iou = iou_matrix(drawing, masks)
            hits = iou > IOU_T
            increments = np.stack((np.where(hits, iou, 0).sum(axis=0),
                                   .5 * np.maximum(1, hits.sum(axis=0))), axis=1)
            contribution += weight[:, None, None] * increments
        return cls(base, contribution)

    def selected(self, indices):
        """Pool terms for an already pixel-disjoint selection."""
        return self.base + self.contribution[:, indices].sum(axis=1)


def select_tree(tree, gains, penalty):
    """Run the released antichain on candidate-aligned gains."""
    return dp_select(tree["parent"], tree["children"], tree["level"], gains - penalty)


def nested_grid(groups, evaluate, penalties=PENALTIES):
    """Choose each recipe's penalty using cross-fitted inner group predictions.

    evaluate(excluded, group) returns [image, recipe, penalty, scenario, term]
    for one group. Model fitting must exclude every group in excluded.
    """
    groups = np.asarray(groups)
    unique = sorted(set(groups))
    if len(unique) < 3:
        raise ValueError("Nested tuning requires at least three groups.")
    result, choices = None, {}
    for outer in unique:
        inner_terms = sum(evaluate(frozenset((outer, inner)), inner).sum(axis=0)
                          for inner in unique if inner != outer)
        selected = pq(inner_terms[:, :, 0, :]).argmax(axis=1)
        held = evaluate(frozenset((outer,)), outer)
        if result is None:
            result = np.empty((len(groups), held.shape[1], *held.shape[3:]))
        result[groups == outer] = held[:, np.arange(len(selected)), selected]
        choices[str(outer)] = [penalties[i] for i in selected]
        print(f"Outer group {outer}: penalties {choices[str(outer)]}", flush=True)
    return result, choices


def comparisons(stems, recipes, baseline):
    """Summarize pooled PQ and paired day-bootstrap deltas in all scenarios."""
    base = recipes[baseline]
    result = {}
    for name, terms in recipes.items():
        paired = np.stack((base, terms), axis=1)
        _, draws = day_bootstrap(stems, paired, 2000)
        scores = pq(paired.sum(axis=0))
        sampled = pq(draws)
        intervals = np.percentile(sampled[:, 1] - sampled[:, 0], [2.5, 97.5], axis=0).T
        result[name] = {scenario: dict(pq=float(value), delta=float(value - reference), ci=ci.tolist())
                        for scenario, value, reference, ci in zip(SCENARIOS, scores[1], scores[0], intervals)}
    return result


def write_result(path, result):
    """Save only aggregate experiment outputs to an ignored path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}", flush=True)
