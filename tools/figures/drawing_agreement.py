"""Compare the frozen submitted masks with independent drawings of the training observations."""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np

from filseg.data import gt_rles_by_image, load_records
from filseg.metric import iou_matrix, score_record
from filseg.rle import counts_to_rle
from filseg.select import dp_select
from tools.cv_map_parity import load_submissions
from tools.cv_sparse import read_pickle
from tools.figures.instance_errors import day_bootstrap
from tools.timing_replay import write_csv


def pq_terms(draw, masks):
    """Return the additive numerator and denominator of the host score."""
    t = score_record(draw, masks)
    return np.array([t.sum_iou, t.tp + (t.fp + t.fn) / 2])


def consensus_triples(pairwise):
    """Return all three-way strict matches unique in each drawing pair."""
    ab, ac, bc = [m > .5 for m in pairwise]
    def unique(m):
        return m & (m.sum(axis=0) == 1) & (m.sum(axis=1)[:, None] == 1)

    ab, ac, bc = map(unique, (ab, ac, bc))
    for i, j in zip(*np.nonzero(ab)):
        for k in np.flatnonzero(ac[i] & bc[j]):
            yield i, j, k


def image_evidence(drawings, masks, tree, penalty):
    """Count support by other drawings, strict consensus, and exact PQ loss terms."""
    matrices = [iou_matrix(d, masks) for d in drawings]
    best = np.array([m.max(axis=0, initial=0) for m in matrices])
    fp, fn, boundary = 0, 0, 0.
    result = {}
    for m in matrices:
        hit = m > .5
        fp += (~hit.any(axis=0)).sum()
        fn += (~hit.any(axis=1)).sum()
        boundary += (1 - m[hit]).sum()
    loss = (fp + fn) / 2 + boundary
    for name, value in (("fp", fp/2), ("fn", fn/2), ("boundary", boundary), ("count", (fp+fn)/2)):
        result["loss_" + name] = [value, loss]
    if len(drawings) > 1:
        other_hit = ((best > .5).sum(axis=0) - (best > .5)) > 0
        for name, eligible in (("fp_other_hit", best <= .5),
                               ("near_other_hit", (best >= .1) & (best <= .5))):
            result[name] = [np.count_nonzero(other_hit & eligible), eligible.sum()]
    if len(drawings) != 3:
        return result

    pairwise = [iou_matrix(drawings[i], drawings[j]) for i,j in combinations(range(3), 2)]
    shape = np.zeros((2, 2))
    for i, j, k in consensus_triples(pairwise):
        expert = (pairwise[0][i,j] + pairwise[1][i,k] + pairwise[2][j,k]) / 3
        predicted = np.mean([m[n].max(initial=0) for m,n in zip(matrices, (i,j,k))])
        shape += [[predicted, 1], [expert, 1]]
    result["consensus_model"], result["consensus_human"] = shape

    candidates = tree["rle"]
    gains = [iou_matrix(d, candidates).max(axis=0, initial=0) for d in drawings]
    gains = np.where(np.array(gains) > .5, gains, 0.)
    totals = np.zeros((3, 2))
    for held, drawing in enumerate(drawings):
        others = [i for i in range(3) if i != held]
        totals[0] += pq_terms(drawing, masks)
        for number in (1, 2):
            selections = []
            for group in combinations(others, number):
                score = gains[list(group)].mean(axis=0)
                keep = dp_select(tree["parent"], tree["children"], tree["level"], score - penalty)
                selections.append(pq_terms(drawing, [candidates[i] for i in keep]))
            totals[number] += np.mean(selections, axis=0)
    for key, val in zip(("held_model", "held_one", "held_two"), totals):
        result[key] = val
    return result


DEFINITIONS = {
    "held_model": ("0.457", "Pooled PQ of submitted masks against each of three held drawings."),
    "held_one": ("0.433", "Pooled held-drawing PQ at reference penalty 0.23. Average additive terms for the two possible one-drawing selectors."),
    "held_two": ("0.450", "Pooled held-drawing PQ at reference penalty 0.23. Select using the mean strict gain of the other two drawings."),
    "consensus_model": ("0.712", "Mean best IoU of any prediction to each member of a unique strict three-drawing clique. A miss contributes zero."),
    "consensus_human": ("0.643", "Mean of three expert-to-expert IoUs on exactly the same cliques."),
    "fp_other_hit": ("0.535", "Fraction of per-drawing false positives on multiply drawn images hitting another drawing at IoU > 0.5."),
    "near_other_hit": ("0.66", "Among prediction-drawing pairs with best IoU in [0.1,0.5], fraction hitting another drawing."),
    "loss_fp": ("", "Half the false-positive count divided by the total pooled PQ shortfall in numerator units."),
    "loss_fn": ("", "Half the false-negative count divided by the total pooled PQ shortfall in numerator units."),
    "loss_boundary": ("", "Sum of 1 minus IoU over strict hits divided by the total pooled PQ shortfall in numerator units."),
    "loss_count": ("0.61-0.62", "Combined false-positive and false-negative shares of the pooled shortfall."),
}


def summarize(evidence, replicates):
    """Use identical resampled observing days for each paired comparison."""
    rows, points, draws = [], {}, {}
    for key,(historical, definition) in DEFINITIONS.items():
        stems = sorted(s for s,vals in evidence.items() if key in vals)
        terms = np.asarray([evidence[s][key] for s in stems], dtype=float)
        _, sums = day_bootstrap(stems, terms, replicates)
        num, den = terms.sum(axis=0)
        if den <= 0 or np.any(sums[:,1] == 0):
            raise ValueError(f"Undefined denominator for {key}.")
        points[key], draws[key] = num/den, sums[:,0]/sums[:,1]
        lo,hi = np.quantile(draws[key], [.025,.975])
        rows.append(dict(metric=key,historical_comparator=historical,estimate=points[key],lower=lo,upper=hi,
                         observations=len(stems),days=len({s[:8] for s in stems}),numerator=num,
                         denominator=den,definition=definition,source="official training drawings and submitted OOF masks"
                         + ("; own OOF candidate trees" if key in ("held_one","held_two") else "")))
    for a,b in (("held_model","held_one"),("held_model","held_two"),("consensus_model","consensus_human")):
        base = next(r for r in rows if r["metric"] == a)
        lo, hi = np.quantile(draws[a] - draws[b], [.025,.975])
        rows.append(dict(base,metric=a+"_minus_"+b,historical_comparator="",estimate=points[a]-points[b],
                         lower=lo,upper=hi,numerator="",denominator="",definition="Paired difference on the same observing days.",
                         source=next(r["source"] for r in rows if r["metric"] == b)))
    return rows


def main():
    """Build the table from official training drawings, held-out masks and their own candidate trees."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--trees", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, default=Path("report/data/oof_submitted.csv"))
    parser.add_argument("--out", type=Path, default=Path("report/data/submitted_drawing_agreement.csv"))
    args = parser.parse_args()
    gt = gt_rles_by_image(load_records(args.annotations))
    trees = read_pickle(args.trees)
    if len(gt) != 707 or sum(len(d)==3 for d in gt.values()) != 151 or set(trees) != set(gt):
        raise ValueError("Expected the official training cohort and its held-out candidate trees.")
    prediction = load_submissions([args.predictions], gt)[0]
    evidence = {s: image_evidence(list(d.values()), [counts_to_rle(r) for r in prediction.get(s, [])],
                                  trees[s], .23) for s,d in gt.items()}
    write_csv(args.out, summarize(evidence, 2000))


if __name__ == "__main__":
    main()
