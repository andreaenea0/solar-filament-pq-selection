"""Score sparse three-verifiers images and test a top-candidate fallback on held-out folds."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from filseg.data import gt_rles_by_image, load_records
from filseg.metric import score_image
from filseg.rle import counts_to_rle
from tools.cv_map_parity import load_submissions
from tools.experiments.mask_topology import results
from tools.figures.instance_errors import metric_values
from training.prepare import read_folds

GATES = (None, -np.inf, .10, .20, .23)
LABELS = ("off", "always", "0.10", "0.20", "0.23")


def load_proposals(path):
    """Read a trusted JSON or the archived top-candidate CSV."""
    if path.suffix.lower() != ".csv":
        return json.loads(path.read_text(encoding="utf-8"))
    proposals = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            stem = row["stem"]
            if stem in proposals:
                raise ValueError(f"Duplicate proposal for {stem}.")
            proposal = None
            if int(row["n_cands"]):
                rle = json.loads(row["rle_json"])
                if rle["size"] != [2048, 2048]:
                    raise ValueError(f"{stem}: expected a 2048 x 2048 RLE.")
                proposal = dict(counts=rle["counts"], gain=float(row["top_blend_gain"]))
            proposals[stem] = proposal
    return proposals


def host_terms(records, masks):
    """Return additive host score terms."""
    t = score_image(list(records.values()), masks)
    return t.sum_iou, t.tp, t.fp, t.fn


def main():
    """Use frozen OOF predictions and, optionally, archived top-candidate scores."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True, help="Official training JSON.")
    parser.add_argument("--pred", type=Path, required=True, help="Frozen held-out prediction CSV.")
    parser.add_argument("--folds", type=Path, help="Official fold CSV, required with --proposals.")
    parser.add_argument("--proposals", type=Path, help="Archived OOF top-gain CSV or equivalent trusted JSON.")
    args = parser.parse_args()
    if bool(args.folds) != bool(args.proposals):
        parser.error("Pass --folds and --proposals together.")
    gt = gt_rles_by_image(load_records(args.gt))
    submission = load_submissions([args.pred], gt)[0]
    stems = sorted(gt)
    empty = {s for s in stems if not submission.get(s)}
    proposals = load_proposals(args.proposals) if args.proposals else {}
    if args.proposals and proposals.keys() != empty:
        parser.error("Provide a top candidate or null for each empty OOF image.")

    groups = np.zeros((3, 4))
    image_counts = np.zeros(3, dtype=int)
    terms = np.empty((len(stems), len(GATES), 4))
    edits = np.zeros(terms.shape[:2], dtype=int)
    for row, stem in enumerate(stems):
        masks = [counts_to_rle(c) for c in submission.get(stem, [])]
        terms[row] = host_terms(gt[stem], masks)
        group = min(len(masks), 2)
        image_counts[group] += 1
        groups[group] += terms[row, 0]
        proposal = proposals.get(stem)
        if proposal is not None:
            gain = float(proposal["gain"])
            if not np.isfinite(gain):
                parser.error(f"{stem}: top-candidate gain must be finite.")
            alternative = host_terms(gt[stem], [counts_to_rle(proposal["counts"])])
            for col, gate in enumerate(GATES[1:], 1):
                if gain > gate:
                    terms[row, col] = alternative
                    edits[row, col] = 1
    for label, count, terms_for_count in zip(("0", "1", "2+"), image_counts, groups):
        _, tp, fp, fn = terms_for_count
        pq, *_ = metric_values(terms_for_count)
        print(f"{label} predictions: {count} images, PQ {pq:.6f}, {int(tp)} TP / {int(fp)} FP / {int(fn)} FN")
    if args.proposals:
        folds = read_folds(args.folds)
        if folds.keys() != gt.keys():
            parser.error("Fold assignments must cover exactly the training stems.")
        summary, *fixed = results(stems, folds, terms, edits, "empty fallback", LABELS)
        print(f"Nested: delta {summary['delta']:+.6f} [{summary['lo']:+.6f}, {summary['hi']:+.6f}], "
              f"{summary['edits']} additions; settings {summary['choices']}.")
        for row in fixed:
            print(f"Fixed {row['recipe']}: delta {row['delta']:+.6f}, {row['edits']} additions.")


if __name__ == "__main__":
    main()
