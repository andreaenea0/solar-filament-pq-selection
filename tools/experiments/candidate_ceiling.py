"""Measure exact host-PQ oracle ceilings of coarse and fine threshold forests."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from pycocotools import mask as mu
from scipy import ndimage as ndi

from filseg.candidates import LADDER, S8
from filseg.maps import read_map
from filseg.metric import iou_matrix, score_image
from filseg.select import dp_select
from tools.cv_sparse import pq, read_pickle
from tools.experiments.cached_selector import Terms, comparisons
from training.prepare import read_folds

FINE = (.15, .20, .25, .30, .35, .40, .45, .50, .55, .60, .65, .70, .75, .80)


def encode_crop(mask, top, left, shape):
    """Encode a Boolean crop at full-image coordinates without a full-size allocation."""
    cols, rows = np.nonzero(mask.T)
    positions = (cols + left) * shape[0] + rows + top
    if not len(positions):
        counts = [int(np.prod(shape))]
    else:
        breaks = np.flatnonzero(np.diff(positions) > 1)
        starts = positions[np.r_[0, breaks + 1]]
        ends = positions[np.r_[breaks, len(positions) - 1]] + 1
        counts = [int(starts[0])]
        for i, (start, end) in enumerate(zip(starts, ends)):
            counts.append(int(end - start))
            if i + 1 < len(starts):
                counts.append(int(starts[i + 1] - end))
        if ends[-1] < np.prod(shape):
            counts.append(int(np.prod(shape) - ends[-1]))
    return mu.frPyObjects(dict(size=list(shape), counts=counts), *shape)


def mask_tree(p, ladder=LADDER, min_area=30):
    """Build the released masks/order/hierarchy, omitting all regression features."""
    tree = dict(parent=[], children=[], level=[], rle=[])
    previous, previous_indices = None, None
    for level, threshold in enumerate(ladder):
        labels, count = ndi.label(p > int(round(threshold * 255)), S8)
        indices = np.full(count + 1, -1, dtype=int)
        for label, region in enumerate(ndi.find_objects(labels), 1):
            if region is None:
                continue
            mask = labels[region] == label
            if mask.sum() < min_area:
                continue
            parent = -1
            if previous is not None:
                y, x = np.argwhere(mask)[0]
                parent = int(previous_indices[previous[region][y, x]])
            index = len(tree["rle"])
            indices[label] = index
            tree["parent"].append(parent)
            tree["children"].append([])
            tree["level"].append(level)
            tree["rle"].append(encode_crop(mask, region[0].start, region[1].start, p.shape))
            if parent >= 0:
                tree["children"][parent].append(index)
        previous, previous_indices = labels, indices
    return tree


def oracle(trees, terms, stems):
    """Maximize a ratio of exact additive host terms over all forest antichains."""
    ratio = 0.
    for iteration in range(100):
        indices, totals = [], []
        for stem in stems:
            tree, term = trees[stem], terms[stem]
            weights = term.contribution[0, :, 0] - ratio * term.contribution[0, :, 1]
            choice = dp_select(tree["parent"], tree["children"], tree["level"], weights)
            indices.append(choice)
            totals.append(term.selected(choice)[0])
        totals = np.asarray(totals)
        updated = float(pq(totals.sum(axis=0)))
        if abs(updated - ratio) < 1e-12:
            return totals, dict(zip(stems, indices)), iteration + 1
        ratio = updated
    raise RuntimeError("Fractional oracle failed to converge.")


def build_job(job):
    """Validate the coarse cache before scoring either pool on one training image."""
    stem, maps, cached, records = job
    p = read_map(maps / f"{stem}.npz")
    if p.dtype != np.uint8 or p.ndim != 2:
        raise ValueError(f"{stem}: expected one held-out uint8 map.")
    coarse, fine = mask_tree(p), mask_tree(p, FINE)
    for key in ("parent", "children", "level", "rle"):
        if coarse[key] != cached[key]:
            raise ValueError(f"{stem}: fast coarse {key} differs from the released cache.")
    if not {r["counts"] for r in coarse["rle"]} <= {r["counts"] for r in fine["rle"]}:
        raise ValueError(f"{stem}: fine pool does not contain every coarse mask.")
    result = {}
    for name, tree in (("coarse", coarse), ("fine", fine)):
        tree["coverage_strict"] = sum((iou_matrix(drawing, tree["rle"]) > .5).any(axis=1).sum()
                                      for drawing in records.values())
        result[name] = tree, Terms.build(records, tree["rle"])
    return stem, result


def verify_choices(trees, terms, choices, gt):
    """Check the exact scorer and pixel disjointness for each chosen oracle mask set."""
    for stem, indices in choices.items():
        masks = [trees[stem]["rle"][i] for i in indices]
        if masks and sum(mu.area(masks)) != mu.area(mu.merge(masks)):
            raise ValueError(f"{stem}: oracle masks overlap.")
        actual = score_image(list(gt[stem].values()), masks)
        expected = terms[stem].selected(indices)[0]
        np.testing.assert_allclose(expected, (actual.sum_iou, actual.tp + .5 * (actual.fp + actual.fn)), atol=1e-10)


def main():
    """Run CPU-only oracle screening; predictions and arrays remain under data/."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, help="Smoke-test only; omit for all five folds.")
    parser.add_argument("--prepared", type=Path, help="Optional ignored checkpoint of built training forests.")
    args = parser.parse_args()
    gt = read_pickle(args.cache / "gt_rles_coco.pkl")
    folds = read_folds(args.cache / "folds.csv")
    cached = read_pickle(args.cache / "cands_train_own.pkl")
    stems = sorted(gt)[:args.limit]
    if args.prepared and args.prepared.is_file():
        prepared = read_pickle(args.prepared)
        if set(prepared) != set(stems):
            raise ValueError("Prepared checkpoint has different training image coverage.")
    else:
        jobs = ((s, args.maps, cached[s], gt[s]) for s in stems)
        with ProcessPoolExecutor(args.workers) as pool:
            prepared = {}
            for i, (stem, value) in enumerate(pool.map(build_job, jobs), 1):
                prepared[stem] = value
                if i % 50 == 0 or i == len(stems):
                    print(f"Built and cache-verified {i}/{len(stems)} images", flush=True)
        if args.prepared:
            args.prepared.parent.mkdir(parents=True, exist_ok=True)
            args.prepared.write_bytes(pickle.dumps(prepared))
    recipes, rows, image_rows = {}, [], []
    for name in ("coarse", "fine"):
        trees = {s: prepared[s][name][0] for s in stems}
        terms = {s: prepared[s][name][1] for s in stems}
        pooled, choices, iterations = oracle(trees, terms, stems)
        verify_choices(trees, terms, choices, gt)
        recipes[name] = pooled[:, None, :]
        rows.append(dict(pool=name, cohort="all", pq=float(pq(pooled.sum(axis=0))),
                         nodes=sum(len(t["rle"]) for t in trees.values()), iterations=iterations,
                         covered=sum(t["coverage_strict"] for t in trees.values())))
        for fold in sorted({folds[s] for s in stems}):
            held = [s for s in stems if folds[s] == fold]
            values, held_choices, n_iter = oracle(trees, terms, held)
            verify_choices(trees, terms, held_choices, gt)
            rows.append(dict(pool=name, cohort=f"fold{fold}", pq=float(pq(values.sum(axis=0))),
                             nodes=sum(len(trees[s]["rle"]) for s in held), iterations=n_iter,
                             covered=sum(trees[s]["coverage_strict"] for s in held)))
        for stem, values in zip(stems, pooled):
            contribution = terms[stem].contribution[0]
            image_rows.append(dict(pool=name, stem=stem, fold=folds[stem], nodes=len(contribution),
                                   covered=trees[stem]["coverage_strict"], selected=len(choices[stem]),
                                   numerator=values[0], denominator=values[1]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    pd.DataFrame(image_rows).to_csv(args.out.with_name(args.out.stem + "_per_image.csv"), index=False)
    paired = comparisons(stems, recipes, "coarse")
    args.out.with_suffix(".json").write_text(json.dumps(paired, indent=2) + "\n")
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(paired["fine"]["full"], flush=True)


if __name__ == "__main__":
    main()
