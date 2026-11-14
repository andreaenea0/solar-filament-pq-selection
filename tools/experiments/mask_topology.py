"""Test two mask-only topology edits on the official held-out three-verifiers folds."""

import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np
from pycocotools import mask as mu
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from filseg.data import gt_rles_by_image, load_records
from filseg.metric import encode, score_image
from filseg.rle import counts_to_rle
from tools.cv_map_parity import load_submissions
from tools.figures.instance_errors import day_bootstrap, metric_values, write_table
from tools.figures.missed_instances import CONNECTIVITY, boundary, crop_mask

GAPS = (4, 8, 16)
NECKS = (1, 2, 3)
ALIGNMENT = np.cos(np.deg2rad(35))


def merge_edges(crops, max_gap=GAPS[-1]):
    """Find elongated, aligned pairs within a boundary gap (default: 16 pixels)."""
    geometry = {}
    for i, (mask, region) in enumerate(crops):
        coords = np.argwhere(mask).astype(float)
        centre = coords.mean(axis=0)
        relative = coords - centre
        variance, directions = np.linalg.eigh(relative.T @ relative / len(relative))
        if variance[-1] >= 4 * max(variance[0], 0.25):
            geometry[i] = (centre + [axis.start for axis in region], directions[:, -1], boundary(mask, region))
    trees, edges = {}, []
    for i, j in combinations(geometry, 2):
        centre_i, axis_i, points_i = geometry[i]
        centre_j, axis_j, points_j = geometry[j]
        centre_offset = centre_j - centre_i
        norm = np.linalg.norm(centre_offset)
        if not norm:
            continue
        direction = centre_offset / norm
        if min(abs(axis_i @ direction), abs(axis_j @ direction), abs(axis_i @ axis_j)) < ALIGNMENT:
            continue
        bbox_gap = np.maximum(points_i.min(0) - points_j.max(0), points_j.min(0) - points_i.max(0))
        if np.any(bbox_gap > max_gap):
            continue
        if i not in trees:
            trees[i] = cKDTree(points_i)
        gap = trees[i].query(points_j)[0].min()
        if gap <= max_gap:
            edges.append((gap, i, j))
    return sorted(edges)


def merge_masks(masks, edges, max_gap):
    """Union disjoint pairs in nearest-first order, without adding bridge pixels."""
    used, merged = set(), []
    for gap, i, j in edges:
        if gap > max_gap:
            break
        if i not in used and j not in used:
            merged.append(mu.merge([masks[i], masks[j]]))
            used.update((i, j))
    return [rle for i, rle in enumerate(masks) if i not in used] + merged, len(merged)


def split_neck(mask, erosion):
    """Use two substantial eroded cores to partition a thin-neck mask."""
    cores, _ = ndi.label(ndi.binary_erosion(mask, structure=CONNECTIVITY, iterations=erosion),
                         structure=CONNECTIVITY)
    areas = np.bincount(cores.ravel())[1:]
    large = np.flatnonzero(areas >= max(25, 0.15 * mask.sum())) + 1
    if len(large) != 2:
        return None
    nearest = ndi.distance_transform_edt(~np.isin(cores, large), return_distances=False, return_indices=True)
    first = mask & (cores[tuple(nearest)] == large[0])
    second = mask & ~first
    if all(ndi.label(part, structure=CONNECTIVITY)[1] == 1 for part in (first, second)):
        return first, second
    return None


def split_masks(masks, crops, erosion):
    """Replace eligible masks with two connected pieces covering the original exactly."""
    result, edits = [], 0
    for rle, (mask, region) in zip(masks, crops):
        pieces = split_neck(mask, erosion)
        if pieces is None:
            result.append(rle)
            continue
        edits += 1
        for piece in pieces:
            full = np.zeros(rle["size"], dtype=bool)
            full[region] = piece
            result.append(encode(full))
    return result, edits


def choose_held_out(stems, folds, terms, edits):
    """Pick a fixed rule on the other folds for each held-out fold."""
    fold_ids = np.array([folds[s] for s in stems])
    if set(fold_ids) != set(range(5)):
        raise ValueError("Expected all five official folds.")
    choices = []
    selected = np.zeros_like(terms[:, 0])
    selected_edits = 0
    for k in range(5):
        held_out = fold_ids == k
        recipe = int(metric_values(terms[~held_out].sum(axis=0))[:, 0].argmax())
        choices.append(recipe)
        selected[held_out] = terms[held_out, recipe]
        selected_edits += int(edits[held_out, recipe].sum())
    return choices, selected, selected_edits


def results(stems, folds, terms, edits, family, labels):
    """Report fixed results and paired day intervals for fold-held-out tuning."""
    choices, selected, selected_edits = choose_held_out(stems, folds, terms, edits)
    paired = np.stack((terms[:, 0], selected), axis=1)
    _, draws = day_bootstrap(stems, paired, 2000)
    base, tuned = metric_values(paired.sum(axis=0))[:, 0]
    draw_scores = metric_values(draws)[:, :, 0]
    lo, hi = np.percentile(draw_scores[:, 1] - draw_scores[:, 0], [2.5, 97.5])
    yield dict(family=family, recipe="fold-held-out", edits=selected_edits, pq=tuned, delta=tuned - base,
               lo=lo, hi=hi, choices="|".join(labels[i] for i in choices))
    for label, pq, count in zip(labels, metric_values(terms.sum(axis=0))[:, 0], edits.sum(axis=0)):
        yield dict(family=family, recipe=label, edits=int(count), pq=pq, delta=pq - base)


def main():
    """Score only the official training drawings and frozen held-out prediction masks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True, help="Official training annotation JSON.")
    parser.add_argument("--pred", type=Path, required=True, help="Frozen three-verifiers OOF submission CSV.")
    parser.add_argument("--folds", type=Path, required=True, help="Official stem-to-fold CSV.")
    parser.add_argument("--out", type=Path, required=True, help="Aggregate results CSV.")
    args = parser.parse_args()
    gt = gt_rles_by_image(load_records(args.gt))
    stems = sorted(gt)
    submission = load_submissions([args.pred], gt)[0]
    with args.folds.open(encoding="utf-8-sig", newline="") as stream:
        folds = {r["stem"]: int(r["fold"]) for r in csv.DictReader(stream)}
    if set(stems) != folds.keys():
        parser.error("The fold file must cover exactly the training stems.")

    terms = np.empty((len(stems), 1 + len(GAPS) + len(NECKS), 4))
    edits = np.zeros(terms.shape[:2], dtype=int)
    for row, stem in enumerate(stems):
        masks = [counts_to_rle(counts) for counts in submission.get(stem, [])]
        records = list(gt[stem].values())
        crops = [crop_mask(rle) for rle in masks]
        edges = merge_edges(crops)
        variants = [(masks, 0)] + [merge_masks(masks, edges, gap) for gap in GAPS]
        variants += [split_masks(masks, crops, erosion) for erosion in NECKS]
        for col, (variant, count) in enumerate(variants):
            t = score_image(records, variant)
            terms[row, col] = (t.sum_iou, t.tp, t.fp, t.fn)
            edits[row, col] = count
        if (row + 1) % 100 == 0:
            print(f"Scored {row + 1}/{len(stems)} images.", flush=True)
    rows = []
    split_start = 1 + len(GAPS)
    for family, values, columns in (
        ("merge", GAPS, list(range(split_start))),
        ("split", NECKS, [0, *range(split_start, terms.shape[1])]),
    ):
        labels = ("off", *(str(n) for n in values))
        rows.extend(results(stems, folds, terms[:, columns], edits[:, columns], family, labels))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_table(args.out, rows)
    for row in rows:
        if row["recipe"] == "fold-held-out":
            print(f"{row['family']}: PQ {row['pq']:.6f}, delta {row['delta']:+.6f} "
                  f"[{row['lo']:+.6f}, {row['hi']:+.6f}], {row['edits']} edits", flush=True)


if __name__ == "__main__":
    main()
