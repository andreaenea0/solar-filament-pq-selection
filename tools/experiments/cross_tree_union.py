"""Measure oracle headroom and held-out selection on the union of two trees."""

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from filseg.metric import iou_matrix
from tools.cv_sparse import pq
from tools.experiments.cached_selector import (
    PENALTIES, SOURCES, Terms, comparisons, load_blend, load_training, nested_grid, select_tree, write_result,
)


def bit_indices(bits):
    """Iterate set-bit indices from low to high."""
    while bits:
        bit = bits & -bits
        yield bit.bit_length() - 1
        bits ^= bit


class ConflictGraph:
    """An undirected graph with one vertex per distinct mask."""

    def __init__(self, conflicts):
        self.neighbors = tuple(sum(1 << int(j) for j in np.flatnonzero(row)) for row in conflicts)

    def components(self, active):
        """Split an active vertex set into disconnected components."""
        while active:
            front, found = active & -active, 0
            while front:
                found |= front
                adjacent = 0
                for i in bit_indices(front):
                    adjacent |= self.neighbors[i]
                front = adjacent & active & ~found
            yield found
            active ^= found

    def _milp(self, nodes, weights):
        """Solve a larger connected component exactly with binary edge constraints."""
        edges = [(i, j) for i, node in enumerate(nodes) for j in range(i + 1, len(nodes))
                 if self.neighbors[node] & (1 << nodes[j])]
        count = len(edges)
        matrix = coo_matrix((np.ones(2 * count), (np.repeat(np.arange(count), 2), np.ravel(edges))),
                            shape=(count, len(nodes))).tocsc()
        fit = milp(-weights[nodes], integrality=np.ones(len(nodes)), bounds=Bounds(0, 1),
                   constraints=LinearConstraint(matrix, -np.inf, 1), options={"mip_rel_gap": 0.})
        if not fit.success:
            raise RuntimeError(f"Exact conflict solver failed: {fit.message}")
        return sum(1 << nodes[i] for i in np.flatnonzero(fit.x > .5))

    def select(self, weights):
        """Return an exact maximum-weight independent set, excluding nonpositive weights."""
        active = sum(1 << int(i) for i in np.flatnonzero(weights > 0))

        @lru_cache(maxsize=None)
        def solve(bits):
            if not bits:
                return 0., 0
            parts = list(self.components(bits))
            if len(parts) > 1:
                values = [solve(part) for part in parts]
                return sum(value for value, _ in values), sum(chosen for _, chosen in values)
            nodes = list(bit_indices(bits))
            degrees = [(self.neighbors[i] & bits).bit_count() for i in nodes]
            if min(degrees) == len(nodes) - 1:
                i = max(nodes, key=lambda n: weights[n])
                return weights[i], 1 << i
            if len(nodes) > 30:
                chosen = self._milp(nodes, weights)
                return sum(weights[i] for i in bit_indices(chosen)), chosen
            i = nodes[int(np.argmax(degrees))]
            rest = bits ^ (1 << i)
            with_i, chosen = solve(rest & ~self.neighbors[i])
            without_i, other = solve(rest)
            if with_i + weights[i] >= without_i:
                return with_i + weights[i], chosen | (1 << i)
            return without_i, other

        return list(bit_indices(solve(active)[1]))


def union_masks(trees):
    """Deduplicate canonical COCO RLEs and retain each source's candidate mapping."""
    masks, keys, mapping = [], {}, []
    for tree in trees:
        source = []
        for rle in tree["rle"]:
            counts = rle["counts"]
            key = (tuple(rle["size"]), counts.encode("ascii") if isinstance(counts, str) else counts)
            if key not in keys:
                keys[key] = len(masks)
                masks.append(rle)
            source.append(keys[key])
        mapping.append(np.array(source, dtype=int))
    return masks, mapping


def fit_affine(scores, targets, weights):
    """Map source gains onto expected true gain using training rows only."""
    mean_x, mean_y = np.average(scores, weights=weights), np.average(targets, weights=weights)
    centered = scores - mean_x
    variance = np.dot(weights, centered ** 2)
    slope = max(0., np.dot(weights, centered * (targets - mean_y)) / variance) if variance else 0.
    return slope, mean_y - slope * mean_x


def merge_gains(size, mapping, gains):
    """Retain the strongest calibrated score of each duplicate mask."""
    merged = np.full(size, -np.inf)
    for indices, scores in zip(mapping, gains):
        np.maximum.at(merged, indices, scores)
    return merged


def oracle(terms, choose):
    """Maximize pooled full host PQ with exact additive numerator and cost."""
    score = 0.
    while True:
        selected = np.stack([t.selected(choose(i, t.contribution[0, :, 0] - score * t.contribution[0, :, 1]))
                             for i, t in enumerate(terms)])
        updated = pq(selected.sum(axis=0))[0]
        if updated - score <= 1e-12:
            return selected
        score = updated


def main():
    """Compare the two released families, exact full-PQ oracles and nested calibrated selectors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    gt, folds, trees = load_training(args.cache)
    blends = [load_blend(args.cache, source, trees[source])[0] for source in SOURCES]
    stems = sorted(gt)
    groups = np.array([folds[s] for s in stems])
    source_trees = [[trees[source][s] for s in stems] for source in SOURCES]
    terms, graphs, mapping = [], [], []
    input_nodes = union_nodes = 0
    for i, stem in enumerate(stems):
        masks, indices = union_masks([source[i] for source in source_trees])
        overlaps = iou_matrix(masks, masks)
        np.fill_diagonal(overlaps, 0.)
        graphs.append(ConflictGraph(overlaps > 0))
        terms.append(Terms.build(gt[stem], masks))
        mapping.append(indices)
        input_nodes += sum(map(len, indices))
        union_nodes += len(masks)

    def pick(source, i, gains, penalty):
        chosen = select_tree(source_trees[source][i], gains, penalty)
        return terms[i].selected(mapping[i][source][chosen])

    recipes = {f"released_{name}": np.stack([pick(source, i, blends[source][s], penalty)
                                             for i, s in enumerate(stems)])
               for source, (name, penalty) in enumerate((("public-ensemble", .27), ("three-verifiers", .25)))}
    recipes["oracle_A"] = oracle(terms, lambda i, gains: mapping[i][0][
        select_tree(source_trees[0][i], gains[mapping[i][0]], 0.)])
    recipes["oracle_union"] = oracle(terms, lambda i, gains: graphs[i].select(gains))
    print(f"Prepared {union_nodes} distinct masks from {input_nodes} nodes; oracle complete.", flush=True)

    @lru_cache(maxsize=None)
    def fit(excluded):
        train = [i for i in range(len(stems)) if groups[i] not in excluded]
        return [fit_affine(np.concatenate([blends[k][stems[i]] for i in train]),
                           np.concatenate([source_trees[k][i]["g"] for i in train]),
                           np.concatenate([np.full(len(source_trees[k][i]["rle"]), source_trees[k][i]["R"])
                                           for i in train]))
                for k in range(2)]

    def evaluate(excluded, group):
        coeffs = fit(excluded)
        rows = []
        for i in np.flatnonzero(groups == group):
            gains = [np.clip(slope * blend[stems[i]] + offset, 0., 1.)
                     for (slope, offset), blend in zip(coeffs, blends)]
            merged = merge_gains(len(graphs[i].neighbors), mapping[i], gains)
            rows.append([[pick(0, i, gains[0], lam) for lam in PENALTIES],
                         [terms[i].selected(graphs[i].select(merged - lam)) for lam in PENALTIES]])
        return np.array(rows)

    selected, penalties = nested_grid(groups, evaluate)
    recipes["calibrated_A"], recipes["calibrated_union"] = np.moveaxis(selected, 1, 0)
    result = dict(images=len(stems), input_nodes=input_nodes, unique_nodes=union_nodes, penalties=penalties,
                  calibration={str(k): [[float(v) for v in co] for co in fit(frozenset((k,)))]
                               for k in sorted(set(groups))},
                  vs_released_A=comparisons(stems, recipes, "released_A"),
                  union_vs_calibrated_A=comparisons(stems, {k: recipes[k] for k in ("calibrated_A", "calibrated_union")},
                                                    "calibrated_A"),
                  oracle_union_vs_A=comparisons(stems, {k: recipes[k] for k in ("oracle_A", "oracle_union")}, "oracle_A"))
    write_result(args.out, result)


if __name__ == "__main__":
    main()
