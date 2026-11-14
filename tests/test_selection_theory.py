"""Check the antichain optimizer against exhaustive feasible-subset search."""

import numpy as np
import pytest

from filseg.select import dp_select
from filseg.metric import encode, score_record


def ancestors(parent, node):
    """Return all strict ancestors in a valid forest."""
    found = set()
    while parent[node] >= 0:
        node = parent[node]
        found.add(node)
    return found


def exhaustive_value(parent, gain):
    """Enumerate all antichains; keep the empty set as a feasible choice."""
    above = [ancestors(parent, i) for i in range(len(parent))]
    best = 0.
    for bits in range(1 << len(parent)):
        chosen = {i for i in range(len(parent)) if bits & (1 << i)}
        if all(not (above[i] & chosen) for i in chosen):
            best = max(best, sum(gain[i] for i in chosen))
    return best


def test_random_forests_equal_exhaustive_search():
    """Different topology, levels, signs and node order must preserve the optimum."""
    rng = np.random.default_rng(20261111)
    for size in range(12):
        for _ in range(12):
            parent = [int(rng.integers(-1, i)) for i in range(size)]
            level = [0] * size
            for i, p in enumerate(parent):
                level[i] = 0 if p < 0 else level[p] + int(rng.integers(1, 4))
            order = rng.permutation(size)
            inverse = np.argsort(order)
            parent = [int(inverse[parent[i]]) if parent[i] >= 0 else -1 for i in order]
            level = np.asarray(level)[order]
            children = [[i for i, p in enumerate(parent) if p == node] for node in range(size)]
            gain = rng.integers(-5, 6, size=size) / 4
            chosen = dp_select(parent, children, level, gain)
            assert len(chosen) == len(set(chosen))
            assert all(not (ancestors(parent, i) & set(chosen)) for i in chosen)
            assert sum(gain[i] for i in chosen) == pytest.approx(exhaustive_value(parent, gain))


def test_ties_choose_a_positive_parent_and_drop_zero_branches():
    """Document the deployed deterministic tie behavior."""
    assert dp_select([-1, 0, 0, -1], [[1, 2], [], [], []], [0, 1, 1, 0], [1, .5, .5, 0]) == [0]


def test_overlapping_truth_requires_generalized_denominator():
    """The one-hit identity fails for two overlapping GT masks hit by one prediction."""
    truth = [encode(np.array([[1, 1, 0]])), encode(np.array([[0, 1, 1]]))]
    total = score_record(truth, [encode(np.ones((1, 3)))])
    assert total.tp == 2
    assert total.pq == pytest.approx(2 / 3)
    summed_iou, gt_count, match_count = 4 / 3, 2, 2
    assert 2 * summed_iou / (gt_count + max(1, match_count)) == pytest.approx(total.pq)
    assert 2 * (2 / 3) / (gt_count + 1) != pytest.approx(total.pq)
