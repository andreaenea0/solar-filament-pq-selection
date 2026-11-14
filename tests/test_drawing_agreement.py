"""Check strict thresholds, ambiguous triples and pooled loss units."""

import numpy as np
import pytest

from filseg.metric import encode
from tools.figures.drawing_agreement import consensus_triples, image_evidence


def test_three_way_matches_require_all_edges_and_unique_pairs():
    identity = np.eye(2)
    ambiguous = np.array([[.8, .8], [0, .9]])
    assert list(consensus_triples((identity, identity, identity))) == [(0, 0, 0), (1, 1, 1)]
    assert not list(consensus_triples((ambiguous, identity, identity)))
    assert list(consensus_triples((identity, identity, np.diag([.5, .6])))) == [(1, 1, 1)]


def test_near_miss_can_be_another_drawings_strict_hit():
    first = np.zeros((6, 6), dtype=np.uint8)
    first[1:3, 1:3] = 1
    half = first.copy()
    half[1] = 0
    records = [[encode(first)], [encode(half)]]
    evidence = image_evidence(records, [encode(first)], None, .23)
    assert evidence["fp_other_hit"] == [1, 1]
    assert evidence["near_other_hit"] == [1, 1]
    assert evidence["loss_fp"] == pytest.approx([.5, 1.])
    assert evidence["loss_fn"] == pytest.approx([.5, 1.])
    assert evidence["loss_boundary"] == pytest.approx([0, 1.])
    assert "held_model" not in evidence
