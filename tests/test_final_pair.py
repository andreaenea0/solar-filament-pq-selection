"""Verify the hedge gate and paired max-score calculation."""

import numpy as np
import pytest

from tools.experiments.final_pair import BASE_A, BASE_B, Recipe, choose, pair_value


def test_gate_uses_overlap_among_qualified_recipes():
    alternate = Recipe("pubown", .2, .27)
    counts = [(100, 100, 100), (80, 100, 100), (90, 100, 100)]
    assert choose([BASE_A, BASE_B, alternate], [.463, .459, .461], counts) == 2
    assert choose([BASE_A, BASE_B, alternate], [.459, .458, .459], counts) == 0


def test_pair_value_keeps_both_finals_on_the_same_sample():
    scores = np.array([[.6, .4], [.4, .6]])
    terms = np.stack((scores, np.ones_like(scores)), axis=-1)
    terms = np.repeat(terms[:, :, None, :], 3, axis=2)
    stems = ["20200101a", "20200102b"]
    pair = pair_value(stems, terms, image_counts=(1,))["1"]["full"]
    assert pair["expected_max"] == pytest.approx(.6)
    assert pair["advantage"] == pytest.approx(.1, abs=.005)
    identical = pair_value(stems, terms[:, [0, 0]], image_counts=(1,))["1"]["full"]
    assert identical["advantage"] == 0.
