"""Check score-mix weights and count calibration with deterministic ties."""

import numpy as np

from tools.experiments.mask_shift import mix_weights, one_to_one_terms, rate_cut


def test_mix_weights_and_variation():
    weights, tv, _, _ = mix_weights(["a1", "a2", "a3", "b1"], ["a4", "b2"], lambda s: s[0])
    np.testing.assert_allclose(weights, [2 / 3, 2 / 3, 2 / 3, 2])
    assert tv == .25


def test_rate_cut_preserves_ties_and_does_not_invent_predictions():
    rows = [np.array([1, 2, 2]), np.array([2, 4])]
    assert rate_cut(rows, 2.) == 2
    assert rate_cut(rows, 3.) == 0


def test_one_to_one_assignment_resolves_two_hits_to_one_prediction():
    terms = one_to_one_terms(np.array([[.7, .1], [.8, .3]]))
    np.testing.assert_array_equal(terms, [.8, 1, 1, 1])
    np.testing.assert_array_equal(one_to_one_terms(np.zeros((2, 0))), [0., 0, 0, 2])
