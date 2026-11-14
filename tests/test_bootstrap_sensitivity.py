"""Longer-block intervals must preserve pairing and the score's ratio of pooled terms."""

import numpy as np
import pytest

from tools.experiments.bootstrap_sensitivity import paired_interval


def test_common_blocks_preserve_a_constant_paired_improvement():
    stems = ["19700101000000Ah", "19700107000000Ah", "19700108000000Ah"]
    terms = np.zeros((3, 2, 4))
    terms[:, :, 1] = 1
    terms[:, 0, 0] = [.6, .7, .8]
    terms[:, 1, 0] = terms[:, 0, 0] + .1
    for width, blocks in ((1, 3), (7, 2), (28, 1)):
        result = paired_interval(stems, terms, width, 100)
        assert result["occupied_blocks"] == blocks
        assert [result[k] for k in ("delta", "lo", "hi")] == pytest.approx([.1, .1, .1])


def test_invalid_resampling_configuration_is_rejected():
    with pytest.raises(ValueError):
        paired_interval([], np.zeros((0, 2, 4)), 0)
    with pytest.raises(ValueError, match="nonempty"):
        paired_interval([], np.zeros((0, 2, 4)), 1)
    with pytest.raises(ValueError, match="finite"):
        paired_interval(["19700101"], np.full((1, 2, 4), np.nan), 1)


def test_pool_terms_before_forming_each_ratio():
    """Unequal image denominators distinguish pooled PQ from mean image PQ."""
    terms = np.zeros((2, 2, 4))
    terms[:, :, 1] = [[1, 1], [9, 9]]
    terms[:, 0, 0] = [.6, 5.4]
    terms[:, 1, 0] = [.8, 8.1]
    result = paired_interval(["19700101000000Ah", "19700101010000Ah"], terms, 1, 20)
    assert [result[k] for k in ("delta", "lo", "hi")] == pytest.approx([.29] * 3)
