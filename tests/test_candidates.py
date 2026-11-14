"""Candidate features must describe the threshold actually used to build each mask."""

import numpy as np
from pycocotools import mask as mu
import pytest

from filseg import candidates


@pytest.fixture
def small_frame(monkeypatch):
    """Use small arrays while preserving the tree builder's full-frame coordinates."""
    monkeypatch.setattr(candidates, "H", 8)
    monkeypatch.setattr(candidates, "W", 8)
    monkeypatch.setattr(candidates, "RHO", np.zeros((8, 8)))
    image = np.zeros((8, 8), dtype=np.uint8)
    image[2:5, 2:5] = 255
    return image


@pytest.mark.parametrize("ladder", (candidates.LADDER, (.05, .1, .2, .4, .6, .8, .9)))
def test_threshold_features_follow_custom_ladder(small_frame, ladder):
    """More than six levels used to crash; other custom levels silently had wrong features."""
    tree = candidates.build_tree(small_frame, ladder=ladder, min_area=1)
    assert [c.feats["t"] for c in tree] == list(ladder)
    assert [c.parent for c in tree] == [-1, *range(len(ladder) - 1)]
    assert [c.feats["level"] for c in tree] == list(range(len(ladder)))
    for node in tree:
        np.testing.assert_array_equal(mu.decode(node.rle), small_frame > 0)


def test_quantized_cutoff_is_strict_and_connectivity_is_eight(small_frame):
    """Values exactly at the cutoff are excluded; diagonal foreground pixels stay connected."""
    pixels = np.zeros_like(small_frame)
    pixels[2, 2], pixels[3, 3], pixels[2, 3] = 39, 39, 38
    tree = candidates.build_tree(pixels, ladder=(.15,), min_area=1)
    assert len(tree) == 1
    assert tree[0].feats["area"] == 2
    np.testing.assert_array_equal(mu.decode(tree[0].rle), pixels > 38)
