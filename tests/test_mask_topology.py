"""Check mask topology edits and the held-out rule selection."""

import numpy as np
from pycocotools import mask as mu

from filseg.metric import encode
from tools.experiments.mask_topology import choose_held_out, merge_edges, merge_masks, split_neck
from tools.figures.missed_instances import crop_mask


def test_merge_preserves_exact_union_and_respects_gap():
    mask = np.zeros((32, 56), dtype=bool)
    mask[10:13, 4:24] = True
    first = encode(mask)
    other = np.roll(mask, 23, axis=1)
    masks = [first, encode(other)]
    edges = merge_edges([crop_mask(rle) for rle in masks])
    assert merge_masks(masks, edges, 3)[1] == 0
    merged, count = merge_masks(masks, edges, 4)
    assert count == len(merged) == 1
    np.testing.assert_array_equal(mu.decode(merged[0]), mask | other)


def test_split_preserves_exact_union_and_disjointness():
    mask = np.zeros((40, 60), dtype=bool)
    mask[8:24, 4:21] = True
    mask[8:24, 33:50] = True
    mask[15, 21:33] = True
    left, right = split_neck(mask, 1)
    assert not (left & right).any()
    np.testing.assert_array_equal(left | right, mask)


def test_held_out_fold_cannot_select_its_own_recipe():
    stems = [str(i) for i in range(5)]
    folds = {stem: int(stem) for stem in stems}
    terms = np.tile([[[.6, 1, 0, 0], [.59, 1, 0, 0]]], (5, 1, 1))
    terms[0, 1, 0] = .9
    choices, selected, _ = choose_held_out(stems, folds, terms, np.zeros((5, 2)))
    assert choices == [0, 1, 1, 1, 1]
    assert selected[0, 0] == .6
