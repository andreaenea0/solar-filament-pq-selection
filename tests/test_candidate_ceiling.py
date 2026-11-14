"""Analysis-only crop encoding and exact fractional oracle verification."""

from itertools import combinations

import numpy as np
import pytest
from pycocotools import mask as mu

from filseg.metric import encode, score_image
from tools.experiments.cached_selector import Terms
from tools.experiments.candidate_ceiling import encode_crop, mask_tree, oracle


@pytest.mark.parametrize("shape,top,left,crop", [
    ((6, 7), 1, 2, [[1, 0, 1], [1, 1, 1], [0, 1, 0]]),
    ((5, 5), 1, 1, [[1, 1, 1], [1, 0, 1], [1, 1, 1]]),  # Interior hole.
    ((3, 2), 0, 0, [[0, 1], [0, 1], [1, 0]]),  # A run crosses a column boundary.
    ((4, 5), 2, 3, [[0, 0], [0, 1]]),  # Final image pixel is foreground.
    ((3, 4), 0, 0, np.ones((3, 4))),
    ((3, 4), 0, 0, np.zeros((3, 4))),
])
def test_crop_encoder_matches_canonical_full_image_bytes(shape, top, left, crop):
    """Compare canonical compressed RLEs, including empty and boundary cases."""
    crop = np.asarray(crop, dtype=bool)
    full = np.zeros(shape, dtype=np.uint8)
    full[top:top + crop.shape[0], left:left + crop.shape[1]] = crop
    assert encode_crop(crop, top, left, shape) == encode(full)


def exhaustive_best(tree, records):
    """Enumerate every pixel-disjoint subset with the actual host scorer."""
    best = 0.
    for count in range(len(tree["rle"]) + 1):
        for indices in combinations(range(len(tree["rle"])), count):
            masks = [tree["rle"][i] for i in indices]
            if masks and sum(mu.area(masks)) != mu.area(mu.merge(masks)):
                continue
            best = max(best, score_image(list(records.values()), masks).pq)
    return best


def test_fractional_oracle_matches_brute_force_with_overlapping_drawings():
    """A parent hits two overlapping GTs; simplified max-IoU g is insufficient."""
    image = np.zeros((6, 16), dtype=np.uint8)
    image[2:4, 1:13] = 100
    image[2:4, 1:5] = image[2:4, 9:13] = 220
    tree = mask_tree(image, (.15, .65), min_area=1)
    first, second = np.zeros_like(image), np.zeros_like(image)
    first[2:4, 1:9], second[2:4, 5:13] = 1, 1
    records = {"a": [encode(first), encode(second)], "b": [tree["rle"][0]], "empty": []}
    terms = Terms.build(records, tree["rle"])
    values, choices, iterations = oracle({"s": tree}, {"s": terms}, ["s"])
    actual = score_image(list(records.values()), [tree["rle"][i] for i in choices["s"]]).pq
    assert actual == pytest.approx(exhaustive_best(tree, records))
    assert values[0, 0] / values[0, 1] == pytest.approx(actual)
    assert iterations < 10


def test_oracle_handles_empty_forest():
    """An empty image pool still carries the GT denominator and converges at zero."""
    tree = mask_tree(np.zeros((5, 5), dtype=np.uint8))
    records = {"a": [encode(np.ones((5, 5), dtype=np.uint8))]}
    terms = Terms.build(records, [])
    values, choices, _ = oracle({"s": tree}, {"s": terms}, ["s"])
    assert choices == {"s": []}
    assert values.tolist() == [[0., .5]]


def test_pooled_oracle_can_drop_an_images_individual_optimum():
    """A low-consensus object improves its own image but lowers pooled PQ."""
    tree = mask_tree(np.full((6, 6), 200, dtype=np.uint8), (.15,))
    certain = {"a": tree["rle"]}
    uncertain = {"a": tree["rle"], "b": [], "c": [], "d": []}
    terms = {"certain": Terms.build(certain, tree["rle"]),
             "uncertain": Terms.build(uncertain, tree["rle"])}
    totals, choices, _ = oracle({s: tree for s in terms}, terms, list(terms))
    assert choices == {"certain": [0], "uncertain": []}
    assert totals.sum(axis=0).tolist() == pytest.approx([1., 1.5])
    # Exhaust all four joint selections with the actual host terms.
    scores = []
    for a, b in (([], []), ([0], []), ([], [0]), ([0], [0])):
        first = score_image(list(certain.values()), [tree["rle"][i] for i in a])
        second = score_image(list(uncertain.values()), [tree["rle"][i] for i in b])
        first += second
        scores.append(first.pq)
    assert max(scores) == pytest.approx(2/3)
