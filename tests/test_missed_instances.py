"""Regression checks for missed-instance geometry and the exact host oracle."""

import unittest

import numpy as np
from scipy.spatial import cKDTree

from filseg.metric import encode, score_image
from tools.figures.missed_instances import boundary, crop_mask, lower_tree, measure, miss_cohorts, oracle_additions


def square(y, x, side=2):
    """Encode a small square for the synthetic observations."""
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[y:y + side, x:x + side] = 1
    return encode(mask)


class FinalBMissesTest(unittest.TestCase):
    def test_contrast_centroid_and_distance(self):
        missed = square(8, 8)
        image = np.full((20, 20), 100)
        image[8:10, 8:10] = 80
        tree = cKDTree(boundary(*crop_mask(square(11, 12))))
        area, rho, contrast, distance, _ = measure(missed, image, (8.5, 8.5, 9), tree, 0)
        self.assertEqual(area, 4)
        self.assertEqual(rho, 0)
        self.assertAlmostEqual(contrast, 0.2)
        self.assertAlmostEqual(distance, np.sqrt(13))

    def test_strict_low_tree_and_pooled_oracle(self):
        missed, matched = square(2, 2), square(14, 14)
        broad = square(13, 13, 4)
        drawings = [[missed, missed, matched], [missed, matched]]
        predictions = [matched]
        exact_half = np.zeros((20, 20), dtype=np.uint8)
        exact_half[2, 2:4] = 1
        candidate = dict(rle=[missed, encode(exact_half), broad, matched], level=[0, 1, 0, 2],
                         parent=[-1, 0, -1, 2], children=[[1], [], [3], []])
        _, groups = zip(*miss_cohorts(drawings, predictions))
        self.assertEqual(list(groups[0]), ["corroborated", "corroborated", None])
        coverage, tree = lower_tree(drawings, predictions, candidate, groups)
        np.testing.assert_array_equal(coverage[0], [[True, True], [True, True], [False, False]])
        np.testing.assert_array_equal(tree[1], [True, False, False, False])
        self.assertEqual(tree[3][0], 1.5)
        baseline = score_image(drawings, predictions)
        pq, additions = oracle_additions(baseline, {"stem": tree})
        self.assertEqual(additions, {"stem": [missed]})
        self.assertEqual(pq, score_image(drawings, [matched, missed]).pq)

    def test_tree_hit_can_conflict_with_frozen_prediction(self):
        candidate_mask, frozen = square(2, 2, 4), square(3, 3)
        drawings = [[candidate_mask], [candidate_mask]]
        candidate = dict(rle=[candidate_mask, frozen], level=[0, 2],
                         parent=[-1, 0], children=[[1], []])
        _, groups = zip(*miss_cohorts(drawings, [frozen]))
        coverage, tree = lower_tree(drawings, [frozen], candidate, groups)
        np.testing.assert_array_equal(coverage, [[[True, False]], [[True, False]]])
        self.assertFalse(tree[1].any())


if __name__ == "__main__":
    unittest.main()
