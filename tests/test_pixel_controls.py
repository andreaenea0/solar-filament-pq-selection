"""Protect the original byte-scale rule boundary and paired cohort requirement."""
import numpy as np
import pytest
from pycocotools import mask as mu

from tools.experiments.pixel_controls import fixed_rule, summarize


def test_byte_scale_mean_is_inclusive_at_the_exact_cutoff():
    # 200 pixels, mean 216.75 == .85 * 255, peak 253 > .99 * 255.
    pixels = np.zeros((30, 30), dtype=np.float32)
    values = np.array([217] * 150 + [216] * 50, dtype=np.float32)
    values[0], values[-1] = 253, 180
    pixels[:10, :20] = values.reshape(10, 20)
    masks = fixed_rule(pixels)
    assert len(masks) == 1 and int(mu.area(masks[0])) == 200
    pixels[9, 19] -= 1
    assert fixed_rule(pixels) == []


def test_missing_alternative_cannot_silently_change_the_scored_population():
    rows = [dict(recipe=recipe, stem=stem, fold=0, records=1,
                 sum_iou=.7, tp=1, fp=0, fn=0)
            for recipe, stem in (("control", "20200101000000Ch"),
                                 ("alternative", "20200101000000Ch"),
                                 ("control", "20200102000000Ch"))]
    with pytest.raises(ValueError, match="same observations"):
        summarize(rows, "control", 20)
