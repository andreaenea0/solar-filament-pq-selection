"""Synthetic checks for the jury figures and report measurements."""

import numpy as np
import pytest

from filseg.disk import fit_limb, profile
from filseg.metric import encode
from tools.figures.annotation_stats import summarize
from tools.figures.overlays import (
    FN, crop_region, disagreement_anchor, failure_candidates, matches, outlined, render, render_composite,
)


def rectangle(shape, rows, columns):
    mask = np.zeros(shape, dtype=np.uint8)
    mask[rows, columns] = 1
    return encode(mask)


def test_overlay_detects_even_split_and_merge():
    shape = (96, 96)
    image = np.full(shape, 127, dtype=np.uint8)
    left = rectangle(shape, slice(40, 50), slice(30, 40))
    right = rectangle(shape, slice(40, 50), slice(40, 50))
    full = rectangle(shape, slice(40, 50), slice(30, 50))
    gt_hit, pred_hit = matches([full], [left, right])
    assert not gt_hit.any() and not pred_hit.any()  # IoU of 0.5 is not a hit.
    assert "split" in {kind for kind, *_ in failure_candidates(image, [full], [left, right])}
    assert "merge" in {kind for kind, *_ in failure_candidates(image, [left, right], [full])}
    assert render(image, [[left], [right]], [left], "Test image").size == (192, 326)


def test_overlay_contours_keep_native_pixels_and_crop_boundaries():
    image = np.full((64, 64), 127, dtype=np.uint8)
    first = rectangle(image.shape, slice(10, 30), slice(10, 30))
    second = rectangle(image.shape, slice(11, 29), slice(28, 38))
    panel = np.asarray(outlined(image, [first], [FN], np.s_[15:35, 10:30], width=1))
    assert panel.shape == (20, 20, 3)
    assert tuple(panel[0, 0]) == FN
    assert tuple(panel[0, 10]) == (127, 127, 127)  # Crop cut through mask interior.
    assert tuple(panel[14, 10]) == FN
    score, box = disagreement_anchor([[first], [second]])
    assert score[0] == 1
    region = crop_region(image.shape, box, size=40)
    assert image[region].shape == (40, 40)
    assert render_composite(image, [[first], [second]], [first], "synthetic", region, proxy=True).size == (84, 198)


def test_limb_fit_and_aperture_profile():
    y, x = np.ogrid[:2048, :2048]
    radius = np.hypot(x - 1014, y - 1029)
    image = np.where(radius <= 922, 160 - 40 * radius / 922, 12).astype(np.uint8)
    disk = fit_limb(image)
    assert (disk.x, disk.y, disk.radius) == pytest.approx((1014, 1029, 922), abs=0.5)
    brightness = profile(image, disk)
    assert brightness["r900_off_disk_fraction"] == 0
    assert 0.05 < brightness["r950_off_disk_fraction"] < 0.07
    assert brightness["median_0_0.2"] > brightness["median_0.9_0.95"]
    with pytest.raises(ValueError, match="full disk"):
        fit_limb(np.full((256, 256), 128, dtype=np.uint8))


def test_limb_fit_ignores_stronger_outer_halo():
    y, x = np.ogrid[:2048, :2048]
    radius = np.hypot(x - 1022, y - 1026)
    image = np.select([radius <= 904, (radius >= 930) & (radius <= 974)], [100, 240], default=45)
    disk = fit_limb(image.astype(np.uint8))
    assert (disk.x, disk.y, disk.radius) == pytest.approx((1022, 1026, 904), abs=0.5)


def test_annotation_stats_pool_pairs_and_exclude_single_record():
    shape = (32, 32)
    shared = rectangle(shape, slice(0, 5), slice(0, 5))
    unique = rectangle(shape, slice(10, 15), slice(0, 5))
    stem = "20260101000000Bh"
    other = "20260102000000Bh"
    draws = {stem: {f"a-{stem}": [shared, unique], f"b-{stem}": [shared]},
             other: {f"a-{other}": [shared]}}
    rows = {(r["section"], r["group"]): r for r in summarize(draws)}
    assert rows["site", "B"]["pq"] == pytest.approx(2 / 3)
    assert rows["year", "2026"]["images"] == 1
    assert rows["singleton", 2]["singleton_share"] == pytest.approx(1 / 3)
    assert rows["annotator", "a"]["mean_per_record"] == 1.5
    assert rows["annotator", "a"]["median_area_px"] == 25
