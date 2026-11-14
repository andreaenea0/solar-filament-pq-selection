"""Check the sparse Multiscale IoU against a dense port of the reference algorithm."""

import numpy as np
import pytest
from scipy import ndimage as ndi

from filseg.metric import encode
from tools.figures.multiscale_iou import (
    BOX_SIZES, edge_pixels, mask_multiscale_iou, multiscale_iou, rle_edge_pixels, scale_ratios,
)


def dense_reference(reference, detected, box_sizes=BOX_SIZES):
    """Follow the published code: Sobel edges, zero padding at the top and left, box counts plus one."""
    edges = []
    for mask in (reference, detected):
        mask = mask.astype(np.int64)
        gradient = np.hypot(ndi.sobel(mask, axis=0, mode="constant"), ndi.sobel(mask, axis=1, mode="constant"))
        edges.append(gradient > 0)
    ratios = []
    for box in box_sizes:
        grids = []
        for edge in edges:
            padded = np.pad(edge, (((-edge.shape[0]) % box, 0), ((-edge.shape[1]) % box, 0)))
            cells = padded.reshape(padded.shape[0] // box, box, padded.shape[1] // box, box)
            grids.append(cells.any(axis=(1, 3)))
        ratios.append(((grids[0] & grids[1]).sum() + 1) / (grids[0].sum() + 1))
    return float(np.trapezoid(ratios, dx=1 / (len(box_sizes) - 1)))


def blobs(rng, shape):
    """Draw a few smooth random regions, some touching the frame."""
    field = ndi.gaussian_filter(rng.normal(size=shape), 6)
    return field > np.quantile(field, rng.uniform(.7, .95))


@pytest.mark.parametrize("shape", [(128, 128), (150, 203), (2048 // 8, 2048 // 4)])
def test_sparse_value_equals_dense_reference(shape):
    rng = np.random.default_rng(sum(shape))
    for _ in range(6):
        reference, detected = blobs(rng, shape), blobs(rng, shape)
        assert mask_multiscale_iou(reference, detected) == pytest.approx(dense_reference(reference, detected), abs=1e-12)
        shifted = np.roll(reference, rng.integers(1, 5), axis=rng.integers(0, 2))
        assert mask_multiscale_iou(reference, shifted) == pytest.approx(dense_reference(reference, shifted), abs=1e-12)


def test_identical_masks_score_one_and_distant_masks_score_low():
    reference = np.zeros((512, 512), dtype=bool)
    reference[40:60, 30:120] = True
    near = np.roll(reference, 3, axis=0)
    far = np.roll(reference, (400, 300), axis=(0, 1))
    assert mask_multiscale_iou(reference, reference) == pytest.approx(1)
    # The plus-one smoothing of the box counts keeps disjoint small objects well above zero.
    assert .05 < mask_multiscale_iou(reference, np.zeros_like(reference)) < mask_multiscale_iou(reference, far) < .3
    assert mask_multiscale_iou(reference, far) < mask_multiscale_iou(reference, near) < 1
    ratios = scale_ratios(edge_pixels(reference), edge_pixels(far), reference.shape)
    assert ratios[0] < .01 and ratios[-1] == 1


def test_boundary_detail_matters_at_equal_area():
    yy, xx = np.mgrid[:256, :256]
    jagged = (np.abs(yy - 128) < 20 + 8 * np.sign(np.sin(xx / 3))) & (np.abs(xx - 128) < 100)
    smooth = (np.abs(yy - 128) < 20) & (np.abs(xx - 128) < 100)
    assert abs(int(jagged.sum()) - int(smooth.sum())) < .05 * smooth.sum()
    assert mask_multiscale_iou(jagged, jagged) > mask_multiscale_iou(jagged, smooth) + .1


def test_rle_edges_match_full_frame_edges_also_on_the_border():
    mask = np.zeros((2048, 2048), dtype=bool)
    mask[0:12, 100:160] = True
    mask[900:930, 2030:2048] = True
    full = edge_pixels(mask)
    sparse = rle_edge_pixels(encode(mask))
    assert set(map(tuple, sparse)) == set(map(tuple, full))
    assert multiscale_iou(sparse, sparse, mask.shape) == pytest.approx(1)
