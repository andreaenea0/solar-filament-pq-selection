"""Multiscale IoU of Ahmadzadeh et al. (ICIP 2021), computed on sparse boundary pixels.

The reference implementation is https://bitbucket.org/gsudmlab/multiscale_iou (MIT licence).
It takes the Sobel edges of the ground-truth and detected masks, covers both with square
grids of side 1, 2, 4, ..., 512 pixels, and at each scale divides the number of cells that
contain both boundaries, plus one, by the number of cells that contain the ground-truth
boundary, plus one. MIoU is the trapezoidal mean of the ten ratios. This module returns the
same value without building full-frame arrays.
"""

import numpy as np
from pycocotools import mask as mu
from scipy import ndimage as ndi

BOX_SIZES = tuple(2 ** n for n in range(10))


def edge_pixels(mask, origin=(0, 0), frame=None):
    """Return in-frame coordinates where the Sobel gradient of a binary mask is non-zero.

    `mask` is the whole frame, or a crop at `origin` of a `frame` that is zero elsewhere.
    """
    padded = np.pad(np.asarray(mask, dtype=np.int32), 1)
    gradient = np.hypot(ndi.sobel(padded, axis=0, mode="constant"), ndi.sobel(padded, axis=1, mode="constant"))
    pixels = np.argwhere(gradient > 0) + (origin[0] - 1, origin[1] - 1)
    frame = np.shape(mask) if frame is None else frame
    return pixels[(pixels >= 0).all(axis=1) & (pixels < frame).all(axis=1)]


def rle_edge_pixels(rle):
    """Decode only the bounding box of a COCO RLE and return its edge pixels."""
    x, y, width, height = map(int, mu.toBbox(rle))
    return edge_pixels(mu.decode(rle)[y:y + height, x:x + width], (y, x), rle["size"])


def occupied_cells(pixels, shape, box):
    """Return sorted unique grid cells; like the reference, the grid is padded at the top and left."""
    rows = (pixels[:, 0] + (-shape[0]) % box) // box
    cols = (pixels[:, 1] + (-shape[1]) % box) // box
    return np.unique(rows * (shape[1] // box + 2) + cols)


def scale_ratios(reference, detected, shape, box_sizes=BOX_SIZES):
    """Return the per-scale ratio of shared boundary cells to ground-truth boundary cells."""
    ratios = []
    for box in box_sizes:
        first = occupied_cells(reference, shape, box)
        second = occupied_cells(detected, shape, box)
        shared = np.intersect1d(first, second, assume_unique=True)
        ratios.append((len(shared) + 1) / (len(first) + 1))
    return np.array(ratios)


def multiscale_iou(reference, detected, shape, box_sizes=BOX_SIZES):
    """Return MIoU of a detected boundary against a ground-truth boundary, both as edge pixels."""
    ratios = scale_ratios(reference, detected, shape, box_sizes)
    return float(np.trapezoid(ratios, dx=1 / (len(box_sizes) - 1)))


def mask_multiscale_iou(reference, detected, box_sizes=BOX_SIZES):
    """Return MIoU for two binary masks of the same shape, the reference being the ground truth."""
    reference, detected = np.asarray(reference), np.asarray(detected)
    if reference.shape != detected.shape:
        raise ValueError("The two masks must share one frame.")
    return multiscale_iou(edge_pixels(reference), edge_pixels(detected), reference.shape, box_sizes)
