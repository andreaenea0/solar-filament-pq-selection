"""The keep-rule baseline on a probability map in [0, 1]."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from .metric import encode

LOW, HIGH, MIN_AREA, MIN_MEAN = 0.35, 0.99, 200, 0.85
CONNECTIVITY = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True)
class Component:
    """A component of the low-threshold map and its rule features."""

    label: int
    bounds: tuple[slice, slice]
    area: int
    peak: np.float32
    mean: np.float32


def components(probability, low=LOW):
    """Return 8-connected labels and component features in scan order."""
    labels, _ = ndi.label(probability > low, structure=CONNECTIVITY)
    found = []
    for label, bounds in enumerate(ndi.find_objects(labels), 1):
        values = probability[bounds][labels[bounds] == label]
        found.append(Component(label, bounds, len(values), values.max(), values.mean()))
    return labels, found


def kept_masks(labels, found, high=HIGH, min_area=MIN_AREA, min_mean=MIN_MEAN):
    """Apply the existing strict peak and inclusive area/mean cutoffs."""
    masks = []
    for c in found:
        if c.peak > high and c.area >= min_area and c.mean >= min_mean:
            mask = np.zeros(labels.shape, dtype=np.uint8)
            mask[c.bounds] = labels[c.bounds] == c.label
            masks.append(encode(mask))
    return masks


def keep_rule(probability, low=LOW, high=HIGH, min_area=MIN_AREA, min_mean=MIN_MEAN):
    """Return COCO RLEs for the disjoint components kept by the baseline."""
    return kept_masks(*components(probability, low), high, min_area, min_mean)
