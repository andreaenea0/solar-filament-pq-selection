"""Submission CSV encoding (COCO compressed RLE counts, column-major)."""
from __future__ import annotations

import csv
import os

import numpy as np
from pycocotools import mask as mask_util

H = W = 2048


def mask_to_counts(m: np.ndarray) -> str:
    """Encode a two-dimensional mask in COCO column-major order."""
    r = mask_util.encode(np.asfortranarray(m.astype(np.uint8)))
    return r["counts"].decode("ascii")


def counts_to_rle(c: str, h: int = H, w: int = W) -> dict:
    """Wrap compressed counts for the host's fixed image shape."""
    return {"size": [h, w], "counts": c.encode("ascii")}


def counts_to_mask(c: str, h: int = H, w: int = W) -> np.ndarray:
    """Decode the compressed counts to a two-dimensional mask."""
    return mask_util.decode(counts_to_rle(c, h, w))


def write_submission(path, predictions):
    """Write (stem, RLEs) in observation order using the competition CSV format."""
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator=os.linesep)
        writer.writerow(("filament_id", "segmentation_rle"))
        for stem, masks in predictions:
            for index, mask in enumerate(masks, 1):
                writer.writerow((f"{stem}_{index}", mask["counts"].decode("ascii")))
