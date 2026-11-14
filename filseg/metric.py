"""Pooled multi-annotator Panoptic Quality, replicating the host's self-evaluation notebook.

For each annotator record of an image, every predicted mask of that image is compared with
that annotator's filaments. A pair is a hit when IoU > 0.5. TP IoUs, FP (predictions hitting
nothing) and FN (GT hitting nothing) are pooled over all records of all images:

    PQ = sum(IoU_TP) / (|TP| + 0.5 |FP| + 0.5 |FN|)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pycocotools import mask as mask_util

IOU_T = 0.5


@dataclass
class Totals:
    sum_iou: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __iadd__(self, o: "Totals"):
        self.sum_iou += o.sum_iou; self.tp += o.tp; self.fp += o.fp; self.fn += o.fn
        return self

    @property
    def pq(self) -> float:
        d = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.sum_iou / d if d > 0 else 0.0

    @property
    def sq(self) -> float:
        return self.sum_iou / self.tp if self.tp else 0.0

    @property
    def rq(self) -> float:
        d = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.tp / d if d > 0 else 0.0


def encode(m: np.ndarray) -> dict:
    return mask_util.encode(np.asfortranarray(m.astype(np.uint8)))


def iou_matrix(gt_rles: list, pred_rles: list) -> np.ndarray:
    if not gt_rles or not pred_rles:
        return np.zeros((len(gt_rles), len(pred_rles)))
    return np.asarray(mask_util.iou(gt_rles, pred_rles, [0] * len(pred_rles)), dtype=np.float64).reshape(
        len(gt_rles), len(pred_rles))


def score_record(gt_rles: list, pred_rles: list) -> Totals:
    t = Totals()
    if not gt_rles:
        t.fp = len(pred_rles); return t
    if not pred_rles:
        t.fn = len(gt_rles); return t
    iou = iou_matrix(gt_rles, pred_rles)
    hit = iou > IOU_T
    t.sum_iou = float(iou[hit].sum()); t.tp = int(hit.sum())
    t.fp = int((hit.sum(0) == 0).sum()); t.fn = int((hit.sum(1) == 0).sum())
    return t


def score_image(gt_records: list[list], pred_rles: list) -> Totals:
    """gt_records: one list of GT RLEs per annotator record of this image."""
    t = Totals()
    for g in gt_records:
        t += score_record(g, pred_rles)
    return t
