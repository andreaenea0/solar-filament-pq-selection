"""Candidate extraction + per-candidate targets on out-of-fold probability maps."""
from __future__ import annotations

import numpy as np
from pycocotools import mask as mu

from .candidates import build_tree


def candidate_targets(cands, gt_by_record: dict) -> tuple[np.ndarray, np.ndarray]:
    """g = mean over annotator records of IoU * 1[IoU > 0.5]; also the hit rate."""
    rles = [c.rle for c in cands]
    R = len(gt_by_record)
    g = np.zeros(len(cands)); hr = np.zeros(len(cands))
    for gts in gt_by_record.values():
        if not gts or not rles:
            continue
        iou = np.asarray(mu.iou(rles, gts, [0] * len(gts))).reshape(len(rles), len(gts))
        best = iou.max(1)
        g += np.where(best > 0.5, best, 0); hr += best > 0.5
    return g / max(R, 1), hr / max(R, 1)


def candidate_record(stem, maps, image, gt_by_record=None, channel="mean"):
    """Build the candidate pickle record used for training and inference."""
    families = maps if maps.ndim == 3 else None
    if families is not None:
        maps = (families.astype(np.float32).mean(0).round().astype(np.uint8)
                if channel == "mean" else families[int(channel)])
    cands = build_tree(maps, image)
    # Only the image itself is used at inference: the host forbids metadata parsed from the file name
    # (observing time, site), so `stem` is used for bookkeeping only, never as a feature.
    img_mean = float(image.mean())
    for c in cands:
        c.feats.update(n_cands=len(cands), img_mean=img_mean)
        if families is not None and len(families) >= 2:
            mask = mu.decode(c.rle).astype(bool)
            v0 = families[0][mask].astype(np.float32) / 255
            v1 = families[1][mask].astype(np.float32) / 255
            c.feats.update(m0=float(v0.mean()), m1=float(v1.mean()),
                           dfam=float(abs(v0.mean() - v1.mean())), min_fam=float(min(v0.mean(), v1.mean())),
                           hi0=float((v0 > 0.5).mean()), hi1=float((v1 > 0.5).mean()))
    g, hr = candidate_targets(cands, gt_by_record) if gt_by_record is not None else (None, None)
    return dict(feats=[c.feats for c in cands], parent=[c.parent for c in cands],
                children=[c.children for c in cands], level=[c.level for c in cands],
                rle=[c.rle for c in cands], g=g, hr=hr,
                R=len(gt_by_record) if gt_by_record is not None else None)
