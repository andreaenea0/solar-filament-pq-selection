"""Candidate filaments from a probability map, organised as a threshold component tree.

Connected regions of p > t are computed for an increasing ladder of thresholds. Every region at a
higher threshold lies inside exactly one region of the level below, so the regions form a tree.
Each region is a candidate filament; a final segmentation is an antichain of this tree (a set of
nested-free, hence pixel-disjoint, regions).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pycocotools import mask as mu
from scipy import ndimage as ndi

H = W = 2048
S8 = np.ones((3, 3), bool)
LADDER = (0.15, 0.25, 0.35, 0.5, 0.65, 0.8)
YY, XX = np.mgrid[:H, :W]
RHO = np.hypot(YY - (H - 1) / 2, XX - (W - 1) / 2) / 900.0   # distance from disk centre in solar radii (approx.)


@dataclass
class Cand:
    level: int
    label: int
    slc: tuple
    feats: dict
    parent: int = -1
    children: list = field(default_factory=list)


def _feats(p, sl, m, level, threshold, img=None):
    pv = p[sl][m].astype(np.float32) / 255.0
    area = int(m.sum())
    ys, xs = np.nonzero(m)
    h, w = m.shape
    if area > 2:
        cov = np.cov(np.stack([ys, xs]).astype(np.float64))
        ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
        elong = float(np.sqrt(max(ev[0], 1e-6) / max(ev[1], 1e-6)))
        length = float(4 * np.sqrt(max(ev[0], 0)))
    else:
        elong, length = 1.0, 1.0
    rho = RHO[sl][m]
    f = dict(level=level, t=threshold, area=area, log_area=np.log1p(area), meanp=float(pv.mean()),
             maxp=float(pv.max()), p25=float(np.quantile(pv, 0.25)), p75=float(np.quantile(pv, 0.75)),
             frac_hi=float((pv > 0.8).mean()), elong=elong, length=length, width=area / max(length, 1.0),
             bbox_h=h, bbox_w=w, fill=area / (h * w), rho_mean=float(rho.mean()), rho_max=float(rho.max()))
    if img is not None:
        iv = img[sl][m].astype(np.float32)
        ring = ndi.binary_dilation(m, S8, iterations=6) & ~m
        rv = img[sl][ring].astype(np.float32) if ring.any() else iv
        f.update(int_mean=float(iv.mean()), contrast=float(rv.mean() - iv.mean()) / max(float(rv.mean()), 1.0))
    return f


def build_tree(p_u8: np.ndarray, img: np.ndarray | None = None, ladder=LADDER, min_area: int = 30,
               pad: int = 8) -> list[Cand]:
    """Build an increasing-threshold forest in level order, then label scan order.

    Each mask uses the strict quantized cutoff ``p_u8 > round(t * 255)``.
    Candidate indices and their feature rows share this order. Supply an
    increasing ladder so each retained component has at most one parent.
    """
    cands: list[Cand] = []
    prev_lab, prev_idx = None, None
    for lv, t in enumerate(ladder):
        lab, n = ndi.label(p_u8 > int(round(t * 255)), structure=S8)
        objs = ndi.find_objects(lab)
        idx = np.full(n + 1, -1, np.int64)
        for c, sl0 in enumerate(objs, 1):
            if sl0 is None:
                continue
            m = lab[sl0] == c
            if m.sum() < min_area:
                continue
            y0 = max(sl0[0].start - pad, 0); y1 = min(sl0[0].stop + pad, H)
            x0 = max(sl0[1].start - pad, 0); x1 = min(sl0[1].stop + pad, W)
            sl = (slice(y0, y1), slice(x0, x1))
            m = lab[sl] == c
            cd = Cand(lv, c, sl, _feats(p_u8, sl, m, lv, t, img))
            if prev_lab is not None:
                yy, xx = np.nonzero(m)
                par_lab = prev_lab[sl][yy[0], xx[0]]
                cd.parent = int(prev_idx[par_lab]) if par_lab > 0 else -1
            idx[c] = len(cands)
            if cd.parent >= 0:
                cands[cd.parent].children.append(len(cands))
            cands.append(cd)
        for cd in cands:
            if cd.level == lv:
                m = np.zeros((H, W), np.uint8)
                m[cd.slc] = lab[cd.slc] == cd.label
                cd.rle = mu.encode(np.asfortranarray(m))
        prev_lab, prev_idx = lab, idx
    return cands
