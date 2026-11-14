"""Exact tree selection for an additive predicted-PQ-gain surrogate.

When each prediction has at most one strict hit per drawing, adding a
pixel-disjoint candidate c to an image annotated by R annotators changes the pooled PQ terms by
    numerator   += R * g_c,      g_c = mean_r IoU_r(c) * 1[IoU_r(c) > 0.5]
    denominator += R * 0.5       (each record gets either a TP that removes a FN, or a FP)
so c raises PQ iff g_c > PQ / 2. Overlapping GT instances can violate
that one-hit condition; evaluation always uses the exact host scorer.
We learn g_c from candidate features and choose, per image, the antichain
maximising sum(g_hat - lam). See docs/SELECTION_THEORY.md for assumptions,
the fractional objective, record weights and the multiple-hit exception.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATS = ["level", "t", "area", "log_area", "meanp", "maxp", "p25", "p75", "frac_hi", "elong", "length",
         "width", "bbox_h", "bbox_w", "fill", "rho_mean", "rho_max", "int_mean", "contrast",
         "n_children", "parent_meanp", "parent_area_ratio", "depth_below"]


EXTRA = ["n_cands", "img_mean", "m0", "m1", "dfam", "min_fam", "hi0", "hi1"]


def feature_frame(c: dict) -> pd.DataFrame:
    """Return image-derived booster features in candidate-row order."""
    if not c["feats"]:
        return pd.DataFrame(columns=FEATS + EXTRA, dtype=float)
    df = pd.DataFrame(c["feats"])
    par = np.array(c["parent"]); kids = c["children"]
    df["n_children"] = [len(k) for k in kids]
    pm = np.array([c["feats"][q]["meanp"] if q >= 0 else np.nan for q in par])
    pa = np.array([c["feats"][q]["area"] if q >= 0 else np.nan for q in par])
    df["parent_meanp"] = pm
    df["parent_area_ratio"] = df["area"].values / pa
    depth = np.zeros(len(par))
    for i in sorted(range(len(par)), key=lambda i: -c["level"][i]):
        depth[i] = 0 if not kids[i] else 1 + max(depth[k] for k in kids[i])
    df["depth_below"] = depth
    extra = [k for k in EXTRA if k in df.columns]
    return df[FEATS + extra]


def dp_select(parent, children, level, gain) -> list[int]:
    """Maximize supplied node weights over a forest's antichains, allowing empty output.

    Parents must have lower levels than their children. Positive ties
    prefer the parent; zero-valued branches remain empty.
    """
    n = len(parent); best = np.zeros(n); choice = [[] for _ in range(n)]
    for i in sorted(range(n), key=lambda i: -level[i]):
        sub = sum(best[k] for k in children[i])
        if gain[i] > 0 and gain[i] >= sub:
            best[i], choice[i] = gain[i], [i]
        elif sub > 0:
            best[i], choice[i] = sub, [j for k in children[i] for j in choice[k]]
    return [j for i in range(n) if parent[i] < 0 for j in choice[i]]


def learned_masks(candidates, model, penalty, verifier=None, *, alpha=0.5, trace=None):
    """Predict candidate gains and return the selected disjoint RLEs."""
    if not candidates["rle"]:
        if trace is not None:
            trace.update(lgb_max=None, verifier_max=None, blended_max=None, above_penalty=0, selected=0, top=None)
        return []
    frame = feature_frame(candidates)
    columns = model.feature_name()
    missing = [name for name in columns if name not in frame]
    if missing:
        raise ValueError(f"Selector expects unavailable features: {', '.join(missing)}.")
    lgb_gains = model.predict(frame[columns])
    gains = lgb_gains
    if verifier is not None:
        if verifier.shape != gains.shape or not np.isfinite(verifier).all():
            raise ValueError("Verifier gains must align with candidate rows and be finite.")
        gains = alpha * lgb_gains + (1 - alpha) * verifier
    indices = dp_select(candidates["parent"], candidates["children"], candidates["level"], gains - penalty)
    if trace is not None:
        top = int(gains.argmax())
        trace.update(
            lgb_max=float(lgb_gains.max()),
            verifier_max=None if verifier is None else float(verifier.max()),
            blended_max=float(gains[top]),
            above_penalty=int((gains > penalty).sum()),
            selected=len(indices),
            top=dict(level=int(candidates["level"][top]), area_px=int(frame.iloc[top]["area"]),
                     lgb=float(lgb_gains[top]), verifier=None if verifier is None else float(verifier[top])),
        )
    return [candidates["rle"][i] for i in indices]
