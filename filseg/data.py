"""MAGFiLO annotation loading and rasterisation."""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

H = W = 2048


@dataclass
class Record:
    """One annotator's drawing of one observation."""
    record_id: str          # "<annotator>-<image_stem>"
    image_stem: str         # e.g. "20140609195854Bh"
    polygons: list = field(default_factory=list)   # list of (N,2) float arrays (x, y)
    spines: list = field(default_factory=list)


def _parse(v):
    return json.loads(v) if isinstance(v, str) else v


def load_records(json_path: str | Path) -> dict[str, Record]:
    d = json.load(open(json_path))
    recs: dict[str, Record] = {}
    for im in d["images"]:
        rid = im["id"]
        recs[rid] = Record(rid, Path(im["file_name"]).stem)
    for a in d["annotations"]:
        seg = _parse(a["segmentation"])
        polys = seg if seg and isinstance(seg[0], list) else [seg]
        r = recs[a["image_id"]]
        # one filament may consist of several polygon parts: keep them together
        r.polygons.append([np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in polys])
        sp = _parse(a.get("spine") or "[]")
        r.spines.append(np.asarray(sp, dtype=np.float64).reshape(-1, 2) if sp else np.zeros((0, 2)))
    return recs


def records_by_image(recs: dict[str, Record]) -> dict[str, list[Record]]:
    out = defaultdict(list)
    for r in recs.values():
        out[r.image_stem].append(r)
    return dict(out)


def filament_rle(parts, h: int = H, w: int = W) -> dict:
    """Rasterise one filament exactly as the host does (pycocotools frPyObjects)."""
    from pycocotools import mask as mu
    rl = mu.frPyObjects([p.reshape(-1).tolist() for p in parts if len(p) >= 3], h, w)
    return mu.merge(rl) if len(rl) > 1 else rl[0]


def gt_rles_by_image(recs: dict[str, Record]) -> dict[str, dict[str, list]]:
    """{image_stem: {record_id: [rle per filament]}}"""
    return {stem: {r.record_id: [filament_rle(p) for p in r.polygons] for r in rs}
            for stem, rs in records_by_image(recs).items()}
