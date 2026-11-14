"""Write report counts, sizes, agreement and singleton shares from training GT."""

import argparse
import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

from filseg.data import gt_rles_by_image, load_records
from filseg.metric import IOU_T, Totals, iou_matrix, score_record


def summarize(by_image):
    """Yield per-annotator summaries, pooled paired PQ and unmatched drawing shares."""
    counts, sizes = defaultdict(list), defaultdict(list)
    agreement, images = defaultdict(Totals), defaultdict(int)
    singleton_images, singletons = defaultdict(int), defaultdict(int)
    for stem, drawings in sorted(by_image.items()):
        entries = sorted(drawings.items())
        masks = [rles for _, rles in entries]
        observed = [np.zeros(len(rles), dtype=bool) for rles in masks]
        for record, rles in entries:
            annotator, separator, suffix = record.rpartition("-")
            if not separator or suffix != stem:
                raise ValueError(f"Unexpected annotation record ID {record!r} for {stem}.")
            counts[annotator].append(len(rles))
            sizes[annotator].extend(mu.area(rles))
        if len(masks) < 2:
            continue
        groups = (("site", stem[14:15] or "unknown"), ("year", stem[:4]), ("all", "all"))
        for group in groups:
            images[group] += 1
        for left, right in combinations(range(len(masks)), 2):
            hits = iou_matrix(masks[left], masks[right]) > IOU_T
            observed[left] |= hits.any(1)
            observed[right] |= hits.any(0)
            pair = score_record(masks[left], masks[right])
            for group in groups:
                agreement[group] += pair
        singleton_images[len(masks)] += 1
        singletons[len(masks), "total"] += sum(map(len, masks))
        singletons[len(masks), "singleton"] += sum(int((~hit).sum()) for hit in observed)

    for annotator, values in sorted(counts.items()):
        areas = sizes[annotator]
        yield dict(section="annotator", group=annotator, records=len(values), filaments=sum(values),
                   mean_per_record=np.mean(values), median_per_record=np.median(values),
                   median_area_px=np.median(areas) if areas else "",
                   p25_area_px=np.percentile(areas, 25) if areas else "",
                   p75_area_px=np.percentile(areas, 75) if areas else "")
    for (section, group), total in sorted(agreement.items()):
        yield dict(section=section, group=group, images=images[section, group], pq=total.pq,
                   sum_iou=total.sum_iou, tp=total.tp, fp_plus_fn=total.fp + total.fn)
    numbers = sorted({n for n, _ in singletons})
    for n in [*numbers, "all"]:
        keys = numbers if n == "all" else [n]
        total = sum(singletons[k, "total"] for k in keys)
        count = sum(singletons[k, "singleton"] for k in keys)
        yield dict(section="singleton", group=n, images=sum(singleton_images[k] for k in keys),
                   filaments=total, singleton_drawings=count, singleton_share=count / total if total else "")


def main():
    """Export one CSV with section and group columns."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True, help="MAGFiLO training annotation JSON.")
    parser.add_argument("--out", type=Path, default=Path("data/annotation_stats.csv"))
    args = parser.parse_args()
    rows = list(summarize(gt_rles_by_image(load_records(args.gt))))
    columns = list(dict.fromkeys(key for row in rows for key in row))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
