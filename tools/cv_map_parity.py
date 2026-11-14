r"""Compare matched-size and full map ensembles on the same held-out images.

Pre-averaged <stem>.npz, key p, uint8 [H,W] or [C,H,W]:
    python tools/cv_map_parity.py GT.pkl --matched MATCHED_DIR --full FULL_DIR --out parity.csv

Raw held-out members (.npy or .npz key p, uint8 [H,W]):
    python tools/cv_map_parity.py GT.pkl --members members.json --counts 1,2 --out parity.csv
The manifest is {stem: [[ordered paths for family 0], [paths for family 1], ...]}.
Paths can be relative to the manifest. Count order must match family order.
Matched averages use the first N members of each family. Full uses all members.
Raw input rounds each family mean to uint8 before equally averaging channels, as
in k02. Pass --unrounded for the k07 keep rule's float32 checkpoint average.

GT.pkl is a trusted training pickle {stem: {record_id: [COCO RLE]}}. By default
all GT stems are required in both map recipes. --stems selects a common subset.
Use members trained without the scored images in both recipes. The script can
compare alternative selected masks via --submissions MATCHED.csv FULL.csv. The
peak gate is always measured from the input maps using the keep-rule cutoffs.
"""

import argparse
import csv
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filseg.maps import channel_mean, matched_family_maps, read_map
from filseg.metric import IOU_T, Totals, iou_matrix, score_image
from filseg.rle import counts_to_rle
from filseg.rule import HIGH, LOW, MIN_AREA, MIN_MEAN, components, kept_masks


def component_matches(left, right, left_components, right_components):
    """Find one-to-one IoU > 0.5 matches using intersecting label pairs."""
    overlap = (left > 0) & (right > 0)
    stride = len(right_components) + 1
    pairs, size = np.unique(left[overlap].astype(np.int64) * stride + right[overlap], return_counts=True)
    li, ri = np.divmod(pairs, stride)
    li, ri = li - 1, ri - 1
    left_area = np.array([c.area for c in left_components])
    right_area = np.array([c.area for c in right_components])
    hits = size / (left_area[li] + right_area[ri] - size) > IOU_T
    return li[hits], ri[hits]


def compare_image(stem, matched, full, records, thresholds, selections=None):
    """Return topology, peak crossings and paired selected-mask PQ terms."""
    if matched.ndim != 2 or matched.shape != full.shape:
        raise ValueError(f"{stem}: the two map recipes must have the same [H,W] shape.")
    low, high, area, mean = thresholds
    left, mc = components(matched, low)
    right, fc = components(full, low)
    mi, fi = component_matches(left, right, mc, fc)
    mp = np.array([c.peak > high for c in mc], dtype=bool)
    fp = np.array([c.peak > high for c in fc], dtype=bool)
    if selections is None:
        masks = (kept_masks(left, mc, high, area, mean), kept_masks(right, fc, high, area, mean))
    else:
        masks = tuple([counts_to_rle(counts, *matched.shape) for counts in s.get(stem, [])] for s in selections)
        if any(np.any(np.triu(iou_matrix(pred, pred), k=1) > 0) for pred in masks):
            raise ValueError(f"{stem}: selected masks in each submission must be pixel-disjoint.")
    overlap = iou_matrix(*masks)
    shared = overlap[overlap > IOU_T]
    row = dict(stem=stem, matched_components=len(mc), full_components=len(fc), shared_components=len(mi),
               peak_pass_to_fail=np.count_nonzero(mp[mi] & ~fp[fi]),
               peak_fail_to_pass=np.count_nonzero(~mp[mi] & fp[fi]),
               unmatched_matched_peak_pass=mp.sum() - mp[mi].sum(),
               unmatched_full_peak_pass=fp.sum() - fp[fi].sum(),
               matched_masks=len(masks[0]), full_masks=len(masks[1]), shared_masks=len(shared),
               shared_mask_iou_sum=shared.sum())
    for name, pred in zip(("matched", "full"), masks):
        total = score_image(list(records.values()), pred)
        row.update({f"{name}_{key}": value for key, value in vars(total).items()})
        row[f"{name}_pq"] = total.pq
    row["delta_pq"] = row["full_pq"] - row["matched_pq"]
    return row


def load_submissions(paths, known_stems):
    """Read optional selected masks in the standard submission CSV format."""
    selections = []
    for path in paths:
        by_image, seen = defaultdict(list), set()
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or not {"filament_id", "segmentation_rle"} <= set(reader.fieldnames):
                raise ValueError(f"{path}: expected filament_id and segmentation_rle columns.")
            for row in reader:
                identifier = row["filament_id"]
                stem, _, index = identifier.rpartition("_")
                if stem not in known_stems or not index.isdecimal() or int(index) < 1 or identifier in seen:
                    raise ValueError(f"{path}: unexpected or duplicate filament ID {identifier}.")
                seen.add(identifier)
                by_image[stem].append(row["segmentation_rle"])
        selections.append(by_image)
    return selections


def paired_interval(rows, replicates):
    """Bootstrap the pooled PQ difference by jointly resampling observing days."""
    days = defaultdict(lambda: np.zeros((2, 2)))
    for row in rows:
        terms = days[row["stem"][:8]]
        for i, name in enumerate(("matched", "full")):
            terms[i, 0] += row[f"{name}_sum_iou"]
            terms[i, 1] += row[f"{name}_tp"] + 0.5 * (row[f"{name}_fp"] + row[f"{name}_fn"])
    n = len(days)
    if n < 2:
        return None
    samples = np.random.default_rng(0).multinomial(n, np.full(n, 1 / n), size=replicates)
    totals = np.tensordot(samples, np.stack(list(days.values())), axes=1)
    numerator, denominator = totals[..., 0], totals[..., 1]
    pq = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    return np.quantile(pq[:, 1] - pq[:, 0], (0.025, 0.975))


def report(rows, bootstrap):
    """Print pooled PQ and aggregate component and mask changes."""
    sums = {key: sum(row[key] for row in rows) for key in rows[0] if key != "stem" and not key.endswith("_pq")}
    matched, full = [Totals(**{key: sums[f"{name}_{key}"] for key in vars(Totals())}) for name in ("matched", "full")]
    print(f"{len(rows)} paired images. PQ matched {matched.pq:.4f}, full {full.pq:.4f}, delta {full.pq - matched.pq:+.4f}")
    interval = paired_interval(rows, bootstrap) if bootstrap else None
    if interval is not None:
        print(f"95% paired day-bootstrap interval for delta [{interval[0]:+.4f}, {interval[1]:+.4f}]")
    print("Components matched/full/shared:", *(sums[f"{key}_components"] for key in ("matched", "full", "shared")))
    print("Peak pass to fail / fail to pass:", sums["peak_pass_to_fail"], sums["peak_fail_to_pass"])
    print("Unmatched peak passes, matched/full:", sums["unmatched_matched_peak_pass"], sums["unmatched_full_peak_pass"])
    print("Selected masks matched/full/shared:", *(sums[f"{key}_masks"] for key in ("matched", "full", "shared")))
    if sums["shared_masks"]:
        print(f"Shared-mask mean IoU {sums['shared_mask_iou_sum'] / sums['shared_masks']:.4f}")


def main():
    """Run a paired map comparison and optionally save per-image diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gt", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--matched", type=Path)
    source.add_argument("--members", type=Path)
    parser.add_argument("--full", type=Path)
    parser.add_argument("--counts", help="Comma-separated matched member counts by family.")
    parser.add_argument("--unrounded", action="store_true", help="Retain raw member means as float32.")
    parser.add_argument("--stems", type=Path, help="Whitespace-separated held-out subset.")
    parser.add_argument("--submissions", nargs=2, type=Path, metavar=("MATCHED", "FULL"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000, help="Paired day draws, or zero to disable.")
    parser.add_argument("--low", type=float, default=LOW)
    parser.add_argument("--high", type=float, default=HIGH)
    parser.add_argument("--min-area", type=int, default=MIN_AREA)
    parser.add_argument("--min-mean", type=float, default=MIN_MEAN)
    args = parser.parse_args()
    if (args.members and (args.full or not args.counts)) or (args.matched and (not args.full or args.counts or args.unrounded)):
        parser.error("Use --members with --counts or --matched with --full.")
    if args.bootstrap < 0:
        parser.error("Bootstrap count must be nonnegative.")

    try:
        with args.gt.open("rb") as stream:
            gt = pickle.load(stream)
        stems = set(args.stems.read_text().split()) if args.stems else set(gt)
        if not stems or stems - gt.keys():
            raise ValueError("The held-out subset must be nonempty and present in GT.")
        selections = load_submissions(args.submissions, gt) if args.submissions else None
        manifest = json.loads(args.members.read_text(encoding="utf-8")) if args.members else None
        counts = list(map(int, args.counts.split(","))) if args.counts else None
        thresholds = (args.low, args.high, args.min_area, args.min_mean)
        rows = []
        for stem in sorted(stems):
            if manifest is None:
                maps = [read_map(directory / f"{stem}.npz") for directory in (args.matched, args.full)]
                if any(p.dtype != np.uint8 or p.ndim not in (2, 3) or not p.size for p in maps):
                    raise ValueError(f"{stem}: expected uint8 [H,W] or [C,H,W] maps.")
            else:
                families = [[read_map(args.members.parent / path) for path in family] for family in manifest[stem]]
                maps = [matched_family_maps(families, n, quantize=not args.unrounded)
                        for n in (counts, [len(f) for f in families])]
            matched, full = (channel_mean(p) / 255 for p in maps)
            rows.append(compare_image(stem, matched, full, gt[stem], thresholds, selections))
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))

    report(rows, args.bootstrap)
    if args.out:
        with args.out.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
