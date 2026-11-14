"""Measure how a one-pixel change of mask thickness moves PQ for each annotator."""

import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from filseg.data import gt_rles_by_image, load_records
from filseg.metric import encode, score_record
from filseg.rle import counts_to_rle
from tools.cv_map_parity import load_submissions
from tools.figures.instance_errors import day_bootstrap, metric_values, write_table
from tools.figures.missed_instances import crop_mask

CROSS = ndi.generate_binary_structure(2, 1)
EDITS = ("none", "dilate", "erode")


def annotator(record_id):
    """Return the annotator code that prefixes a training record ID."""
    return record_id.split("-", 1)[0]


def edit_masks(masks, edit):
    """Grow or shrink each mask by a 3x3 cross; earlier masks keep contested pixels."""
    if edit not in EDITS:
        raise ValueError(f"Unknown edit {edit}.")
    if edit == "none" or not masks:
        return list(masks)
    morphology = ndi.binary_dilation if edit == "dilate" else ndi.binary_erosion
    taken = np.zeros(masks[0]["size"], dtype=bool)
    edited = []
    for rle in masks:
        mask, region = crop_mask(rle, pad=1)
        mask = morphology(mask, structure=CROSS) & ~taken[region]
        if mask.any():
            taken[region] |= mask
            full = np.zeros(rle["size"], dtype=bool)
            full[region] = mask
            edited.append(encode(full))
    return edited


def style_terms(gt, submission):
    """Accumulate host PQ terms by image, annotator and thickness edit."""
    stems = sorted(gt)
    codes = sorted({annotator(record) for records in gt.values() for record in records})
    index = {code: i for i, code in enumerate(codes)}
    terms = np.zeros((len(stems), len(codes), len(EDITS), 4))
    records = np.zeros(len(codes), dtype=int)
    for row, stem in enumerate(stems):
        masks = [counts_to_rle(counts) for counts in submission.get(stem, [])]
        variants = [edit_masks(masks, edit) for edit in EDITS]
        for record, drawings in gt[stem].items():
            col = index[annotator(record)]
            records[col] += 1
            for k, variant in enumerate(variants):
                t = score_record(drawings, variant)
                terms[row, col, k] += (t.sum_iou, t.tp, t.fp, t.fn)
    return stems, codes, records, terms


def ratio_row(stems, terms, replicates=2000):
    """Report PQ and the edited-to-unedited PQ ratios with paired day intervals."""
    _, draws = day_bootstrap(stems, terms, replicates)
    pq = metric_values(terms.sum(axis=0))[:, 0]
    drawn = metric_values(draws)[:, :, 0]
    row = dict(pq=pq[0])
    for k, edit in enumerate(EDITS[1:], 1):
        valid = drawn[:, 0] > 0
        lo, hi = np.percentile(drawn[valid, k] / drawn[valid, 0], [2.5, 97.5])
        row.update({f"{edit}_ratio": pq[k] / pq[0], f"{edit}_lo": lo, f"{edit}_hi": hi})
    return row


def main():
    """Score frozen held-out masks and their thickness edits on the official training drawings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True, help="Official training annotation JSON.")
    parser.add_argument("--pred", type=Path, required=True, help="Held-out submission CSV on training images.")
    parser.add_argument("--out", type=Path, required=True, help="Per-annotator results CSV.")
    parser.add_argument("--min-records", type=int, default=20, help="Smallest annotator sample to report.")
    args = parser.parse_args()
    gt = gt_rles_by_image(load_records(args.gt))
    stems, codes, records, terms = style_terms(gt, load_submissions([args.pred], gt)[0])
    rows = [dict(annotator="all", records=int(records.sum()), **ratio_row(stems, terms.sum(axis=1)))]
    for col, code in enumerate(codes):
        if records[col] >= args.min_records:
            rows.append(dict(annotator=code, records=int(records[col]), **ratio_row(stems, terms[:, col])))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_table(args.out, rows)
    for row in rows:
        print(f"{row['annotator']:>6} {row['records']:4d} records: PQ {row['pq']:.4f}, "
              f"dilated x{row['dilate_ratio']:.3f}, eroded x{row['erode_ratio']:.3f}")


if __name__ == "__main__":
    main()
