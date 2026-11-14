"""Validate a Solar Filament submission against the official test JPEG stems."""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filseg.metric import iou_matrix
from filseg.rle import H, W, counts_to_rle

FIELDS = {"filament_id", "segmentation_rle"}


def foreground_area(counts, pixels):
    """Check COCO signed delta runs before passing untrusted counts to the C decoder."""
    runs = []
    offset = total = area = 0
    while offset < len(counts):
        value = 0
        for group in range(6):
            if offset == len(counts):
                raise ValueError("Truncated compressed RLE.")
            digit = ord(counts[offset]) - 48
            offset += 1
            if not 0 <= digit <= 63:
                raise ValueError("Invalid compressed RLE character.")
            value |= (digit & 31) << (5 * group)
            if not digit & 32:
                if digit & 16:
                    value -= 1 << (5 * (group + 1))
                break
        else:
            raise ValueError("Overlong compressed RLE run.")
        if len(runs) > 2:
            value += runs[-2]
        total += value
        if value < 0 or total > pixels:
            raise ValueError("Invalid compressed RLE run length.")
        if len(runs) % 2:
            area += value
        runs.append(value)
    if total != pixels:
        raise ValueError(f"RLE covers {total} pixels, expected {pixels}.")
    return area


def check_submission(path, stems, shape=(H, W), *, allow_empty_images=False):
    """Check mask validity and require image coverage unless explicitly relaxed."""
    if not stems:
        raise ValueError("The expected test stem set is empty.")
    seen, by_image = set(), defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if not reader.fieldnames or len(reader.fieldnames) != 2 or set(reader.fieldnames) != FIELDS:
            raise ValueError("Expected the filament_id and segmentation_rle columns.")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Line {reader.line_num}: expected two CSV fields.")
            identifier, counts = row["filament_id"], row["segmentation_rle"]
            stem, separator, suffix = identifier.rpartition("_")
            if not separator or stem not in stems or not re.fullmatch(r"0*[1-9][0-9]*", suffix):
                raise ValueError(f"Line {reader.line_num}: unexpected ID {identifier!r}.")
            if identifier in seen:
                raise ValueError(f"Duplicate filament ID {identifier!r}.")
            seen.add(identifier)
            try:
                area = foreground_area(counts, shape[0] * shape[1])
                if area == 0:
                    raise ValueError("Empty mask.")
                rle = counts_to_rle(counts, *shape)
                mask = mu.decode(rle)
                if mask.shape != shape or not mask.flags.f_contiguous or mask.sum() != area:
                    raise ValueError("COCO decode did not produce the expected column-major mask.")
            except ValueError as exc:
                raise ValueError(f"{identifier}: {exc}") from exc
            by_image[stem].append((identifier, rle))

    for rows in by_image.values():
        ids = [identifier for identifier, _ in rows]
        rles = [rle for _, rle in rows]
        overlap = np.triu(iou_matrix(rles, rles), 1)
        pairs = np.argwhere(overlap > 0)
        if len(pairs):
            i, j = pairs[0]
            problem = "Duplicate masks" if overlap[i, j] == 1 else "Overlapping masks"
            raise ValueError(f"{problem}: {ids[i]} and {ids[j]}.")
    missing = stems - by_image.keys()
    if missing and not allow_empty_images:
        raise ValueError(f"Validated {len(seen)} masks across {len(by_image)} images. "
                         f"{len(missing)} test stems have no rows: {', '.join(sorted(missing)[:8])}.")
    return len(seen), len(by_image)


def main():
    """Check the CSV without loading any test labels or model checkpoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Submission CSV to validate.")
    parser.add_argument("--images", type=Path, required=True, help="Official test JPEG directory.")
    parser.add_argument("--allow-empty-images", action="store_true",
                        help="Allow images with zero predictions, as the host scorer does.")
    args = parser.parse_args()
    csv.field_size_limit(max(csv.field_size_limit(), 6 * (H * W + 1)))
    try:
        stems = [p.stem for p in args.images.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg")]
        if len(stems) != len(set(stems)):
            raise ValueError("The test JPEG directory contains duplicate stems.")
        masks, images = check_submission(args.csv, set(stems), allow_empty_images=args.allow_empty_images)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"Valid submission: {masks} disjoint nonempty masks across {images} test images.")
    if images != len(stems):
        print(f"{len(stems) - images} of {len(stems)} test images have zero prediction rows.")


if __name__ == "__main__":
    main()
