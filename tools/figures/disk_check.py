"""Summarize limb fits and radial display brightness on competition JPEGs."""

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from filseg.disk import fit_limb, profile


def main():
    """Write per-image diagnostics and print robust summaries across fits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows, failures = [], []
    for path in sorted(args.images.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg"}:
            continue
        with Image.open(path) as jpeg:
            image = np.asarray(jpeg.convert("L"))
        try:
            disk = fit_limb(image)
        except ValueError as exc:
            failures.append((path.stem, str(exc)))
            continue
        radial = profile(image, disk)
        centre = radial["median_0_0.2"]
        edge = radial["median_0.9_0.95"]
        offset = np.hypot(disk.x - (image.shape[1] - 1) / 2, disk.y - (image.shape[0] - 1) / 2)
        rows.append(dict(stem=path.stem, **asdict(disk), offset=offset,
                         edge_to_centre=edge / centre if centre else np.nan, **radial))
    for stem, error in failures:
        print(f"Fit failed for {stem}. {error}")
    if not rows:
        parser.error("No usable limb fits.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} fits, {len(failures)} rejected. Median and 5th–95th percentiles")
    for key in list(rows[0])[1:]:
        low, median, high = np.nanpercentile([row[key] for row in rows], [5, 50, 95])
        print(f"{key:32s} {median:9.4f}  [{low:.4f}, {high:.4f}]")
    q1, q3 = np.percentile([row["radius"] for row in rows], [25, 75])
    print(f"Radius IQR {q1:.3f}–{q3:.3f} px")


if __name__ == "__main__":
    main()
