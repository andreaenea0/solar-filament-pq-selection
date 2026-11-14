"""Compare the solar geometry of held-out predicted masks with the training annotations."""

import argparse
import csv
from pathlib import Path

import numpy as np

from filseg.data import gt_rles_by_image, load_records
from filseg.disk import Disk
from filseg.rle import counts_to_rle
from tools.cv_map_parity import load_submissions
from tools.figures.filament_physics import measure_mask, quantiles, solar_b0, summarize

FIELDS = ("rho", "latitude", "tilt", "area_px", "area_msh")


def as_drawings(stems, submission):
    """Present one submission as a single drawing per image, empty images included."""
    return {stem: {"prediction": [counts_to_rle(counts) for counts in submission.get(stem, [])]} for stem in stems}


def pooled_rows(measurements):
    """Summarize the tilt and size cuts of the bin tables over all latitudes and years."""
    tilts = [m.tilt for m in measurements if m.tilt is not None]
    central = [m for m in measurements if m.area_msh is not None]
    yield dict(section="tilt_all", filaments=len(measurements), eligible=len(tilts), **quantiles(tilts, "tilt_deg"))
    yield dict(section="size_all", filaments=len(measurements), eligible=len(central),
               **quantiles([m.area_px for m in central], "area_px"),
               **quantiles([m.area_msh for m in central], "area_msh"))


def compare(sources, disks, b0_angles):
    """Return the aggregate rows of every source and its per-instance measurements.

    `sources` maps a name to {stem: {drawing: [rle]}}. Densities are per drawing: one annotator
    record for the annotations and one image for a submission.
    """
    rows, instances = [], []
    for name, drawings in sources.items():
        measured = [(stem, measure_mask(rle, disks[stem], b0_angles[stem]))
                    for stem, records in drawings.items() for rles in records.values() for rle in rles]
        aggregate = list(summarize(drawings, disks, b0_angles)) + list(pooled_rows([m for _, m in measured]))
        rows.extend(dict(source=name, **row) for row in aggregate)
        instances.extend(dict(source=name, stem=stem, **{field: getattr(m, field) for field in FIELDS})
                         for stem, m in measured)
    return rows, instances


def write_rows(path, rows):
    """Write rows with a stable column order and compact numbers."""
    columns = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: "" if v is None else f"{v:.6g}" if isinstance(v, (float, np.floating)) else v
                          for k, v in row.items()} for row in rows)


def main():
    """Apply the physics cuts of filament_physics.py to the annotations and to each named submission."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True, help="Official training annotation JSON.")
    parser.add_argument("--disks", type=Path, required=True, help="CSV from tools.figures.disk_check on training JPEGs.")
    parser.add_argument("--pred", nargs="+", required=True, metavar="NAME=CSV",
                        help="Held-out submission CSVs on training images, each with a short model name.")
    parser.add_argument("--out", type=Path, required=True, help="Aggregate comparison CSV.")
    parser.add_argument("--instances", type=Path, help="Optional per-instance measurements for figures.")
    args = parser.parse_args()
    named = dict(item.split("=", 1) for item in args.pred)
    gt = gt_rles_by_image(load_records(args.gt))
    stems = sorted(gt)
    with args.disks.open(encoding="utf-8", newline="") as stream:
        disks = {r["stem"]: Disk(float(r["x"]), float(r["y"]), float(r["radius"]), float(r["rms"]), int(r["rays"]))
                 for r in csv.DictReader(stream)}
    if gt.keys() - disks.keys():
        parser.error(f"Missing disk fits for {len(gt.keys() - disks.keys())} annotated images.")
    sources = {"annotations": gt}
    for name, submission in zip(named, load_submissions([Path(path) for path in named.values()], gt)):
        sources[name] = as_drawings(stems, submission)
    rows, instances = compare(sources, disks, solar_b0(stems))
    write_rows(args.out, rows)
    if args.instances:
        write_rows(args.instances, instances)
    print(f"Wrote {len(rows)} aggregate rows for {len(sources)} sources to {args.out}")


if __name__ == "__main__":
    main()
