"""Measure radial density, heliographic tilt and yearly size from training GT."""

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

from filseg.data import gt_rles_by_image, load_records
from filseg.disk import Disk

RADIAL_BINS = (0, .2, .4, .6, .8, .9, 1.)
LATITUDE_BINS = (-90, -60, -45, -30, -15, 0, 15, 30, 45, 60, 90)


@dataclass(frozen=True)
class Measurement:
    """One drawn instance, with angles in degrees and surface area in MSH."""

    rho: float
    latitude: float | None
    tilt: float | None
    area_px: int
    area_msh: float | None


def surface_points(x, north, b0):
    """Map normalized disk coordinates to an Earth-facing unit sphere."""
    z = np.sqrt(np.maximum(0, 1 - x ** 2 - north ** 2))
    c, s = np.cos(b0), np.sin(b0)
    return np.stack((x, c * north + s * z, c * z - s * north), axis=-1)


def measure_mask(rle, disk, b0):
    """Measure the host-rasterized area, centroid and deprojected major axis."""
    x0, y0, width, height = map(int, mu.toBbox(rle))
    rows, cols = np.nonzero(mu.decode(rle)[y0:y0 + height, x0:x0 + width])
    if not len(rows):
        raise ValueError("Training GT includes an empty rasterized instance.")
    x = (cols + x0 - disk.x) / disk.radius
    north = (disk.y - (rows + y0)) / disk.radius
    radial = np.hypot(x, north)
    cx, cy = x.mean(), north.mean()
    rho = float(np.hypot(cx, cy))
    if rho >= 1:
        return Measurement(rho, None, None, len(rows), None)

    centroid = surface_points(cx, cy, b0)
    lat = float(np.arcsin(np.clip(centroid[1], -1, 1)))
    area_msh = None
    if rho < .8 and radial.max() < .95:
        area_msh = float(np.sum(1 / np.sqrt(1 - radial ** 2)) * 1e6 / (2 * np.pi * disk.radius ** 2))

    tilt = None
    if rho < .85 and radial.max() < .95 and len(rows) >= 3:
        lon = np.arctan2(centroid[0], centroid[2])
        west = np.array([np.cos(lon), 0, -np.sin(lon)])
        pole = np.cross(centroid, west)
        surface = surface_points(x, north, b0)
        tangent = surface @ np.stack((west, pole), axis=-1)
        variances, axes = np.linalg.eigh(np.cov(tangent, rowvar=False))
        if variances[1] > 0 and variances[1] >= 1.5 ** 2 * variances[0]:
            tilt = float(np.degrees(np.arctan2(abs(axes[1, 1]), abs(axes[0, 1]))))
    return Measurement(rho, float(np.degrees(lat)), tilt, len(rows), area_msh)


def quantiles(values, name):
    """Return labeled quartiles, with blanks for an empty bin."""
    q = np.percentile(values, [25, 50, 75]) if values else ("", "", "")
    return {f"{name}_{tag}": value for tag, value in zip(("p25", "median", "p75"), q)}


def cycle_year(stem):
    """Assign the observing year and the cycle, split at December 2019."""
    return stem[:4], 24 if stem[:6] < "201912" else 25


def summarize(gt, disks, b0_angles):
    """Aggregate drawn instances by radius, latitude and observing year."""
    rows = [(cycle_year(stem), measure_mask(rle, disks[stem], b0_angles[stem]))
            for stem, records in gt.items() for rles in records.values() for rle in rles]
    records = sum(len(x) for x in gt.values())
    for low, high in zip(RADIAL_BINS, RADIAL_BINS[1:]):
        count = sum(low <= m.rho < high for _, m in rows)
        annulus = high ** 2 - low ** 2
        hemisphere = np.sqrt(1 - low ** 2) - np.sqrt(1 - high ** 2)
        yield dict(section="radial", lower=low, upper=high, filaments=count, records=records,
                   density_per_record_per_disk=count / (records * annulus),
                   density_per_record_per_hemisphere=count / (records * hemisphere))
    yield dict(section="radial_outside", lower=1., filaments=sum(m.rho >= 1 for _, m in rows), records=records)

    for low, high in zip(LATITUDE_BINS, LATITUDE_BINS[1:]):
        band = [m for _, m in rows if m.latitude is not None and low <= m.latitude < high]
        tilts = [m.tilt for m in band if m.tilt is not None]
        yield dict(section="tilt_by_latitude", lower=low, upper=high, filaments=len(band), eligible=len(tilts),
                   **quantiles(tilts, "tilt_deg"))

    images_by_year = Counter(cycle_year(stem) for stem in gt)
    records_by_year = Counter()
    for stem, records_for_image in gt.items():
        records_by_year[cycle_year(stem)] += len(records_for_image)
    for year, cycle in sorted(images_by_year):
        population = [m for key, m in rows if key == (year, cycle)]
        central = [m for m in population if m.area_msh is not None]
        yield dict(section="size_by_year", year=year, cycle=cycle, images=images_by_year[year, cycle],
                   records=records_by_year[year, cycle], filaments=len(population), eligible=len(central),
                   **quantiles([m.area_px for m in central], "area_px"),
                   **quantiles([m.area_msh for m in central], "area_msh"))


def solar_b0(stems):
    """Compute Earth-view B0 in radians for the UTC exposure timestamps."""
    from astropy.time import Time
    from astropy.utils.iers import conf
    from sunpy.coordinates.sun import B0

    times = [datetime.strptime(stem[:14], "%Y%m%d%H%M%S") for stem in stems]
    with conf.set_temp("auto_download", False):
        angles = B0(Time(times, scale="utc")).rad
    return dict(zip(stems, angles))


def main():
    """Write aggregate report CSV from official training JSON and disk fits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True, help="Official training annotation JSON.")
    parser.add_argument("--disks", type=Path, required=True, help="CSV from tools.figures.disk_check on training JPEGs.")
    parser.add_argument("--out", type=Path, default=Path("data/physics.csv"))
    args = parser.parse_args()
    gt = gt_rles_by_image(load_records(args.gt))
    if not gt:
        parser.error("No training observations.")
    with args.disks.open(encoding="utf-8", newline="") as f:
        disks = {r["stem"]: Disk(float(r["x"]), float(r["y"]), float(r["radius"]), float(r["rms"]), int(r["rays"]))
                 for r in csv.DictReader(f)}
    if gt.keys() - disks.keys():
        parser.error(f"Missing disk fits for {len(gt.keys() - disks.keys())} annotated images.")
    rows = list(summarize(gt, disks, solar_b0(sorted(gt))))
    columns = list(dict.fromkeys(key for row in rows for key in row))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: f"{v:.6g}" if isinstance(v, (float, np.floating)) else v for k, v in row.items()}
                         for row in rows)
    print(f"Wrote {len(rows)} aggregate rows from {len(gt)} training images to {args.out}")


if __name__ == "__main__":
    main()
