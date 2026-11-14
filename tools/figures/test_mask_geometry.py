"""Compare test prediction geometry with the published training geometry."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion
from pycocotools import mask as mu

from filseg.disk import Disk
from filseg.rle import counts_to_rle
from tools.cv_map_parity import load_submissions
from tools.figures.filament_physics import measure_mask, quantiles, solar_b0
from tools.figures.instance_errors import write_table
from tools.figures.missed_instances import crop_mask, measure


def shape_screen(rle, image, disk):
    """Measure local contrast, aspect ratio and an effective projected width."""
    mask, _ = crop_mask(rle)
    coords = np.argwhere(mask)
    area = len(coords)
    if area < 3:
        return np.nan, 0., np.inf
    eigenvalues, directions = np.linalg.eigh(np.cov(coords, rowvar=False))
    length = np.ptp(coords @ directions[:, -1]) + 1
    aspect = np.sqrt(eigenvalues[-1] / max(eigenvalues[0], .25))
    contrast = measure(rle, image, (disk.x, disk.y, disk.radius), None, 0.)[2]
    return contrast, aspect, area / length


def comparison(rows, training):
    """Use the radius, latitude and year bins in the existing training table."""
    total = sum(int(r["filaments"]) for r in training if r["section"].startswith("radial"))
    for row in training:
        section = row["section"]
        if section.startswith("radial"):
            low, high = float(row["lower"]), float(row["upper"] or "inf")
            count = sum(low <= m.rho < high for _, m in rows)
            yield dict(section=section, lower=low, upper=row["upper"], train_n=int(row["filaments"]),
                       test_n=count, train_share=int(row["filaments"]) / total, test_share=count / len(rows))
        if section == "tilt_by_latitude":
            low, high = float(row["lower"]), float(row["upper"])
            values = [m.tilt for _, m in rows if m.tilt is not None and low <= m.latitude < high]
            yield dict(section=section, lower=low, upper=high, train_n=int(row["eligible"]), test_n=len(values),
                       train_median=row["tilt_deg_median"], **quantiles(values, "test"))
        if section == "size_by_year":
            year = row["year"]
            values = [m for stem, m in rows if stem.startswith(year) and m.area_msh is not None]
            yield dict(section=section, year=year, train_n=int(row["eligible"]), test_n=len(values),
                       train_median=row["area_msh_median"], train_area_px_median=row["area_px_median"],
                       **quantiles([m.area_msh for m in values], "test"),
                       **quantiles([m.area_px for m in values], "test_area_px"))


def panel(path, rle):
    """Render the same native-image crop with and without its contour."""
    x, y, w, h = mu.toBbox(rle).astype(int)
    mask = mu.decode(rle).astype(bool)
    side = min(2048, max(96, max(w, h) + 36))
    x0 = min(2048 - side, max(0, x + (w - side) // 2))
    y0 = min(2048 - side, max(0, y + (h - side) // 2))
    region = np.s_[y0:y0 + side, x0:x0 + side]
    with Image.open(path) as jpeg:
        original = np.asarray(jpeg.convert("RGB"))[region].copy()
    crop = mask[region]
    overlay = original.copy()
    overlay[crop & ~binary_erosion(crop, iterations=max(1, side // 110))] = (0, 242, 255)
    return [Image.fromarray(p).resize((175, 175), Image.Resampling.LANCZOS) for p in (original, overlay)]


def render_examples(ranked, paths, output):
    """Render up to two distinct observations for each screening category."""
    used, selected = set(), []
    for label, entries in ranked.items():
        for _, stem, index, rle in sorted(entries, key=lambda row: row[:3], reverse=True):
            if stem in used:
                continue
            selected.append((label, stem, index, rle))
            used.add(stem)
            if sum(item[0] == label for item in selected) == 2:
                break
    if not selected:
        raise ValueError("No examples met the screening criteria.")
    sheet = Image.new("RGB", (3 * 372, 225 * ((len(selected) + 2) // 3) + 36), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), "three-verifiers test predictions | JPEG crop at left, cyan contour at right", fill="black")
    for i, (label, stem, index, rle) in enumerate(selected):
        x, y = (i % 3) * 372 + 10, (i // 3) * 225 + 37
        draw.text((x, y), f"{label} | {stem}_{index}", fill="black")
        left, right = panel(paths[stem], rle)
        sheet.paste(left, (x, y + 20))
        sheet.paste(right, (x + 177, y + 20))
    sheet.save(output / "three_verifiers_test_screen.png")


def main():
    """Audit frozen test masks and render screened examples without test labels."""
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("pred", "images", "disks", "train-physics", "out"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    args = parser.parse_args()
    paths = {p.stem: p for p in args.images.iterdir() if p.suffix.lower() in (".jpg", ".jpeg")}
    predictions = load_submissions([args.pred], paths)[0]
    with args.disks.open(encoding="utf-8", newline="") as f:
        disks = {r["stem"]: Disk(float(r["x"]), float(r["y"]), float(r["radius"]), float(r["rms"]), int(r["rays"]))
                 for r in csv.DictReader(f)}
    with args.train_physics.open(encoding="utf-8", newline="") as f:
        training = list(csv.DictReader(f))
    angles = solar_b0(sorted(paths))
    rows, ranked = [], defaultdict(list)
    for stem, masks in predictions.items():
        with Image.open(paths[stem]) as jpeg:
            image = np.asarray(jpeg.convert("L"))
        disk = disks[stem]
        for i, counts in enumerate(masks, 1):
            rle = counts_to_rle(counts)
            measurement = measure_mask(rle, disk, angles[stem])
            rows.append((stem, measurement))
            contrast, aspect, width = shape_screen(rle, image, disk)
            if measurement.rho >= .85:
                ranked["Near limb"].append((measurement.rho, stem, i, rle))
            elif aspect < 1.5 and np.isfinite(contrast):
                ranked["Compact and dark"].append((contrast, stem, i, rle))
            elif aspect >= 3:
                ranked["Thin and elongated"].append((-width, stem, i, rle))
    args.out.mkdir(parents=True, exist_ok=True)
    write_table(args.out / "test_mask_geometry.csv", list(comparison(rows, training)))
    render_examples(ranked, paths, args.out)
    print(f"Measured {len(rows)} masks on {len(predictions)} test images.")


if __name__ == "__main__":
    main()
