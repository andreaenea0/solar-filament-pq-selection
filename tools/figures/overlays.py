"""Render per-annotator PQ overlays and shortlist failures on training JPEGs."""

import argparse
from html import escape
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mu
from scipy.ndimage import binary_erosion

from filseg.data import gt_rles_by_image, load_records
from filseg.disk import fit_limb
from filseg.metric import IOU_T, iou_matrix
from filseg.rle import counts_to_rle
from filseg.rule import keep_rule
from tools.cv_map_parity import load_submissions

TP, FP, FN = (64, 231, 155), (255, 99, 80), (97, 181, 255)


def matches(gt, predictions):
    """Return masks with at least one strict host-PQ match."""
    hits = iou_matrix(gt, predictions) > IOU_T
    return hits.any(1), hits.any(0)


def split_anchors(anchors, parts, unmatched):
    """Find unmatched masks mostly covered by at least two substantial parts."""
    if not anchors or len(parts) < 2:
        return
    areas = np.asarray(mu.area(anchors), dtype=float)
    part_areas = np.asarray(mu.area(parts), dtype=float)
    iou = iou_matrix(anchors, parts)
    intersection = iou * (areas[:, None] + part_areas) / (1 + iou)
    for index in np.flatnonzero(unmatched):
        substantial = (intersection[index] >= 0.1 * areas[index]) & (intersection[index] >= 0.5 * part_areas)
        if substantial.sum() >= 2:
            union = mu.merge([part for part, keep in zip(parts, substantial) if keep])
            coverage = float(mu.area(mu.merge([union, anchors[index]], intersect=True))) / areas[index]
            if coverage >= 0.8:
                yield index, coverage


def failure_candidates(image, gt, predictions, disk=None):
    """Yield visual screening candidates as category, score and bounding box."""
    gt_hit, pred_hit = matches(gt, predictions)
    for kind, anchors, parts, hit in (("split", gt, predictions, gt_hit),
                                       ("merge", predictions, gt, pred_hit)):
        for index, score in split_anchors(anchors, parts, ~hit):
            yield kind, score, mu.toBbox(anchors[index]).tolist()
    h, w = image.shape
    centre_x, centre_y, radius = (disk.x, disk.y, disk.radius) if disk else (
        (w - 1) / 2, (h - 1) / 2, min(h, w) * 900 / 2048
    )
    for masks, hit, is_prediction in ((predictions, pred_hit, True), (gt, gt_hit, False)):
        for rle, matched in zip(masks, hit):
            if matched:
                continue
            x, y, bw, bh = mu.toBbox(rle).astype(int)
            if not bw or not bh:
                continue
            box = [int(v) for v in (x, y, bw, bh)]
            rho = np.hypot(x + bw / 2 - centre_x, y + bh / 2 - centre_y) / radius
            if 0.8 <= rho <= 1.06:
                yield "limb", float(1 - abs(1 - rho)), box
            if not is_prediction:
                continue
            ys = slice(max(y - 24, 0), min(y + bh + 24, h))
            xs = slice(max(x - 24, 0), min(x + bw + 24, w))
            mask = mu.decode(rle)[ys, xs].astype(bool)
            context = image[ys, xs]
            background = context[~mask]
            if len(background) < 2 or mask.sum() < 3:
                continue
            contrast = (np.median(background) - np.median(context[mask])) / max(np.median(background), 1)
            eigenvalues = np.linalg.eigvalsh(np.cov(np.nonzero(mask)))
            elongation = np.sqrt(eigenvalues[-1] / max(eigenvalues[0], 0.01))
            fill = mask.sum() / (bw * bh)
            if elongation <= 2 and fill >= 0.5 and contrast >= 0.10:
                yield "sunspot-like", float(contrast * fill / max(elongation, 1)), box
            if elongation >= 3 and contrast > 0 and rho < 0.8 and background.std() >= 10:
                yield "fibril-like", float(elongation * contrast * background.std()), box


def crop_region(shape, box, size=512):
    """Center a native-pixel crop on a box while staying inside the JPEG."""
    h, w = shape
    x, y, bw, bh = box
    width, height = min(w, size), min(h, size)
    x0 = int(np.clip(round(x + bw / 2 - width / 2), 0, w - width))
    y0 = int(np.clip(round(y + bh / 2 - height / 2), 0, h - height))
    return np.s_[y0:y0 + height, x0:x0 + width]


def outlined(image, masks, colors, region, width):
    """Draw thin internal contours, including where a mask crosses a crop edge."""
    h, w = image.shape
    y0, y1, _ = region[0].indices(h)
    x0, x1, _ = region[1].indices(w)
    ya, yb = max(0, y0 - width), min(h, y1 + width)
    xa, xb = max(0, x0 - width), min(w, x1 + width)
    pixels = np.repeat(image[region][..., None], 3, axis=2)
    for rle, color in zip(masks, colors):
        x, y, bw, bh = map(int, mu.toBbox(rle))
        if x + bw <= x0 or x >= x1 or y + bh <= y0 or y >= y1:
            continue
        top, bottom = max(ya, y), min(yb, y + bh)
        left, right = max(xa, x), min(xb, x + bw)
        mask = mu.decode(rle)[top:bottom, left:right].astype(bool)
        boundary = mask & ~binary_erosion(mask, iterations=width)
        draw_y0, draw_y1 = max(y0, top), min(y1, bottom)
        draw_x0, draw_x1 = max(x0, left), min(x1, right)
        visible = boundary[draw_y0 - top:draw_y1 - top, draw_x0 - left:draw_x1 - left]
        pixels[draw_y0 - y0:draw_y1 - y0, draw_x0 - x0:draw_x1 - x0][visible] = color
    return Image.fromarray(pixels)


def panel_font(width):
    """Choose readable labels for native-pixel crops and full disks."""
    try:
        return ImageFont.load_default(size=24 if width >= 1024 else 16)
    except TypeError:  # Pillow 10.0
        return ImageFont.load_default()


def render(image, records, predictions, title, region=None, first_drawing=1):
    """Draw one native-pixel GT and prediction panel per record."""
    if region is None:
        region = (slice(None), slice(None))
    height, width = image[region].shape
    font = panel_font(width)
    header, label = 62, 36
    panel = Image.new("RGB", (2 * width, header + len(records) * (height + label)), "#101820")
    draw = ImageDraw.Draw(panel)
    draw.text((12, 6), title, fill="white", font=font)
    for x, name, color in ((12, "TP", TP), (82, "FP", FP), (152, "FN", FN)):
        draw.rectangle((x, 41, x + 10, 51), fill=color)
        draw.text((x + 16, 36), name, fill="white", font=font)
    for row, gt in enumerate(records, 1):
        gt_hit, pred_hit = matches(gt, predictions)
        top = header + (row - 1) * (height + label)
        for col, (masks, hit, error_color, caption) in enumerate(
                ((gt, gt_hit, FN, "GT"), (predictions, pred_hit, FP, "prediction"))):
            preview = outlined(image, masks, [TP if flag else error_color for flag in hit], region,
                               width=2 if width >= 1024 else 1)
            panel.paste(preview, (col * width, top + label))
            draw.text((col * width + 12, top + 4), f"Drawing {first_drawing + row - 1} | {caption} | {hit.sum()}/{len(hit)} matched",
                      fill="white", font=font)
    return panel


def write_gallery_index(directory, manifest, stems, composite=None):
    """Link the generated PNGs in a local, standalone inspection page."""
    cards = []
    if composite is not None:
        cards.append(f'<p><a href="{escape(composite, quote=True)}">Aligned annotator comparison</a></p>')
    for stem in stems:
        if (directory / f"{stem}.png").is_file():
            cards.append(f'<p><a href="{escape(stem)}.png">Whole disk: {escape(stem)}</a></p>')
    for item in manifest:
        title = f'{item["category"]} | {item["stem"]} | drawing {item["drawing"]}'
        path = f'gallery/{item["file"]}'
        cards.append(f'<figure><figcaption>{escape(title)}</figcaption>'
                     f'<a href="{escape(path, quote=True)}"><img src="{escape(path, quote=True)}" '
                     f'alt="{escape(title, quote=True)}"></a></figure>')
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Training morphology gallery</title>'
            '<style>body{max-width:1100px;margin:2rem auto;font:17px sans-serif;background:#101820;color:white}'
            'a{color:#a8d8ff}img{max-width:100%}figure{margin:2rem 0}</style>'
            '<h1>Training morphology gallery</h1><p>Selected examples; these do not estimate error prevalence. '
            'Colors mark strict IoU &gt; 0.5 matches against the displayed drawing. Match counts cover the '
            'whole image, including masks outside a crop. Category labels are geometric screening hints, '
            'not physical classifications.</p>' + ''.join(cards) + '</html>')
    (directory / "index.html").write_text(page, encoding="utf-8")


def disagreement_anchor(records):
    """Pick a visible GT instance that other drawings omit or draw differently."""
    drawings = records[:3]
    best = None
    for i, masks in enumerate(drawings):
        if not masks:
            continue
        other_hits = [iou_matrix(masks, other).max(1, initial=0) for j, other in enumerate(drawings) if j != i]
        missed = np.count_nonzero(np.asarray(other_hits) <= IOU_T, axis=0)
        for rle, n in zip(masks, missed):
            box = mu.toBbox(rle)
            score = (n, len(drawings), max(box[2:]) <= 512, int(mu.area(rle)))
            if best is None or score > best[0]:
                best = score, box
    return best


def render_composite(image, records, predictions, stem, region, proxy=False):
    """Show up to three annotators and the prediction on the identical crop."""
    if len(records) < 2:
        raise ValueError("A composite needs at least two drawings.")
    height, width = image[region].shape
    font = panel_font(width)
    header, label, gap = 54, 30, 4
    panel = Image.new("RGB", (2 * width + gap, header + 2 * (height + label) + gap), "#101820")
    draw = ImageDraw.Draw(panel)
    draw.text((12, 6), f"Annotator comparison | {stem}", fill="white", font=font)
    draw.text((12, 28), f"Same crop for every panel | x={region[1].start}, y={region[0].start}",
              fill="#cbd5df", font=font)
    entries = [(f"Drawing {i}", masks, FN) for i, masks in enumerate(records[:3], 1)]
    if len(entries) == 2:
        entries.append(("JPEG", [], FN))
    entries.append(("Proxy prediction (drawing 1)" if proxy else "Prediction", predictions, TP))
    for i, (title, masks, color) in enumerate(entries):
        x, y = (i % 2) * (width + gap), header + (i // 2) * (height + label + gap)
        draw.text((x + 12, y + 4), title, fill="white", font=font)
        panel.paste(outlined(image, masks, [color] * len(masks), region, width=1), (x, y + label))
    return panel


def main():
    """Render selected images, then export a ranked gallery with crop provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True, help="MAGFiLO training annotation JSON.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stems", type=Path, help="Optional text file, one training stem per line.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="Submission-format predictions on training images.")
    source.add_argument("--maps", type=Path, help="Per-stem uint8 NPZ (p = [H,W] or [C,H,W]); uses the keep rule.")
    source.add_argument("--fake-gt", action="store_true", help="Use the first drawing as a smoke-test prediction.")
    parser.add_argument("--gallery-only", action="store_true", help="Scan all selected images, save only the gallery.")
    parser.add_argument("--examples", type=int, default=3, help="Gallery examples per failure type.")
    parser.add_argument("--zooms", type=int, default=4, help="Maximum native 512 px zooms per image.")
    parser.add_argument("--composite", type=Path, help="Save one aligned multi-annotator report figure.")
    args = parser.parse_args()
    if args.zooms < 0:
        parser.error("--zooms must be nonnegative.")
    gt = gt_rles_by_image(load_records(args.gt))
    stems = sorted(set(args.stems.read_text().split()) if args.stems else gt)
    submissions = load_submissions([args.csv], gt)[0] if args.csv else None
    jpeg_paths = {p.stem: p for p in sorted(args.images.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg"}}
    args.out.mkdir(parents=True, exist_ok=True)
    best, composite = {}, None
    zooms = args.out / "zooms"
    if args.zooms and not args.gallery_only:
        zooms.mkdir(exist_ok=True)
    for stem in stems:
        with Image.open(jpeg_paths[stem]) as jpeg:
            image = np.asarray(jpeg.convert("L"))
        records = [gt[stem][key] for key in sorted(gt[stem])]
        if args.csv:
            predictions = [counts_to_rle(s) for s in submissions.get(stem, ())]
        elif args.maps:
            with np.load(args.maps / f"{stem}.npz") as archive:
                p = archive["p"].astype(np.float32) / 255
            predictions = keep_rule(p.mean(0) if p.ndim == 3 else p)
        else:
            predictions = records[0]
        if not args.gallery_only:
            render(image, records, predictions, stem).save(args.out / f"{stem}.png")
            ranked = sorted([rle for gt in records for rle in gt], key=mu.area, reverse=True)
            used = set()
            for rle in ranked:
                if len(used) >= args.zooms:
                    break
                region = crop_region(image.shape, mu.toBbox(rle))
                start = (region[0].start, region[1].start)
                if all((start[0] - y) ** 2 + (start[1] - x) ** 2 >= 128 ** 2 for y, x in used):
                    used.add(start)
                    render(image, records, predictions, f"{stem} | zoom {len(used)}", region).save(
                        zooms / f"{stem}_{len(used)}.png")
        if args.composite and len(records) >= 2:
            anchor = disagreement_anchor(records)
            if anchor and (composite is None or anchor[0] > composite[0]):
                composite = (anchor[0], stem, crop_region(image.shape, anchor[1]), records, predictions)
        try:
            disk = fit_limb(image)
        except ValueError:
            disk = None
        for row, record in enumerate(records, 1):
            for kind, score, box in failure_candidates(image, record, predictions, disk):
                group = best.setdefault(kind, {})
                if stem not in group or score > group[stem][0]:
                    group[stem] = (score, row, box, record, predictions)
    gallery = args.out / "gallery"
    gallery.mkdir(exist_ok=True)
    manifest = []
    for kind, group in sorted(best.items()):
        for stem in sorted(group, key=lambda s: (-group[s][0], s))[:args.examples]:
            score, row, box, record, predictions = group[stem]
            with Image.open(jpeg_paths[stem]) as jpeg:
                image = np.asarray(jpeg.convert("L"))
            region = crop_region(image.shape, box)
            name = f"{kind}_{stem}.png"
            render(image, [record], predictions, f"{kind} | {stem} | original drawing {row}",
                   region, first_drawing=row).save(gallery / name)
            manifest.append(dict(category=kind, stem=stem, drawing=row, score=score, box=box, file=name))
    (gallery / "index.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.composite:
        if composite is None:
            parser.error("No selected image has a usable multi-annotator crop.")
        _, stem, region, records, predictions = composite
        with Image.open(jpeg_paths[stem]) as jpeg:
            image = np.asarray(jpeg.convert("L"))
        args.composite.parent.mkdir(parents=True, exist_ok=True)
        render_composite(image, records, predictions, stem, region, proxy=args.fake_gt).save(args.composite)
        print(f"Composite {stem}, x={region[1].start}, y={region[0].start}: {args.composite}")
    write_gallery_index(args.out, manifest, stems,
                        os.path.relpath(args.composite, args.out).replace(os.sep, "/") if args.composite else None)


if __name__ == "__main__":
    main()
