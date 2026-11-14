"""Audit every predicted mask from JPEG pixels, without annotation labels.

Sunspot hits are a declared dark compact-core proxy, not physical labels.
Nothing in this diagnostic changes inference or the prediction archive.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from pycocotools import mask as mu

from filseg.disk import fit_limb
from filseg.rle import counts_to_rle
from tools.timing_replay import read_rows, write_csv

DETECTOR = dict(radial_bins=100, intensity_ratio=.65, local_deficit=.12,
                background_sigma=12, min_area=9, max_area=2500,
                min_solidity=.65, min_circularity=.35, max_aspect=2.5,
                max_radius=.95, opening_size=3)


def radial_coordinates(shape, disk):
    """Measure image-derived projected distance in fitted solar radii."""
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return np.hypot(xx - disk.x, yy - disk.y) / disk.radius


def compact_cores(image, rho):
    """Flag dark, locally depressed, compact cores inside 0.95 solar radii."""
    sampled, radius = image[::4, ::4], rho[::4, ::4]
    edges = np.linspace(0, 1, DETECTOR['radial_bins'] + 1)
    medians = np.array([np.median(sampled[(radius >= a) & (radius < b)])
                        for a, b in zip(edges[:-1], edges[1:])])
    if not np.isfinite(medians).all() or np.any(medians <= 0):
        raise ValueError('radial background has an empty or nonpositive annulus')
    radial = np.interp(rho, (edges[:-1] + edges[1:]) / 2, medians)
    smooth = cv2.GaussianBlur(image.astype(np.float32), (0, 0), DETECTOR['background_sigma'])
    raw = ((image < DETECTOR['intensity_ratio'] * radial)
           & ((smooth - image) > DETECTOR['local_deficit'] * radial)
           & (rho < DETECTOR['max_radius']))
    size = DETECTOR['opening_size']
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    opened = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    keep = np.zeros(image.shape, dtype=np.uint8)
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if not DETECTOR['min_area'] <= area <= DETECTOR['max_area']:
            continue
        region = (labels[y:y+h, x:x+w] == index).astype(np.uint8)
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        hull = cv2.contourArea(cv2.convexHull(contour))
        area_contour = cv2.contourArea(contour)
        circularity = 4 * np.pi * area_contour / max(perimeter**2, 1)
        solidity = area_contour / max(hull, 1)
        if (circularity >= DETECTOR['min_circularity']
                and solidity >= DETECTOR['min_solidity']
                and max(w, h) / min(w, h) <= DETECTOR['max_aspect']):
            keep[y:y+h, x:x+w] |= region
    return keep


def measure_mask(mask, rho, occupancy, cores):
    """Use 8-connected foreground and 4-connected background topology."""
    yy, xx = np.nonzero(mask)
    if not len(xx):
        raise ValueError('empty predicted mask')
    x0, x1, y0, y1 = int(xx.min()), int(xx.max())+1, int(yy.min()), int(yy.max())+1
    crop = mask[y0:y1, x0:x1].astype(np.uint8)
    parts, _, stats, _ = cv2.connectedComponentsWithStats(crop, connectivity=8)
    filled = ndi.binary_fill_holes(crop, structure=ndi.generate_binary_structure(2, 1))
    holes, count = ndi.label(filled & ~crop.astype(bool), structure=ndi.generate_binary_structure(2, 1))
    hole_areas = np.bincount(holes.ravel())[1:]
    radii = rho[yy, xx]
    core_pixels = int(cores[yy, xx].sum())
    overlap_pixels = int((occupancy[yy, xx] > 1).sum())
    return dict(area=len(xx), parts=parts-1, largest_part=int(stats[1:, cv2.CC_STAT_AREA].max()),
                holes=count, hole_area=int(hole_areas.sum()),
                largest_hole=int(hole_areas.max(initial=0)),
                centroid_x=float(xx.mean()), centroid_y=float(yy.mean()),
                rho_min=float(radii.min()), rho_median=float(np.median(radii)), rho_max=float(radii.max()),
                limb_pixels=int((radii >= .98).sum()), off_disk_pixels=int((radii > 1).sum()),
                overlap_pixels=overlap_pixels, core_pixels=core_pixels,
                core_fraction=core_pixels/len(xx), x=x0, y=y0, width=x1-x0, height=y1-y0)


def draw_contours(image, masks, cores, disk):
    """Show prediction contours in cyan, proxy cores orange and fitted limb magenta."""
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    contours = [c for mask in masks for c in cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]]
    cv2.drawContours(rgb, contours, -1, (0, 255, 255), 2)
    cv2.drawContours(rgb, cv2.findContours(cores, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (255, 150, 0), 2)
    cv2.circle(rgb, (round(disk.x), round(disk.y)), round(disk.radius), (255, 0, 220), 2)
    return rgb


def contact_sheets(tiles, folder, prefix, columns=4, rows=3):
    """Write bounded JPEG pages with readable IDs and criterion labels."""
    width, height = 384, 438
    for page, start in enumerate(range(0, len(tiles), columns*rows), 1):
        sheet = Image.new('RGB', (columns*width, rows*height), (18,18,18))
        draw = ImageDraw.Draw(sheet)
        for index, (pixels, lines) in enumerate(tiles[start:start+columns*rows]):
            col, row = index % columns, index // columns
            tile = Image.fromarray(pixels)
            tile.thumbnail((width, width), Image.Resampling.LANCZOS)
            ox, oy = col*width, row*height
            sheet.paste(tile, (ox+(width-tile.width)//2, oy+(width-tile.height)//2))
            for line, label in enumerate(lines[:3]):
                draw.text((ox+5, oy+width+line*16), label, fill='white')
        sheet.save(folder / f'{prefix}-{page:02d}.jpg', quality=82, optimize=True)


def main():
    """Measure the complete reference archive and render independent review sheets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, default=Path('docs/reference/submitted_test.csv'))
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--figures', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    args = parser.parse_args()
    if any(path.exists() for path in (args.csv, args.figures, args.metadata)):
        parser.error('all output paths must be new')
    paths = {p.stem:p for p in args.images.glob('*.jpeg')}
    predictions = read_rows(args.predictions, set(paths))
    grouped = {stem:[] for stem in paths}
    for key, counts in predictions.items():
        stem = key.rsplit('_',1)[0]
        if stem not in paths:
            raise ValueError('prediction has no input JPEG')
        grouped[stem].append((key, counts))
    args.figures.mkdir(parents=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    measured, image_rows, hashes = [], [], []
    full_tiles = []
    for index, stem in enumerate(sorted(paths)):
        image = cv2.imread(str(paths[stem]), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (2048,2048):
            raise ValueError('expected native 2048-square JPEG')
        disk = fit_limb(image)
        rho = radial_coordinates(image.shape, disk)
        cores = compact_cores(image, rho)
        masks = [mu.decode(counts_to_rle(c)).astype(np.uint8) for _,c in grouped[stem]]
        occupancy = np.zeros(image.shape, dtype=np.uint16)
        for mask in masks:
            occupancy += mask
        for (key,_), mask in zip(grouped[stem], masks):
            row = dict(filament_id=key, stem=stem, **measure_mask(mask,rho,occupancy,cores))
            row['rho_centroid'] = float(np.hypot(row['centroid_x']-disk.x,row['centroid_y']-disk.y)/disk.radius)
            row.update(disk_x=disk.x,disk_y=disk.y,disk_radius=disk.radius,disk_rms=disk.rms,disk_rays=disk.rays)
            measured.append(row)
        image_rows.append(dict(stem=stem,masks=len(masks),core_pixels=int(cores.sum()),
                               overlap_pixels=int((occupancy>1).sum()),disk_rms=disk.rms))
        hashes.append(dict(file=paths[stem].name,sha256=hashlib.sha256(paths[stem].read_bytes()).hexdigest()))
        whole=draw_contours(image,masks,cores,disk)
        full_tiles.append((cv2.resize(whole,(384,384),interpolation=cv2.INTER_AREA),
                           [stem,f'{len(masks)} masks; {int(cores.sum())} proxy pixels']))
        print(f'{index+1}/{len(paths)} {stem}: {len(masks)} masks',flush=True)
    contact_sheets(full_tiles,args.figures,'all-disks')
    rng = np.random.default_rng(20261113)
    selected = {}
    def add(rows, reason):
        for row in rows:
            selected.setdefault(row['filament_id'],[]).append(reason)
    for field,reverse,name in (('area',True,'largest'),('area',False,'smallest'),
                               ('holes',True,'holes'),('hole_area',True,'hole-area'),
                               ('parts',True,'parts'),('rho_max',True,'limb')):
        add(sorted(measured,key=lambda row:row[field],reverse=reverse)[:10],name)
    for field in ('overlap_pixels','core_pixels','off_disk_pixels'):
        add([row for row in measured if row[field]>0],field)
    add([measured[i] for i in rng.choice(len(measured),min(20,len(measured)),replace=False)],'random')
    tiles = []
    for row in measured:
        key,stem = row['filament_id'],row['stem']
        if key not in selected:
            continue
        image=cv2.imread(str(paths[stem]),cv2.IMREAD_GRAYSCALE)
        disk=fit_limb(image);rho=radial_coordinates(image.shape,disk)
        cores=compact_cores(image,rho)
        mask=mu.decode(counts_to_rle(predictions[key])).astype(np.uint8)
        drawn=draw_contours(image,[mask],cores,disk)
        margin=64
        x0=max(0,row['x']-margin);y0=max(0,row['y']-margin)
        x1=min(2048,row['x']+row['width']+margin);y1=min(2048,row['y']+row['height']+margin)
        tiles.append((drawn[y0:y1,x0:x1].copy(),[key,','.join(selected[key]),
                     f"A={row['area']} parts={row['parts']} holes={row['holes']} rmax={row['rho_max']:.3f}"]))
    contact_sheets(tiles,args.figures,'extremes-random')
    for row in measured:
        row['sheet_reasons']='|'.join(selected.get(row['filament_id'],[]))
    write_csv(args.csv,measured)
    write_csv(args.csv.with_name(args.csv.stem+'_images.csv'),image_rows)
    write_csv(args.csv.with_name(args.csv.stem+'_inputs.csv'),hashes)
    metadata=dict(detector=DETECTOR,random_seed=20261113,images=len(paths),masks=len(measured),
                  selected_masks=len(selected),empty_images=[r['stem'] for r in image_rows if not r['masks']],
                  prediction_sha256=hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
                  topology='8-connected foreground, 4-connected holes',limb_band='rho >= 0.98',
                  sheet_order=[r['filament_id'] for r in measured if r['filament_id'] in selected])
    args.metadata.write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in metadata.items() if k not in ('sheet_order','detector')},indent=2))


if __name__=='__main__':
    main()
