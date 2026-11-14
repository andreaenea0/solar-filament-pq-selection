"""Build threshold-tree candidates and optional targets from saved probability maps."""

import argparse
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filseg.maps import read_map
from filseg.oof import candidate_record


def build_one(stem, maps, images, channel, gt):
    """Read an observation and build the training or inference candidate record."""
    with Image.open(images / f"{stem}.jpeg") as image:
        pixels = np.asarray(image.convert("L"))
    return stem, candidate_record(stem, read_map(maps / f"{stem}.npz"), pixels, gt, channel)


def main():
    """Write {stem: candidate record} using the existing pickle schema."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gt", type=Path, help="Optional {stem: {record_id: [COCO RLE]}} pickle.")
    parser.add_argument("--channel", default="mean")
    args = parser.parse_args()
    gt = None
    if args.gt:
        with args.gt.open("rb") as stream:
            gt = pickle.load(stream)
    stems = sorted(path.stem for path in args.maps.glob("*.npz"))
    records = (gt[stem] if gt is not None else None for stem in stems)
    with ProcessPoolExecutor(4) as pool:
        result = dict(pool.map(build_one, stems, repeat(args.maps), repeat(args.images),
                               repeat(args.channel), records, chunksize=4))
    with args.out.open("wb") as stream:
        pickle.dump(result, stream)
    print(len(result), sum(len(row["feats"]) for row in result.values()))


if __name__ == "__main__":
    main()
