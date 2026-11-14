"""Time independent CPU runs of an own-weight recipe and compare them with its archive."""

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter

from tools.timing_replay import replay_summary


def read_rows(path):
    """Read the ID-to-counts mapping from a known submission CSV."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {row["filament_id"]: row["segmentation_rle"] for row in csv.DictReader(stream)}


def compare(reference, actual):
    """Compare row IDs and RLE bytes, then compare geometry irrespective of IDs."""
    stems = {key.rsplit("_", 1)[0] for rows in (reference, actual) for key in rows}
    result = replay_summary(reference, actual, stems)
    minimum = f"{result['minimum_iou']:.6f}" if result['minimum_iou'] is not None else "n/a"
    return (
        f"{len(actual)} CPU / {len(reference)} archive masks, {result['shared_ids']} shared IDs, "
        f"{result['exact_rles']} byte-identical, {result['strict_matches']} strict IoU matches, "
        f"min matched IoU {minimum}"
    )


def select_images(directory, stems=None):
    """Choose explicit observations or the first, middle and last JPEGs."""
    images = sorted((p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg")),
                    key=lambda path: path.stem)
    by_stem = {p.stem: p for p in images}
    if len(by_stem) != len(images):
        raise ValueError("Use unique image stems.")
    if stems is not None:
        if not stems or len(set(stems)) != len(stems):
            raise ValueError("Choose distinct image stems.")
        missing = set(stems) - by_stem.keys()
        if missing:
            raise ValueError(f"JPEGs not found: {', '.join(sorted(missing))}.")
        return [by_stem[stem] for stem in stems]
    if len(images) < 3:
        raise ValueError("Provide at least three JPEGs, or choose one or more with --stems.")
    return [images[0], images[len(images) // 2], images[-1]]


def main():
    """Copy chosen JPEGs into isolated inputs and time the documented release CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Official test JPEG directory.")
    parser.add_argument("--final", choices=("submitted", "three-verifiers", "single-unet"), default="submitted",
                        help="Recipe to replay (default: submitted, the final).")
    parser.add_argument("--weights-dir", type=Path, required=True, help="Directory of verified release files.")
    parser.add_argument("--reference", type=Path, required=True, help="Archived submission CSV of that recipe.")
    parser.add_argument("--out-dir", type=Path, required=True, help="New directory for the results.")
    parser.add_argument("--stems", nargs="+", help="Replay these image stems instead of the default three.")
    args = parser.parse_args()
    try:
        images = select_images(args.images, args.stems)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.out_dir.exists():
        parser.error("Choose a new output directory.")

    reference = read_rows(args.reference)
    root = Path(__file__).resolve().parents[1]
    weights = args.weights_dir.resolve()
    unet_seeds = range(1) if args.final == "single-unet" else range(2)
    verifier_seeds = range(3) if args.final == "three-verifiers" else range(1)
    for image in images:
        expected = {key: counts for key, counts in reference.items() if key.rsplit("_", 1)[0] == image.stem}
        source = (args.out_dir / image.stem).resolve()
        source.mkdir(parents=True)
        shutil.copy2(image, source / image.name)
        out = source / "submission.csv"
        start = perf_counter()
        subprocess.run([
            sys.executable, str(root / "submit.py"), "--device", "cpu", "--tta-batch", "1",
            "--images", str(source), "--final", args.final,
            "--weights", *(str(weights / f"effv2s_s{seed}_fold-1.pt") for seed in unet_seeds),
            "--selector", str(weights / "own_ver3_a05_l025_nometa_lgb.txt"),
            "--verifier-weights", *(str(weights / f"verifier_resnet18_s{seed}.pt") for seed in verifier_seeds),
            "--out", str(out),
        ], check=True)
        print(f"{image.stem}: {perf_counter() - start:.2f} s, {compare(expected, read_rows(out))}", flush=True)


if __name__ == "__main__":
    main()
