"""Prepare three-verifiers map channels, verifier targets and seed averages."""

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np

from filseg.data import gt_rles_by_image, load_records
from filseg.maps import matched_family_maps, read_map


def load_pickle(path):
    """Read a trusted intermediate from this pipeline."""
    with path.open("rb") as stream:
        return pickle.load(stream)


def dump_pickle(path, payload):
    """Save a pipeline intermediate."""
    with path.open("wb") as stream:
        pickle.dump(payload, stream)


def read_folds(path):
    """Read the released image-level fold assignment."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    folds = {row["stem"]: int(row["fold"]) for row in rows}
    if len(folds) != len(rows) or set(folds.values()) != set(range(5)):
        raise ValueError("Expected one row per stem and all five folds.")
    return folds


def write_maps(sources, out, folds=None):
    """Duplicate one held-out map or the rounded mean of two test-seed maps."""
    by_dir = [{path.stem: path for path in source.glob("*.npz")} for source in sources]
    if folds is not None:
        for maps, allowed in zip(by_dir, ((0,), (1, 2), (3, 4)), strict=True):
            if maps.keys() != {stem for stem, fold in folds.items() if fold in allowed}:
                raise ValueError(f"OOF map stems must match folds {allowed}.")
        members = {stem: [path] for maps in by_dir for stem, path in maps.items()}
    else:
        first, second = by_dir
        if not first or first.keys() != second.keys():
            raise ValueError("The two test-seed directories must contain identical nonempty stem sets.")
        members = {stem: [path, second[stem]] for stem, path in first.items()}
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("*.npz")):
        raise FileExistsError(f"Use an empty output map directory: {out}")
    for stem, paths in sorted(members.items()):
        maps = [read_map(path) for path in paths]
        mean = matched_family_maps([maps], [len(maps)])[0]
        np.savez_compressed(out / f"{stem}.npz", p=np.stack((mean, mean)))


def write_targets(candidates, crops, folds, out):
    """Save candidate gains and annotator weights in the crop archive's row order."""
    if candidates.keys() != folds.keys():
        raise ValueError("Training candidates must cover the official folds.")
    with np.load(crops, allow_pickle=False) as archive:
        stems, indices = archive["stem"], archive["idx"]
    gains, weights, fold_ids = [], [], []
    start = 0
    for stem, record in sorted(candidates.items()):
        count = len(record["rle"])
        stop = start + count
        gain = np.asarray(record["g"])
        if (gain.shape != (count,) or not np.isfinite(gain).all() or record["R"] < 1
                or not (stems[start:stop] == stem).all()
                or not np.array_equal(indices[start:stop], np.arange(count))):
            raise ValueError(f"Invalid targets or crop alignment for {stem}.")
        gains.append(gain)
        weights.append(np.full(count, record["R"]))
        fold_ids.append(np.full(count, folds[stem]))
        start = stop
    if stems.shape != (start,) or indices.shape != (start,):
        raise ValueError("Unexpected crop row count.")
    np.savez(out, g=np.concatenate(gains), w=np.concatenate(weights), fold=np.concatenate(fold_ids))


def mean_gains(predictions):
    """Average the three aligned verifier exports in seed order."""
    first = predictions[0]
    if any(pred.keys() != first.keys() for pred in predictions[1:]):
        raise ValueError("Verifier seed exports must cover identical stems.")
    result = {}
    for stem in sorted(first):
        arrays = [np.asarray(pred[stem]) for pred in predictions]
        if any(a.ndim != 1 or a.shape != arrays[0].shape or not np.isfinite(a).all() for a in arrays):
            raise ValueError(f"Invalid verifier seed gains for {stem}.")
        result[stem] = np.stack(arrays).mean(axis=0)
    return result


def main():
    """Run one local preparation step with explicit input and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gt = commands.add_parser("gt", help="Convert official training JSON to scorer RLEs.")
    gt.add_argument("json", type=Path)
    gt.add_argument("out", type=Path)
    for mode, count in (("oof-maps", 3), ("test-maps", 2)):
        maps = commands.add_parser(mode, help="Write two identical own-model channels per stem.")
        maps.add_argument("out", type=Path)
        maps.add_argument("sources", type=Path, nargs=count)
        if mode == "oof-maps":
            maps.add_argument("--folds", type=Path, required=True)
    targets = commands.add_parser("targets", help="Align verifier training targets with crops.")
    for arg in ("candidates", "crops", "folds", "out"):
        targets.add_argument(arg, type=Path)
    mean = commands.add_parser("mean-gains", help="Average three verifier gain exports.")
    mean.add_argument("out", type=Path)
    mean.add_argument("seeds", type=Path, nargs=3)
    args = parser.parse_args()
    if args.command == "gt":
        dump_pickle(args.out, gt_rles_by_image(load_records(args.json)))
    elif args.command in ("oof-maps", "test-maps"):
        folds = read_folds(args.folds) if args.command == "oof-maps" else None
        write_maps(args.sources, args.out, folds)
    elif args.command == "targets":
        write_targets(load_pickle(args.candidates), args.crops, read_folds(args.folds), args.out)
    else:
        dump_pickle(args.out, mean_gains([load_pickle(path) for path in args.seeds]))


if __name__ == "__main__":
    main()
