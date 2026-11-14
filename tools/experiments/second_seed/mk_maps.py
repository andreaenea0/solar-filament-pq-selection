"""Build the original rounded two-seed study maps without import-time file access."""

import argparse
import os
from pathlib import Path
import numpy as np


def build_maps(seed0, seed1_directories, output):
    """Preserve round-to-even uint8 means and duplicated own-map channels."""
    seed1 = {path.stem: path for directory in seed1_directories for path in directory.glob("*.npz")}
    if not seed1:
        raise ValueError("No held-out seed-1 maps found.")
    for name in ("maps_mean2", "maps_s1"):
        (output/name).mkdir(parents=True, exist_ok=True)
    for stem, path in sorted(seed1.items()):
        with np.load(seed0/f"{stem}.npz", allow_pickle=False) as archive:
            first = archive["p"].astype(np.float32)
        with np.load(path, allow_pickle=False) as archive:
            second = archive["p"].astype(np.float32)
        if first.ndim != 2 or first.shape != second.shape:
            raise ValueError(f"{stem}: expected two same-size 2D probability maps.")
        mean = ((first+second)/2).round().astype(np.uint8)
        second = second.astype(np.uint8)
        np.savez_compressed(output/"maps_mean2"/f"{stem}.npz", p=np.stack((mean, mean)))
        np.savez_compressed(output/"maps_s1"/f"{stem}.npz", p=np.stack((second, second)))
    return len(seed1)


def main(argv=None):
    """Accept explicit paths or the documented environment-variable defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=os.environ.get("FILSEG_WORK"))
    parser.add_argument("--cache", type=Path, default=os.environ.get("FILSEG_CACHE"))
    parser.add_argument("--seed1-dir", type=Path, action="append", help="Repeat for each held-out export directory.")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.work is None or args.cache is None:
        parser.error("Pass --work/--cache or set FILSEG_WORK/FILSEG_CACHE.")
    directories = args.seed1_dir or [args.work/name/"oof_control" for name in ("k20out", "k20f23out")]
    print(build_maps(args.cache/"maps_own_oof", directories, args.out or args.work))


if __name__ == "__main__":
    main()
