"""Run the three-verifiers training recipes locally and write a new weights directory."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def unique_input(root, name, directory=False):
    """Find an unambiguous input before launching expensive work."""
    paths = [p for p in root.rglob(name) if p.is_dir() == directory]
    if len(paths) != 1:
        raise ValueError(f"Expected one {name} under {root}, found {len(paths)}.")
    return paths[0]


def run(root, *args, env=None):
    """Invoke a checked-in recipe in the current Python environment."""
    subprocess.run([sys.executable, *map(str, args)], cwd=root, env=env, check=True)


def run_recipe(root, source, out, script, **defaults):
    """Set the local paths that replace the Kaggle mounts."""
    out.mkdir()
    env = dict(os.environ, FILSEG_INPUT=str(source), FILSEG_OUTPUT=str(out),
               FILSEG_TMP=str(out / "scratch"), **defaults)
    run(root, script, env=env)


def training_inputs(source):
    """Find the official inputs and validate the required neighbour layout."""
    train = unique_input(source, "train_images", directory=True)
    test = unique_input(source, "test_images", directory=True)
    annotations = unique_input(source, "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    folds = unique_input(source, "folds.csv")
    if not next(source.rglob("+2_*.jpg"), None):
        raise ValueError("Mount the GONG neighbour files required by the training recipes.")
    return train, test, annotations, folds


def main():
    """Train the OOF controls, both final U-Nets, the verifier and LightGBM."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Directory with the official data, folds.csv and GONG neighbours.")
    parser.add_argument("--out", type=Path, required=True, help="New directory for training artifacts.")
    args = parser.parse_args()
    source, out = args.input.resolve(), args.out.resolve()
    if not source.is_dir() or out.exists() or source == out or source in out.parents:
        parser.error("Use an existing input directory and a new output directory outside it.")
    try:
        train, test, annotations, folds = training_inputs(source)
    except ValueError as exc:
        parser.error(str(exc))

    import torch

    if torch.cuda.device_count() < 2:
        parser.error("The original training recipes require two CUDA devices.")

    root = Path(__file__).resolve().parents[1]
    out.mkdir(parents=True)
    k06, k09, k11, k07, k17 = (out / name for name in ("k06", "k09", "k11", "k07", "k17"))
    run_recipe(root, source, k06, "kernels/k06-dice/train.py", FOLD="0", STEPS="10500")
    run_recipe(root, source, k09, "training/k09-oof12/train.py", FOLDS="1,2", STEPS="10500")
    run_recipe(root, source, k11, "training/k11-oof34/train.py", FOLDS="3,4", STEPS="10500")
    run_recipe(root, source, k07, "kernels/k07-final/final.py", FOLD="-1", STEPS="12000")

    gt, maps_train, maps_test = (out / name for name in ("gt.pkl", "maps-train", "maps-test"))
    cands_train, cands_test = (out / name for name in ("cands-train.pkl", "cands-test.pkl"))
    run(root, "-m", "training.prepare", "gt", annotations, gt)
    run(root, "-m", "training.prepare", "oof-maps", maps_train,
        *(folder / "oof_control" for folder in (k06, k09, k11)), "--folds", folds)
    run(root, "-m", "training.prepare", "test-maps", maps_test,
        *(k07 / f"test_effv2s_s{seed}" for seed in range(2)))
    run(root, "tools/build_cands.py", "--maps", maps_train, "--images", train, "--out", cands_train, "--gt", gt)
    run(root, "tools/build_cands.py", "--maps", maps_test, "--images", test, "--out", cands_test)

    crops = out / "crops"
    crops.mkdir()
    crops_train, crops_test = crops / "crops_train.npz", crops / "crops_test.npz"
    run(root, "training/crops4.py", cands_train, maps_train, train, crops_train)
    run(root, "training/crops4.py", cands_test, maps_test, test, crops_test)
    run(root, "-m", "training.prepare", "targets", cands_train, crops_train, folds, crops / "targets.npz")
    run_recipe(root, crops, k17, "training/k17-verifier-own-final/verifier.py")
    verifier = out / "verifier-test.pkl"
    run(root, "-m", "training.prepare", "mean-gains", verifier,
        *(k17 / f"verif_test_resnet18_s{seed}.pkl" for seed in range(3)))
    run(root, "training/make_sub_blend.py", cands_train, cands_test, verifier,
        out / "own_ver3_a05_l025_nometa.csv", "0.5", "0.25")

    weights = out / "weights"
    weights.mkdir()
    files = [
        *(k07 / f"effv2s_s{seed}_fold-1.pt" for seed in range(2)),
        *(k17 / f"verifier_resnet18_s{seed}.pt" for seed in range(3)),
        out / "own_ver3_a05_l025_nometa_lgb.txt",
    ]
    for file in files:
        shutil.copy2(file, weights / file.name)
    print(f"Trained three-verifiers weights: {weights}")


if __name__ == "__main__":
    main()
