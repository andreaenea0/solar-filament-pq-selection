"""Retrain only seed 0 with the released k07 all-data recipe on one GPU."""
import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]


def load_recipe(source=ROOT / "kernels/k07-final/final.py"):
    """Import the archived recipe without launching its two-process main block."""
    spec = importlib.util.spec_from_file_location("seed0_recipe", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    """Use all official observations; do not fit the selector or another seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source, out = args.input.resolve(), args.out.resolve()
    if not source.is_dir() or out.exists() or source == out or source in out.parents:
        parser.error("Use an existing input directory and a new output outside it.")
    from training.retrain_team_models import unique_input
    inputs = {name: unique_input(source, name, directory=directory)
              for name, directory in (("train_images", True), ("test_images", True),
                  ("MAGFiLO_1.0_Annotations_kaggle2026_train.json", False))}
    data = json.loads(inputs["MAGFiLO_1.0_Annotations_kaggle2026_train.json"].read_text(encoding="utf-8"))
    stems = sorted({Path(record["file_name"]).stem for record in data["images"]})
    if len(stems) != 707:
        parser.error(f"Expected 707 official training observations, found {len(stems)}.")
    import torch
    if not torch.cuda.is_available():
        parser.error("This training diagnostic requires one CUDA GPU.")
    out.mkdir(parents=True)
    scratch = out / "scratch"
    scratch.mkdir()
    # With FOLD=-1 every nonnegative fold is included and no validation fold exists.
    folds = scratch / "folds.csv"
    folds.write_text("stem,fold\n" + "".join(f"{stem},0\n" for stem in stems), encoding="utf-8")
    inputs["folds.csv"] = folds
    os.environ.update(FOLD="-1", STEPS="12000", FILSEG_INPUT=str(source),
                      FILSEG_OUTPUT=str(out), FILSEG_TMP=str(scratch))
    recipe = load_recipe()
    recipe.find = lambda name, d=False: str(inputs[name])
    started = perf_counter()
    recipe.prepare(include_neighbors=False)
    prepared = perf_counter()
    torch.backends.cudnn.benchmark = True
    recipe.run(0, "effv2s_s0", False, 0)
    finished = perf_counter()
    metadata = dict(seed=0, fold=-1, steps=12000, observations=len(stems),
                    device_name=torch.cuda.get_device_name(0),
                    preparation_seconds=prepared-started,
                    training_and_map_export_seconds=finished-prepared,
                    total_seconds=finished-started,
                    recipe_sha256=hashlib.sha256((ROOT / "kernels/k07-final/final.py").read_bytes()).hexdigest(),
                    training_json_sha256=hashlib.sha256(inputs["MAGFiLO_1.0_Annotations_kaggle2026_train.json"].read_bytes()).hexdigest(),
                    versions={name: importlib.metadata.version(name) for name in
                              ("torch", "segmentation-models-pytorch", "timm", "numpy")},
                    scope="new nondeterministic fit; no change to released final",
                    neighbors="not decoded; temporal=False always samples index[0]",
                    folds="all zero: equivalent all-data inclusion at FOLD=-1")
    (out / "training_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
