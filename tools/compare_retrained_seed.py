"""Compare a retrained seed with released maps and frozen submitted predictions.

Agreement with released predictions is a reproducibility diagnostic, not
test PQ. The official test annotations are neither read nor required.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from tools.timing_replay import ROOT, WEIGHTS, read_rows, replay_summary, verify_weights, write_csv


def map_metrics(released, retrained):
    """Compare uint8 probability maps on the probability scale [0,1]."""
    if released.shape != (2048, 2048) or retrained.shape != released.shape:
        raise ValueError("probability maps must both be 2048 x 2048")
    if released.dtype != np.uint8 or retrained.dtype != np.uint8:
        raise ValueError("probability maps must be uint8")
    delta = (retrained.astype(np.float64) - released) / 255.
    return dict(mae=float(np.abs(delta).mean()), rmse=float(np.sqrt((delta**2).mean())),
                max_absolute_error=float(np.abs(delta).max()),
                exact_pixel_fraction=float((retrained == released).mean()))


def export_released(images, weights, destination, device, tta_batch):
    """Export each released seed separately, retaining its uint8 rounding."""
    import torch
    from filseg.inference import load_model, read_jpeg, normalize, predict_map
    torch.backends.cudnn.benchmark = device.type == "cuda"
    for seed in range(2):
        directory = destination / str(seed)
        directory.mkdir(parents=True)
        model = load_model(weights / WEIGHTS[seed], device)
        for path in images:
            probability = predict_map(model, normalize(read_jpeg(path)), device, tta_batch)
            np.savez_compressed(directory / f"{path.stem}.npz", p=probability)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def cached_submission(images_dir, maps, weights, output, device, tta_batch):
    """Replay the unchanged entry point with two explicitly supplied seed maps."""
    from filseg import inference
    import submit
    current = {}
    original_read = inference.read_jpeg

    def read_image(path):
        current["stem"] = path.stem
        return original_read(path)

    def load_model(path, _device):
        return WEIGHTS.index(Path(path).name)

    def predict_map(seed, _normalized, _device, _tta_batch):
        with np.load(maps[seed] / f"{current['stem']}.npz") as archive:
            probability = archive["p"]
        if probability.shape != (2048, 2048) or probability.dtype != np.uint8:
            raise ValueError("invalid cached probability map")
        return probability

    command = ["submit.py", "--images", str(images_dir), "--final", "submitted",
               "--device", str(device), "--tta-batch", str(tta_batch),
               "--weights", *(str(weights / name) for name in WEIGHTS[:2]),
               "--selector", str(weights / WEIGHTS[2]),
               "--verifier-weights", str(weights / WEIGHTS[3]), "--out", str(output)]
    with patch.object(inference, "read_jpeg", read_image), \
         patch.object(inference, "load_model", load_model), \
         patch.object(inference, "predict_map", predict_map), patch.object(sys, "argv", command):
        submit.main()


def main():
    """Export release maps once and compare masks with all downstream models frozen."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--trained-maps", type=Path, required=True)
    parser.add_argument("--trained-checkpoint", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "weights")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tta-batch", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--expected-images", type=int, default=180)
    args = parser.parse_args()
    images = sorted(p for p in args.images.iterdir() if p.is_file() and p.suffix.lower() in (".jpeg", ".jpg"))
    stems = {path.stem for path in images}
    if args.expected_images < 1 or len(images) != args.expected_images or len(stems) != len(images):
        parser.error("unexpected test image count or duplicate stems")
    if args.out.exists():
        parser.error("--out must be new")
    for stem in stems:
        with np.load(args.trained_maps / f"{stem}.npz") as archive:
            probability = archive["p"]
        if probability.shape != (2048, 2048) or probability.dtype != np.uint8:
            parser.error(f"invalid trained probability map: {stem}")
    hashes = verify_weights(args.weights_dir)
    with args.trained_checkpoint.open("rb") as stream:
        trained_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    import torch
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    args.out.mkdir(parents=True)
    released = args.out / "released_maps"
    export_released(images, args.weights_dir, released, device, args.tta_batch)
    rows = []
    for path in images:
        with np.load(released / "0" / f"{path.stem}.npz") as old, \
             np.load(args.trained_maps / f"{path.stem}.npz") as new:
            rows.append(dict(stem=path.stem, **map_metrics(old["p"], new["p"])))
    write_csv(args.out / "probability_agreement.csv", rows)
    baseline, retrained = args.out / "released_replay.csv", args.out / "retrained_replay.csv"
    cached_submission(args.images, [released/"0", released/"1"], args.weights_dir, baseline, device, args.tta_batch)
    cached_submission(args.images, [args.trained_maps, released/"1"], args.weights_dir, retrained, device, args.tta_batch)
    from tools.check_submission import check_submission
    for path in (baseline, retrained):
        check_submission(path, stems, allow_empty_images=True)
    reference = ROOT / "docs/reference/submitted_test.csv"
    ref = read_rows(reference, stems, subset=True)
    old, new = read_rows(baseline, stems), read_rows(retrained, stems)
    summaries = [dict(comparison=name, **replay_summary(a, b, stems)) for name, a, b in (
        ("archive_vs_released_replay", ref, old),
        ("released_replay_vs_retrained", old, new),
        ("archive_vs_retrained", ref, new))]
    write_csv(args.out / "mask_agreement.csv", summaries)
    metadata = dict(released_weights_sha256=hashes, trained_checkpoint_sha256=trained_hash,
                    device=str(device), torch_version=torch.__version__,
                    scope="seed 0 replaced; seed 1, selector and verifier frozen; prediction agreement only")
    (args.out / "comparison_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
