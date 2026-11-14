"""Run local checkpoints on test JPEGs and write a submission CSV."""

import argparse
import json
from pathlib import Path

import numpy as np

from filseg.rle import write_submission
from filseg.rule import HIGH, LOW, MIN_AREA, MIN_MEAN, keep_rule


def main():
    """Run the keep rule or a released finalist recipe with explicit model paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Directory of test .jpeg or .jpg files.")
    parser.add_argument("--weights", type=Path, nargs="+", required=True, help="Own k07 .pt state dicts.")
    parser.add_argument("--out", type=Path, required=True, help="Output submission CSV.")
    parser.add_argument("--device", default="cuda", help="Torch device such as cuda:0 or cpu (default: cuda).")
    parser.add_argument("--tta-batch", type=int, choices=(1, 2, 4, 8), default=4,
                        help="U-Net transforms per forward pass (default: 4). Use 1 for a CPU check.")
    parser.add_argument("--low", type=float, default=LOW, help="Keep-rule component threshold.")
    parser.add_argument("--high", type=float, default=HIGH, help="Keep-rule peak threshold.")
    parser.add_argument("--min-area", type=int, default=MIN_AREA, help="Keep-rule minimum mask area in pixels.")
    parser.add_argument("--min-mean", type=float, default=MIN_MEAN, help="Keep-rule minimum mean probability.")
    parser.add_argument("--selector-model", type=Path, help="Optional own-map LightGBM model in text format.")
    parser.add_argument("--final", choices=("public-ensemble", "three-verifiers", "submitted", "single-unet"), default="submitted",
                        help="Finalist settings with --selector and --verifier-weights (default: submitted). "
                             "three-verifiers adds two verifiers; single-unet drops one U-Net; public-ensemble adds public maps.")
    parser.add_argument("--selector", type=Path, metavar="BOOSTER", help="Finalist LightGBM booster in text format.")
    parser.add_argument("--verifier-weights", type=Path, nargs="+", metavar="PT",
                        help="ResNet-18 checkpoints: one (seed 0) for submitted and single-unet, three for three-verifiers and public-ensemble.")
    parser.add_argument("--public-receipt", type=Path, help="Local k02 ensemble receipt for the public-ensemble recipe.")
    parser.add_argument("--penalty", type=float, help="Learned selection penalty (team-only: 0.25, public-ensemble: 0.27).")
    parser.add_argument("--trace", type=Path, help="Optional per-image finalist diagnostics JSON.")
    args = parser.parse_args()

    if not args.images.is_dir():
        parser.error(f"Image directory does not exist: {args.images}")
    images = sorted((p for p in args.images.iterdir() if p.is_file() and p.suffix.lower() in {".jpeg", ".jpg"}),
                    key=lambda path: path.stem)
    if not images or len({p.stem for p in images}) != len(images):
        parser.error("Provide at least one JPEG and use unique image stems.")
    uses_public_maps = args.final == "public-ensemble"
    default_penalty = 0.27 if uses_public_maps else 0.25
    penalty = default_penalty if args.penalty is None else args.penalty
    if not np.isfinite(penalty):
        parser.error("Selector penalty must be finite.")
    if bool(args.selector) != bool(args.verifier_weights):
        parser.error("--selector and --verifier-weights must be passed together.")
    if uses_public_maps != bool(args.public_receipt) or uses_public_maps and not args.selector:
        parser.error("Recipe public-ensemble requires --public-receipt, --selector and --verifier-weights.")
    n_unets = 1 if args.final == "single-unet" else 2
    if args.selector and (args.selector_model or penalty != default_penalty or len(args.weights) != n_unets):
        parser.error(f"Recipe {args.final} requires {'one' if n_unets == 1 else 'two'} own U-Net "
                     f"weight{'' if n_unets == 1 else 's'} and penalty {default_penalty}.")
    n_verifiers = 1 if args.final in ("submitted", "single-unet") else 3
    if args.verifier_weights and len(args.verifier_weights) != n_verifiers:
        parser.error(f"Recipe {args.final} requires {'one' if n_verifiers == 1 else 'three'} verifier weight{'' if n_verifiers == 1 else 's'}.")
    if args.trace and not args.selector:
        parser.error("--trace requires the finalist selector and verifiers.")

    import torch
    from filseg.inference import load_model, normalize, predict_map, read_jpeg

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable. Install a CUDA build of PyTorch or pass --device cpu.")
        torch.backends.cudnn.benchmark = True
    if uses_public_maps:
        from filseg.public_maps import load_public_models, public_map

        try:
            public_models = load_public_models(args.public_receipt, device)
        except (OSError, ValueError, KeyError) as exc:
            parser.error(str(exc))
    models = [load_model(path, device) for path in args.weights]
    selector = None
    if args.selector or args.selector_model:
        import lightgbm as lgb
        from filseg.oof import candidate_record
        from filseg.select import learned_masks

        selector = lgb.Booster(model_file=str(args.selector or args.selector_model))
    if args.selector:
        from filseg.verifier import candidate_crops, load_verifier, verifier_gains

        verifiers = [load_verifier(path, device) for path in args.verifier_weights]

    predictions, traces = [], {}
    final_candidates, final_crops = [], []
    for path in images:
        image = read_jpeg(path)
        normalized = normalize(image)
        total = 0
        seed_maxima = []
        for model in models:
            probability_map = predict_map(model, normalized, device, args.tta_batch)
            total = total + probability_map.astype(np.float32)
            if args.trace:
                seed_maxima.append(float(probability_map.max()) / 255)
        average = total / len(models)
        if selector is None:
            masks = keep_rule(average / 255, args.low, args.high, args.min_area, args.min_mean)
        else:
            probability = average.round().astype(np.uint8)
            if args.selector:
                first = public_map(public_models, normalized, device, args.tta_batch) if uses_public_maps else probability
                maps = np.stack((first, probability))
            else:
                maps = probability
            candidates = candidate_record(path.stem, maps, image)
            if args.selector:
                if args.trace:
                    traces[path.stem] = dict(
                        own_max_probability=float(probability.max()) / 255,
                        own_seed_max_probabilities=seed_maxima,
                        tree_max_probability=float(maps.astype(np.float32).mean(0).round().max()) / 255,
                        candidates=len(candidates["rle"]),
                        levels=np.bincount(np.asarray(candidates["level"], dtype=int), minlength=6).tolist(),
                        penalty=penalty,
                    )
                final_candidates.append((path.stem, candidates))
                final_crops.append(candidate_crops(image, maps, candidates["rle"]))
                continue
            masks = learned_masks(candidates, selector, penalty)
        predictions.append((path.stem, masks))

    if args.selector:
        # k17 consumes batches across stems in the sorted crops4 row order.
        torch.backends.cudnn.benchmark = False
        gains = verifier_gains(verifiers, np.concatenate(final_crops), device)
        offset = 0
        for stem, candidates in final_candidates:
            end = offset + len(candidates["rle"])
            blend = {"alpha": 0.35} if uses_public_maps else {}
            if args.trace:
                blend["trace"] = traces[stem]
            predictions.append((stem, learned_masks(candidates, selector, penalty, gains[offset:end], **blend)))
            offset = end

    write_submission(args.out, predictions)
    if args.trace:
        args.trace.write_text(json.dumps(traces, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {sum(len(masks) for _, masks in predictions)} filaments from {len(images)} images to {args.out}")


if __name__ == "__main__":
    main()
