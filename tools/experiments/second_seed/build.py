"""Rebuild own-map test finals from cached per-seed maps on CPU."""

import argparse
import os
import pickle
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def verifier_path(output):
    """Keep the derived verifier cache separate even when the output has no suffix."""
    return output.with_name(f"{output.stem}_verif.pkl")


def rebuild(mode, images, maps_directory, weights, output, threads=4):
    """Run the original candidate/verifier/DP path on fixed cached test maps."""
    import torch
    import lightgbm as lgb
    from filseg.inference import read_jpeg
    from filseg.oof import candidate_record
    from filseg.select import learned_masks
    from filseg.verifier import candidate_crops, load_verifier, verifier_gains
    from filseg.rle import write_submission

    device = torch.device("cpu")
    torch.set_num_threads(threads)
    selector = lgb.Booster(model_file=str(weights/"own_ver3_a05_l025_nometa_lgb.txt"))
    verifiers = [load_verifier(weights/"verifier_resnet18_s0.pt", device)]
    candidates, crops = [], []
    for path in images:
        image = read_jpeg(path)
        with np.load(maps_directory/f"{path.stem}.npz", allow_pickle=False) as archive:
            probabilities = archive["p"]
        if probabilities.shape != (2, *image.shape):
            raise ValueError(f"{path.stem}: expected two cached seed maps.")
        probability = (probabilities.astype(np.float32).mean(0).round().astype(np.uint8)
                       if mode == "mean" else probabilities[int(mode[1])])
        maps = np.stack((probability, probability))
        tree = candidate_record(path.stem, maps, image)
        candidates.append((path.stem, tree))
        crops.append(candidate_crops(image, maps, tree["rle"]))
    gains = verifier_gains(verifiers, np.concatenate(crops), device)
    offset, predictions, saved_gains = 0, [], {}
    for stem, tree in candidates:
        end = offset+len(tree["rle"])
        predictions.append((stem, learned_masks(tree, selector, .25, gains[offset:end])))
        saved_gains[stem] = gains[offset:end]
        offset = end
    write_submission(output, predictions)
    with verifier_path(output).open("wb") as stream:
        pickle.dump(saved_gains, stream)
    return predictions


def main(argv=None):
    """Validate inputs before loading models; retain the original MODE OUT syntax."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("mean", "s0", "s1"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--images", type=Path, default=os.environ.get("FILSEG_TEST_IMAGES"))
    cache = os.environ.get("FILSEG_CACHE")
    parser.add_argument("--maps", type=Path, default=Path(cache)/"maps_own_test2" if cache else None)
    parser.add_argument("--weights", type=Path, default=ROOT/"weights")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.images is None or args.maps is None:
        parser.error("Pass --images/--maps or set FILSEG_TEST_IMAGES/FILSEG_CACHE.")
    if not args.images.is_dir() or args.threads <= 0:
        parser.error("Provide an existing image directory and a positive thread count.")
    images = sorted((p for p in args.images.iterdir() if p.is_file() and p.suffix.lower() in {".jpeg", ".jpg"}), key=lambda p:p.stem)
    if not images or len({p.stem for p in images}) != len(images):
        parser.error("Provide JPEGs with unique image stems.")
    predictions = rebuild(args.mode, images, args.maps, args.weights, args.out, args.threads)
    print(args.mode, sum(len(m) for _,m in predictions), "filaments", len(predictions), "images")


if __name__ == "__main__":
    main()
