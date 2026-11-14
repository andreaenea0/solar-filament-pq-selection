"""Fit candidate gains, blend verifier scores and select the additive-surrogate antichain."""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filseg.select import learned_masks
from tools.make_sub import fit_gain_model


def main(argv=None):
    """Preserve the six positional training arguments and exported booster/CSV format."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train", type=Path)
    parser.add_argument("test", type=Path)
    parser.add_argument("verifier", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("alpha", type=float, help="Weight of LightGBM gains in the blend.")
    parser.add_argument("penalty", type=float)
    args = parser.parse_args(argv)
    loaded = []
    for path in (args.train, args.test, args.verifier):
        with path.open("rb") as stream:
            loaded.append(pickle.load(stream))
    training, test, verifier = loaded
    for stem, candidates in test.items():
        if not candidates["rle"]:
            continue
        if stem not in verifier:
            raise ValueError(f"Verifier predictions missing for {stem}.")
        scores = np.asarray(verifier[stem])
        if scores.shape != (len(candidates["rle"]),) or not np.isfinite(scores).all():
            raise ValueError(f"Verifier/candidate misalignment or nonfinite scores on {stem}.")

    model = fit_gain_model(training)
    model.save_model(str(args.out).rsplit(".", 1)[0] + "_lgb.txt")
    rows = []
    for stem, candidates in sorted(test.items()):
        scores = np.asarray(verifier[stem]) if candidates["rle"] else None
        masks = learned_masks(candidates, model, args.penalty, scores, alpha=args.alpha)
        rows.extend((f"{stem}_{i}", mask["counts"].decode("ascii"))
                    for i, mask in enumerate(masks, 1))
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(args.out, index=False)
    print(args.out, len(rows), "filaments on", len({row[0].rsplit("_", 1)[0] for row in rows}), "images")


if __name__ == "__main__":
    main()
