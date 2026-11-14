"""Fit the gain model on training candidates and write CSVs for given penalties."""

import argparse
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filseg.rle import write_submission
from filseg.select import feature_frame, learned_masks


def fit_gain_model(training):
    """Fit the existing 600-round, record-weighted LightGBM regression."""
    params = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=40,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1)
    records = [c for c in training.values() if len(c["g"])]
    features = pd.concat([feature_frame(c) for c in records])
    targets = np.concatenate([c["g"] for c in records])
    weights = np.concatenate([np.full(len(c["g"]), c["R"]) for c in records])
    return lgb.train(params, lgb.Dataset(features, targets, weight=weights), 600)


def main():
    """Keep the positional submission CLI and optionally export the fitted booster."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train", type=Path)
    parser.add_argument("test", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("penalties", help="Comma-separated penalties.")
    parser.add_argument("--save-model", type=Path, help="Write a LightGBM text model for submit.py.")
    args = parser.parse_args()
    with args.train.open("rb") as stream:
        training = pickle.load(stream)
    with args.test.open("rb") as stream:
        test = pickle.load(stream)
    model = fit_gain_model(training)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.save_model:
        model.save_model(str(args.save_model))
    for penalty in map(float, args.penalties.split(",")):
        predictions = [(stem, learned_masks(c, model, penalty)) for stem, c in sorted(test.items())]
        write_submission(args.out / f"sub_lam{penalty:.2f}.csv", predictions)
        print(penalty, sum(len(masks) for _, masks in predictions), "filaments on",
              sum(bool(masks) for _, masks in predictions), "images")


if __name__ == "__main__":
    main()
