"""Fit the original four-fold seed-study selector without import-time jobs."""

import argparse
import os
import pickle
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from filseg.select import dp_select, feature_frame

PARAMS = dict(objective="regression", learning_rate=.03, num_leaves=31, min_data_in_leaf=40,
              feature_fraction=.8, bagging_fraction=.8, bagging_freq=1, lambda_l2=1., verbose=-1)
LAMS = np.round(np.arange(.13, .34, .02), 2)


def tot(trees, predictions, penalty, stems, gt):
    """Preserve the historical one-hit shortcut; use score.py for exact host PQ."""
    numerator = denominator = count = 0
    for stem in stems:
        tree = trees[stem]
        records = len(gt[stem])
        denominator += sum(len(values) for values in gt[stem].values())
        if len(predictions[stem]) == 0:
            continue
        selected = dp_select(tree["parent"], tree["children"], tree["level"], predictions[stem]-penalty)
        numerator += 2*records*float(np.sum(np.asarray(tree["g"])[selected]))
        denominator += records*len(selected)
        count += len(selected)
    return numerator/denominator, count


def read_pickle(path):
    """Load an explicitly supplied trusted local study artifact."""
    with path.open("rb") as stream:
        return pickle.load(stream)


def main(argv=None):
    """Retain the original four fits, row order, parameters and penalty diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=os.environ.get("FILSEG_WORK"))
    parser.add_argument("--cache", type=Path, default=os.environ.get("FILSEG_CACHE"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.work is None or args.cache is None:
        parser.error("Pass --work/--cache or set FILSEG_WORK/FILSEG_CACHE.")
    import pandas as pd
    import lightgbm as lgb
    seed0 = read_pickle(args.cache/"cands_train_own.pkl")
    mean = read_pickle(args.work/"cands_mean2.pkl")
    seed1 = read_pickle(args.work/"cands_s1.pkl")
    gt = read_pickle(args.work/"gt_rles_coco.pkl")
    folds = pd.read_csv(args.work/"folds.csv").set_index("stem")["fold"].to_dict()
    stems = sorted(mean)
    if not stems or mean.keys() != seed1.keys() or not set(stems) <= seed0.keys() & gt.keys() & folds.keys():
        parser.error("Study artifacts need the same nonempty paired image coverage.")
    print("images", len(stems), "cands/img s0", np.mean([len(seed0[s]["g"]) for s in stems]).round(1),
          "mean2", np.mean([len(mean[s]["g"]) for s in stems]).round(1), "s1", np.mean([len(seed1[s]["g"]) for s in stems]).round(1))
    frames = {s: feature_frame(seed0[s]) for s in seed0}
    predictions = {name: {} for name in ("s0", "mean2", "s1")}
    for fold in range(4):
        training = [s for s in seed0 if folds[s] != fold and len(seed0[s]["g"])]
        x = pd.concat([frames[s] for s in training])
        y = np.concatenate([seed0[s]["g"] for s in training])
        weights = np.concatenate([np.full(len(seed0[s]["g"]), seed0[s]["R"]) for s in training])
        model = lgb.train(PARAMS, lgb.Dataset(x, y, weight=weights), 600)
        for stem in stems:
            if folds[stem] != fold:
                continue
            for name, trees in (("s0", seed0), ("mean2", mean), ("s1", seed1)):
                frame = feature_frame(trees[stem])
                predictions[name][stem] = model.predict(frame[x.columns]) if len(frame) else np.zeros(0)
    output = args.out or args.work/"lgb_pred.pkl"
    with output.open("wb") as stream:
        pickle.dump(predictions, stream)
    for name, trees in (("s0", seed0), ("mean2", mean), ("s1", seed1)):
        values = {float(p): tot(trees, predictions[name], p, stems, gt) for p in LAMS}
        print(name, "PQ:", {p: round(v[0], 4) for p, v in values.items()})
        print(name, "masks/img:", {p: round(v[1]/len(stems), 2) for p, v in values.items()})
        diagnostic = tot(trees, {s: np.asarray(trees[s]["g"]) for s in stems}, .23, stems, gt)
        print(name, "true-g diagnostic", round(diagnostic[0], 4))
    for penalty in (.19, .21, .23, .25):
        differences = []
        for fold in range(4):
            held = [s for s in stems if folds[s] == fold]
            differences.append(round(tot(mean, predictions["mean2"], penalty, held, gt)[0]-tot(seed0, predictions["s0"], penalty, held, gt)[0], 4))
        print("lam", penalty, "per-fold mean2 - s0:", differences)


if __name__ == "__main__":
    main()
