# Second U-Net seed ablation

Question: does averaging the seed-0 and seed-1 U-Net maps (three-verifiers) beat one
U-Net? The main CV cannot answer it, because each held-out image is predicted
by one model.

1. `mk_maps.py` builds, for the 566 images of folds 0–3, the seed-1 held-out
   maps and the rounded mean of the seed-0 and seed-1 maps.
2. `run.sh` builds the candidate trees on both map sets with `tools/build_cands.py`.
3. `eval_lgb.py` trains the LightGBM selector per fold on seed-0 candidates
   (the released recipe) and scores seed 0, seed 1 and the mean with the
   exact antichain DP over a grid of λ. Its printed PQ uses the one-hit
   gain shortcut; `score.py` evaluates the selected masks with the exact
   host scorer, including rare multiple GT hits.
4. `build.py` rebuilds own-map test finals from cached per-seed test maps
   for the mean-map and single-seed variants. This is separate from an
   end-to-end JPEG replay against `docs/reference/`.

Result: [`report/data/second_seed_ablation.csv`](../../../report/data/second_seed_ablation.csv).
Set `FILSEG_WORK` (held-out seed-1 maps, GT pickle, `folds.csv`),
`FILSEG_CACHE` (seed-0 maps and candidate trees) and `TRAIN` (training
JPEGs); the cached maps are not in git.

The exact host rerun covers 566 images and 528 observing days. At lambda
0.23, seed 0 scores 0.44808, seed 1 0.44535 and the mean map 0.45172.
The paired gain over the arithmetic mean of the two single-seed PQs is
0.00500 [0.00217, 0.00808], with 2,000 common day draws and seed 23.
Per-fold mean-map minus seed-0 differences are +0.0008, +0.0081,
+0.0064 and -0.0002.

This is a four-fold, LightGBM-only comparison. The verifier and fifth
second-seed fold are unavailable. Each outer selector trains on all
seed-0 candidates outside that fold, including fold 4. The tree is rebuilt
on the rounded mean maps; this is not averaging finished predictions.
Several penalties were explored on the same cohort. The result cannot
be added to the final's five-fold PQ or pass the full-pipeline acceptance
rule. The original fixed-0.23, true-gain-informed diagnostic is also not
an optimized fractional oracle upper bound.

Recompute the paired summary and per-fold scores using only the public
[per-image terms](../../../report/data/second_seed_per_image.csv):

```sh
python -m tools.experiments.second_seed.score \
  --terms report/data/second_seed_per_image.csv \
  --summary data/report/second_seed_ablation.csv
```

The command also writes a sibling `_folds.csv`.
[Published fold scores](../../../report/data/second_seed_ablation_folds.csv)
reveal the negative fourth contrast. To regenerate terms from trusted
original intermediates, use `--work "$FILSEG_WORK" --cache "$FILSEG_CACHE"
--out data/report/second_seed_per_image.csv --summary data/report/second_seed_ablation.csv`
instead of `--terms`. All annotations are from the official training split.

The preparation, selector and cached-test drivers have guarded entry
points. Importing them or requesting `--help` loads no data or models.
The existing environment defaults and `run.sh` remain supported. Explicit
paths also work from the repository root:

```sh
python -m tools.experiments.second_seed.mk_maps --work WORK --cache CACHE
python -m tools.experiments.second_seed.eval_lgb --work WORK --cache CACHE
python -m tools.experiments.second_seed.build mean data/mean.csv --images TEST_IMAGES --maps CACHE/maps_own_test2
```

Repeat `--seed1-dir` with each exported `oof_control` directory when its
parent folder names differ from the original layout. The cached-test
driver writes verifier scores to a separate sibling `_verif.pkl`, even
if the submission path has no suffix. It does not run the pixel model.
