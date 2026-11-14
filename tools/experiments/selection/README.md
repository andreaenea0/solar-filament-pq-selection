# Frozen selection analyses

These commands analyze official **training** OOF predictions. They do not
change the release or run neural inference. Inputs are trusted pipeline
pickles: candidate trees, candidate-aligned blended gains and training
annotations as COCO RLEs. The folds CSV is the released image assignment.
Raw maps, labels and score arrays are not bundled with the public snapshot.

Install the optional calibration dependency after the project environment:

```sh
python -m pip install -r tools/experiments/selection/requirements.txt
```

## Current penalty curves

Use one `--model NAME CANDIDATES SCORES` triple per recipe. public-ensemble needs its
three-ResNet blend; three-verifiers needs three own-map verifiers; submitted needs seed 0.
All include the appropriate frozen LightGBM scores. Empty trees remain in
the cohort. The default grid is 0.15 through 0.35 in steps of 0.01.

```sh
python -m tools.experiments.selection.lambda_sweep_finals \
  --gt "$TRAIN_RLE" \
  --model public-ensemble "$PUBOWN_TREES" "$public-ensemble_GAINS" \
  --model three-verifiers "$OWN_TREES" "$B_GAINS" \
  --model submitted "$OWN_TREES" "$SUBMITTED_GAINS" \
  --out data/report/lambda_sweep_finals.csv
```

The [published table](../../../report/data/lambda_sweep_finals.csv) uses
the exact host scorer, 2,000 complete UTC-day resamples and seed 23.
The same days serve every penalty and recipe. Intervals condition on
frozen fits and selected settings. The old untagged, metadata-era
`lambda_sweep.csv` is retired. `sweep.py` is a generic single-recipe
wrapper with the historical 0.10–0.40 grid and 1,000 image draws, seed 0;
it does not select a historical cache implicitly.

## Calibration and the empirical fixed-point control

```sh
python -m tools.experiments.selection.calib \
  --name submitted --candidates "$OWN_TREES" --scores "$SUBMITTED_GAINS" \
  --gt "$TRAIN_RLE" --folds "$FOLDS" --penalty 0.25 \
  --out data/report/fixed_point_submitted.csv
```

This writes a pooled summary, a sibling `_folds.csv` with choices and
convergence status, and `_bins.csv` with drawing-count-weighted gain
reliability. Use `--penalty 0.27` for the public ensemble. `--score-mode both` is the
default: `host` preserves all strict hits; `one-hit` reproduces the
historical best-IoU shortcut, which misses rare multiple GT hits.

For each outer fold, isotonic fitting and penalty choice use the other
four folds. Upstream OOF maps and verifier fits remain fixed. This is
conditional downstream tuning, not a fully nested end-to-end estimate.
The empirical update is lambda <- measured PQ(lambda)/2, starting at
0.2, with 30 updates and tolerance 1e-5. Learned gains do not give the
oracle Dinkelbach guarantee. Some folds cycle; their reported score is
the last iterate, and `converged=False` and the residual are retained.

The [combined summary](../../../report/data/selection_fixed_points.csv)
and [fold choices](../../../report/data/selection_fixed_point_folds.csv)
record input precision and scikit-learn version. Preserve archived score
dtype: isotonic handling of nearly equal scores changes between float32
and float64. The raw reference logs are retained as
[`reference_calib_submitted.log`](reference_calib_submitted.log) and
[`reference_calib_public_ensemble.log`](reference_calib_public_ensemble.log); their one-hit quantities
must not be presented as exact host PQ.

`calib2.py` reproduces the local +/-0.03 gain window, 0.04-wide bins and
isotonic inverse lookup. It takes the same explicit single-recipe paths,
`--penalty`, optional `--targets`, `--bootstrap` and `--resampling`.
Its historical default is 1,000 **image** draws; the local table labels
the unit and empty draws. These are conditional calibration diagnostics.
