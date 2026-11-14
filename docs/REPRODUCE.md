# Reproduce the submission and ablations

Run the inference commands from the repository root after the
[environment setup](../README.md#run-the-pipeline-notebook). Set
`TEST` to the directory containing the official test JPEGs. The
checkout includes the submitted model, its team-trained variants and the
public-ensemble ablation.

Pass the model paths in each complete command below. `--final` selects
settings and does not load those paths automatically. For the own-map
recipes, omitting both selector options runs the legacy keep-rule baseline.

## Submitted replay

```sh
PYTHON=python bash make_release.sh weights
python submit.py --images "$TEST" --final submitted \
  --weights weights/effv2s_s0_fold-1.pt weights/effv2s_s1_fold-1.pt \
  --selector weights/own_ver3_a05_l025_nometa_lgb.txt \
  --verifier-weights weights/verifier_resnet18_s0.pt \
  --out submission-submitted.csv
```

The archive for comparison is
[`submitted_test.csv`](reference/submitted_test.csv).
The archived file uses LF line endings. Its SHA-256 is `3ab827b24424b2c9162e83d59038389b7dd0790eaa356af4cab90f0c24b18ba7` (312,730 bytes). The historical score receipt preserves the hash of its Windows CRLF copy. On the documented CPU observation, one mask encoding differed from the original GPU output; both masks matched at IoU above 0.999.

For the three-verifier variant, pass `--final three-verifiers` and all three verifier seeds.
For single-unet, pass `--final single-unet` and only the seed-0 U-Net and
seed-0 verifier.
[Three-verifier archive](reference/three_verifiers_test.csv)
· [single-unet archive](reference/single_unet_test.csv)

## Public-ensemble ablation replay

Set `FINAL = "public-ensemble"` in `pipeline.ipynb` or use `FILSEG_FINAL=public-ensemble`
when launching it. The [Public-ensemble manifest](../release_weights_public_ensemble.sha256)
verifies the team files, public checkpoints, and ensemble receipt.
[Afshar attribution and weight licence](../WEIGHTS_LICENSE.md)

```sh
PYTHON=python bash make_release.sh --final public-ensemble weights
python submit.py --images "$TEST" --final public-ensemble \
  --weights weights/effv2s_s0_fold-1.pt weights/effv2s_s1_fold-1.pt \
  --public-receipt weights/public/solar_unet_receipt.json \
  --selector weights/public_ensemble/lgb_pubown_nometa.txt \
  --verifier-weights weights/public_ensemble/verifier_resnet18_s0.pt weights/public_ensemble/verifier_resnet18_s1.pt weights/public_ensemble/verifier_resnet18_s2.pt \
  --out submission-a.csv
```

The archive for comparison is
[`public_ensemble_test.csv`](reference/public_ensemble_test.csv).

## CPU smoke test

Pass `--device cpu --tta-batch 1` and point `TEST` at a directory
with one or two original test JPEGs. The notebook validates the CSV
and displays the masks. For CLI output, run:

```sh
python tools/check_submission.py submission-submitted.csv --images "$TEST" --allow-empty-images
```

The [methods and results](RESULTS.md#replay-and-cost) report the
matched CPU timings and mask agreement.

To time one observation and compare its IDs and masks with the release
archive, use the same Python environment as inference:

```sh
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m tools.jury_cpu_check \
  --images "$TEST" --weights-dir weights --final submitted \
  --reference docs/reference/submitted_test.csv \
  --stems 20110329082654Uh --out-dir data/cpu-check-submitted
```

The output directory must be new. Omit `--stems` for the first, middle
and last JPEGs, or give two stems for a two-image check. Times include
process startup and model loading. The summary reports shared IDs,
byte-identical RLEs and strict IoU matches. The replay reproduces the
archive up to floating-point differences: bit-exact identity depends on
hardware and libraries, because floating-point arithmetic can move
boundary pixels. On one T4 GPU the snapshot's submission command matched
all 1,150 archived masks across 180 images (minimum IoU 0.890) and returned
one extra mask. [Measured GPU parity and timings](RESULTS.md#one-t4-gpu-all-180-test-images).

In PowerShell, use `$python` from the README environment setup:

```powershell
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
& $python -m tools.jury_cpu_check --images $env:FILSEG_IMAGES --weights-dir weights --final submitted --reference docs/reference/submitted_test.csv --stems 20110329082654Uh --out-dir data/cpu-check-submitted
```

## One-GPU timing replay

The [timing notebook](../kernels/release/submitted_timing.ipynb) runs all
180 test JPEGs on one CUDA device using the public snapshot and released
submitted weights. Edit its three mounted input paths. It uses no private
caches or test labels and does not upload a submission. In an installed
CUDA environment, the equivalent command is:

~~~sh
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m tools.timing_replay \
  --images "$TEST" --weights-dir weights --out-dir data/submitted-gpu-timing
~~~

The output directory must be new. The tool verifies the four checkpoint
hashes, calls the released inference entry point with temporary wrappers,
then validates the CSV and compares it with the bundled submitted predictions.
It writes flat stage timings, a replay summary and version/hash metadata.
Agreement is between predictions, not a measurement of test PQ.

Stage times are exclusive. Feature extraction inside tree construction
and selection is counted once under features. Tree time includes candidate
RLE encoding; RLE/CSV measures final serialization. GPU stages synchronize
CUDA. Total time covers runtime imports through CSV writing; checksum
preflight and reference comparison are excluded. Peak CPU memory is this
process's OS high-water resident memory before comparison; GPU allocated
and reserved peaks are reported separately. This is one instrumented run,
without a warm-up image; timer and synchronization overhead are included.

For a CPU parity check, point at a one-image JPEG directory and add
`--device cpu --tta-batch 1 --expected-images 1`. Mask matching is restricted
to each observation and uses strict IoU >0.5, independently of instance IDs.
The summary also reports shared IDs and byte-identical RLEs. A GPU run is
still required to report GPU timings; CPU measurements do not establish
GPU speed.

## Optional single-seed training diagnostic

The [one-GPU training notebook](../kernels/release/retrain_seed0.ipynb)
repeats only k07 seed 0, FOLD=-1 and 12,000 updates. It needs official
training/test images, the official training JSON and the released weights.
Training also needs the public ImageNet encoder initialization, downloaded
with Internet enabled or supplied in a compatible local cache.

~~~sh
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m training.retrain_seed0 \
  --input "$INPUT" --out data/seed0-training
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m tools.compare_retrained_seed \
  --images "$TEST" --trained-maps data/seed0-training/test_effv2s_s0 \
  --trained-checkpoint data/seed0-training/effv2s_s0_fold-1.pt \
  --weights-dir weights --out data/seed0-comparison
~~~

Both output directories must be new and outside the input tree. The
driver imports the released k07 functions and calls its seed-0 worker
directly. It skips unused neighbour decoding: with temporal=False, the
sampler always reads the annotated frame at index zero, without an extra
random draw. The original two-worker launcher retains its default
preparation. For FOLD=-1 the bookkeeping table may use any nonnegative
fold value; all 707 observations are included and no validation fold is
exported. The diagnostic writes that table itself.

Planning estimate for one T4: roughly 3-5 wall hours, expected to fit
10 GPU hours with one allocated GPU. This is not a measured single-T4
benchmark. The archived concurrent two-device job printed 7,775 s for
seed-0 training plus test-map export and 464 s for shared preparation.
The estimate allows additional released-map exports and mask comparisons;
hardware, dependencies and download time can change it.
[Original timer evidence](../report/data/training_timings.csv)

The comparison keeps released seed 1, LightGBM and verifier fixed. It
exports released maps, then calls the released submission entry point with
each pair of cached seed maps. It reports per-image probability MAE/RMSE
and exact pixel fraction, plus mask matches, mean/min matched IoU and
PQ-style agreement. A released-map replay versus the archive separates
inference precision differences from retraining differences. These are
agreements between predictions, never test scores. Training metadata
records the recipe and training-JSON hashes, device, versions and timings.

A new CUDA fit need not match checkpoint bytes or masks exactly. This
diagnostic does not qualify a model improvement and does not replace
shipped weights. Full end-to-end retraining of both U-Nets and downstream
models is the separate procedure below.

## Optional training of the team-only recipes

Install the project requirements and a CUDA PyTorch build on Linux
with two CUDA devices. Put the official training JSON, the training
and test JPEG directories, Afshar's `folds.csv`, and the GONG
neighbour-frame directories beneath one input directory. Keep the
original JPEG names. The training loader expects the neighbour
directories even for the annotated-frame control.
Keep the input directory separate from the new training-output directory.

```sh
INPUT=/path/to/inputs
TRAIN="$INPUT/MAGFiLO_1.0_Kaggle_2026/train/train_images"
python training/fetch_gong_neighbours.py --images "$TRAIN" --out "$INPUT/neighbours"
python -m training.retrain_team_models --input "$INPUT" --out data/retrained-b
```

The [training driver](../training/retrain_team_models.py) runs the
[pixel control](../kernels/k06-dice/train.py), the remaining
held-out folds, and [both final U-Net seeds](../kernels/k07-final/final.py).
It prepares the candidate tree and crops, trains
[the verifier](../training/k17-verifier-own-final/verifier.py),
and [fits the LightGBM selector](../training/make_sub_blend.py).
Use the resulting `data/retrained-b/weights` paths to replay that
fit. The release manifest identifies the shipped checkpoints.

The driver writes the full three-verifiers checkpoint bundle. Its exported training
submission averages all three verifiers. To replay the selected
submitted recipe after retraining, use the submitted command above with
the two retrained U-Nets, retrained booster and seed-0 verifier.

## Training recipe

These settings describe the executing sources. The GPU training
commands above produce a new fit; the published hashes identify the
shipped fit. CUDA kernels and training randomness can prevent exact
weight-byte reproduction.

| Stage | Settings | Source |
| --- | --- | --- |
| Pixel targets and input | One grayscale channel at 2048 × 2048. Union each drawing's polygons, average the drawings, round the soft target to uint8. Normalize using per-image mean/std in the 950 px aperture, with std floor 1e−3. | [Pixel recipe](../kernels/k07-final/final.py) |
| Crop sampling | 512 px crops; 70% foreground-centering branch, sampling from a stride-8 target grid; ±128 px inclusive jitter, with coordinates clipped to the image. Empty grids fall back to random centering. | `sample_crop` in the pixel recipe |
| Augmentation | Uniform quarter-turns and horizontal flip; gamma U(0.8,1.25), contrast U(0.9,1.1) before normalization. | `sample_crop` |
| Pixel architecture and loss | EfficientNetV2-S U-Net with ImageNet initialization (`in21k_ft_in1k`); BCE plus soft Dice pooled over the full batch, with additive smoothing 1. | `dice_loss` and `run` |
| Pixel optimization | AdamW, batch 16, lr 3e−4, weight decay 1e−4, 300-step warmup and cosine schedule; fp16 mixed precision. | Pixel recipe |
| Fits and folds | 10,500 steps for each of five held-out image folds; 12,000 steps for each all-data seed 0 and 1. k06 supplies fold 0, k09 folds 1–2, k11 folds 3–4, k07 both final seeds. | [Training driver](../training/retrain_team_models.py) |
| Training frames | Final pixel models select annotated frames only (`temporal=False`). The archived preparation code still decodes mounted neighbour files; these are not sampled by the final models. | Pixel `prepare` / `sample_crop` / launcher |
| Probability export | Eight D4 views, average inverse-aligned probabilities, round to uint8. At test time round each seed map first, then average/round the two maps. | [Inference](../filseg/inference.py), [map preparation](../training/prepare.py) |
| Candidate tree | Eight-connected components at 0.15, 0.25, 0.35, 0.50, 0.65, 0.80. The byte-map comparison is strict `p_u8 > round(255 × threshold)`. Minimum area 30 px; feature padding 8 px. | [Tree extraction](../filseg/candidates.py) |
| Verifier input | Four channels: JPEG, own probability, own probability, candidate mask. Crop side `int(1.5 × max(width,height)) + 16`; resize to 96 × 96, then divide byte values by 255. | [Crop preparation](../training/crops4.py), [release verifier](../filseg/verifier.py) |
| Verifier fit | ImageNet-initialized ResNet-18, one raw gain output, MSE weighted by annotation-record count R; 12 epochs, batch 256, AdamW lr 1e−3 / weight decay 1e−4, OneCycle with 10% warmup; incomplete final batches are discarded; rotations/flips in training and D4 raw-output mean at prediction. The driver fits seeds 0,1,2; submitted reads seed 0. | [Verifier training](../training/k17-verifier-own-final/verifier.py) |
| LightGBM fit | 31 image-derived features, target mean matched IoU per drawing, row weight R; regression, lr .03, 31 leaves, minimum 40 rows/leaf, feature/bagging fraction .8, bagging frequency 1, L2 1, 600 rounds. | [Selector fit](../training/make_sub_blend.py) |
| Submitted selection | Equal LightGBM/verifier weights; penalty .25; exact maximum-weight tree antichain for the estimated penalized gains. | [Entry point](../submit.py), [selection](../filseg/select.py) |

The [design table](DESIGN_CHOICES.md) distinguishes
measured ablations from fixed defaults and historical run summaries.

The public sources include the original [k05 per-crop Dice control](../kernels/k05-recipe/train.py)
and [k06 batch-Dice control](../kernels/k06-dice/train.py), plus
[k03](../kernels/k03-train/train.py). The k05/k06 comparison changes
the Dice reduction. k03 also differs in augmentation and step count,
so its difference from k06 is a bundled recipe comparison. These
archived controls retain their original Kaggle mount paths.
[Per-image fold-0 evidence](../report/data/dice_ablation_fold0_per_image.csv)

### Recorded training cost

The original Kaggle log events were retrieved again and parsed with
[the timing extractor](../tools/figures/training_timings.py). These are
the jobs' printed timers, not controlled hardware benchmarks or total
job wall times. Pixel workers run in parallel on two CUDA devices.

| Pixel recipe | Shared preparation (s) | Longest worker: training and map export (s) |
| --- | ---: | ---: |
| k06, fold-0 control/comparison job | 438 | 6,923 |
| k09, folds 1–2 | 497 | 6,645 |
| k11, folds 3–4 | 525 | 6,399 |
| k07, both final seeds | 464 | 7,775 |

The verifier's printed final-epoch timers are 138, 139 and 131 s for
seeds 0, 1 and 2. They exclude subsequent prediction/export and do
not include process startup. Shared preparation is separate from
worker timers. Repeated worker labels in the source log are numbered
by event order, without inferring their GPU or fold identity.
[Per-worker timing evidence](../report/data/training_timings.csv)

To regenerate the CSV from saved structured logs:

```sh
python -m tools.figures.training_timings --logs LOG_DIRECTORY --out data/training-timings.csv
```

The parser expects `k06-training.log`, `k09-training.log`,
`k11-training.log`, `k07-training.log`, and `k17-training.log`, each
containing its original JSON log-event list. The repository reports
no measured end-to-end time for the full local retraining driver.

## Replay the submitted verifier ablation

The public per-image host terms reproduce both removal controls and
their paired day/block intervals without training or a private cache:

~~~sh
python -m tools.experiments.submitted_verifier_ablation \
  --terms report/data/submitted_verifier_per_image.csv --out data/submitted-verifier-check
~~~

To rebuild from masks, use the official training JSON, the frozen OOF
candidate/gain cache and the published submitted OOF predictions:

~~~sh
python -m tools.experiments.submitted_verifier_ablation \
  --cache data/cache --gt "$TRAIN_JSON" \
  --predictions report/data/oof_submitted.csv --out data/submitted-verifier-rebuild
~~~

The output directory must be new. The cache uses the same candidate
trees and no-metadata OOF LightGBM gains as the existing selector studies.
The rebuild checks disjointness and calls the exact host scorer; it
writes input hashes, per-image terms, fold results, penalty choices and
1/7/28-day paired intervals. No new model or test labels are used.
The intervals keep predictions and penalty choices fixed.

## Replay recovered historical pixel controls

These public per-image host terms need no private maps or GPU:

~~~sh
python -m tools.experiments.pixel_controls --terms report/data/pixel_capacity_per_image.csv --control k06 --out data/pixel-capacity-check
python -m tools.experiments.pixel_controls --terms report/data/pixel_multiscale_per_image.csv --control scale1 --out data/pixel-scale-check
~~~

For a native reconstruction, replace `--terms` with `--gt "$TRAIN_JSON"`,
`--folds data/cache/folds.csv` and repeated `--map NAME=DIRECTORY` arguments
pointing to the original own-model held-out maps. The source validates the
complete released fold-0 cohort and native uint8 shapes. Use
`--mean NAME=MEMBER,MEMBER` for the declared unrounded byte-scale means.
Every contrast retains the original fixed rule; it does not refit a model,
retune its thresholds or estimate a five-fold final improvement.
[Scope and results](ABLATIONS.md#recovered-fold-0-pixel-controls)

## Rebuild the manuscript figures

The [paper map](PAPER_MAP.md) links all six print-ready figures to their inputs and source functions. From the repository root:

~~~sh
OUT=report/figs
python -m report.observed_errors --images "$TRAIN_IMAGES" --annotations "$TRAIN_JSON" --predictions report/data/oof_submitted.csv --out "$OUT"
python -m report.overlap_distributions --out "$OUT"
python -m report.cost_accuracy --out "$OUT"
python -m tools.figures.physics_figure --table report/data/physics_pred_vs_gt.csv --instances report/data/physics_instances.csv --radius report/data/pq_by_radius.csv --out "$OUT/observed_morphology.pdf"
~~~

Figures 1 and 2 additionally need the team's training-time held-out map, tree, booster scores and aligned seed-0 verifier export. Pass their paths as follows:

~~~sh
for module in observed_selection observed_pipeline; do
  python -m report.$module --cache "$CACHE" --verifier "$VERIFIER" --images "$TRAIN_IMAGES" --annotations "$TRAIN_JSON" --predictions report/data/oof_submitted.csv --out "$OUT"
done
~~~

The generator requires an exact match with all nine held-out masks on the selected training observation before it draws a crop. To rebuild the separate verifier-control plot, minority-support example and numerical provenance table, run:

~~~sh
python -m tools.figures.paper_submitted --gt "$TRAIN_JSON" --images "$TRAIN_IMAGES" --out data/figure-evidence
~~~

Each entry in the [claim ledger](../report/data/paper_claims.csv) records its source and full numerical value.

## Run the checks

After installing the inference environment, install pytest and run:

```sh
python -m pip install pytest
python -m pytest -q
```

The configuration collects `tests/` only, so saved clones or snapshots
under ignored `data/` do not enter test discovery. Set
`MAGFILO_TRAIN_JSON` to the official **training** annotation JSON to
also exercise its schema and rasterization check. Synthetic tests
need no competition data. Optional selection-analysis tests use the
dependencies in `tools/experiments/selection/requirements.txt`.

The public snapshot builder also checks that local Markdown links
point to selected release files or directories. A guide present only
in the private checkout cannot satisfy a public link.
