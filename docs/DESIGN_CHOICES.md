# Design choices: every constant of the pipeline

One row per constant used by the final submitted, its variants three-verifiers and single-unet, and the public-ensemble ablation.

**Origin** is one of four classes:

- **Task**: fixed by the challenge (data, metric, format).
- **Theory**: derived from the metric or the data geometry.
- **CV**: tuned on the 5-fold held-out predictions; the evidence column gives the source and the sensitivity.
- **A priori**: inherited from a public baseline or fixed without tuning. We say so honestly; where we later tested an alternative, the row says what happened.

Numerical comparisons below link to released CSVs. Historical pixel controls
use one frozen training seed and one fold; their day intervals exclude
training-seed variation. Unrecovered run-log numbers are not used as
quantitative evidence for the current design.
Code references are to the release commit. CV numbers are host PQ pooled over the 1,154 annotator records of the 707 training images.

## 1. Data and metric

A [held-out decision example](DECISION_EXAMPLE.md) shows why one prediction
can improve PQ even when it matches only one of three drawings. It links
the metric-level decision to visible annotation disagreement.

| Constant | Value | Where | Origin | Evidence |
| --- | --- | --- | --- | --- |
| Image size | 2048 × 2048 | `filseg/inference.py` | Task | GONG H-alpha JPEGs as distributed. |
| Match threshold | IoU > 0.5 | `filseg/metric.py` | Task | Host PQ; our scorer reproduces the host self-evaluation (annotator-vs-annotator PQ 0.3399, [`report/data/ablation.csv`](../report/data/ablation.csv)). |
| Pooling | PQ terms pooled over annotator records | `filseg/metric.py` | Task | Host metric, including all strict hits when drawn instances overlap. See the [multiple-hit exception](SELECTION_THEORY.md#overlapping-ground-truth-drawings-the-exception). |
| Inference inputs | Image pixels only, no file-name metadata | `filseg/oof.py`, `tests/test_no_filename_metadata.py` | Task | Host ruling (Kaggle topic 753665). Site and year features parsed from the file name were removed on 9 Nov; CV moved 0.4558 → 0.4557 (three-verifiers). |

## 2. Solar disk and normalisation

| Constant | Value | Where | Origin | Evidence |
| --- | --- | --- | --- | --- |
| Normalisation aperture | 950 px radius around the image centre | `filseg/inference.py`, `kernels/k07-final/final.py` | A priori (inherited from Afshar's public recipe) | Covers the disk (radius ≈ 904 px) plus a margin. Not tuned. |
| Normalisation | Per-image mean and std inside the aperture | `filseg/inference.py` | A priori (inherited) | Same as training. Not tuned. |
| Disk radius for the `rho` features | 900 px | `filseg/candidates.py` | Task (approximate) | 900 px is the feature-scale approximation. Across 707 training images the fitted radius has mean 903.6 px and population standard deviation 0.9 px ([fits](../report/data/disks_train.csv), [script](../tools/figures/disk_check.py)). The 0.4% difference only rescales two LightGBM features. |
| Disk centre | Image centre ((2047)/2, (2047)/2) | `filseg/candidates.py`, `filseg/inference.py` | Task | Fitted centre (1023, 1024) ([`tools/figures/disk_check.py`](../tools/figures/disk_check.py)). |

## 3. Segmentation U-Net (own maps, used by all recipes)

| Constant | Value | Where | Origin | Evidence |
| --- | --- | --- | --- | --- |
| Architecture | U-Net, EfficientNetV2-S encoder (ImageNet-21k), 1 input channel | `filseg/inference.py`, `kernels/k07-final/final.py` | A priori (inherited), exploratory check | Native fold-0 read-back: EfficientNetV2-M 0.4467 and longer EfficientNetV2-S training 0.4445 vs 0.4525 control. Both paired day intervals include zero. These single-seed points do not establish an architecture ranking or five-fold gain. [Data](../report/data/pixel_capacity_summary.csv). |
| Target | Soft mean of the annotators' masks | `kernels/k07-final/final.py` | A priori (inherited) | Not tuned. |
| Loss | BCE + soft Dice summed over the whole batch | `kernels/k07-final/final.py` | CV | Single-factor ablation on fold 0 (k05 per-crop Dice vs k06 batch Dice, the only difference is the Dice reduction; fixed keep rule): 0.4111 → 0.4525, **+0.041 [0.031, 0.052]** (paired day bootstrap, 2,000 draws; [data](../report/data/dice_ablation_fold0_per_image.csv), [script](../tools/experiments/dice_ablation_fold0.py), [fold-0 table](../report/data/ablation_fold0.csv)). Recomputed k03 versus k06 is +0.03090 [0.01852, 0.04191] and bundles augmentation and training-step changes (see the next rows); it is not the single-factor Dice effect. |
| Crop | 512 px, 70% centred on a filament, ±128 px jitter | `kernels/k07-final/final.py` | A priori (inherited) | No isolated current five-fold crop comparison. Exploratory resized-context runs are not used as quantitative justification here. |
| Augmentation | rot90/flip, gamma U(0.8, 1.25), contrast U(0.9, 1.1) before normalisation | `kernels/k07-final/final.py` | A priori; bundled fold-0 evidence, single seed | Switching from gain/offset jitter after normalisation to this, together with 9,000 → 10,500 steps, gave −0.0105 [−0.0206, −0.0001] under per-crop Dice (k03 0.4216 vs k05 0.4111). This bundles augmentation and update count; it was not retested under batch Dice. [Per-image terms](../report/data/dice_ablation_fold0_per_image.csv). |
| Optimiser | AdamW, lr 3e-4, weight decay 1e-4, batch 16 | `kernels/k07-final/final.py` | A priori (inherited) | Not tuned. |
| Schedule | 300 warm-up steps, cosine decay | `kernels/k07-final/final.py` | A priori (inherited) | Not tuned. |
| Training steps | 10,500 for held-out models; 12,000 for the final all-data models | `training/retrain_team_models.py` | A priori; exploratory fold-0 checks | The augmentation/update-count contrast is bundled, as above. The recovered 21,000-step control has a lower point estimate but an unresolved paired interval ([data](../report/data/pixel_capacity_summary.csv)). The all-data update count was not compared with the held-out setting. |
| Temporal neighbours | Off (annotated frames only) | `kernels/k07-final/final.py` (`temporal=False`) | CV | Neighbour frames lost −0.008 on fold 0: 0.4447 vs 0.4525 ([`report/data/ablation_fold0.csv`](../report/data/ablation_fold0.csv)). |
| Precision | fp16 mixed precision in training | `kernels/k07-final/final.py` | A priori | Not tuned. |
| Test-time augmentation | D4 (8 views), mean of probabilities | `filseg/inference.py` | A priori (inherited), exploratory check | Recovered fold-0 scale maps: averaging scales 1.0 and 1.25 gives +0.00209 [−0.00479, +0.00890] under the fixed keep rule. This does not estimate a five-fold full-pipeline gain or training-seed variation. [Data](../report/data/pixel_multiscale_summary.csv). No paired no-D4 ablation exists. |
| Number of U-Net seeds | 2 (seeds 0 and 1), probabilities averaged; single-unet keeps seed 0 only | `kernels/k07-final/final.py`, `submit.py` | A priori, then checked by a dedicated CV ablation | The original CV predicts each image with ONE U-Net, so the second seed needed its own test. Ablation on the 566 images of folds 0–3 (both seeds have held-out maps there): the tree is built on the rounded mean of the two maps, as at test time; LightGBM is trained on seed-0 candidates, as in the released selector. At λ 0.23, seed 0 0.4481, seed 1 0.4453, mean of two 0.4517: +0.005 [0.002, 0.008] over the average single seed (+0.004 [0.001, 0.007] at λ 0.21 and 0.25; paired day bootstrap, 2,000 draws). Per fold vs seed 0: +0.0008, +0.0081, +0.0064, −0.0002; three of four folds improve. Limits: LightGBM-only selector (per-fold verifiers not saved), 4 of 5 folds. Cost: one more 22.1M-parameter U-Net and twice the U-Net time. This partial study does not estimate the gain of the full five-fold final or justify adding its effect to the reported CV. [Data](../report/data/second_seed_ablation.csv) · [scripts](../tools/experiments/second_seed/README.md). |
| Map quantisation | uint8, `clip(round(p·255))`, NumPy round half to even | `filseg/inference.py`, `submit.py` | A priori (inherited from the k02 map format) | Quantisation and precision can affect components or selection near a cutoff. Independent within-image read-back of historical metadata-three-verifiers k21 confirms minimum IoU 0.83 and median 0.9995 among changed masks ([data](../report/data/historical_three_verifiers_replay.csv)); this is not the current final. Current submitted CPU agreement is reported separately in [results](RESULTS.md#replay-and-cost). |

## 4. Candidate tree

| Constant | Value | Where | Origin | Evidence |
| --- | --- | --- | --- | --- |
| Threshold ladder | 0.15, 0.25, 0.35, 0.50, 0.65, 0.80 | `filseg/candidates.py` | A priori, then checked | On all 707 own-map OOF images, fourteen interleaved levels raise the exact pooled host oracle from 0.56048 to 0.56142: +0.000939 [+0.000660, +0.001273], below the predeclared 0.01 screen ([data](../report/data/candidate_ceiling_paired.csv)). Selecting across the released levels beats the best single level by +0.0027 (three-verifiers) and +0.0022 (A) ([ablations](ABLATIONS.md)). |
| Connectivity | 8-connected | `filseg/candidates.py` | A priori | Same as the public keep rule. |
| Minimum candidate area | 30 px | `filseg/candidates.py` | A priori, data-backed | No sweep. Only 1 of 8,199 annotated filaments is below 30 px (6 below 100 px, minimum 9 px). The floor suppresses tiny components; it can also remove genuine small filaments or low-confidence fragments. |
| Feature bounding-box padding | 8 px | `filseg/candidates.py` | A priori | Only affects the context ring of the `contrast` feature (6 dilations). Not tuned. |
| Tree map for public-ensemble | Rounded mean of the public and own channels | `filseg/oof.py` | A priori | 50/50 weight not tuned. The separate public-map keep rule scores 0.4522 with a different candidate/selector recipe ([`report/data/ablation.csv`](../report/data/ablation.csv)). |

## 5. Gain model (LightGBM)

| Constant | Value | Where | Origin | Evidence |
| --- | --- | --- | --- | --- |
| Target | g = mean over records of best IoU·1[IoU > 0.5] | `filseg/oof.py` | Theory | The per-candidate change of the PQ numerator under the one-hit assumption; see the Penalty row and [multiple-hit exception](SELECTION_THEORY.md#overlapping-ground-truth-drawings-the-exception). |
| Sample weight | Number of annotator records R of the image | `tools/make_sub.py` | Theory | The image contributes R·g to the numerator summed over its records. |
| Features | 31 image-derived features (23 tree/shape/photometric + 8 per-image and map-family) | `filseg/select.py` | A priori, exploratory checks | Historical extra features raised R² 0.463 → 0.476; LGB-only PQ was 0.4542 vs 0.4531. The 0.4628 blend also added verifier v12, with paired change +0.0008 [−0.0019, +0.0034] over an older public-ensemble selector. It does not isolate a feature gain in current submitted. [Data](../report/data/postproc_ablation.csv). Metadata removed by host rule. |
| Parameters | regression, lr 0.03, 31 leaves, min 40 per leaf, feature/bagging fraction 0.8, L2 1.0, 600 rounds | `tools/make_sub.py` (`fit_gain_model`, shared by the training CLIs) | A priori | No isolated current parameter sweep. A historical second-level stacker scored 0.4589 vs 0.4620 ([data](../report/data/postproc_ablation.csv)); upstream fits were reused. |

## 6. CNN verifier

Keeping one verifier has a current submitted removal control: +0.0066 PQ
[+0.0037, +0.0097] over LightGBM alone with its penalty chosen on the other
four folds; all five differences are positive. This supports the
verifier's presence. The particular ResNet-18 architecture remains an
inherited choice, with the larger bundled variant checked below.
[Paired control](../report/data/submitted_verifier_summary.csv)


| Constant | Value | Where | Origin | Evidence |
| --- | --- | --- | --- | --- |
| Architecture | timm `resnet18.a1_in1k`, 4 input channels, 1 output | `filseg/verifier.py` | A priori, exploratory check | A historical bundled expansion (5 channels, 128 px, 2× context, additional ResNet-34 and blending) gave +0.0011 [−0.0010, +0.0033] on an older public-ensemble selector. It does not isolate architecture or establish a current submitted gain. [Data](../report/data/postproc_ablation.csv), verifier v12. |
| Input channels | image, public map, own map, candidate mask (three-verifiers: own map twice) | `filseg/verifier.py` | A priori | Not tuned. |
| Crop | Nominal side int(1.5 × max(w, h)) + 16 px around the candidate, clipped to the image and resized to 96 × 96 (INTER_AREA) | `filseg/verifier.py`, `training/crops4.py` | A priori | No isolated crop sweep; the larger-input/context variant above bundles several changes. |
| Training | 12 epochs, batch 256, AdamW lr 1e-3, weight decay 1e-4, OneCycle (10% warm-up), MSE weighted by R | `training/k17-verifier-own-final/verifier.py` | A priori | Never swept: later verifier runs changed inputs or splits, not these values. |
| Test-time augmentation | D4 mean of raw outputs | `filseg/verifier.py` | A priori | Not tuned. |
| Number of verifier seeds | 1 (seed 0) in the final submitted; 3 in three-verifier variant and in A | `submit.py` | CV on every rubric item | PQ: three-verifiers 0.4557 vs submitted 0.4550; A 0.4631 vs 0.4618 ([reconstructed public-one-verifier/public-ensemble terms](../report/data/single_verifier.csv)). Paired submitted − three-verifiers over all rubric items (day bootstrap, 2,000 draws): PQ −0.0007 [−0.0024, +0.0010], mean IoU of overlapping pairs −0.0006 [−0.0017, +0.0005], mean Dice of overlapping pairs −0.0005 [−0.0015, +0.0006], one-to-many +0.09 pt [−0.05, +0.23], many-to-one +0.01 pt [−0.07, +0.09]. The reported paired intervals include zero on each item. We choose one verifier for simplicity under unresolved differences; these intervals do not establish equivalence. ([data](../report/data/rubric_paired_vs_three_verifiers.csv), [script](../tools/experiments/rubric_paired.py).) |

## 7. Selection

| Constant | Value | Where | Origin | Evidence |
| --- | --- | --- | --- | --- |
| Blend weight α on LightGBM | three-verifiers, submitted 0.5; A 0.35 | `filseg/select.py`, `submit.py` | CV | Historical public-four-verifiers: α 0.50 vs 0.35 −0.0012 [−0.0031, +0.0007]; three-verifiers: α 0.35 vs 0.50 −0.0004 [−0.0024, +0.0017] ([ablations](ABLATIONS.md)). These tested contrasts leave small differences unresolved; they do not establish a flat surface over every blend weight. |
| Penalty λ | three-verifiers, submitted 0.25; A 0.27 | `submit.py` | Theory motivates true gain; CV chooses the estimated-score operating point | Under the one-hit condition, adding a disjoint mask improves PQ when its true gain exceeds PQ/2: 0.22748 at current submitted PQ. Estimated gains have imperfect calibration. The current [day-bootstrap sweep](../report/data/lambda_sweep_finals.csv) gives submitted 0.45495 at 0.25 and max 0.45530 at 0.27; within 0.0015 of max for 0.22–0.31. three-verifiers: max 0.45631 at 0.29, range 0.23–0.30. public-ensemble: max 0.46314 at 0.27, range 0.26–0.28. The fold-excluded empirical PQ/2 control gives exact host PQ 0.45409 (submitted) / 0.46082 (A), versus 0.45483 / 0.46314 for per-fold tuning. Some 30-update controls do not converge; these are last-iterate values, not oracle optima ([summary](../report/data/selection_fixed_points.csv), [choices](../report/data/selection_fixed_point_folds.csv), [scripts and reference logs](../tools/experiments/selection/README.md)). All upstream OOF fits stay fixed. Current [weighted gain bins](../report/data/selection_gain_bins.csv) label each recipe. the three-verifier model's [rubric sweep](../report/data/rubric_vs_penalty_three_verifiers.csv) trades hit rate and per-record Dice for pair IoU; we retain 0.25. |
| Selection algorithm | Max-weight antichain of the tree by bottom-up DP | `filseg/select.py` | Theory | Exact maximiser of Σ(ĝ − λ) under pixel-disjointness, which the tree's antichains guarantee. |
| Thickness | No post-hoc dilation or erosion | — | CV | On current submitted OOF, 1-px cross dilation multiplies PQ by 0.94293 [0.93441, 0.95074] and erosion by 0.93695 [0.92783, 0.94672] ([data](../report/data/annotator_style_submitted.csv), [script](../tools/experiments/annotator_style.py)). |
| Empty images | Allowed (no forced mask) | `filseg/select.py` | Theory | The DP permits the empty antichain when no positive surrogate value is available. An older fallback experiment had incomplete candidate coverage and does not establish a current submitted effect. |

## 8. Public maps (public-ensemble ablation only)

| Constant | Value | Where | Origin | Evidence |
| --- | --- | --- | --- | --- |
| Public ensemble | Afshar's 10 U-Nets (5 folds × ResNet34, EfficientNetV2-S) | `filseg/public_maps.py`, `weights/public/` | A priori (third-party weights, CC BY-NC-SA 4.0) | The public-map keep rule scores 0.4522 in our cohort. Current A vs three-verifiers is +0.0075 PQ for 232.6M more neural parameters and also changes the selector fit. [Data](../report/data/complexity_ablation.csv). |
| Family averaging | Mean of 5 folds per family, then mean of the 2 families, each rounded to uint8 | `filseg/public_maps.py` | A priori (k02 convention) | Verified to reproduce the cached maps pixel for pixel ([`weights/public_ensemble/README.md`](../weights/public_ensemble/README.md)). |

## Notes on historical comparisons

The [historical postprocessing table](../report/data/postproc_ablation.csv)
uses an earlier public-ensemble recipe. Its broader 0.05–0.95 ladder, fold-0
diagnostics and fixed true-g penalties differ from the current own-map,
all-image fractional oracle. Those historical values are not a paired
comparison with the current ceiling check above.

Older low-threshold run logs do not provide a paired current-cohort
comparison.

Five rows were first marked "not verified" and then traced: batch Dice (single-factor fold-0 ablation recomputed, see the Loss row), final steps (a priori, never compared), minimum area (data-backed), verifier training and LightGBM parameters (a priori, never tuned). The label "in21k init" in [`report/data/ablation_fold0.csv`](../report/data/ablation_fold0.csv) is wrong: timm's default `tf_efficientnetv2_s` weights are already `in21k_ft_in1k`, so k03 and k05 share the same initial weights.
