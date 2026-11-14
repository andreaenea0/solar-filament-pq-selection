# Methods and held-out results

The selected submission is **submitted**. The checkout also replays three-verifiers,
single-unet, and the public-ensemble ablation. Every constant, its
origin and its evidence are listed in [design choices](DESIGN_CHOICES.md);
the component controls are in [ablations](ABLATIONS.md).

## Data and pipeline

The official training split contains 707 GONG H-alpha images and
1,154 drawings. PQ compares the prediction with each drawing of
its image and pools matched IoU, true positives, false positives,
and false negatives. The match threshold is strictly above IoU 0.5.
The submission checker verifies pixel disjointness.
[Training counts](../tools/figures/annotation_stats.py)
· [PQ scorer](../filseg/metric.py) · [CSV validator](../tools/check_submission.py)

MAGFiLO's review process used time-evolution views when adjudicating
whether nearby pieces formed one instance.
A single-image method therefore faces uncertainty about grouping as well
as boundaries. The component tree supplies alternative groupings; the
gain target follows the challenge's per-drawing score.
[MAGFiLO protocol](https://www.nature.com/articles/s41597-024-03876-y)
· [Selection derivation](SELECTION_THEORY.md)

submitted normalizes the solar disk and averages two own EfficientNetV2-S
U-Net maps with eight-way D4 augmentation. single-unet uses one own
U-Net. public-ensemble adds five public ResNet-34 and five public EfficientNetV2-S
fold models. The public checkpoints and their receipt are bundled.
[Own inference](../filseg/inference.py) · [Public inference](../filseg/public_maps.py)
· [Weight attribution](../WEIGHTS_LICENSE.md)

A threshold component tree proposes masks. The crop and gain
arrays share image-stem order and each tree's candidate indices.
LightGBM and the verifier estimate gain. Dynamic programming
selects a disjoint set. The team-only recipes use LightGBM weight 0.50
and penalty 0.25. submitted and single-unet use the seed-0 verifier.
three-verifiers averages three verifier seeds. public-ensemble uses three public-ensemble-specific
verifiers, LightGBM weight 0.35, and penalty 0.27. The selector
uses image-derived features.
[Candidate tree](../filseg/candidates.py) · [Verifier](../filseg/verifier.py)
· [Antichain selection](../filseg/select.py) · [Release settings](../submit.py)

The final's single verifier has a direct removal control on all 707
observations. At the same penalty 0.25, PQ is 0.45495 with the verifier
and 0.44851 with LightGBM alone: +0.00644 [0.00352, 0.00939], positive
in all five folds. Giving LightGBM its own penalty selected on the other
four folds yields 0.44835: +0.00660 [0.00365, 0.00969], again positive
in all five folds. The 7-day and 28-day block intervals also stay above
zero. These comparisons keep the upstream OOF fits fixed; the intervals
exclude refitting and model-selection variation.
[Summary](../report/data/submitted_verifier_summary.csv)
· [Fold results](../report/data/submitted_verifier_folds.csv)
· [Block sensitivity](../report/data/submitted_verifier_block_sensitivity.csv)
· [Rebuild and public-statistics replay](../tools/experiments/submitted_verifier_ablation.py)

## Held-out results

PQ is pooled across five image folds. The Dice and IoU columns
summarize positive-overlap pairs, including pairs below the PQ
match threshold. A split is a drawn mask that overlaps multiple
predictions. A merge is a prediction that overlaps multiple drawn masks.

The folds group drawings of the same observation together, but do
not keep observing days together: 41 days containing 83 training
images span folds. A pixel audit found no decoded-image duplicates
across folds or between training and test. These checks do not rule
out dependence from persistent solar regions.
[Protocol measurements](../report/data/protocol_summary.csv)
· [Pixel and split audit](../tools/experiments/protocol_audit.py)

| Recipe | PQ | Pair Dice | Pair IoU | Split rate | Merge rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| **submitted** | **0.45495** | 0.72848 | 0.59794 | 3.98% | 1.86% |
| three-verifiers | 0.45567 | 0.72893 | 0.59851 | 3.89% | 1.85% |
| public-ensemble | 0.46314 | 0.73564 | 0.60585 | 3.11% | 2.21% |

[PQ and relation table](../report/data/rubric_finals.csv)
· [PQ and rubric implementation](../tools/figures/rubric_table.py)
· [Pair summary](../report/data/notebook_summary.csv)
· [Full distributions](../report/data/notebook_distributions.csv)
· [Distribution implementation](../tools/figures/notebook_distributions.py)

submitted minus three-verifiers is −0.00071 PQ with paired observing-day
95% interval [−0.00242, +0.00099]. The reported Dice, IoU, and
split and merge differences between these recipes also have
paired intervals that include zero. submitted uses one verifier
at inference. The main held-out maps use one U-Net per image
and do not isolate the effect of the second inference seed. A
separate check on folds 0–3, where both seeds have held-out maps,
builds the tree on the mean of the two maps as at test time:
+0.005 [0.002, 0.008] PQ over the average single seed with a
LightGBM-only selector at λ 0.23.
[Second-seed check](../tools/experiments/second_seed/README.md)
· [Data](../report/data/second_seed_ablation.csv)
[Paired differences](../report/data/rubric_paired_vs_three_verifiers.csv)
· [Scoring](../tools/figures/rubric_table.py)
· [Held-out training driver](../training/retrain_team_models.py)

These are staged out-of-fold estimates. The selector studies reuse
upstream held-out maps and verifier predictions; refitting a selector
inside an inner loop does not by itself exclude the outer fold from
every upstream training dependency. The reported scores are not a
fully nested evaluation of the entire training pipeline.

The distribution files include a human-to-human reference and
model distributions restricted to the same multi-drawing images.
Use that shared cohort when comparing human and model curves.
[Cohort selection](../tools/figures/notebook_distributions.py)

### What the uncertainty covers

Intervals resample complete UTC observing days and keep fitted OOF
predictions, selected settings and fold assignments fixed. They describe
sampling variation on this cohort. They exclude training-seed variation,
the repeated model/setting search and a change in the private-test
population. OOF exclusion is stage-specific: downstream training reuses
upstream maps of other folds, whose pixel models can include the outer
fold. The pipeline is not fully nested end to end.
An interval containing zero does not establish equivalence.

For submitted minus three-verifiers, using 7-day or 28-day calendar blocks gives intervals
[-0.00255, +0.00102] and [-0.00225, +0.00079], respectively. Both include
zero. This sensitivity check does not establish independent solar events.
[Block sensitivity](../report/data/bootstrap_block_sensitivity.csv)
· [Implementation](../tools/experiments/bootstrap_sensitivity.py)

The four-fold second-seed check is exploratory and omits the verifier.
Its gain cannot be added to the main five-fold final score. Human curves
also weight images differently: there are R(R-1) ordered drawing pairs
versus R model-to-drawing comparisons on an image with R drawings.
Human agreement is a reference rather than a performance ceiling.

### What the controls show

On the three-verifier model's frozen candidate tree and held-out scores, dropping the
verifier at penalty 0.25 changes PQ by −0.00716 [−0.01004, −0.00428].
Using only one tree level, chosen on the other folds, changes PQ by
−0.00274 [−0.00567, +0.00023]. Setting the penalty to zero changes
PQ from 0.45567 to 0.36773. The controls hold the input maps fixed.
[Selector controls and paired intervals](../tools/experiments/selector_ablation.py)


### Penalty and calibration controls

The exact host sweep holds all upstream OOF predictions fixed. Ranges
below are grid points within 0.0015 PQ of that recipe's maximum.

| Recipe | Released penalty / PQ | Grid maximum penalty / PQ | Range |
| --- | --- | --- | --- |
| submitted | 0.25 / 0.45495 | 0.27 / 0.45530 | 0.22–0.31 |
| three-verifiers | 0.25 / 0.45567 | 0.29 / 0.45631 | 0.23–0.30 |
| public-ensemble | 0.27 / 0.46314 | 0.27 / 0.46314 | 0.26–0.28 |

[Current curves and intervals from 2,000 bootstrap draws over observing days](../report/data/lambda_sweep_finals.csv)
· [Portable commands](../tools/experiments/selection/README.md)

Half of released submitted PQ is 0.22748. The empirical PQ/2 iteration on
all OOF data settles at penalty 0.22703 for submitted and 0.23038 for the public ensemble.
Choosing this control separately outside each held-out fold gives exact
host PQ 0.45409 / 0.46082, versus 0.45483 / 0.46314 for per-fold penalty
tuning and 0.45495 / 0.46314 at the fixed release points. Some fold
iterations cycle; these controls report their last value after 30 updates.
They are not oracle optimizers. Only downstream choices exclude the
outer fold; cached upstream maps and verifier scores are reused.
[Summary, precision and scoring mode](../report/data/selection_fixed_points.csv)
· [Per-fold choices and convergence](../report/data/selection_fixed_point_folds.csv)

The older submitted values 0.4540 / 0.4548 use the one-hit gain shortcut;
the table preserves those results separately from exact host scoring.
Gain calibration bins use drawing-count weights. Isotonic controls also
depend on score precision, which the commands preserve and record.
[Current gain bins](../report/data/selection_gain_bins.csv)


submitted's PQ is 0.0850 for masks under 200 px² and 0.1163 for
masks from 0.9 to 1.0 fitted solar radii. Matched pairs are binned
by the drawn mask. Unmatched masks are binned by their own geometry.
The strata table reports the sample sizes and intervals.
[submitted strata](../report/data/strata_submitted.csv)
· [Stratification and intervals](../tools/figures/instance_errors.py)

Cross-tree selection, factored gain, and score offsets did not
support replacing the release selector. Drawing-style checks favored
keeping a shared operating point. A separate portfolio screen found
high overlap among its candidates.
[Cross-tree test](../tools/experiments/cross_tree_union.py)
· [Gain test](../tools/experiments/style_gain.py)
· [Offset test](../tools/experiments/selector_shift.py)
· [Drawing-style test](../tools/experiments/annotator_style.py)
· [Portfolio screen](../tools/experiments/final_pair.py)

## Replay and cost

The released exports contain 1,150 submitted, 1,156 three-verifiers, 1,187 single-unet,
and 1,165 public-ensemble masks over 180 test images. The strict-match Dice between
submitted and three-verifiers is 98.70%. Between submitted and public-ensemble it is 94.86%. These
comparisons use only predicted masks.
[Mask comparison](../tools/experiments/mask_shift.py)
· [Submitted archive](reference/submitted_test.csv)
· [Three-verifier archive](reference/three_verifiers_test.csv)
· [Single-U-Net archive](reference/single_unet_test.csv)
· [Public-ensemble archive](reference/public_ensemble_test.csv)

The submitted, three-verifiers, single-unet, and public-ensemble networks use approximately 55M, 78M,
33M, and 310M parameters, respectively.
[Component model counts](../report/data/complexity_ablation.csv)
· [Release model selection](../submit.py)
· [Reproduction commands](REPRODUCE.md)

An offline replay used the same official JPEG, `20110120105534Ch`,
for all four recipes on an Intel N95 with four CPU threads and D4
batch size 1. Times include model loading and CSV writing. Every run
reproduced both archived IDs and passed the disjoint-mask check.

| Recipe | CPU inference (s/image) | Exact RLEs | Strict matches | Minimum matched IoU |
| --- | ---: | ---: | ---: | ---: |
| submitted | 224.03 | 1/2 | 2/2 | 0.999245 |
| three-verifiers | 237.27 | 1/2 | 2/2 | 0.999245 |
| single-unet | 115.70 | 2/2 | 2/2 | 1.000000 |
| public-ensemble | 990.34 | 1/2 | 2/2 | 0.894585 |

[Replay notebook](../pipeline.ipynb)
· [Mask comparison helper](../tools/jury_cpu_check.py)
· [Recorded timestamps and parity](../report/data/cpu_replay_evidence.csv)

### One T4 GPU, all 180 test images

Each recipe was regenerated twice on one Tesla T4 (four CPU threads,
PyTorch 2.11 with CUDA 13.0): run 1 from the release tree, run 2 from the
published repository snapshot. All 23 released weight checksums matched
and every command exited with code 0. Times are wall-clock seconds per
image, including model loading; peak GPU memory was 6.1 GB. Parity
columns are for run 2, against the archives in `docs/reference/`.

| Recipe | s/image, run 1 / run 2 | Masks (archive / replay) | Byte-identical RLEs | Archived masks recovered | Minimum matched IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| submitted | 7.06 / 7.47 | 1,150 / 1,151 | 1,023 | 1,150 | 0.890 |
| three-verifiers | 6.97 / 7.48 | 1,156 / 1,156 | 1,009 | 1,156 | 0.852 |
| single-unet | 5.47 / 5.69 | 1,187 / 1,187 | 1,033 | 1,187 | 0.694 |
| public-ensemble | 18.27 / 20.85 | 1,165 / 1,166 | 1,032 | 1,165 | 0.863 |

Times vary by up to about 15% between runs on Kaggle T4 machines: the
submitted model takes about 7 s per image and the public-ensemble
variant about 20 s. In run 2 the submitted model's stages take 3.52
s/image for the two U-Nets with D4 augmentation, 3.46 s for the
threshold tree and its features (CPU), 0.34 s for the verifier crops
and forward pass, and under 0.01 s for the gain model and tree
selection.

The snapshot's submission command recovered all 1,150 archived masks and
returned one extra. Run 1 recovered 1,149. The separate timing run on
the snapshot returned 1,150 masks. Of those, 1,149 strictly matched an
archived mask, with mean IoU 0.999 over those matches. On
CPU (Intel N95, [above](#replay-and-cost)) both masks of the checked
image matched at IoU 0.9992 and one of the two encodings was
byte-identical.
[Run 1 timings and parity](../report/data/gpu_t4_replay.json)
· [Run 2 timings and parity](../report/data/gpu_t4_replay_snapshot.json)
· [Run 2 matched-IoU summary](../report/data/gpu_t4_replay_snapshot_summary.csv)
· [Timing tool](../tools/timing_replay.py)

Retraining both U-Nets from scratch with
[the released recipe](../kernels/k07-final/final.py) took 8,263 s on two
T4 GPUs. With LightGBM and the verifier unchanged, the retrained model
produced 1,150 test masks; 1,086 of the 1,150 archived masks have a
strict match (mutual PQ 0.885, pixel Dice 0.937). Kaggle displayed 0.39
for the retrained submission (ref 58329929, 14 November), as for the
selected submission. Training is not
bit-reproducible on GPU, so this measures the variation expected from
retraining, not a different method.
[Retraining check](../report/data/unet_retrain_check.json)

## Recompute the reported evidence

The [claim-audit runner](../tools/figures/claim_audit.py) recomputes
annotation counts, PQ/SQ/RQ, positive-overlap statistics, paired
day-bootstrap differences, the controlled batch-Dice comparison,
test-prediction agreement and exact architecture parameter counts.
It reads the official **training** annotations and the checked-in
prediction CSVs; it does not read test annotations.

```sh
python -m tools.figures.claim_audit --gt /path/to/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
  --parameters --out data/claim-audit.json
```

The calibration-bin table uses annotation-record-count weights and
the older four-verifier A ensemble (public-four-verifiers), alongside three-verifiers. Rebuild those
bins with `--cache /path/to/trusted-candidate-cache`. Current fixed-point
controls have [rebuild commands and scoring-mode details](../tools/experiments/selection/README.md),
[pooled results](../report/data/selection_fixed_points.csv) and
[fold choices/convergence](../report/data/selection_fixed_point_folds.csv).
Historical observations without preserved terms remain labelled as run logs.

To add executed-notebook CPU evidence, pass `--replay-dir` and
`--replay-stems` together. List every processed observation, including
images with zero predicted masks. Geometry matches are counted only
within the same image; identical coordinates in another image cannot match.

The [training disk fits](../report/data/disks_train.csv) and
[current submitted thickness control](../report/data/annotator_style_submitted.csv)
provide the inputs/results for the geometry and one-pixel checks.
Recomputing solar-coordinate strata additionally used SunPy 8.0.1
and Astropy 8.0.2; install these in an analysis environment. They are
not required for prediction or the claim-audit command above.

## What a second drawing changes

We held out each drawing in turn on 151 triply drawn training observations (149 observing days). The pixel-only submitted model scored 0.4468 PQ. For a diagnostic selector, we replaced the predicted candidate gains with strict overlaps from one or two other drawings of the same image, keeping the candidate trees and the historical diagnostic penalty 0.23. One other drawing yielded 0.4305 PQ and two yielded 0.4481. The paired differences (submitted minus drawing-informed selector) were +0.0164 [0.0063, 0.0281] and −0.0013 [−0.0123, 0.0104]. A one-drawing selector averages the two possible choices in pooled numerator and denominator units.

Across 352 unique three-way strict matches, mean best predicted-to-drawn IoU was 0.6814, versus 0.6478 between drawings of the same objects. Misses contribute zero. Each of the three drawing members is matched independently to the predictions, so a single predicted instance need not explain all three. This conditional difference was +0.0336 [0.0199, 0.0464].

On the 296 multiply drawn observations, 874/1,646 false-positive prediction/drawing events (53.1%) hit another drawing at IoU > 0.5. Among near misses with best IoU from 0.1 through 0.5, 523/936 (55.9%) did so. Across all 707 observations, false positives accounted for 27.9% of the shortfall from perfect pooled PQ, false negatives for 33.8% and imperfect overlap on strict matches for 38.3%. Intervals resample observing days 2,000 times with the predictions fixed.

[Estimates, counts and paired intervals](../report/data/submitted_drawing_agreement.csv) · [Training-only evaluation](../tools/figures/drawing_agreement.py)
