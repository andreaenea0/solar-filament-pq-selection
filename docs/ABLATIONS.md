# Component ablations

The selected submission is submitted. public-ensemble is the public-U-Net ablation;
three-verifiers is the three-verifier variant. The older component CSV retains
its historical final labels. It reports **0.4631 PQ** for public-ensemble with three
ResNet-18 verifiers and **0.4557** for three-verifiers. The paired cache
experiment reproduces three-verifiers at 0.45567 and uses **public-four-verifiers**, the earlier
four-verifier A, at 0.46333. The cache contains the public-four-verifiers OOF verifier
scores. Paired public-four-verifiers results below apply to that version.
[Component table](../report/data/complexity_ablation.csv)

| Recipe | Component | With | Control | Evidence | Inference cost |
| --- | --- | ---: | --- | --- | --- |
| public-ensemble | Public U-Net maps | 0.4631 | 0.4557 for the full three-verifiers recipe | Recipe comparison. The tree and verifier training also differ. | Ten public U-Nets, 232.6M parameters, each with D4 |
| public-ensemble | Own U-Net maps | 0.4631 | Paired removal unavailable | Public-map keep rule scores 0.4522 with a different selector. | Two own U-Nets, 44.2M, each with D4 |
| public-ensemble | Two own U-Net seeds | 0.4631 | Paired single-seed A unavailable | Own-map evidence in the three-verifiers row below; in A the own maps are one of two tree inputs. | One extra 22.1M-parameter U-Net |
| public-ensemble | D4 augmentation | 0.4631 | No no-D4 replay recorded | The recovered own-model scale study below does not isolate D4 or quantify an public-ensemble-specific gain. | Eight transforms per U-Net and verifier |
| public-ensemble | Cross-level tree selection | 0.46333 public-four-verifiers | 0.46115 at one held-out level | Cached control, Δ −0.00218 [−0.00520, 0.00074]. | Six candidate levels and antichain DP. Feature and crop costs still include the tree. |
| public-ensemble | LightGBM | 0.46333 public-four-verifiers | 0.45824 with verifier alone | Cached control at λ 0.27, Δ −0.00508 [−0.00855, −0.00200]. | One 31-feature booster per candidate |
| public-ensemble | Verifier | 0.46333 public-four-verifiers | 0.45404 with LightGBM alone | Cached control at λ 0.27, Δ −0.00928 [−0.01315, −0.00552]. | public-four-verifiers has about 37.5M verifier parameters; the public-ensemble variant's three current ResNet-18s have 33.5M |
| public-ensemble | Blend weight | 0.46333 public-four-verifiers at α 0.35 | 0.46215 at α 0.50 | Cached control at λ 0.27, Δ −0.00117 [−0.00313, 0.00073]. | One weighted sum per candidate |
| public-ensemble | Penalty | 0.46333 public-four-verifiers at λ 0.27 | 0.36377 at λ 0 and 0.46195 at λ 0.30 | Cached controls. Δ at 0.30 is −0.00137 [−0.00372, 0.00082]. | One subtraction per candidate |
| three-verifiers | Public U-Net maps | 0.4631 for the full public-ensemble selector | 0.4557 for three-verifiers | Recipe comparison. It changes more than the maps. | three-verifiers uses the two own U-Nets |
| three-verifiers | Own U-Net maps | 0.4557 | Paired removal unavailable | The own-map keep-rule reference is 0.4450. | Two 22.1M-parameter U-Nets |
| three-verifiers | Two own U-Net seeds | 0.4517 (mean of two maps, LightGBM only, λ 0.23, folds 0–3) | 0.4481 seed 0 / 0.4453 seed 1 | Dedicated ablation, Δ +0.005 [0.002, 0.008] vs the average single seed; per fold vs seed 0 +0.0008, +0.0081, +0.0064, −0.0002. [Data](../report/data/second_seed_ablation.csv). | One extra 22.1M-parameter U-Net |
| three-verifiers | D4 augmentation | 0.4557 | No no-D4 replay recorded | The recovered fold-0 scale study below retains D4; it does not isolate D4's contribution. | Eight transforms per U-Net and verifier |
| three-verifiers | Cross-level tree selection | 0.45567 | 0.45292 at one held-out level | Cached control, Δ −0.00274 [−0.00567, 0.00023]. | Six candidate levels and antichain DP |
| three-verifiers | LightGBM | 0.45567 | 0.45240 with verifier alone | Cached control at λ 0.25, Δ −0.00326 [−0.00680, 0.00004]. | One 31-feature booster per candidate |
| three-verifiers | Verifier | 0.45567 | 0.44851 with LightGBM alone | Cached control at λ 0.25, Δ −0.00716 [−0.01004, −0.00428]. | Three ResNet-18s add 33.5M parameters |
| submitted (final) | Verifier, seed 0 | 0.45495 | 0.44851 with LightGBM alone at the same λ 0.25 | Direct removal control, Δ +0.00644 [+0.00352, +0.00939], positive in 5/5 folds. [Data](../report/data/submitted_verifier_summary.csv) | One ResNet-18, 11.2M parameters |
| submitted (final) | Penalty for the simpler control | 0.45495 | 0.44835 with LightGBM alone and its penalty chosen on the other four folds | Δ +0.00660 [+0.00365, +0.00969], positive in 5/5 folds. [Choices](../report/data/submitted_verifier_penalty_choices.csv) | Same one-verifier cost |
| submitted (final) | Two extra verifier seeds (three-verifier variant) | 0.4557 (three-verifiers) | 0.4550 (submitted) | Paired submitted − three-verifiers: PQ −0.0007 [−0.0024, +0.0010]; pair IoU −0.0006 [−0.0017, +0.0005]; pair Dice −0.0005 [−0.0015, +0.0006]; one-to-many +0.09 pt; many-to-one +0.01 pt. The reported paired intervals include zero; we choose the smaller ensemble for simplicity. [Data](../report/data/rubric_paired_vs_three_verifiers.csv) | Two more ResNet-18s, 22.4M parameters |
| three-verifiers | Blend weight | 0.45567 at α 0.50 | 0.45526 at α 0.35 | Cached control at λ 0.25, Δ −0.00040 [−0.00244, 0.00165]. | One weighted sum per candidate |
| three-verifiers | Penalty | 0.45567 at λ 0.25 | 0.36773 at λ 0 and 0.45549 at λ 0.27 | Cached controls. Δ at 0.27 is −0.00018 [−0.00158, 0.00120]. Rubric: λ 0.29 − 0.25 PQ +0.0006 [−0.0014, +0.0029], pair IoU +0.005, drawn filaments hit −3.4 pt, per-record Dice −0.010. This is a rubric trade-off; we retain 0.25 rather than claiming a unique joint optimum ([data](../report/data/rubric_vs_penalty_three_verifiers.csv)). | One subtraction per candidate |

The submitted removal controls use the current seed-0 prediction CSV and
the frozen 31-feature OOF LightGBM gains, the same candidate trees and
released DP. Every mask set is checked for disjointness and scored with
the exact host scorer. The fixed control keeps λ 0.25; the tuned control
chooses from the existing penalty grid using the other four folds.
These are staged OOF comparisons with upstream fits fixed.
[Per-image host terms](../report/data/submitted_verifier_per_image.csv)
· [Input hashes](../report/data/submitted_verifier_inputs.csv)
· [Paired 1/7/28-day intervals](../report/data/submitted_verifier_block_sensitivity.csv)

The single-level control tunes its level on the other four folds,
retains that level's disjoint masks with scores above the release
penalty, and scores the held-out fold. It uses the original tree
features. Cached selector intervals use the 707 training images; the
second-seed check uses 566 images and 528 observing days. Intervals are
paired 95% observing-day bootstrap intervals conditional on frozen OOF
scores. The cache has D4-averaged maps and lacks per-transform
predictions for a paired no-D4 ablation.

The smallest reported verifier gains are from increasing the seed
count. the three-verifier model's three-verifier blend scores 0.4557 versus 0.4553 with one
ResNet-18 and a separately tuned penalty. Recovered raw fold exports
reproduce A at 0.463136 versus 0.461814 with one ResNet-18 at the same
penalty, a paired public-one-verifier-minus-public-ensemble difference of -0.001322
[-0.003549, +0.000792]. These intervals condition on frozen staged OOF
predictions. [Summary](../report/data/single_verifier.csv) ·
[Per-image terms](../report/data/single_verifier_per_image.csv).
The reported difference from adding the public ensemble's
EfficientNet-B0 was +0.0002 and the released public ensemble omits it. The paired
cache controls also leave the modest blend-weight and penalty
differences within the measured uncertainty. The zero-penalty control
has a large loss for both recipes.

| Recipe | Parameters | Measured end-to-end time |
| --- | ---: | --- |
| submitted (final) | 55.4M | 224.03 s/image on one official JPEG, Intel N95 CPU, four threads |
| three-verifiers | 77.7M | 237.27 s/image on the same CPU replay; 7.7 s/image on one T4 in a 180-image replay of the earlier three-verifiers |
| single-unet | 33.3M | 115.70 s/image on the same CPU replay |
| public-ensemble | 310.3M | 990.34 s/image on the same CPU replay |

Component costs above describe work and parameter counts. The
measured times are whole-pipeline wall times including model loading.
[Replay and cost](RESULTS.md#replay-and-cost) · [Replay instructions](REPRODUCE.md)

Reproduce the cached selector comparisons from the repository root:

```sh
python -m tools.experiments.selector_ablation --cache data/cache --out data/report/selector_ablation.json --csv data/report/selector_controls.csv
```

Pass `--public-oof-verifier` with the candidate-aligned three-verifier
archive to repeat the selector controls for the current public ensemble.
The [flat control table](../report/data/selector_control_summary.csv)
labels the cached four-verifier recipe as public-four-verifiers, separately from current public ensemble.

Replay the public-one-verifier/public-ensemble comparison directly from its public sufficient statistics:

```sh
python -m tools.experiments.single_verifier --terms report/data/single_verifier_per_image.csv --out data/a1_replay.csv
```

For a fresh reconstruction, supply `--cache` and `--verifier-folds` with
the five disjoint seed-0 OOF verifier exports instead of `--terms`.
The script blends 0.35 LightGBM + 0.65 verifier at penalty 0.27 and
scores the actual host terms against all training drawings.

## Remove the full learned decision stage

On the same 707 held-out own-model maps, the unchanged keep rule scores
0.445037 PQ and frozen submitted scores 0.454951: **+0.009914
[0.006343, 0.013558]**, positive in all five folds. The 2,000 paired
resamples keep observing days together; 7-day and 28-day block intervals
also stay positive. This measures the existing decision stage over the
fixed inherited rule. Its thresholds were not retuned for these maps,
so the comparison does not establish superiority over all simple rules.
Both recipes use one held-out U-Net per image here; the second deployed
seed is not part of this contrast. There is no new final-model gain.

The fixed rule uses probability >0.35, peak >0.99, area >=200 pixels
and mean probability >=0.85, exactly as in `filseg/rule.py`.
Every prediction set is checked for disjointness and scored against
all drawings with the exact host scorer. The rebuilt rule score agrees
with the previously rounded 0.4450 complexity-table row.

[Paired summary](../report/data/submitted_rule_summary.csv) ·
[Fold results](../report/data/submitted_rule_folds.csv) ·
[Per-image terms](../report/data/submitted_rule_per_image.csv) ·
[Input hashes](../report/data/submitted_rule_inputs.csv)

Repeat statistics without a cache:

~~~sh
python -m tools.experiments.submitted_rule_ablation --terms report/data/submitted_rule_per_image.csv --out data/submitted-rule-check
~~~

For a native rebuild, replace `--terms` with
`--maps data/cache/maps_own_oof --gt "$TRAIN_JSON" --folds data/cache/folds.csv`.
Use a new output directory. The dates group evaluation resamples only;
they are not inference features. The comparison conditions on frozen
staged OOF predictions and excludes refitting/model-search uncertainty.

## Recovered fold-0 pixel controls

The original held-out capacity maps were retrieved again and scored against
the official training drawings with the unchanged keep rule. All 142 images
and 235 drawings align with released fold 0. EfficientNetV2-M scores 0.44668
and the 21,000-update EfficientNetV2-S scores 0.44448, versus 0.45249 for
the control. Both paired day intervals include zero. These are single-seed,
single-fold observations, not an established architecture ranking.
[Capacity summary](../report/data/pixel_capacity_summary.csv) ·
[Per-image host terms](../report/data/pixel_capacity_per_image.csv) ·
[Input hashes](../report/data/pixel_capacity_inputs.csv)

The recovered multi-scale export uses the same held-out checkpoint at
scales 0.75, 1.0 and 1.25. The fixed unrounded mean of 1.0 and 1.25 gives
0.45458 versus 0.45249: +0.00209 [−0.00479, +0.00890]. The mean of all
three gives 0.42219. Means are computed on float32 byte-scale maps before
the keep rule, without another rounding step. The separately exported
scale-1.0 control differs slightly from the original k06 export; each
contrast uses its own stated control. No full five-fold submitted gain or
training-seed uncertainty is estimated, and the final stays unchanged.
[Scale summary](../report/data/pixel_multiscale_summary.csv) ·
[Per-image host terms](../report/data/pixel_multiscale_per_image.csv) ·
[Input hashes](../report/data/pixel_multiscale_inputs.csv)

Intervals use 2,000 paired resamples of the 139 observing days, with
upstream fits fixed. The [rebuild/replay tool](../tools/experiments/pixel_controls.py)
also checks disjointness and paired cohort coverage.
