# From the paper to the code

The paper asks when a dark structure in a solar H-alpha image should count as one filament. The table links refer to held-out training observations. The competition test drawings are unavailable.

| Configuration in the paper | `--final` or `FILSEG_FINAL` | Weights loaded at inference |
| --- | --- | --- |
| Submitted model (default) | `submitted` | Two team U-Nets, one booster and one ResNet-18 verifier |
| Three-verifier variant | `three-verifiers` | The same maps and booster with three verifiers |
| Public-ensemble variant | `public-ensemble` | Adds ten public U-Nets and uses the corresponding booster and three verifiers |
| Single-U-Net control | `single-unet` | Submitted selector with a single team U-Net |

## Methods and evaluation

| Paper location | Implementation and numerical evidence |
| --- | --- |
| Introduction and training cohort | [Drawing and observation counts](../tools/figures/annotation_stats.py), [agreement by disk position](../report/data/pq_by_radius.csv) |
| Data, preprocessing and pixel model | [Training targets, crops, augmentation and batch Dice](../kernels/k07-final/final.py), [disk normalization and inference](../filseg/inference.py), [fold checks](../tools/experiments/protocol_audit.py) |
| Component thresholds, candidate crops, features and scoring | [Component tree](../filseg/candidates.py), [candidate features and selection](../filseg/select.py), [verifier input](../filseg/verifier.py) |
| Eq. 1, marginal gain and the tree recursion | [Definition, assumptions and derivation](SELECTION_THEORY.md), [pooled host metric](../filseg/metric.py), [gain targets](../filseg/oof.py), [dynamic programming](../filseg/select.py) |
| Table 1, rubric metrics | [Metric definitions and implementation](../tools/figures/rubric_table.py), [five-fold pooled estimates and intervals](../report/data/rubric_finals.csv), [split/merge and pair distributions](../report/data/notebook_summary.csv) |
| Table 2, pixel-model ablations | [Fold-0 ablations](../report/data/ablation_fold0.csv), [paired image terms for batch Dice](../report/data/dice_ablation_fold0_per_image.csv), [capacity](../report/data/pixel_capacity_summary.csv) and [multiscale control](../report/data/pixel_multiscale_summary.csv) |
| Complexity and limitations | [Paired fixed-rule control](../report/data/submitted_rule_summary.csv), [paired verifier control](../report/data/submitted_verifier_summary.csv), [one versus three verifiers](../report/data/rubric_paired_vs_three_verifiers.csv), [public ensemble versus submitted](../report/data/rubric_public_ensemble_vs_submitted.csv), [four-fold second-seed control](../report/data/second_seed_ablation.csv), [CPU replay](../report/data/cpu_replay_evidence.csv) |

On the 151 triply drawn training observations, the [held-drawing comparisons](../report/data/submitted_drawing_agreement.csv) use the [training-only evaluation script](../tools/figures/drawing_agreement.py). The same table reports conditional shape IoU, corroborated false positives and the exact pooled PQ-loss decomposition.

## Figures

| Illustration | Generator | Inputs |
| --- | --- | --- |
| [1. A parent component and two retained descendants](../report/figs/observed_selection.pdf) | [Observed selection](../report/observed_selection.py) | Training observation `20130314165714Mh`, both official drawings, held-out tree, predicted gains and [archived masks](../report/data/oof_submitted.csv) |
| [2. Agreement on 296 multiply drawn observations](../report/figs/overlap_distributions.pdf) | [Distribution panels](../report/overlap_distributions.py) | [Binned positive overlaps and counterpart counts](../report/data/notebook_distributions.csv) |
| [3. Merges, splits and the limb](../report/figs/observed_errors.pdf) | [Native crop comparison](../report/observed_errors.py) | [Exact observations, drawings and crop locations](MORPHOLOGY.md), official training JPEGs and [held-out masks](../report/data/oof_submitted.csv) |
| [4. PQ versus deployed neural parameters](../report/figs/deployment_cost.pdf) | [Cost plot](../report/cost_accuracy.py) | [Seven plotted configurations at their stated settings](../report/data/complexity.csv) |
| [5. From the H-alpha image to selected instances](../report/figs/observed_pipeline.pdf) | [Pipeline panels](../report/observed_pipeline.py) | Same observation, held-out probability map and gains. The generator checks all nine selected masks against the archive. The bars compare the parent weight with the sum of the two child-branch weights after the per-mask penalty. |
| [6. Morphology and disk position](../report/figs/observed_morphology.pdf) | [Morphology panels](../tools/figures/physics_figure.py) | [Aggregate](../report/data/physics_pred_vs_gt.csv), [individual training instances](../report/data/physics_instances.csv) and [radial PQ](../report/data/pq_by_radius.csv) |

Figures 1 and 5 use training intermediates from the original staged validation. They require the own-model candidate cache, its out-of-fold booster predictions and the aligned seed-0 verifier scores. Final inference weights alone cannot recover the held-out fits. Figures 2, 3, 4 and 6 use the archived predictions, training images or the linked numerical tables.

The [reproduction guide](REPRODUCE.md) gives the commands for final inference. The [released checksums](../release_weights.sha256) identify its weights and the [submitted test CSV](reference/submitted_test.csv) fixes its output.
