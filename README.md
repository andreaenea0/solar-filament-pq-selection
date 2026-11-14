# Solar Filament Segmentation Challenge 2026

The pipeline segments individual solar filaments in full-resolution, 2048 × 2048 GONG H-alpha JPEGs. It writes disjoint instance masks in the competition CSV format with `filament_id` and compressed COCO `segmentation_rle`.

Drawings of the same image can disagree about boundaries and whether
nearby pieces form one filament. We estimate each candidate's expected
strict-match quality across drawings, then choose disjoint instances
with a tree dynamic program.
[Decision example](docs/DECISION_EXAMPLE.md) · [Derivation](docs/SELECTION_THEORY.md)
· **[From the paper to the code](docs/PAPER_MAP.md)**

On identical held-out maps, this decision stage raises PQ from 0.4450
to 0.4550 over the unchanged inherited keep rule: +0.0099
[0.0063, 0.0136], positive in all five folds. The fixed rule is not
retuned; [the paired control](docs/ABLATIONS.md#remove-the-full-learned-decision-stage)
states the comparison's scope and conditional uncertainty.

The submitted model uses only team-trained weights: two
U-Nets, a LightGBM booster and one ResNet-18 verifier. The same code
replays the documented variants and the public-U-Net ablation.

| Recipe | Role | Inference recipe | Parameters | 5-fold PQ | Replay |
| --- | --- | --- | ---: | ---: | --- |
| submitted | **Final** (default) | Two own U-Nets, LightGBM and one ResNet-18 verifier | 55M | 0.4550 | [Release verification and command](docs/REPRODUCE.md#submitted-replay) |
| three-verifiers | Variant: three verifiers | submitted with verifier seeds 0, 1 and 2 | 78M | 0.4557 | `FILSEG_FINAL=three-verifiers` |
| single-unet | Variant: one U-Net | submitted with the seed-0 U-Net only | 33M | see note | `FILSEG_FINAL=single-unet` |
| public-ensemble | Ablation: public U-Nets | Own and ten public U-Nets, LightGBM and three public-ensemble-specific verifiers | 310M | 0.4631 | [Ablation: adding the public U-Nets](#ablation-adding-the-public-u-nets) |

The organisers' rubric favours the simpler model unless extra
complexity is justified. The retained single verifier adds +0.0066 PQ
[+0.0037, +0.0097] over LightGBM alone when the simpler control's penalty
is selected on the other four folds, with positive differences in all
five folds. This uses frozen staged OOF predictions.
[Paired control](report/data/submitted_verifier_summary.csv)
For the three-verifier model's two extra verifiers (22.4M parameters),
submitted minus three-verifiers is −0.0007 PQ
[−0.0024, +0.0010], and the pair Dice, pair IoU, split and merge
differences also have paired intervals that include zero. We choose
one verifier for simplicity under these unresolved small differences;
the intervals do not establish equivalence. The public ensemble gains
+0.008 PQ over submitted for 5.6 times the parameters and third-party weights, so it is
kept as an ablation. The held-out maps use one U-Net per image, so the
full cross-validation does not isolate the second U-Net seed; a separate
four-fold LightGBM-only check gives +0.005 [0.002, 0.008] PQ for it.
[Methods and results](docs/RESULTS.md) · [Design choices](docs/DESIGN_CHOICES.md) · [Ablations](docs/ABLATIONS.md)

## Run the pipeline notebook

Use Python 3.11 or 3.12 in a virtual environment. Choose the setup
that matches your platform and device.

### Windows x64 CPU with the tested Python 3.12 lock

Install CPython 3.12 and keep the checkout path short on Windows
(for example, `C:\filseg`) to avoid legacy path-length limits in wheel
installation. Run these commands in PowerShell from the repository root.
If the launcher does not list Python 3.12, use that interpreter's full
path in place of `py -3.12`.

```powershell
py -3.12 -m venv .venv
$python = ".\.venv\Scripts\python.exe"
& $python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0+cpu torchvision==0.21.0+cpu
& $python -m pip install -r requirements-lock.txt
& $python -m pip check
```

### Linux CPU setup in Bash

On WSL, keep the checkout and environment in the Linux filesystem for
faster installation. [Filesystem guide](https://learn.microsoft.com/en-us/windows/wsl/filesystems)

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

In Git Bash on Windows, create the environment with
`py -3.12 -m venv .venv` and activate with
`source .venv/Scripts/activate` before the pip commands.
For CUDA, install the PyTorch build for your CUDA runtime before
installing the remaining requirements.

### Test images and finalist selection

Extract the official competition test JPEGs from the
[Kaggle competition data](https://www.kaggle.com/competitions/filament-segmentation-2026/data).
For a CPU smoke test, copy one or two JPEGs into a separate directory.
Keep their filenames so the output IDs match the references.

Open [`pipeline.ipynb`](pipeline.ipynb) from the repository root, set
`IMAGES` to the directory containing the JPEGs and run all cells
with the environment you installed above. The notebook verifies the
bundled checkpoints, runs on CPU by default, and saves a
validated CSV plus overlays.

All checkpoints are in the clone; nothing is downloaded. The default
recipe is the final, submitted. Select another one with `FINAL` in the notebook's first
cell or with `FILSEG_FINAL` at the terminal (`submitted`, `three-verifiers`, `single-unet`
or `public-ensemble`). Each recipe has a reference CSV for the 180 test images:
[submitted](docs/reference/submitted_test.csv),
[three-verifiers](docs/reference/three_verifiers_test.csv),
[single-unet](docs/reference/single_unet_test.csv) and
[public-ensemble](docs/reference/public_ensemble_test.csv).
Run the notebook from PowerShell using the `$python` from the Windows
setup above:

```powershell
$env:FILSEG_IMAGES = (Resolve-Path "C:\path\to\test_images").Path
$env:FILSEG_FINAL = "submitted"
$env:FILSEG_OUT = "data/pipeline/final-submitted.csv"
& $python -m nbconvert --execute --to notebook --ExecutePreprocessor.timeout=-1 --output-dir data/pipeline --output pipeline-submitted.ipynb pipeline.ipynb
```

The corresponding Bash command is:

```sh
mkdir -p data/pipeline
FILSEG_IMAGES=/path/to/test_images FILSEG_FINAL=submitted FILSEG_OUT=data/pipeline/final-submitted.csv \
  python -m nbconvert --execute --to notebook --ExecutePreprocessor.timeout=-1 \
  --output-dir data/pipeline --output pipeline-submitted.ipynb pipeline.ipynb
```

Use distinct output filenames when you replay a second recipe.

See the [component ablations, rubric and timing](docs/RESULTS.md).

## Run from the command line

The command below replays the final, submitted. For three-verifiers pass `--final three-verifiers` and
all three verifiers; for single-unet pass `--final single-unet` and only
`effv2s_s0_fold-1.pt`. The reproduction guide gives the
[public-ensemble ablation command](docs/REPRODUCE.md#public-ensemble-ablation-replay).

```sh
PYTHON=python bash make_release.sh weights
python submit.py --images test_images \
  --weights weights/effv2s_s0_fold-1.pt weights/effv2s_s1_fold-1.pt \
  --selector weights/own_ver3_a05_l025_nometa_lgb.txt \
  --verifier-weights weights/verifier_resnet18_s0.pt \
  --out submission.csv
```

Use `--device cpu --tta-batch 1` for a one-image CPU check. A CUDA device is recommended for the full image set. The [Kaggle notebook](kernels/release/submitted.ipynb) runs the same entry point with attached code and weight files. Set the notebook's mount paths, enable the GPU, and leave internet access off.

The team files are listed in the [three-verifiers-recipe SHA-256 manifest](release_weights.sha256)
and the [public-ensemble ablation SHA-256 manifest](release_weights_public_ensemble.sha256).
To assemble local inference archives with verified weights, use
`bash make_release.sh weights final-submitted.zip` for the team-only recipes or
`bash make_release.sh --final public-ensemble weights public-ensemble.zip` for the public-ensemble ablation.

## Model card

### Submitted model

| Item | Details |
| --- | --- |
| Training data | The competition's MAGFiLO training split contains 707 GONG observations from 2011–2022, 1,154 drawings by 37 annotators, and 8,199 drawn instances. |
| Pixel model | Two team-trained EfficientNetV2-S U-Nets with ImageNet initialization, one grayscale channel, 512 px crops, and annotator-mean masks as soft targets. The loss is binary cross-entropy plus batch Dice. See the [k07 training recipe](kernels/k07-final/final.py). |
| Inference | Disk normalization, eight-way D4 test-time augmentation and averaging the two U-Net maps. A six-level component tree proposes masks. LightGBM and the seed-0 ResNet-18 verifier estimate PQ gain. The final antichain uses equal LightGBM and verifier weights and penalty 0.25 (three-verifiers: mean of three verifier seeds). |
| Validation | Five image folds, scored against each drawing with strict IoU greater than 0.5 and pooled host PQ. The [submitted OOF masks](report/data/oof_submitted.csv) score **0.4550 PQ** (95% interval 0.4452–0.4649, 2,000 bootstrap draws over 656 observing days), **0.6854 SQ** and **0.6637 RQ**. [three-verifiers](report/data/oof_three_verifiers.csv) scores 0.4557 [0.4463, 0.4656] in the same [rubric analysis](report/data/rubric_finals.csv). |
| Cost | 55M parameters at test time; about 7 s per image on one Tesla T4 (all 180 test images, model loading included; 7.1 and 7.5 s in two runs), peak GPU memory 6.1 GB. [GPU timings](docs/RESULTS.md#one-t4-gpu-all-180-test-images) |
| Reproducibility | On one T4 GPU the snapshot's submission command matched all 1,150 [archived masks](docs/reference/submitted_test.csv) at IoU above 0.5 and produced one extra mask. Minimum matched IoU was 0.890. [GPU and CPU checks](docs/RESULTS.md#one-t4-gpu-all-180-test-images) |
| Limits | Recognition is weaker for very small filaments and near the solar limb. Drawings of the same image can differ in count and extent. [Held-out strata](report/data/strata_submitted.csv) |

The booster uses the same rounded own-model map in both family feature channels. The verifier crops are 96 × 96 pixels with four channels (JPEG, own map, own map, candidate mask). The verifier averages raw gains over D4 transforms (and, for three-verifiers, over the three seeds) before selection.

## Ablation: adding the public U-Nets

Recipe public-ensemble adds Afshar's ten public U-Nets (CC BY-NC-SA 4.0, in
`weights/public/` with their receipt; [attribution](WEIGHTS_LICENSE.md))
as a second map family. The methods page reports its held-out
accuracy and cost alongside submitted.

| Item | Details |
| --- | --- |
| Pixel models | The same two team-trained EfficientNetV2-S U-Nets as three-verifiers, plus Afshar's five ResNet-34 and five EfficientNetV2-S U-Net folds. See the [team k07 recipe](kernels/k07-final/final.py) and [public model receipt and weights](https://www.kaggle.com/datasets/dariushafshar/solar-filament-unet-weights). |
| Inference | Disk normalization and eight-way D4 augmentation. Maps are rounded after D4, then averaged and rounded within each family. The tree receives the public and own maps. The selector combines 0.35 LightGBM with 0.65 mean gain from three public-ensemble-specific ResNet-18 verifiers and uses penalty 0.27. |
| Validation | Five-fold pooled PQ is **0.4631**. [Methods and evidence](docs/RESULTS.md#held-out-results) |
| Reproducibility | Use the [public-ensemble replay instructions](docs/REPRODUCE.md#public-ensemble-ablation-replay) and compare against the [release archive](docs/reference/public_ensemble_test.csv). |

## Licensing and attribution

- Original source code is distributed under the [MIT License](LICENSE).
- Team weights use [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). The [weight notice](WEIGHTS_LICENSE.md) also credits Afshar's public U-Net checkpoints used in public-ensemble ablation and gives their upstream license.
- Credit MAGFiLO v1.0 to Ahmadzadeh and colleagues, and the H-alpha observations to the NSO Global Oscillation Network Group. [MAGFiLO data descriptor](https://doi.org/10.1038/s41597-024-03876-y) · [NSO GONG H-alpha](https://nso.edu/data/nisp-data/h-alpha/)
