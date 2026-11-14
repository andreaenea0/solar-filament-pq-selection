# Inspect submitted morphology

These three official training images provide a reproducible visual
inspection of the frozen held-out submitted masks. The selection continues
the existing report examples. It is illustrative, not a random sample
or an estimate of error prevalence. No test drawings are used.

From the repository root, with the official training files extracted:

```sh
python -m tools.figures.overlays --images data/kaggle/MAGFiLO_1.0_Kaggle_2026/train/train_images --gt data/kaggle/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json --csv report/data/oof_submitted.csv --stems tools/figures/morphology_stems.txt --out data/morphology --composite data/morphology/composite.png --examples 1 --zooms 4
```

Open `data/morphology/index.html` locally. Whole disks retain native
pixels; zooms and gallery panels show 512-by-512 crops without resizing.
The composite places up to three drawings and the prediction on the
same crop. JPEGs, drawings and rendered panels stay in ignored `data/`.

| View | Image | Drawing / native crop | What to inspect |
| --- | --- | --- | --- |
| Annotator comparison | 20120519070814Th | Drawings 1–3; x=782, y=296 | The same dark structure has different drawn endpoints and widths. The predicted extent lies between some drawings. This supports disagreement, not a claim that any drawing is physically wrong. |
| Merge | 20120519070814Th | Drawing 2; x=594, y=1224 | One prediction joins two drawn pieces through a thin connection. The two drawn masks are missed by strict PQ matching. |
| Split | 20130719140154Bh | Drawing 2; x=1176, y=712 | A gap divides a long drawn filament into predicted fragments. Neither fragment strictly matches the complete drawing. |
| Limb | 20140604195854Bh | Drawing 1; x=1536, y=518 | Inspect projected morphology near the limb and the visible disk edge. Geometric distance alone does not identify a filament. |

Drawing numbers follow sorted annotation-record IDs, as in the overlay
tool. Coordinates are zero-based top-left pixel positions. The generated
`gallery/index.json` records each selected anchor box and screening score.
Green marks a strict IoU > 0.5 match to the displayed drawing, blue a
missed drawn instance, and red an unmatched prediction. Match counts in
panel headers cover the full image, including masks outside the crop.
The composite uses blue for drawings and green for predictions without
match coloring.

The gallery's `split` and `merge` hints require an unmatched anchor
mostly covered by multiple substantial parts. They are stricter than
the rubric's any-positive-overlap relation. The optional `fibril-like`
and `sunspot-like` hints are geometry/intensity screens, not physical
classifications. Population rates and the overlap rule belong to the
[results](RESULTS.md), rather than this selected gallery.
