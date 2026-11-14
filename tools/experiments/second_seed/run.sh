set -e
# FILSEG_WORK: held-out seed-1 maps (k20out/, k20f23out/), gt_rles_coco.pkl, folds.csv
# FILSEG_CACHE: cached seed-0 maps (maps_own_oof/) and candidate trees; TRAIN: official training JPEGs
HERE=$(cd -- "$(dirname -- "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
cd "$FILSEG_WORK"
python "$HERE/mk_maps.py"
cd "$ROOT"
for v in mean2 s1; do
  python tools/build_cands.py --maps "$FILSEG_WORK/maps_$v" --images "$TRAIN" --out "$FILSEG_WORK/cands_$v.pkl" --gt "$FILSEG_WORK/gt_rles_coco.pkl"
done
cd "$FILSEG_WORK" && python "$HERE/eval_lgb.py"
