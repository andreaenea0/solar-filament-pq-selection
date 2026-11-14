"""Fold-0 single-factor ablation of the Dice reduction (k03, k05, k06 held-out maps, fixed keep rule).

usage: FILSEG_WORK=DIR python dice_ablation_fold0.py LOW HIGH MIN_AREA MIN_MEAN
DIR holds gt_rles_coco.pkl and k03out/, k05out/, k06out/ with oof_control/*.npz.
"""
import pickle, os, sys, numpy as np
WORK = os.environ["FILSEG_WORK"]
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from concurrent.futures import ProcessPoolExecutor
from scipy import ndimage as ndi
from pycocotools import mask as mu
from filseg.metric import Totals, score_image
GT = pickle.load(open(f'{WORK}/gt_rles_coco.pkl', 'rb')); S8 = np.ones((3, 3), bool)
PAR = tuple(float(x) for x in sys.argv[1:5])
def rule(a):
    d, stem = a; t_lo, t_hi, min_area, min_meanp = PAR
    p = np.load(f'{d}/{stem}.npz')['p'].astype(np.float32)
    if p.ndim == 3: p = p.mean(0)
    lab, n = ndi.label(p > t_lo * 255, structure=S8); out = []
    for c, sl in enumerate(ndi.find_objects(lab), 1):
        m = lab[sl] == c; v = p[sl][m]
        if v.max() > t_hi * 255 and m.sum() >= min_area and v.mean() >= min_meanp * 255:
            mm = np.zeros(p.shape, np.uint8); mm[sl] = m; out.append(mu.encode(np.asfortranarray(mm)))
    t = score_image(list(GT[stem].values()), out); return (t.sum_iou, t.tp, t.fp, t.fn)
if __name__ == '__main__':
    dirs = {k: f'{WORK}/{k}out/oof_control' for k in ('k03', 'k05', 'k06')}
    stems = sorted(f[:-4] for f in os.listdir(dirs['k06'])); R = {}
    with ProcessPoolExecutor(4) as ex:
        for k, d in dirs.items(): R[k] = np.array(list(ex.map(rule, [(d, s) for s in stems], chunksize=8)))
    pq = lambda a: a[:, 0].sum() / (a[:, 1].sum() + 0.5 * a[:, 2].sum() + 0.5 * a[:, 3].sum())
    print(PAR, {k: round(pq(v), 4) for k, v in R.items()})
    days = np.array([s[:8] for s in stems]); ud = np.unique(days); idx = {d: np.where(days == d)[0] for d in ud}; rng = np.random.default_rng(0)
    for a, b in [('k05', 'k06'), ('k03', 'k06'), ('k03', 'k05')]:
        ds = []
        for _ in range(2000):
            ii = np.concatenate([idx[d] for d in rng.choice(ud, len(ud))]); ds.append(pq(R[b][ii]) - pq(R[a][ii]))
        print(a, '->', b, round(pq(R[b]) - pq(R[a]), 4), np.round(np.percentile(ds, [2.5, 97.5]), 4))
    pickle.dump((stems, R), open(f'{WORK}/dice_abl.pkl', 'wb'))
