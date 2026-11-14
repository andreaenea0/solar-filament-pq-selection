"""Extract 4-channel candidate crops (image, public-ensemble prob, own-model prob, candidate mask) for the verifier.
usage: crops4.py CANDS.pkl MAPDIR IMAGEDIR OUT.npz   (maps: p uint8 [2,H,W] = [public, own])
Writes one uint8 array X (N,4,S,S) plus the stem and candidate index of every row."""
import pickle, sys, numpy as np, cv2
from concurrent.futures import ProcessPoolExecutor
from pycocotools import mask as mu
from PIL import Image
S = 96
C = pickle.load(open(sys.argv[1], 'rb')); MAPS, IMGS, OUT = sys.argv[2:5]
def one(stem):
    """Return this image's crops in the original candidate row order."""
    c = C[stem]; n = len(c['rle'])
    if n == 0: return stem, np.zeros((0, 4, S, S), np.uint8)
    with Image.open(f'{IMGS}/{stem}.jpeg') as image:
        img = np.asarray(image.convert('L'))
    with np.load(f'{MAPS}/{stem}.npz') as maps:
        p = maps['p']
    out = np.zeros((n, 4, S, S), np.uint8)
    for i, r in enumerate(c['rle']):
        x, y, w, h = mu.toBbox(r).astype(int)
        side = int(max(w, h) * 1.5) + 16; cx, cy = x + w // 2, y + h // 2
        x0, y0 = max(cx - side // 2, 0), max(cy - side // 2, 0); x1, y1 = min(x0 + side, 2048), min(y0 + side, 2048)
        m = mu.decode(r)[y0:y1, x0:x1] * 255
        for k, a in enumerate((img[y0:y1, x0:x1], p[0, y0:y1, x0:x1], p[1, y0:y1, x0:x1], m)):
            out[i, k] = cv2.resize(np.ascontiguousarray(a), (S, S), interpolation=cv2.INTER_AREA)
    return stem, out
if __name__ == '__main__':
    stems = sorted(C)
    with ProcessPoolExecutor(4) as ex: res = dict(ex.map(one, stems, chunksize=8))
    X = np.concatenate([res[s] for s in stems]); st = np.concatenate([[s] * len(res[s]) for s in stems]).astype('U32')
    idx = np.concatenate([np.arange(len(res[s])) for s in stems])
    np.savez(OUT, X=X, stem=st, idx=idx); print(OUT, X.shape)
