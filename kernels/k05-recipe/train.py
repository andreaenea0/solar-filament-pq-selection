"""k03: controlled ablation of temporal-neighbour augmentation.

Two U-Nets (EfficientNetV2-S encoder, 1 input channel, soft annotator-mean target) are trained with the
same recipe and seed on the folds != FOLD, one on GPU0 (control: annotated frames only) and one on GPU1
(temporal: with prob P_NEIGH the input is a GONG frame taken 2-6 min before/after the annotated one,
with the same target). Both are then evaluated out-of-fold on fold FOLD: their D4-TTA probability maps
are saved for scoring offline with the host's PQ.
"""
import glob, json, os, sys, time, math, random, subprocess, importlib
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import torch.multiprocessing as mp

FOLD = int(os.environ.get("FOLD", 0))
STEPS = int(os.environ.get("STEPS", 10500))
P_NEIGH = 0.3
BS, CROP, LR, WD, WARM = 16, 512, 3e-4, 1e-4, 300
H = W = 2048; R = 950
INPUT, OUT, TMP = "/kaggle/input", "/kaggle/working", "/tmp/filseg"

def find(name, d=False):
    hits = [h for h in sorted(glob.glob(os.path.join(INPUT, "**", name), recursive=True)) if os.path.isdir(h) == d]
    assert len(hits) >= 1, name; return hits[0]

try:
    import segmentation_models_pytorch as smp
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch==0.5.0"], check=True)
    importlib.invalidate_caches(); import segmentation_models_pytorch as smp
from PIL import Image
from pycocotools import mask as mu

YY, XX = np.ogrid[:H, :W]
DISK = ((YY - (H - 1) / 2) ** 2 + (XX - (W - 1) / 2) ** 2) <= R ** 2

def load(p):
    with Image.open(p) as im: return np.asarray(im.convert("L"), np.uint8)

def prepare():
    """Decode annotated frames and valid optional neighbors into memory maps."""
    os.makedirs(TMP, exist_ok=True)
    TR = find("train_images", True); J = find("MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    folds = pd.read_csv(find("folds.csv")).set_index("stem")["fold"].to_dict()
    nbh = sorted(glob.glob(os.path.join(INPUT, "**", "+2_*.jpg"), recursive=True))
    assert nbh, "neighbour frames not found"
    NB = os.path.dirname(os.path.dirname(nbh[0]))
    d = json.load(open(J)); recs = {}
    for im in d["images"]: recs[im["id"]] = (im["file_name"][:-5], [])
    for a in d["annotations"]:
        seg = json.loads(a["segmentation"]) if isinstance(a["segmentation"], str) else a["segmentation"]
        recs[a["image_id"]][1].append(seg if isinstance(seg[0], list) else [seg])
    by = {}
    for rid, (stem, fils) in recs.items(): by.setdefault(stem, []).append(fils)
    stems = sorted(by)
    tgt = np.lib.format.open_memmap(f"{TMP}/tgt.npy", "w+", np.uint8, (len(stems), H, W))
    paths = []
    for i, s in enumerate(stems):
        acc = np.zeros((H, W), np.float32)
        for fils in by[s]:
            m = np.zeros((H, W), np.uint8)
            for parts in fils:
                rl = mu.frPyObjects([p for p in parts if len(p) >= 6], H, W)
                m |= mu.decode(mu.merge(rl) if len(rl) > 1 else rl[0])
            acc += m
        tgt[i] = np.round(acc / len(by[s]) * 255).astype(np.uint8)
        nb = sorted(p for p in glob.glob(f"{NB}/{s}/*.jpg") if os.path.basename(p).split("_")[0] in ("-4", "-2", "+2", "+4"))
        paths.append([f"{TR}/{s}.jpeg"] + nb)
    flat = [p for ps in paths for p in ps]
    img = np.lib.format.open_memmap(f"{TMP}/img.npy", "w+", np.uint8, (len(flat), H, W))
    stats = np.zeros((len(flat), 2), np.float32); k = 0; index = []
    for i, ps in enumerate(paths):
        ids = []
        for frame_index, p in enumerate(ps):
            try:
                x = load(p)
                if x.shape != (H, W):
                    raise ValueError(f"Expected {(H, W)} pixels, got {x.shape}.")
            except (OSError, ValueError) as exc:
                if frame_index == 0:
                    raise RuntimeError(f"Cannot read annotated frame {p}.") from exc
                continue
            img[k] = x; v = x[DISK].astype(np.float32) / 255; stats[k] = (v.mean(), max(v.std(), 1e-3))
            ids.append(k); k += 1
        index.append(ids)
    img.flush(); tgt.flush()
    meta = dict(stems=stems, folds=[int(folds[s]) for s in stems], index=index, n=k)
    np.save(f"{TMP}/stats.npy", stats); json.dump(meta, open(f"{TMP}/meta.json", "w"))
    print(f"prepared {len(stems)} obs, {k} frames ({k - len(stems)} neighbours)", flush=True)

def sample_crop(img, tgt, stats, meta, obs_ids, rng, temporal):
    o = obs_ids[rng.integers(len(obs_ids))]
    ids = meta["index"][o]
    k = ids[0] if (not temporal or len(ids) == 1 or rng.random() > P_NEIGH) else ids[1 + rng.integers(len(ids) - 1)]
    t = tgt[o]
    if rng.random() < 0.7:
        ys, xs = np.nonzero(t[::8, ::8])
        if len(ys):
            j = rng.integers(len(ys)); cy, cx = ys[j] * 8, xs[j] * 8
        else:
            cy, cx = rng.integers(CROP // 2, H - CROP // 2, 2)
    else:
        cy, cx = rng.integers(CROP // 2, H - CROP // 2, 2)
    y0 = int(np.clip(cy - CROP // 2 + rng.integers(-128, 129), 0, H - CROP)); x0 = int(np.clip(cx - CROP // 2 + rng.integers(-128, 129), 0, W - CROP))
    raw = img[k, y0:y0 + CROP, x0:x0 + CROP].astype(np.float32) / 255
    raw = np.clip(raw, 0, 1) ** rng.uniform(0.8, 1.25)                     # mild gamma
    raw = (raw - raw.mean()) * rng.uniform(0.9, 1.1) + raw.mean()          # mild contrast
    x = (raw - stats[k, 0]) / stats[k, 1]
    y = tgt[o, y0:y0 + CROP, x0:x0 + CROP].astype(np.float32) / 255
    r = rng.integers(4); x, y = np.rot90(x, r), np.rot90(y, r)
    if rng.random() < 0.5: x, y = x[:, ::-1], y[:, ::-1]
    return np.ascontiguousarray(x), np.ascontiguousarray(y)

def dice_loss(logit, y):
    p = torch.sigmoid(logit); num = 2 * (p * y).sum((1, 2, 3)) + 1; den = (p + y).sum((1, 2, 3)) + 1
    return (1 - num / den).mean()

D4 = [(0, False), (0, True), (2, False), (2, True), (1, False), (1, True), (3, False), (3, True)]

@torch.no_grad()
def predict(model, x, dev):
    z = torch.from_numpy(x).to(dev)[None, None]
    acc = torch.zeros((H, W), device=dev)
    for j in range(0, 8, 4):
        ch = D4[j:j + 4]; inp = []
        for k, f in ch:
            t = torch.flip(z, dims=[3]) if f else z
            inp.append(torch.rot90(t, k, dims=[2, 3]) if k else t)
        with torch.autocast("cuda", dtype=torch.float16):
            pr = torch.sigmoid(model(torch.cat(inp)).float())
        for q, (k, f) in enumerate(ch):
            p = pr[q:q + 1]
            if k: p = torch.rot90(p, -k, dims=[2, 3])
            if f: p = torch.flip(p, dims=[3])
            acc += p[0, 0]
    return (acc / 8).cpu().numpy()

def run(rank, name, temporal):
    dev = torch.device(f"cuda:{rank}"); torch.manual_seed(0)
    img = np.load(f"{TMP}/img.npy", mmap_mode="r"); tgt = np.load(f"{TMP}/tgt.npy", mmap_mode="r")
    stats = np.load(f"{TMP}/stats.npy"); meta = json.load(open(f"{TMP}/meta.json"))
    tr = [i for i, f in enumerate(meta["folds"]) if f != FOLD]; va = [i for i, f in enumerate(meta["folds"]) if f == FOLD]
    model = smp.Unet(encoder_name="tu-tf_efficientnetv2_s.in21k_ft_in1k", encoder_weights="imagenet", in_channels=1, classes=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / WARM) * 0.5 * (1 + math.cos(math.pi * min(1, s / STEPS))))
    scaler = torch.cuda.amp.GradScaler(); rng = np.random.default_rng(0); t0 = time.time()
    import concurrent.futures as cf
    pool = cf.ThreadPoolExecutor(4)
    def batch():
        seeds = rng.integers(1 << 31, size=BS)
        out = list(pool.map(lambda sd: sample_crop(img, tgt, stats, meta, tr, np.random.default_rng(sd), temporal), seeds))
        return (torch.from_numpy(np.stack([o[0] for o in out]))[:, None], torch.from_numpy(np.stack([o[1] for o in out]))[:, None])
    nxt = pool.submit(batch)
    model.train()
    for step in range(STEPS):
        xb, yb = nxt.result(); nxt = pool.submit(batch)
        xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            lg = model(xb)
        lg = lg.float(); loss = F.binary_cross_entropy_with_logits(lg, yb) + dice_loss(lg, yb)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        if step % 500 == 0: print(f"[{name}] step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    torch.save(model.state_dict(), f"{OUT}/{name}_fold{FOLD}.pt")
    model.eval(); os.makedirs(f"{OUT}/oof_{name}", exist_ok=True)
    for o in va:
        k = meta["index"][o][0]
        x = ((img[k].astype(np.float32) / 255 - stats[k, 0]) / stats[k, 1]).astype(np.float32)
        p = predict(model, x, dev)
        np.savez_compressed(f"{OUT}/oof_{name}/{meta['stems'][o]}.npz", p=np.clip(np.round(p * 255), 0, 255).astype(np.uint8))
    print(f"[{name}] done {time.time() - t0:.0f}s", flush=True)

if __name__ == "__main__":
    t = time.time(); prepare(); print("prep", round(time.time() - t), "s", flush=True)
    torch.backends.cudnn.benchmark = True
    ps = [mp.Process(target=run, args=(0, "control", False)), mp.Process(target=run, args=(1, "temporal", True))]
    [p.start() for p in ps]; [p.join() for p in ps]
    assert all(p.exitcode == 0 for p in ps), [p.exitcode for p in ps]
