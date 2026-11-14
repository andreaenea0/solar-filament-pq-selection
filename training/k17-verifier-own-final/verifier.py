"""k17: final own-maps verifier: ResNet-18 seeds 0, 1, 2 trained on all folds; saves weights and test predictions.

k14: CNN verifier for the PQ-optimal selection.

Each candidate (node of the threshold component tree) is a 4-channel 96x96 crop: image, public-ensemble probability,
own-model probability, candidate mask. An ImageNet-pretrained ResNet-18 (timm, 4 input channels) regresses the
candidate's expected PQ gain g = mean_r IoU_r * 1[IoU_r > 0.5] with MSE weighted by the number of annotator records.
5-fold OOF predictions (same folds as everything else) + a model on all folds for the test candidates. D4 TTA.
"""
import glob, os, time, pickle, numpy as np, torch, torch.nn as nn, torch.multiprocessing as mp
import timm
IN = os.environ.get("FILSEG_INPUT", "/kaggle/input")
OUT = os.environ.get("FILSEG_OUTPUT", "/kaggle/working")
def find(n): h = sorted(glob.glob(f"{IN}/**/{n}", recursive=True)); assert h, n; return h[0]
E, BS, LR = 12, 256, 1e-3
def train_predict(rank, jobs):
    dev = torch.device(f"cuda:{rank}")
    tr = np.load(find("crops_train.npz")); X = torch.from_numpy(tr["X"]); st, idx = tr["stem"], tr["idx"]
    T = np.load(find("targets.npz")); y = torch.from_numpy(T["g"]).float(); w = torch.from_numpy(T["w"]).float(); fold = T["fold"]
    te = np.load(find("crops_test.npz")); Xt = torch.from_numpy(te["X"])
    for k, arch, seed in jobs:
        torch.manual_seed(seed); np.random.seed(seed)
        tri = np.where(fold != k)[0]; vai = np.where(fold == k)[0] if k >= 0 else None
        m = timm.create_model(arch, pretrained=True, in_chans=4, num_classes=1).to(dev).to(memory_format=torch.channels_last)
        opt = torch.optim.AdamW(m.parameters(), LR, weight_decay=1e-4); steps = E * (len(tri) // BS)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.1)
        scaler = torch.cuda.amp.GradScaler(); t0 = time.time()
        for e in range(E):
            m.train(); perm = np.random.permutation(tri); tot = 0
            for b in range(len(tri) // BS):
                ii = perm[b * BS:(b + 1) * BS]; xb = X[ii].to(dev).float() / 255
                if np.random.rand() < .5: xb = xb.flip(3)
                xb = torch.rot90(xb, np.random.randint(4), (2, 3))
                with torch.autocast("cuda", dtype=torch.float16): pr = m(xb).squeeze(1).float()
                loss = (w[ii].to(dev) * (pr - y[ii].to(dev)) ** 2).mean()
                opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step(); tot += loss.item()
            print(f"[{k} {arch} {seed}] epoch {e} loss {tot / (b + 1):.4f} {time.time() - t0:.0f}s", flush=True)
        m.eval()
        def pred(Z):
            out = []
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                for b in range(0, len(Z), 1024):
                    xs = Z[b:b + 1024].to(dev).float() / 255
                    out.append(torch.stack([m(torch.rot90(xs.flip(3) if f else xs, r, (2, 3))).squeeze(1).float() for r in range(4) for f in (0, 1)]).mean(0).cpu())
            return torch.cat(out).numpy()
        if k >= 0:
            p = pred(X[vai]); res = {}
            for s in np.unique(st[vai]):
                sel = vai[st[vai] == s]; res[str(s)] = p[st[vai] == s][np.argsort(idx[sel])]
            pickle.dump(res, open(f"{OUT}/verif_oof_fold{k}_{arch.split('.')[0]}_s{seed}.pkl", "wb"))
        else:
            p = pred(Xt); res = {}
            for s in np.unique(te["stem"]):
                msk = te["stem"] == s; res[str(s)] = p[msk][np.argsort(te["idx"][msk])]
            torch.save(m.state_dict(), f"{OUT}/verifier_{arch.split('.')[0]}_s{seed}.pt")
            pickle.dump(res, open(f"{OUT}/verif_test_{arch.split('.')[0]}_s{seed}.pkl", "wb"))
if __name__ == "__main__":
    J = [(k, a, sd) for a, sd in [("resnet18.a1_in1k", 0), ("resnet18.a1_in1k", 1), ("resnet18.a1_in1k", 2)] for k in [-1]]
    ps = [mp.Process(target=train_predict, args=(r, J[r::2])) for r in range(2)]
    [p.start() for p in ps]; [p.join() for p in ps]
    assert all(p.exitcode == 0 for p in ps)
