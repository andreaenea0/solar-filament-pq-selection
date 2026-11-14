"""Build every data-driven figure and table of the report from plain CSV/NPZ files.

Usage:  python make_figures.py [--data data] [--out .]

Inputs (all optional: whatever is missing is skipped with a message), in --data:
  ablation.csv        [group,]config,pq,pq_lo,pq_hi,sq,rq  one row per ablation step, in order; optional group header
  complexity.csv      component,cv,dpq,params,time,lb   cost-benefit table (TBD -> red); built from complexity_ablation.csv
  rubric.csv          metric,value                         free-form rows (Dice, IoU, 1-to-many, ...)
  matched_iou.csv     iou                                  one row per matched (TP) pair, per annotator record
  dice_per_image.csv  image_id,dice                        per-image Dice of the final OOF prediction
  lambda_sweep_finals.csv model,lambda,pq,pq_lo,pq_hi,masks current frozen recipes
  examples/*.npz      image (H,W) float, ann (R,H,W) int instance labels per annotator,
                      pred (H,W) int instance labels, optional box=[y0,y1,x0,x1], optional title (str)

Outputs, in --out:
  figs/examples.pdf       Figure 2 (rows = examples; columns = image | annotators | ours)
  figs/distributions.pdf  Figure 3 (a) matched IoU + per-image Dice, (b) PQ vs lambda with half of the displayed grid maximum
  tables/ablation.tex     Table 1 (full tabular; rows with config 'Final...' go under a rule)
  tables/rubric.tex       Table 2 (full tabular)
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, GREY = "#1F5FAD", "#B5521B", "#8A8F98"
ANN_COLORS = ["#E6A100", "#2A9D8F", "#C2185B", "#6A4C93", "#43AA8B"]
COL_W = 3.33  # ACM sigconf column width, inches

plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def _read(path):
    if not os.path.exists(path):
        print(f"skip: {path} not found")
        return None
    return pd.read_csv(path)


def fmt_ci(v, lo, hi):
    if pd.isna(lo) or pd.isna(hi):
        return f"{v:.2f}" if abs(v * 100 - round(v * 100)) < 1e-9 else f"{v:.3f}"  # LB scores have 2 decimals
    return f"{v:.3f} \\scriptsize[{lo:.3f}, {hi:.3f}]"


def table_ablation(df, out):
    """Full tabular. Rows whose config starts with 'Final' go below a midrule (e.g. public LB)."""
    body, tail = [], []
    group = None
    for _, r in df.iterrows():
        if "group" in df.columns and r["group"] != group:
            group = r["group"]
            if body:
                body.append("        \\midrule")
            body.append(f"        \\multicolumn{{4}}{{@{{}}l}}{{\\emph{{{group}}}}} \\\\")
        sq = "--" if pd.isna(r.get("sq")) else f"{r['sq']:.3f}"
        rq = "--" if pd.isna(r.get("rq")) else f"{r['rq']:.3f}"
        line = f"        {r['config']} & {fmt_ci(r['pq'], r.get('pq_lo'), r.get('pq_hi'))} & {sq} & {rq} \\\\"
        (tail if str(r["config"]).startswith("Final") else body).append(line)
    tex = ["\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{0.43\\columnwidth}p{0.31\\columnwidth}rr@{}}", "        \\toprule",
           "        Configuration & PQ [95\\% CI] & SQ & RQ \\\\", "        \\midrule", *body]
    if tail:
        tex += ["        \\midrule", *tail]
    tex += ["        \\bottomrule", "    \\end{tabular}"]
    open(out, "w").write("\n".join(tex) + "\n")
    print("wrote", out)


def table_rubric(df, out):
    """metric + one or more value columns (e.g. A,B). 'TBD' cells are shown in red."""
    cols = [c for c in df.columns if c != "metric"]
    cell = lambda v: "\\textcolor{red}{[TBD]}" if str(v).strip() in ("TBD", "nan") else str(v)
    rows = [f"        {r['metric']} & " + " & ".join(cell(r[c]) for c in cols) + " \\\\" for _, r in df.iterrows()]
    head = "        Metric & " + " & ".join("Value" if c == "value" else c for c in cols) + " \\\\"
    tex = ["\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{@{}l" + "c" * len(cols) + "@{}}", "        \\toprule", head, "        \\midrule",
           *rows, "        \\bottomrule", "    \\end{tabular}"]
    open(out, "w").write("\n".join(tex) + "\n")
    print("wrote", out)



def table_complexity(df, out):
    """Cost-benefit table. Columns: component,cv,dpq,params,time,lb (strings; TBD shown in red)."""
    cell = lambda v: "\\textcolor{red}{[TBD]}" if str(v).strip() == "TBD" else str(v)
    rows = [f"        {r['component']} & {cell(r['cv'])} & {cell(r['dpq'])} & {cell(r['params'])} & {cell(r['time'])} & {cell(r['lb'])} \\\\"
            for _, r in df.iterrows()]
    tex = ["\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{@{}lccccc@{}}", "        \\toprule",
           "        Pipeline & OOF PQ & $\\Delta$ & Par.\\ (M) & s/img & LB \\\\", "        \\midrule",
           *rows, "        \\bottomrule", "    \\end{tabular}"]
    open(out, "w").write("\n".join(tex) + "\n")
    print("wrote", out)


def recipe_curve(sweep, name):
    """Select the named current curve so report panels cannot share an untagged legacy sweep."""
    if sweep is None:
        return None
    if "model" not in sweep:
        raise ValueError("Use the tagged lambda_sweep_finals.csv.")
    curve = sweep.loc[sweep["model"] == name].copy()
    if curve.empty:
        raise ValueError(f"No penalty curve for {name}.")
    return curve


def fig_distributions(iou, dice, sweep, out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL_W, 1.45), gridspec_kw={"wspace": 0.45})
    # (a) distributions
    bins = np.linspace(0, 1, 26)
    if iou is not None:
        a.hist(iou["iou"], bins=bins, color=BLUE, alpha=0.85, label="matched IoU", density=True)
        a.axvline(np.median(iou["iou"]), color=BLUE, lw=0.8, ls="--")
    if dice is not None:
        a.hist(dice["dice"], bins=bins, histtype="step", color=ORANGE, lw=1.1, label="Dice / image", density=True)
    a.axvline(0.5, color=GREY, lw=0.6, ls=":")
    a.set_xlabel("value"); a.set_ylabel("density"); a.set_xlim(0, 1)
    a.legend(loc="upper left", frameon=True, framealpha=0.85, edgecolor="none"); a.set_title("(a)", loc="left")
    # (b) PQ vs lambda
    if sweep is not None:
        sweep = sweep.sort_values("lambda")
        if {"pq_lo", "pq_hi"} <= set(sweep.columns):
            b.fill_between(sweep["lambda"], sweep["pq_lo"], sweep["pq_hi"], color=ORANGE, alpha=0.18, lw=0)
        b.plot(sweep["lambda"], sweep["pq"], color=ORANGE, lw=1.2)
        i = int(sweep["pq"].values.argmax())
        pq_star, lam_best = sweep["pq"].values[i], sweep["lambda"].values[i]
        b.axvline(pq_star / 2, color=GREY, lw=0.8, ls="--")
        b.plot([lam_best], [pq_star], "o", color=ORANGE, ms=3)
        b.annotate("max grid PQ / 2", (pq_star / 2, b.get_ylim()[0]), xytext=(2, 2),
                   textcoords="offset points", color=GREY, fontsize=6.5)
    b.set_xlabel(r"selection penalty $\lambda$"); b.set_ylabel("OOF PQ"); b.set_title("(b)", loc="left")
    fig.savefig(out); plt.close(fig)
    print("wrote", out)


def fig_rubric(pairs, m2n, ref_pairs, ref_m2n, out, name="final", ref_name="public rule"):
    """Organisers' Self-Evaluation plots: pairwise Dice and IoU of every overlapping (GT, prediction) pair
    (score > 0), and GT->pred (1:n) / pred->GT (n:1) overlap degrees. Reference pipeline as grey outline."""
    fig, axs = plt.subplots(1, 3, figsize=(COL_W, 1.35), gridspec_kw={"wspace": 0.42})
    bins = np.linspace(0, 1, 26)
    for ax, col, lab in ((axs[0], "dice", "pairwise Dice"), (axs[1], "iou", "pairwise IoU")):
        w = np.ones(len(pairs)) / len(pairs)
        ax.hist(pairs[col], bins=bins, weights=w, color=BLUE, alpha=0.85, label=name)
        ax.axvline(pairs[col].mean(), color=BLUE, lw=0.8, ls="--")
        if ref_pairs is not None:
            ax.hist(ref_pairs[col], bins=bins, weights=np.ones(len(ref_pairs)) / len(ref_pairs),
                    histtype="step", color=GREY, lw=0.9, label=ref_name)
        ax.set_xlim(0, 1); ax.set_xlabel(lab); ax.set_xticks([0, 0.5, 1])
        ax.text(0.03, 0.97, f"mean\n{pairs[col].mean():.3f}", transform=ax.transAxes, va="top", fontsize=5.5, color=BLUE)
        if ref_pairs is not None and ref_name == "annotators":
            ax.text(0.03, 0.72, f"{ref_pairs[col].mean():.3f}", transform=ax.transAxes, va="top", fontsize=5.5, color=GREY)
    axs[0].set_ylabel("probability")
    axs[0].set_title("(a)", loc="left"); axs[1].set_title("(b)", loc="left")
    axs[0].legend(loc="upper left", bbox_to_anchor=(-0.02, 0.62) if ref_name == "annotators" else (-0.02, 0.76), frameon=False, fontsize=5, handlelength=0.9, handletextpad=0.3, labelspacing=0.2)
    ax = axs[2]
    K = 3  # degrees 0,1,2,>=3
    def frac(df, side):
        d = df.loc[df.side == side, "degree"].clip(upper=K).values
        return np.array([(d == k).mean() for k in range(K + 1)])
    xs = np.arange(K + 1) + 0.5
    for side, sgn in (("gt", 1), ("pred", -1)):
        f = frac(m2n, side)
        ax.bar(sgn * xs, f, width=0.8, color=[ORANGE if k == 0 else BLUE for k in range(K + 1)], alpha=0.85)
        if ref_m2n is not None:
            for k, v in enumerate(frac(ref_m2n, side)):
                ax.plot([sgn * xs[k] - 0.4, sgn * xs[k] + 0.4], [v, v], color=GREY, lw=0.9)
    ax.set_yscale("log"); ax.set_ylim(1e-3, 1.2)
    ax.set_xticks(np.r_[-xs[::-1], xs]); ax.set_xticklabels([f"{k}" if k < K else f"{K}+" for k in range(K, -1, -1)] + [f"{k}" if k < K else f"{K}+" for k in range(K + 1)], fontsize=5.5)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel(r"$\leftarrow$pred.: #GT  GT: #pred.$\rightarrow$", fontsize=5.5)
    ax.set_title("(c)", loc="left")
    fig.savefig(out); plt.close(fig)
    print("wrote", out)


def fig_lambda(sweep, out):
    fig, b = plt.subplots(figsize=(COL_W * 0.62, 1.3))
    sweep = sweep.sort_values("lambda")
    if {"pq_lo", "pq_hi"} <= set(sweep.columns):
        b.fill_between(sweep["lambda"], sweep["pq_lo"], sweep["pq_hi"], color=ORANGE, alpha=0.18, lw=0)
    b.plot(sweep["lambda"], sweep["pq"], color=ORANGE, lw=1.2)
    i = int(sweep["pq"].values.argmax())
    pq_star, lam_best = sweep["pq"].values[i], sweep["lambda"].values[i]
    b.axvline(pq_star / 2, color=GREY, lw=0.8, ls="--")
    b.plot([lam_best], [pq_star], "o", color=ORANGE, ms=3)
    b.annotate("max grid PQ / 2", (pq_star / 2, b.get_ylim()[0]), xytext=(2, 2), textcoords="offset points", color=GREY, fontsize=6.5)
    b.set_xlabel(r"selection penalty $\lambda$"); b.set_ylabel("OOF PQ")
    fig.savefig(out); plt.close(fig)
    print("wrote", out)


def _outline(ax, lab, color, lw=0.7):
    for k in np.unique(lab):
        if k == 0:
            continue
        ax.contour((lab == k).astype(float), levels=[0.5], colors=[color], linewidths=lw)


def fig_examples(files, out):
    n = len(files)
    fig, axes = plt.subplots(n, 3, figsize=(COL_W, 1.12 * n + 0.1), squeeze=False,
                             gridspec_kw={"wspace": 0.03, "hspace": 0.08})
    rng = np.random.default_rng(0)
    for row, f in enumerate(files):
        d = np.load(f, allow_pickle=True)
        img, ann, pred = d["image"], d["ann"], d["pred"]
        if "box" in d:
            y0, y1, x0, x1 = [int(v) for v in d["box"]]
            img, ann, pred = img[y0:y1, x0:x1], ann[:, y0:y1, x0:x1], pred[y0:y1, x0:x1]
        lo, hi = np.percentile(img, [1, 99.5])
        for c in range(3):
            ax = axes[row, c]
            ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, interpolation="lanczos")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        for r in range(ann.shape[0]):
            _outline(axes[row, 1], ann[r], ANN_COLORS[r % len(ANN_COLORS)])
        ids = [k for k in np.unique(pred) if k]
        rgba = np.zeros(pred.shape + (4,))
        for k in ids:
            col = matplotlib.colors.to_rgba(plt.cm.tab10(rng.integers(10)))
            rgba[pred == k] = (*col[:3], 0.45)
        axes[row, 2].imshow(rgba, interpolation="nearest")
        _outline(axes[row, 2], pred, "white", lw=0.4)
        title = (str(d["title"]) if "title" in d else os.path.splitext(os.path.basename(f))[0]).split(" (")[0].split("_")[0]
        axes[row, 0].set_ylabel(title, fontsize=6)
    for c, t in enumerate(["H$\\alpha$ image", f"annotators", "ours"]):
        axes[0, c].set_title(t)
    fig.savefig(out, dpi=300); plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "figs"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)
    j = lambda *p: os.path.join(a.data, *p)  # noqa: E731

    abl = _read(j("ablation.csv"))
    if abl is not None:
        table_ablation(abl, os.path.join(a.out, "tables", "ablation.tex"))
    rub = _read(j("rubric_en.csv")) if os.path.exists(j("rubric_en.csv")) else _read(j("rubric.csv"))
    if rub is not None:
        table_rubric(rub, os.path.join(a.out, "tables", "rubric.tex"))
    cx = _read(j("complexity.csv"))
    if cx is not None:
        table_complexity(cx, os.path.join(a.out, "tables", "complexity.tex"))
    curves = _read(j("lambda_sweep_finals.csv"))
    sweep = recipe_curve(curves, "submitted")
    iou, dice = _read(j("matched_iou.csv")), _read(j("dice_per_image.csv"))
    if any(x is not None for x in (iou, dice, sweep)):
        fig_distributions(iou, dice, sweep, os.path.join(a.out, "figs", "distributions.pdf"))
    for tag in ("public-ensemble", "three-verifiers", "submitted"):  # per-final figures from data/A, data/B (lambda sweep shared)
        d = os.path.join(a.data, tag)
        if os.path.isdir(d):
            r = lambda n: _read(os.path.join(d, n))  # noqa: E731
            fig_distributions(r("matched_iou.csv"), r("dice_per_image.csv"), recipe_curve(curves, tag),
                              os.path.join(a.out, "figs", f"distributions_{tag}.pdf"))
            key = tag.replace("-", "")
            pr, mm = _read(j(f"pairs_{key}.csv")), _read(j(f"m2n_{key}.csv"))
            if pr is not None and mm is not None:
                fig_rubric(pr, mm, _read(j("pairs_rule.csv")), _read(j("m2n_rule.csv")),
                           os.path.join(a.out, "figs", f"rubric_dist_{tag}.pdf"), name=f"final {tag}")
            # same plots on the multi-annotator images, with annotator-vs-annotator agreement as reference
            p296, m296 = j(f"pairs_{key}_296.csv"), j(f"m2n_{key}_296.csv")
            if os.path.exists(p296) and os.path.exists(j("pairs_annot.csv")):
                fig_rubric(_read(p296), _read(m296), _read(j("pairs_annot.csv")), _read(j("m2n_annot.csv")),
                           os.path.join(a.out, "figs", f"rubric_dist_{tag}_annot.pdf"), name=("our model" if tag == "submitted" else tag), ref_name="annotators")
            exs = sorted(glob.glob(os.path.join(d, "examples", "*.npz")))
            if exs:
                fig_examples(exs[:4], os.path.join(a.out, "figs", f"examples_{tag}.pdf"))
    if sweep is not None:
        fig_lambda(sweep, os.path.join(a.out, "figs", "lambda.pdf"))
    ex = sorted(glob.glob(j("examples", "*.npz")))
    if ex:
        fig_examples(ex[:4], os.path.join(a.out, "figs", "examples.pdf"))
    else:
        print("skip: no examples/*.npz")


if __name__ == "__main__":
    main()
