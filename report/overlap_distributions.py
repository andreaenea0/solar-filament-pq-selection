"""Plot host overlap distributions on the same 296 multi-drawing observations."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools.figures.palette import BLUE, COMPARISON_GREEN
from report.observed_pipeline import STYLE, save


def draw(table, out):
    """Retain all bins; show the two overlap directions as separate panels."""
    fig, grid = plt.subplots(2, 2, figsize=(3.33, 3.05))
    fig.subplots_adjust(left=.155, right=.965, bottom=.15, top=.84, hspace=.84, wspace=.49)
    labels = ("Pairwise Dice", "Pairwise IoU", "Per prediction", "Per drawing mask")
    for ax, key, title in zip(grid.flat, ("dice","iou","pred_degree","gt_degree"), labels):
        for source, label, color, style in (("submitted@shared","Submitted model",BLUE,"-"),
                                           ("annotator_pairs","Annotator pairs",COMPARISON_GREEN,"--")):
            rows = table[(table.source == source) & (table.section == key)].sort_values("lower")
            if rows.empty or not np.isclose(rows.share.sum(), 1):
                raise ValueError(f"Missing or incomplete distribution: {source}, {key}")
            if key in ("dice", "iou"):
                edges = np.r_[rows.lower, rows.upper.iloc[-1]]
                ax.stairs(100*rows.share, edges, color=color, ls=style, lw=1.05, label=label)
                ax.set(xlim=(0,1), xticks=(0,.5,1))
            else:
                bins = np.minimum(rows.lower.astype(int), 3)
                values = rows.share.groupby(bins).sum().reindex(range(4), fill_value=0)
                shift = -.10 if style == "-" else .10
                ax.plot(np.arange(4)+shift, 100*values, linestyle="none", color=color,
                        marker="o" if style == "-" else "s", markersize=3.5)
                ax.set(xticks=range(4), xticklabels=["0","1","2","3+"], xlim=(-.35,3.35), yscale="log")
        ax.set_title(title, fontsize=8, pad=3)
        ax.grid(axis="y", color="#ddd", lw=.45)
        ax.tick_params(labelsize=7, pad=1, length=2)
    grid[0,0].set_ylabel("Pairs (%)", fontsize=7.5)
    grid[1,0].set_ylabel("Masks (%)", fontsize=7.5)
    for ax in grid[1]:
        ax.set_ylim(.025, 130)
        ax.set_yticks([.1, 1, 10, 100], labels=["0.1", "1", "10", "100"])
        ax.set_xlabel("Overlap partners", fontsize=7.5)
    fig.legend(*grid[0,0].get_legend_handles_labels(), loc="upper center", frameon=False, ncol=2,
               handlelength=1.5, columnspacing=1, fontsize=7.5)
    save(fig, out, "overlap_distributions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("report/data/notebook_distributions.csv"))
    parser.add_argument("--out", type=Path, default=Path("report/figs"))
    args = parser.parse_args()
    with plt.rc_context(STYLE):
        draw(pd.read_csv(args.data), args.out)
