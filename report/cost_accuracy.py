"""Plot the seven configurations compared in the paper directly from its cost table."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

from tools.figures.palette import BLUE, COMPARISON_GREEN
from report.observed_pipeline import STYLE, save


def draw(path, out):
    """Read named rows; omit the verifier-only and partial-fold seed controls."""
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    def point(key):
        hits = [r for r in rows if key == r["component"]]
        if len(hits) != 1:
            raise ValueError(f"Expected exactly one cost-table row for {key}")
        row = hits[0]
        number = lambda value: float(value.replace(r"\textbf{", "").replace("}", ""))
        return number(row["params"]), number(row["cv"])

    families = [
        ("Study U-Nets", BLUE, "o", [r"\textbf{Our U-Nets}, fixed rule", r"\quad + GBM",
          r"\quad \textbf{+ GBM + 1 verifier (submitted)}", r"\quad + GBM + 3 verifiers"]),
        ("With public U-Nets", COMPARISON_GREEN, "s", [r"\textbf{10 public U-Nets}, fixed rule",
          r"\quad + our U-Nets, GBM + 1 verifier", r"\quad + our U-Nets, GBM + 3 verifiers"])]
    fig, ax = plt.subplots(figsize=(3.33, 2.40))
    fig.subplots_adjust(left=.19, right=.975, top=.80, bottom=.23)
    for label, color, marker, keys in families:
        points = [point(key) for key in keys]
        x, y = zip(*points)
        ax.plot(x, y, marker=marker, color=color, markersize=4.5, lw=1.05, label=label)
    x, y = point(families[0][3][2])
    ring, = ax.plot(x, y, linestyle="none", marker="o", ms=8, mfc="white", mec=BLUE, mew=1.1)
    dot, = ax.plot(x, y, linestyle="none", marker="o", ms=3.5, color=BLUE)
    ax.set(xlabel="Neural parameters at deployment (millions)", ylabel="Held-out PQ",
           xlim=(20,330), ylim=(.442,.466), xticks=(50,100,150,200,250,300), yticks=(.445,.450,.455,.460,.465))
    ax.grid(axis="y", lw=.45, color="#d7dadd")
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend([*handles, (ring, dot)], [*labels, "Submitted"],
               handler_map={tuple: HandlerTuple(ndivide=1)}, loc="upper center",
               bbox_to_anchor=(.52, .97), ncol=3, frameon=False, fontsize=7.2,
               handlelength=1.25, handletextpad=.45, columnspacing=.85)
    save(fig, out, "deployment_cost")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=Path("report/data/complexity.csv"), type=Path)
    parser.add_argument("--out", default=Path("report/figs"), type=Path)
    args = parser.parse_args()
    with plt.rc_context(STYLE):
        draw(args.data, args.out)
