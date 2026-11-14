"""Draw the appendix figure that compares the solar geometry of predictions and annotations."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

if __package__:
    from .palette import AMBER, BLUE, COMPARISON_GREEN
else:  # Also support running this file directly.
    from palette import AMBER, BLUE, COMPARISON_GREEN
ANNOTATIONS, PAIRS = "annotations", "annotator_pairs"
MARKS = ((COMPARISON_GREEN, "s"), (BLUE, "o"), (AMBER, "^"))
MODELS = ("submitted=Submitted model",)
INK, MUTED, GRID = "#1b1b1f", "#5d6470", "#e3e5e8"
AREA_BINS = np.logspace(np.log10(40), np.log10(6400), 12)
MIN_TILTS = 30


def day_interval(frame, value, band, replicates=1000):
    """Return a 95% interval of the per-band median, resampling observing days."""
    days = {day: group for day, group in frame.assign(day=frame["stem"].str[:8]).groupby("day")}
    keys, rng = list(days), np.random.default_rng(23)
    draws = []
    for _ in range(replicates):
        sample = pd.concat([days[keys[i]] for i in rng.integers(len(keys), size=len(keys))])
        draws.append(sample.groupby(band, observed=False)[value].median())
    return np.nanpercentile(np.array(draws), [2.5, 97.5], axis=0)


def series(axis, x, y, style, label=True):
    name, colour, marker = style
    axis.plot(x, y, color=colour, marker=marker, lw=1.2, ms=3.6, mec="white", mew=.5, label=name if label else None)


def draw(table, instances, radius, out, models=MODELS):
    """Write the four panels: radial density, tilt by latitude, corrected area and PQ by radius.

    `models` lists one or two SOURCE=LABEL pairs of the tables, drawn after the annotations.
    """
    labels = {ANNOTATIONS: "Annotators", **dict(item.split("=", 1) for item in models)}
    styles = {source: (label, *mark) for (source, label), mark in zip(labels.items(), MARKS)}
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 7.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
                         "text.color": INK, "axes.linewidth": .6, "pdf.fonttype": 42})
    figure, axes = plt.subplots(2, 2, figsize=(3.33, 3.9))
    (density, tilt), (area, quality) = axes
    for axis in axes.flat:
        axis.grid(axis="y", color=GRID, lw=.5)
        axis.set_axisbelow(True)
        axis.tick_params(length=2, width=.5)

    for source, style in styles.items():
        rows = table[table["source"] == source]
        ring = rows[rows["section"] == "radial"]
        series(density, (ring["lower"] + ring["upper"]) / 2, ring["density_per_record_per_hemisphere"], style)

        band = rows[(rows["section"] == "tilt_by_latitude") & (rows["eligible"] >= MIN_TILTS)]
        centre = (band["lower"] + band["upper"]) / 2
        if source == ANNOTATIONS:
            drawn = instances[(instances["source"] == source) & instances["tilt"].notna()]
            edges = sorted(set(band["lower"]) | set(band["upper"]))
            lo, hi = day_interval(drawn.assign(band=pd.cut(drawn["latitude"], edges, right=False)), "tilt", "band")
            tilt.fill_between(centre, lo, hi, color=style[1], alpha=.16, lw=0)
        series(tilt, centre, band["tilt_deg_median"], style, label=False)

        central = instances[(instances["source"] == source) & instances["area_msh"].notna()]
        counts, _ = np.histogram(central["area_msh"], AREA_BINS)
        drawings = ring["records"].iloc[0]
        series(area, np.sqrt(AREA_BINS[:-1] * AREA_BINS[1:]), counts / drawings / np.diff(np.log10(AREA_BINS)),
               style, label=False)

    for shift, (source, style) in zip((-.015, 0, .015), styles.items()):
        ring = radius[(radius["source"] == (PAIRS if source == ANNOTATIONS else source)) & radius["upper"].notna()]
        centre = (ring["lower"] + ring["upper"]) / 2 + shift
        quality.errorbar(centre, ring["pq"], yerr=(ring["pq"] - ring["pq_lo"], ring["pq_hi"] - ring["pq"]),
                         fmt="none", ecolor=style[1], elinewidth=.7, alpha=.8)
        series(quality, centre, ring["pq"], style, label=False)

    density.set(xlabel="Radius $r/R_\\odot$", ylabel="Filaments / drawing\nper unit hemisphere",
                xlim=(0, 1), ylim=(0, None), title="(a) Radial density")
    tilt.set(xlabel="Latitude (deg)", ylabel="Median tilt\nfrom E–W (deg)", xticks=(-60, 0, 60),
             title="(b) Tilt")
    area.set(xscale="log", xlabel="Corrected area (MSH)", ylabel="Filaments per drawing\nper decade",
             ylim=(0, None), title="(c) Area")
    quality.set(xlabel="Radius $r/R_\\odot$", ylabel="PQ", xlim=(0, 1), ylim=(0, .55),
                title="(d) Radial PQ")
    figure.subplots_adjust(left=.19, right=.938, bottom=.115, top=.855, wspace=.77, hspace=.91)
    figure.legend(*density.get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False,
                  handlelength=1.6, columnspacing=1.4, bbox_to_anchor=(.5, .927))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=300)
    plt.close(figure)


def main():
    """Read the tables of physics_compare and radius_agreement and write one figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True, help="Aggregate CSV from tools.figures.physics_compare.")
    parser.add_argument("--instances", type=Path, required=True, help="Per-instance CSV from the same tool.")
    parser.add_argument("--radius", type=Path, required=True, help="CSV from tools.figures.radius_agreement.")
    parser.add_argument("--out", type=Path, required=True, help="Figure path; the extension selects the format.")
    parser.add_argument("--models", nargs="+", default=MODELS, metavar="SOURCE=LABEL",
                        help="One or two model sources of the tables with their legend labels.")
    args = parser.parse_args()
    if len(args.models) > 2:
        parser.error("The figure has room for two models next to the annotations.")
    draw(pd.read_csv(args.table), pd.read_csv(args.instances), pd.read_csv(args.radius), args.out, args.models)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
