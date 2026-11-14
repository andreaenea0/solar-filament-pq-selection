"""Reproduce the three distributions of the host Self Evaluation notebook for held-out submissions.

The notebook plots the IoU and the Dice of every overlapping pair of one annotated filament and one
predicted mask, and counts for every annotated filament and every mask how many partners it
overlaps. Annotator records scored against each other on the same quantities give a human reference.
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from filseg.data import gt_rles_by_image, load_records
from filseg.metric import IOU_T, iou_matrix
from filseg.rle import counts_to_rle
from tools.cv_map_parity import load_submissions
from tools.figures.instance_errors import day_bootstrap, write_table

HUMAN, SHARED = "annotator_pairs", "@shared"
BINS = np.linspace(0, 1, 51)  # the notebook's 50 bins
TOP_DEGREE = 3
TERMS = ("pairs", "iou", "dice", "below", "gt", "gt_missed", "gt_split", "pred", "pred_spurious", "pred_merged")
METRICS = {
    "mean_iou": ("iou", "pairs"), "mean_dice": ("dice", "pairs"), "pairs_at_or_below_half": ("below", "pairs"),
    "gt_missed": ("gt_missed", "gt"), "gt_split": ("gt_split", "gt"),
    "pred_spurious": ("pred_spurious", "pred"), "pred_merged": ("pred_merged", "pred"),
}


def record_scores(truth, predictions):
    """Return the IoU of the overlapping pairs and the number of partners of each drawing and each mask."""
    iou = iou_matrix(truth, predictions)
    related = iou > 0
    return iou[related], related.sum(axis=1), related.sum(axis=0)


def image_scores(drawings, predictions):
    """Score one set of masks against every record, or every record against the others when it is None."""
    if predictions is None:
        cases = [(truth, drawn) for i, drawn in enumerate(drawings.values())
                 for j, truth in enumerate(drawings.values()) if i != j]
    else:
        cases = [(truth, predictions) for truth in drawings.values()]
    scores = [record_scores(truth, drawn) for truth, drawn in cases]
    return tuple(np.concatenate(part) if part else np.zeros(0) for part in zip(*scores))


def terms(pairs, gt_degree, pred_degree):
    """Return the additive terms behind every summary ratio."""
    return np.array([len(pairs), pairs.sum(), (2 * pairs / (1 + pairs)).sum(), (pairs <= IOU_T).sum(),
                     len(gt_degree), (gt_degree == 0).sum(), (gt_degree >= 2).sum(),
                     len(pred_degree), (pred_degree == 0).sum(), (pred_degree >= 2).sum()], dtype=float)


def distribution_rows(source, scores):
    """Yield the notebook histograms of IoU and Dice and the partner counts of both sides."""
    pairs = np.concatenate([s[0] for s in scores])
    for section, values in (("iou", pairs), ("dice", 2 * pairs / (1 + pairs))):
        counts, _ = np.histogram(values, BINS)
        for lower, upper, count in zip(BINS, BINS[1:], counts):
            yield dict(source=source, section=section, lower=lower, upper=upper, count=int(count),
                       share=count / max(len(values), 1))
    for section, column in (("gt_degree", 1), ("pred_degree", 2)):
        degree = Counter(np.minimum(np.concatenate([s[column] for s in scores]), TOP_DEGREE).astype(int).tolist())
        total = max(sum(degree.values()), 1)
        for value in range(TOP_DEGREE + 1):  # the last degree is open: TOP_DEGREE partners or more
            yield dict(source=source, section=section, lower=value, upper="" if value == TOP_DEGREE else value,
                       count=degree[value], share=degree[value] / total)


def summary_rows(source, stems, scores, replicates):
    """Yield each summary ratio with a 95% day-bootstrap interval, then the sample sizes."""
    table = np.array([terms(*s) for s in scores])
    _, draws = day_bootstrap(stems, table, replicates)
    totals, index = table.sum(axis=0), {name: i for i, name in enumerate(TERMS)}
    for metric, (top, bottom) in METRICS.items():
        ratio = draws[:, index[top]] / np.maximum(draws[:, index[bottom]], 1)
        lo, hi = np.percentile(ratio, [2.5, 97.5])
        yield dict(source=source, metric=metric, value=totals[index[top]] / max(totals[index[bottom]], 1), lo=lo, hi=hi)
    for metric, value in (("images", len(stems)), ("pairs", totals[index["pairs"]]),
                          ("drawings", totals[index["gt"]]), ("masks", totals[index["pred"]])):
        yield dict(source=source, metric=metric, value=int(value))


def write_detail(directory, source, scores):
    """Write every pair and every partner count of one source, as report/make_figures.py reads them."""
    directory.mkdir(parents=True, exist_ok=True)
    name = source.replace("@", "_")
    pairs = np.concatenate([s[0] for s in scores])
    write_table(directory / f"pairs_{name}.csv",
                [{"iou": round(v, 5), "dice": round(2 * v / (1 + v), 5)} for v in pairs])
    write_table(directory / f"m2n_{name}.csv", [{"side": side, "degree": int(degree)}
                                                for side, column in (("gt", 1), ("pred", 2))
                                                for s in scores for degree in s[column]])


def tables(gt, submissions, replicates, detail_dir=None):
    """Return the distribution and summary rows of every submission and of the annotator pairs.

    A submission is reported on all images and, as NAME@shared, on the images with two or more
    records, which are the only ones where annotators can be compared with each other.
    """
    stems = sorted(gt)
    shared = [stem for stem in stems if len(gt[stem]) >= 2]
    distributions, summaries = [], []
    cases = [(name, stems, submission) for name, submission in submissions.items()]
    cases += [(name + SHARED, shared, submission) for name, submission in submissions.items()]
    for source, images, submission in [*cases, (HUMAN, shared, None)]:
        scores = [image_scores(gt[stem], None if submission is None else
                               [counts_to_rle(counts) for counts in submission.get(stem, [])]) for stem in images]
        distributions.extend(distribution_rows(source, scores))
        if detail_dir:
            write_detail(detail_dir, source, scores)
        summaries.extend(summary_rows(source, images, scores, replicates))
    return distributions, summaries


def draw(distributions, summaries, out, model="submitted", label="Submitted model"):
    """Draw the three notebook plots for one submission next to the annotator pairs, on the shared images."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    ink, muted, grid = "#1b1b1f", "#5d6470", "#e3e5e8"
    styles = {HUMAN: ("Annotator vs. annotator", "#1F5FAD"), model + SHARED: (label, "#1BAF7A")}
    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted, "xtick.labelcolor": ink, "ytick.labelcolor": ink,
                         "text.color": ink, "axes.linewidth": .6, "pdf.fonttype": 42})
    rows, means = pd.DataFrame(distributions), pd.DataFrame(summaries).set_index(["source", "metric"])["value"]
    figure = plt.figure(figsize=(3.33, 3.1))
    layout = figure.add_gridspec(2, 2, height_ratios=(1, .9))
    panels = {"iou": figure.add_subplot(layout[0, 0]), "dice": figure.add_subplot(layout[0, 1])}
    relations = figure.add_subplot(layout[1, :])
    for axis in (*panels.values(), relations):
        axis.grid(axis="y", color=grid, lw=.5)
        axis.set_axisbelow(True)
        axis.tick_params(length=2, width=.5)

    for section, axis in panels.items():
        for source, (_, colour) in styles.items():
            part = rows[(rows["source"] == source) & (rows["section"] == section)]
            share = 100 * part["share"].to_numpy()
            if source == HUMAN:
                axis.stairs(share, BINS, color=colour, lw=1.1)
            else:
                axis.bar(part["lower"], share, width=.02, align="edge", color=colour, alpha=.75, lw=0)
            mean = means[source, f"mean_{section}"]
            axis.axvline(mean, color=colour, lw=.9, ls=(0, (3, 2)))
            axis.text(mean + (-.03 if source == HUMAN else .03), 1.02, f"{mean:.2f}", color=ink, fontsize=6.5,
                      ha="right" if source == HUMAN else "left", va="bottom", transform=axis.get_xaxis_transform())
        axis.set(xlim=(0, 1), ylim=(0, None), xlabel=f"{'IoU' if section == 'iou' else 'Dice'} of overlapping pairs",
                 xticks=(0, .5, 1))
    panels["iou"].set_ylabel("Share of pairs (%)")

    names = ("1:0\nmissed", "1:1", "1:2+\nsplit", "0:1\nspurious", "1:1", "2+:1\nmerged")
    for shift, (source, (name, colour)) in zip((-.2, .2), styles.items()):
        share = []
        for section in ("gt_degree", "pred_degree"):
            part = rows[(rows["source"] == source) & (rows["section"] == section)].set_index("lower")["share"]
            share += [part[0], part[1], part[2:].sum()]
        position = np.arange(6) + (np.arange(6) >= 3) * .5 + shift
        relations.bar(position, 100 * np.array(share), width=.38, color=colour, alpha=.75 if source != HUMAN else 1,
                      lw=0, label=name)
        for x, value in zip(position, share):
            relations.text(x, 100 * value + 1.5, f"{100 * value:.0f}" if value >= .1 else f"{100 * value:.1f}",
                           ha="center", va="bottom", fontsize=5.8, color=ink)
    relations.set(xticks=np.arange(6) + (np.arange(6) >= 3) * .5, xticklabels=names, ylim=(0, 118),
                  yticks=(0, 25, 50, 75, 100), ylabel="Share (%)")
    relations.text(1, 116, "Annotated filaments", ha="center", va="top", fontsize=6.5, color=ink)
    relations.text(4.5, 116, "Predicted masks", ha="center", va="top", fontsize=6.5, color=ink)
    figure.tight_layout(pad=.3, w_pad=.8, h_pad=1.)
    figure.legend(*relations.get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False,
                  handlelength=1.2, columnspacing=1.4, bbox_to_anchor=(.5, 1.))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=300, bbox_inches="tight", pad_inches=.02)
    plt.close(figure)


def main():
    """Score each named held-out submission and the annotator pairs as the host notebook does."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True, help="Official training annotation JSON.")
    parser.add_argument("--pred", nargs="+", required=True, metavar="NAME=CSV",
                        help="Held-out submission CSVs on training images, each with a short model name.")
    parser.add_argument("--out", type=Path, required=True, help="CSV of histograms and partner counts.")
    parser.add_argument("--summary", type=Path, required=True, help="CSV of means and rates with intervals.")
    parser.add_argument("--figure", type=Path, help="Optional figure of one submission against the annotator pairs.")
    parser.add_argument("--model", default="submitted=Submitted model", metavar="NAME=LABEL",
                        help="Submission of the figure.")
    parser.add_argument("--detail-dir", type=Path, help="Also write pairs_SOURCE.csv and m2n_SOURCE.csv per source.")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    named = dict(item.split("=", 1) for item in args.pred)
    gt = gt_rles_by_image(load_records(args.gt))
    submissions = dict(zip(named, load_submissions([Path(path) for path in named.values()], gt)))
    distributions, summaries = tables(gt, submissions, args.bootstrap, args.detail_dir)
    for path, rows in ((args.out, distributions), (args.summary, summaries)):
        path.parent.mkdir(parents=True, exist_ok=True)
        write_table(path, rows)
    if args.figure:
        draw(distributions, summaries, args.figure, *args.model.split("=", 1))
    print(f"Wrote {len(distributions)} distribution rows and {len(summaries)} summary rows")


if __name__ == "__main__":
    main()
