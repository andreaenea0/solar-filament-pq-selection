"""Draw one held-out training crop through the released selector, checking its archived masks."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocotools import mask as coco

from filseg.candidates import LADDER
from filseg.data import gt_rles_by_image, load_records
from filseg.select import dp_select
from tools.figures.palette import AMBER, BLUE
from tools.cv_sparse import read_pickle
from tools.timing_replay import read_rows

STEM, PARENT = "20130314165714Mh", 13
PENALTY = .25
STYLE = {"font.family": "serif", "font.size": 8, "pdf.fonttype": 42,
         "mathtext.fontset": "dejavuserif", "axes.spines.top": False, "axes.spines.right": False}


def arguments():
    """Read the training-only sources used by both views of this example."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "verifier", "images", "annotations", "predictions", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    return parser.parse_args()


def observed(args):
    """Recover the exact tree decision and verify every selected RLE on this image."""
    tree = read_pickle(args.cache / "cands_train_own.pkl")[STEM]
    booster = read_pickle(args.cache / "cands_train_own_cv_nometa.pkl")["oof_pred"][STEM]
    verifier = read_pickle(args.verifier)[STEM]
    gains = .5 * np.asarray(booster) + .5 * np.asarray(verifier)
    keep = dp_select(tree["parent"], tree["children"], tree["level"], gains - PENALTY)
    archived = read_rows(args.predictions, {STEM}, subset=True)
    recovered = {f"{STEM}_{i}": tree["rle"][node]["counts"].decode("ascii")
                 for i, node in enumerate(keep, 1)}
    if recovered != archived:
        raise ValueError("This tree and verifier must reproduce the submitted model's held-out masks.")
    def branch(node):
        if node in keep:
            return [node]
        return [item for child in tree["children"][node] for item in branch(child)]
    branches = [branch(child) for child in tree["children"][PARENT]]
    if len(branches) != 2 or any(len(b) != 1 for b in branches):
        raise ValueError("Expected exactly one retained mask in each of two child branches.")
    nodes = [PARENT, branches[0][0], branches[1][0]]
    x, y, w, h = map(int, coco.toBbox(tree["rle"][PARENT]))
    size = max(w, h) + 30
    top, left = y + h // 2 - size // 2, x + w // 2 - size // 2
    crop = np.s_[top:top + size, left:left + size]
    with Image.open(args.images / f"{STEM}.jpeg") as frame:
        pixels = np.asarray(frame.convert("L"))[crop]
    with np.load(args.cache / "maps_own_oof" / f"{STEM}.npz") as archive:
        probability = archive["p"][crop]
    drawings = gt_rles_by_image(load_records(args.annotations))[STEM]
    outlines = [[coco.decode(mask)[crop] for mask in record] for record in drawings.values()]
    masks = [coco.decode(tree["rle"][n])[crop] for n in nodes]
    print(f"{STEM}: {len(keep)} archived masks matched; nodes {nodes}, scores {gains[nodes]}")
    return pixels, probability, masks, gains[nodes], [LADDER[tree["level"][n]] for n in nodes], outlines


def contour(ax, mask, color, width=.9):
    """Trace native mask geometry over light and dark pixels."""
    if np.any(mask) and not np.all(mask):
        ax.contour(mask, levels=[.5], colors=["white"], linewidths=width + .7)
        ax.contour(mask, levels=[.5], colors=[color], linewidths=width)


def save(fig, directory, name):
    """Keep vector contours/text and write a matching print-size preview."""
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / (name + ".pdf"), dpi=300, metadata={"CreationDate": None})
    fig.savefig(directory / (name + ".png"), dpi=300)
    plt.close(fig)


def draw(data, directory):
    """Show six views of the same field through inference and the training inputs below."""
    image, p, masks, scores, _, _ = data
    fig, axes = plt.subplots(1, 6, figsize=(7, 1.98))
    fig.subplots_adjust(left=.008, right=.992, top=.82, bottom=.30, wspace=.17)
    titles = ("Hα image", "Pixel probability", "Candidate masks", "Choice of masks", "Selected pair", "On the image")
    for ax, title in zip(axes, titles):
        ax.set_box_aspect(1)
        ax.set_axis_off()
        ax.set_title(title, fontsize=8, pad=5)
    for i in (0, 2, 5):
        axes[i].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[1].imshow(p, cmap="cividis", vmin=0, vmax=255)
    child_colors = (BLUE, "#57859c")
    contour(axes[2], masks[0], AMBER)
    for mask, color in zip(masks[1:], child_colors):
        contour(axes[2], mask, color)
    ax = axes[3]
    weights = scores - PENALTY
    ax.set(xlim=(0, .33), ylim=(0, 1))
    ax.text(0, .80, f"Parent   {weights[0]:.3f}", fontsize=7.4)
    ax.barh(.66, weights[0], height=.105, color=AMBER)
    ax.text(0, .41, f"Pair   {weights[1:].sum():.3f}", fontsize=7.4)
    ax.barh(.27, weights[1], height=.105, color=child_colors[0])
    ax.barh(.27, weights[2], left=weights[1], height=.105, color=child_colors[1])
    axes[4].imshow(masks[1] + 2*masks[2], cmap=matplotlib.colors.ListedColormap(["#f3f4f5", *child_colors]),
                   vmin=0, vmax=2, interpolation="nearest")
    for mask, color in zip(masks[1:], child_colors):
        contour(axes[5], mask, color)
    fig.add_artist(plt.Line2D([.01,.99], [.25,.25], transform=fig.transFigure, color="#bfc3c7", lw=.6))
    fig.text(.01, .12, "Training", weight="bold", fontsize=8)
    fig.text(.097, .12, "U-Net: fraction of drawings covering each pixel", fontsize=7.5)
    fig.text(.50, .12, "Scorers: candidate overlap with held-out drawings", fontsize=7.5)
    save(fig, directory, "observed_pipeline")


if __name__ == "__main__":
    args = arguments()
    with plt.rc_context(STYLE):
        draw(observed(args), args.out)
