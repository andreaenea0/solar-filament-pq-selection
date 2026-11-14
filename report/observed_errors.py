"""Show aligned training drawings and submitted-model masks for three published error crops."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocotools import mask as coco

from filseg.data import gt_rles_by_image, load_records
from filseg.rle import counts_to_rle
from tools.figures.palette import BLUE, COMPARISON_GREEN
from report.observed_pipeline import STYLE, contour, save
from tools.timing_replay import read_rows

# Coordinates and ordinal drawings are those published in docs/MORPHOLOGY.md.
CASES = [("Merge", "20120519070814Th", 2, 594, 1224),
         ("Split", "20130719140154Bh", 2, 1176, 712),
         ("Limb", "20140604195854Bh", 1, 1536, 518)]


def draw(images, annotations, predictions, out):
    """Use identical intensity scaling for both views of each native 512-pixel crop."""
    drawings = gt_rles_by_image(load_records(annotations))
    archive = read_rows(predictions, {stem for _,stem,*_ in CASES}, subset=True)
    fig, axes = plt.subplots(2, 3, figsize=(3.33, 2.45))
    fig.subplots_adjust(left=.10, right=.99, bottom=.025, top=.86, wspace=.055, hspace=.06)
    for i, (name, stem, record, x, y) in enumerate(CASES):
        crop = np.s_[y:y+512, x:x+512]
        with Image.open(images / f"{stem}.jpeg") as observation:
            pixels = np.asarray(observation.convert("L"))[crop]
        truth = drawings[stem][sorted(drawings[stem])[record-1]]
        pred = [counts_to_rle(v) for k,v in archive.items() if k.rpartition("_")[0] == stem]
        lo, hi = np.percentile(pixels, [1,99.5])
        axes[0,i].set_title(name, fontsize=8, pad=2)
        for ax, masks, color in zip(axes[:,i], (truth,pred), (COMPARISON_GREEN,BLUE)):
            ax.imshow(pixels, cmap="gray", vmin=lo, vmax=hi)
            for m in masks:
                contour(ax, coco.decode(m)[crop], color, .65)
            ax.set_axis_off()
    for ax,label in zip(axes[:,0], ("Drawing","Prediction")):
        ax.text(-.12,.5,label,rotation=90,ha="center",va="center",transform=ax.transAxes,fontsize=7.5)
    save(fig,out,"observed_errors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("images","annotations","predictions","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    with plt.rc_context(STYLE):
        draw(args.images,args.annotations,args.predictions,args.out)
