"""Show why a minority-supported filament can still improve pooled PQ."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from pycocotools import mask as mu

from filseg.data import gt_rles_by_image, load_records
from filseg.metric import iou_matrix
from filseg.rle import counts_to_rle
from tools.cv_map_parity import load_submissions


def contributions(drawings, masks):
    """Return exact per-mask numerator/denominator increments and record support."""
    numerator=np.zeros(len(masks)); denominator=np.zeros(len(masks))
    support=np.zeros(len(masks),dtype=int); multiple=np.zeros(len(masks),dtype=int)
    for drawing in drawings:
        iou=iou_matrix(drawing,masks)
        hit=iou>.5; count=hit.sum(axis=0)
        numerator+=np.where(hit,iou,0).sum(axis=0)
        denominator+=.5*np.maximum(1,count)
        support+=count>0; multiple+=count>1
    return numerator,denominator,support,multiple


def render(image, drawings, prediction, row, pq, output):
    """Plot native pixels and aligned contours; PDF exports keep labels and contours vector."""
    x,y,w,h=map(int,mu.toBbox(prediction))
    top,left=max(0,y-32),max(0,x-32)
    region=np.s_[top:min(y+h+32,image.shape[0]),left:min(x+w+32,image.shape[1])]
    pixels=image[region]
    ncols = 2
    nrows = (len(drawings) + ncols) // ncols
    fig,axes=plt.subplots(nrows,ncols,figsize=(3.33, 1.08*nrows),squeeze=False)
    for unused in list(axes.flat)[len(drawings)+1:]:
        unused.set_axis_off()
    for index,(axis,drawing) in enumerate(zip(axes.flat,drawings+[None])):
        axis.imshow(pixels,cmap="gray",vmin=0,vmax=255,interpolation="nearest")
        for rle in ([prediction] if drawing is None else drawing):
            mask=mu.decode(rle)[region]
            if mask.any() and not mask.all():
                axis.contour(mask,levels=[.5],colors="white",linewidths=1.4,alpha=.8)
                axis.contour(mask,levels=[.5],colors=["#245C83" if drawing is None else "#656970"],linewidths=.9)
        if drawing is None:
            title="Held-out prediction"
        else:
            best=float(iou_matrix(drawing,[prediction]).max(initial=0))
            title=f"Drawing {index+1} · IoU {best:.2f}"
        axis.set_title(title,fontsize=8.5, fontfamily='serif', pad=2)
        if index == 0:
            # Native pixel units only: JPEGs do not supply a calibrated angular scale.
            length=next(v for v in (100,50,20,10,5,2,1) if v <= pixels.shape[1]/4)
            start,baseline=.06*pixels.shape[1],.88*pixels.shape[0]
            for color,width in (("black",2.4),("white",1.2)):
                axis.plot([start,start+length],[baseline,baseline],color=color,
                          linewidth=width,solid_capstyle="butt")
            axis.text(start+length/2,baseline-4,f"{length} px",ha="center",va="bottom",
                      fontsize=8,color="white",bbox=dict(facecolor="black",alpha=.45,
                                                       edgecolor="none",pad=.5))
        axis.set_axis_off()
    fig.subplots_adjust(left=.015,right=.985,bottom=.025,top=.89,wspace=.08,hspace=.46)
    with plt.rc_context({"pdf.fonttype": 42}):
        fig.savefig(output,dpi=300)
    plt.close(fig)


def main():
    """Audit frozen held-out masks and choose one explicit, reproducible illustration."""
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("gt","pred","images","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--detail",action="store_true",help="Also save per-mask derived statistics.")
    args=parser.parse_args()
    gt=gt_rles_by_image(load_records(args.gt))
    reference=load_submissions([args.pred],gt)[0]
    rows=[]; truth_count=0
    for stem in sorted(gt):
        masks=[counts_to_rle(v) for v in reference.get(stem,())]
        drawings=list(gt[stem].values()); r=len(drawings)
        numerator,denominator,support,multiple=contributions(drawings,masks)
        truth_count+=sum(map(len,drawings))
        for index,mask in enumerate(masks):
            x,y,w,h=map(int,mu.toBbox(mask))
            rows.append(dict(stem=stem,index=index,records=r,hit_records=support[index],
                             h=support[index]/r,g=numerator[index]/r,d=denominator[index]/r,
                             numerator=numerator[index],denominator=denominator[index],
                             multiple=multiple[index],area=int(mu.area(mask)),x=x,y=y,width=w,height=h))
    frame=pd.DataFrame(rows)
    pq=float(frame.numerator.sum()/(.5*truth_count+frame.denominator.sum()))
    frame["pq_surplus"]=frame.numerator-pq*frame.denominator
    minority=frame.query("0<h<.5")
    positive=minority.query("pq_surplus>0 and multiple==0")
    summary=dict(pq=pq,masks=len(frame),with_strict_support=int((frame.h>0).sum()),
                 minority_support=len(minority),minority_positive_surplus=len(positive))
    args.out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([summary]).to_csv(args.out/"summary.csv",index=False)
    if args.detail:
        frame.to_csv(args.out/"per_mask.csv",index=False)
    # Largest native-size illustration with exactly one drawing overlapping at all.
    # The cohort rule is explicit; this is an explanation, not an estimate of prevalence.
    eligible=positive.query("area>200 and width<384 and height<384").sort_values(["area","stem"],ascending=[False,True])
    for row in eligible.itertuples():
        prediction=counts_to_rle(reference[row.stem][row.index])
        drawings=list(gt[row.stem].values())
        best=[float(iou_matrix(d,[prediction]).max(initial=0)) for d in drawings]
        if sum(value>0 for value in best)!=1:
            continue
        pd.DataFrame([row._asdict()]).drop(columns="Index").to_csv(args.out/"example.csv",index=False)
        with Image.open(args.images/f"{row.stem}.jpeg") as source:
            image=np.asarray(source.convert("L"))
        render(image,drawings,prediction,row,pq,args.out/"minority_support.png")
        print(dict(summary=summary,example=row.stem,index=row.index,per_drawing_best_iou=best),flush=True)
        break
    else:
        raise ValueError("No example satisfies the declared display cohort.")


if __name__=="__main__":
    main()
