"""Build current-report numbers and figures from released evidence and training pixels."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from PIL import Image

from filseg.data import gt_rles_by_image, load_records
from filseg.metric import iou_matrix
from filseg.rle import counts_to_rle
from tools.cv_map_parity import load_submissions
from tools.figures.claim_audit import annotation_summary, dice_ablation
from tools.figures.minority_support import render
from tools.timing_replay import write_csv

BLUE, ORANGE = "#245C83", "#B5521B"


def read_csv(path):
    """Load public, derived numerical evidence."""
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def pipeline(output):
    """Draw the deployed path without suggesting an extra model."""
    fig, ax = plt.subplots(figsize=(7, .92))
    ax.set(xlim=(0, 10), ylim=(0, 1))
    ax.axis("off")
    labels = ["H-alpha JPEG\n2048 x 2048", "2 own U-Nets\nD4 ensemble mean", "Component tree\n6 thresholds",
              "LightGBM + 1 CNN\npredicted utility", "Antichain DP\ndisjoint masks"]
    for i, text in enumerate(labels):
        x = i * 2 + .06
        patch = FancyBboxPatch((x, .17), 1.73, .63, boxstyle="round,pad=.025",
                               facecolor="#F5F7F9", edgecolor="#7D97AA", linewidth=.8)
        ax.add_patch(patch)
        ax.text(x+.865, .485, text, ha="center", va="center", fontsize=8, color="#172B3A")
        if i < 4:
            ax.add_patch(FancyArrowPatch((x+1.78, .485), (x+1.97, .485),
                                         arrowstyle="-|>", mutation_scale=7, linewidth=.9,
                                         color=BLUE, shrinkA=0, shrinkB=0))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    with plt.rc_context({"pdf.fonttype": 42}):
        fig.savefig(output)
    plt.close(fig)


def controls(output, one, extra):
    """Keep the sign and paired uncertainty of the two component comparisons."""
    values = [float(one["delta"]), float(extra["delta"])]
    lower = [float(one["lo"]), float(extra["lo"])]
    upper = [float(one["hi"]), float(extra["hi"])]
    fig, ax = plt.subplots(figsize=(3.38, 1.65))
    ax.axvline(0, color=".5", lw=.8, zorder=1)
    ax.grid(axis="x", color="#E8ECEF", linewidth=.6, zorder=0)
    for i, color in enumerate((BLUE, "#68747D")):
        ax.errorbar(values[i]*1000, 1-i,
                    xerr=np.array([[values[i]-lower[i]], [upper[i]-values[i]]])*1000,
                    fmt="o", color=color, markerfacecolor=color if i == 0 else "white",
                    markeredgewidth=1.1, markersize=4.3, capsize=2.8, lw=1.4, zorder=3)
    ax.set_yticks([1, 0], ["1 vs 0 verifiers\ncontrol tuned", "1 vs 3 verifiers\nsame penalty"])
    ax.set_xlabel(r"Change in PQ ($10^{-3}$)", fontsize=8.7)
    ax.set_xticks(np.arange(-2, 11, 2))
    ax.set_xlim(min(-3., min(lower)*1000-.7), max(10.5, max(upper)*1000+.7))
    ax.set_ylim(-.5, 1.5)
    ax.tick_params(labelsize=8)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set(color="#73808A", linewidth=.7)
    fig.subplots_adjust(left=.35, right=.985, bottom=.28, top=.96)
    with plt.rc_context({"pdf.fonttype": 42}):
        fig.savefig(output)
    plt.close(fig)


def main():
    """Export deterministic TeX, traced scalar values and three real-data figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True, help="Official competition training JSON only.")
    parser.add_argument("--images", type=Path, required=True, help="Official training JPEGs.")
    parser.add_argument("--data", type=Path, default=Path("report/data"))
    parser.add_argument("--out", type=Path, default=Path("report"))
    args = parser.parse_args()
    figures, tables = args.out/"figs", args.out/"tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    (args.out/"data").mkdir(parents=True, exist_ok=True)
    gt = gt_rles_by_image(load_records(args.gt))
    stats = annotation_summary(gt)
    if stats["images"] != 707 or stats["records"] != 1154:
        raise ValueError("Expected the complete official training-only cohort.")
    rubric = {(r["model"], r["metric"]): r for r in read_csv(args.data/"rubric_finals.csv")}
    paired = next(r for r in read_csv(args.data/"rubric_paired_vs_three_verifiers.csv")
                  if r["model"] == "submitted" and r["metric"] == "pq")
    verifier = next(r for r in read_csv(args.data/"submitted_verifier_summary.csv")
                    if r["control"] == "LGB-only-tuned")
    fixed = next(r for r in read_csv(args.data/"submitted_verifier_summary.csv") if r["control"] == "LGB-only")
    rule = next(r for r in read_csv(args.data/"submitted_rule_summary.csv") if int(r["block_days"]) == 1)
    dice = dice_ablation(args.data/"dice_ablation_fold0_per_image.csv", 2000)
    example = read_csv(args.data/"minority_support_example.csv")[0]
    source_params = read_csv(args.data/"complexity_ablation.csv")
    params = [float(next(r for r in source_params if key in r["variant"])["params_M_at_test"])
              for key in ("LGB + 1 r18 (0.5/0.5)", "[three-verifiers]", "[new public-ensemble]")]
    values, trace = {}, []
    def number(name, value, source, field, digits=4):
        values[name] = f"{float(value):.{digits}f}"
        trace.append(dict(macro=name, source=source, field=field, value=float(value), rendered=values[name]))
    public_score = read_csv(args.data/"public_selected_score.csv")[0]
    number("PublicScore",public_score["public_score_display"],"public_selected_score.csv","public_score_display",2)
    for macro, field in (("NImages","images"),("NRecords","records"),("NInstances","instances"),
                         ("NAnnotators","annotators"),("NDays","observing_days")):
        number(macro,stats[field],args.gt.name,field,0)
    for macro, metric, field in (("SubmittedPQ","pq","value"),("SubmittedLo","pq","lo"),("SubmittedHi","pq","hi"),
                                  ("SubmittedSQ","sq","value"),("SubmittedRQ","rq","value")):
        number(macro,rubric["submitted",metric][field],"rubric_finals.csv","submitted/"+metric+"/"+field)
    for macro, row, field in (("VerifierGain",verifier,"delta"),("VerifierLo",verifier,"lo"),
                              ("VerifierHi",verifier,"hi"),("LgbTunedPQ",verifier,"control_pq"),
                              ("LgbFixedPQ",fixed,"control_pq")):
        number(macro,row[field],"submitted_verifier_summary.csv",row["control"]+"/"+field)
    for macro, field in (("SubmittedMinusThreeVerifiers","delta"),("SubmittedMinusThreeVerifiersLo","lo"),("SubmittedMinusThreeVerifiersHi","hi")):
        number(macro,paired[field],"rubric_paired_vs_three_verifiers.csv","submitted/three-verifiers/pq/"+field)
    for macro, field in (("RulePQ","rule_pq"),("RuleGain","delta"),("RuleLo","lo"),("RuleHi","hi")):
        number(macro,rule[field],"submitted_rule_summary.csv","1-day/"+field)
    for macro, value in (("DiceCropPQ",dice["k05"]),("DiceBatchPQ",dice["k06"]),
                         ("DiceGain",dice["comparisons"]["k05_to_k06"]["delta"]),
                         ("DiceLo",dice["comparisons"]["k05_to_k06"]["lo"]),
                         ("DiceHi",dice["comparisons"]["k05_to_k06"]["hi"])):
        number(macro,value,"dice_ablation_fold0_per_image.csv","paired-day/"+macro)
    number("PublicEnsemblePQ",rubric["public_ensemble","pq"]["value"],"rubric_finals.csv","public_ensemble/pq")
    for macro, value in zip(("SubmittedParams","ThreeVerifierParams","PublicEnsembleParams"), params):
        number(macro,value,"complexity_ablation.csv","params_M_at_test/"+macro,1)
    number("ExampleIoU",float(example["numerator"])/int(example["hit_records"]),
           "minority_support_example.csv","numerator/hit_records")
    number("ExampleGain",example["g"],"minority_support_example.csv","g")
    number("HalfPQ",float(verifier["submitted_pq"])/2,"submitted_verifier_summary.csv","submitted_pq/2")
    (tables/"submitted_numbers.tex").write_text(
        "% Generated by tools.figures.paper_submitted; values traced in paper_claims.csv.\n"
        +"".join("\\newcommand{\\"+k+"}{"+v+"}\n" for k,v in values.items()),encoding="utf-8")
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Recipe & PQ [95\% interval] & Pair Dice & Pair IoU & Split (\%) & Merge (\%) & Params (M) \\",
             r"\midrule"]
    for model, title, parameter in zip(("submitted","three_verifiers","public_ensemble"),("submitted (selected)","three-verifiers","Public ensemble"),params):
        get=lambda key:float(rubric[model,key]["value"])
        q=rubric[model,"pq"]
        cells=[title,f"{get('pq'):.4f} [{float(q['lo']):.4f}, {float(q['hi']):.4f}]",
               f"{get('overlap_dice_mean'):.4f}", f"{get('overlap_iou_mean'):.4f}",
               f"{100*get('one_to_many'):.2f}", f"{100*get('many_to_one'):.2f}", f"{parameter:.1f}"]
        lines.append(" & ".join(cells)+r" \\")
    lines += [r"\bottomrule",r"\end{tabular}"]
    (tables/"submitted_results.tex").write_text("\n".join(lines)+"\n",encoding="utf-8")
    pipeline(figures/"submitted_pipeline.pdf")
    controls(figures/"submitted_controls.pdf",verifier,paired)
    predictions=load_submissions([args.data/"oof_submitted.csv"],gt)[0]
    stem=example["stem"]
    prediction=counts_to_rle(predictions[stem][int(example["index"])])
    drawings=list(gt[stem].values())
    best=[float(iou_matrix(d,[prediction]).max(initial=0)) for d in drawings]
    assert sum(v>.5 for v in best)==int(example["hit_records"])==1
    assert np.isclose(np.mean([v if v>.5 else 0 for v in best]),float(example["g"]))
    with Image.open(args.images/(stem+".jpeg")) as image:
        pixels=np.asarray(image.convert("L"))
    row=SimpleNamespace(hit_records=int(example["hit_records"]),records=int(example["records"]),g=float(example["g"]))
    render(pixels,drawings,prediction,row,float(verifier["submitted_pq"]),figures/"submitted_decision.png")
    render(pixels,drawings,prediction,row,float(verifier["submitted_pq"]),figures/"submitted_decision.pdf")
    write_csv(args.out/"data/paper_claims.csv",trace)
    inputs=[args.gt,args.images/(stem+".jpeg"),*[args.data/name for name in (
        "oof_submitted.csv","rubric_finals.csv","rubric_paired_vs_three_verifiers.csv",
        "submitted_verifier_summary.csv","dice_ablation_fold0_per_image.csv",
        "minority_support_example.csv","complexity_ablation.csv","public_selected_score.csv",
        "submitted_rule_summary.csv")]]
    (tables/"submitted_build.json").write_text(json.dumps(dict(
        images=stats["images"],records=stats["records"],example=stem,per_drawing_best_iou=best,
        inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}),indent=2)+"\n",encoding="utf-8")
    print(f"Wrote {len(trace)} traced macros, current result table and 3 figures.")


if __name__=="__main__":
    main()
