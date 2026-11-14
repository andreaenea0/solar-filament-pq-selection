"""Recompute public numerical evidence from training labels and released prediction CSVs."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

from filseg.data import gt_rles_by_image, load_records
from filseg.metric import Totals, iou_matrix, score_image, score_record
from filseg.rle import counts_to_rle
from tools.cv_map_parity import load_submissions
from tools.figures.instance_errors import day_bootstrap, metric_values
from tools.figures import notebook_distributions as nd

RECIPES = {"submitted": "oof_submitted.csv", "three-verifiers": "oof_three_verifiers.csv", "public-ensemble": "oof_public_ensemble.csv"}


def annotation_summary(gt):
    """Count the actual rasterized training instances and annotation cohort."""
    areas = np.array([float(mu.area(rle)) for records in gt.values() for masks in records.values() for rle in masks])
    return dict(images=len(gt), records=sum(map(len, gt.values())), instances=len(areas),
                annotators=len({record.rsplit("-", 1)[0] for records in gt.values() for record in records}),
                observing_days=len({stem[:8] for stem in gt}), multi_drawing_images=sum(len(r) > 1 for r in gt.values()),
                minimum_area=float(areas.min()), below_30=int((areas < 30).sum()), below_100=int((areas < 100).sum()))


def human_reference(gt):
    """Pool every ordered pair of distinct drawings on the shared cohort."""
    totals = Totals()
    images, pairs = 0, 0
    for records in gt.values():
        drawings = list(records.values())
        if len(drawings) < 2:
            continue
        images += 1
        for i, truth in enumerate(drawings):
            for j, prediction in enumerate(drawings):
                if i != j:
                    totals += score_record(truth, prediction)
                    pairs += 1
    return dict(images=images, ordered_pairs=pairs, pq=totals.pq)


def score_recipe(gt, predictions, replicates):
    """Return pooled PQ, overlap distributions and paired bootstrap draws."""
    stems = sorted(gt)
    host, overlap, multi_hit = [], [], 0
    for stem in stems:
        masks = [counts_to_rle(c) for c in predictions.get(stem, [])]
        total = score_image(list(gt[stem].values()), masks)
        host.append([total.sum_iou, total.tp, total.fp, total.fn])
        overlap.append(nd.terms(*nd.image_scores(gt[stem], masks)))
        multi_hit += sum(bool(((iou_matrix(truth, masks) > .5).sum(axis=0) > 1).any())
                         for truth in gt[stem].values())
    host, overlap = np.array(host), np.array(overlap)
    _, host_draws = day_bootstrap(stems, host, replicates)
    _, overlap_draws = day_bootstrap(stems, overlap, replicates)
    values, bounds = metric_values(host.sum(0)), np.percentile(metric_values(host_draws), [2.5, 97.5], axis=0)
    result = {name: dict(value=float(value), lo=float(lo), hi=float(hi))
              for name, value, lo, hi in zip(("pq", "sq", "rq", "fp", "fn"), values, *bounds)}
    draw_metrics = {name: metric_values(host_draws)[:, i] for i, name in enumerate(("pq", "sq", "rq"))}
    index = {name: i for i, name in enumerate(nd.TERMS)}
    totals = overlap.sum(0)
    for name, (top, bottom) in nd.METRICS.items():
        sampled = overlap_draws[:, index[top]] / np.maximum(overlap_draws[:, index[bottom]], 1)
        lo, hi = np.percentile(sampled, [2.5, 97.5])
        result[name] = dict(value=float(totals[index[top]] / max(totals[index[bottom]], 1)), lo=float(lo), hi=float(hi))
        draw_metrics[name] = sampled
    result.update(mask_count=sum(map(len, predictions.values())), multi_hit_records=multi_hit,
                  tp=int(host[:, 1].sum()), sum_iou=float(host[:, 0].sum()))
    return result, draw_metrics


def paired_difference(left, right, left_draws, right_draws):
    """Use the identical day draws for each recipe, preserving pairing."""
    return {name: dict(delta=left[name]["value"] - right[name]["value"],
                       lo=float(np.percentile(left_draws[name] - right_draws[name], 2.5)),
                       hi=float(np.percentile(left_draws[name] - right_draws[name], 97.5))) for name in left_draws}


def dice_ablation(path, replicates):
    """Reproduce all three original paired comparisons from published per-image terms."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    configs = sorted({row["config"] for row in rows})
    selected = [next(c for c in configs if tag in c) for tag in ("k03", "k05", "k06")]
    tables = [{r["stem"]: [float(r[c]) for c in ("sum_iou", "tp", "fp", "fn")]
               for r in rows if r["config"] == config} for config in selected]
    if any(table.keys() != tables[0].keys() for table in tables[1:]):
        raise ValueError("Paired Dice-ablation observations must match.")
    stems = sorted(tables[0])
    paired = np.array([[table[stem] for table in tables] for stem in stems])
    days = sorted({stem[:8] for stem in stems})
    groups = {day: np.flatnonzero([s[:8] == day for s in stems]) for day in days}
    rng = np.random.default_rng(0)
    values = metric_values(paired.sum(0))[:, 0]
    comparisons = {}
    names = ("k03", "k05", "k06")
    for left, right in ((1, 2), (0, 2), (0, 1)):
        differences = []
        for _ in range(replicates):
            indices = np.concatenate([groups[day] for day in rng.choice(days, len(days))])
            sampled = metric_values(paired[indices].sum(0))[:, 0]
            differences.append(sampled[right] - sampled[left])
        comparisons[f"{names[left]}_to_{names[right]}"] = dict(
            delta=float(values[right] - values[left]), lo=float(np.percentile(differences, 2.5)),
            hi=float(np.percentile(differences, 97.5)))
    return dict(k03=float(values[0]), k05=float(values[1]), k06=float(values[2]),
                comparisons=comparisons, images=len(stems), observing_days=len(days))


def archive_agreement(left, right):
    """Compare only predicted masks using strict-match count Dice."""
    hits = 0
    for stem in left.keys() | right.keys():
        a, b = ([counts_to_rle(c) for c in rows.get(stem, [])] for rows in (left, right))
        hits += int((iou_matrix(a, b) > .5).sum())
    counts = [sum(map(len, rows.values())) for rows in (left, right)]
    return dict(left=counts[0], right=counts[1], strict_matches=hits, count_dice=2 * hits / sum(counts))


def parameter_counts():
    """Count trainable architecture parameters with downloads disabled."""
    import segmentation_models_pytorch as smp
    import timm

    counts = {}
    for encoder in ("tu-tf_efficientnetv2_s.in21k_ft_in1k", "resnet34"):
        model = smp.Unet(encoder_name=encoder, encoder_weights=None, in_channels=1, classes=1)
        counts[encoder] = sum(p.numel() for p in model.parameters())
        del model
    model = timm.create_model("resnet18.a1_in1k", pretrained=False, in_chans=4, num_classes=1)
    counts["resnet18_verifier"] = sum(p.numel() for p in model.parameters())
    own, public, verifier = counts["tu-tf_efficientnetv2_s.in21k_ft_in1k"], counts["resnet34"], counts["resnet18_verifier"]
    counts.update({"submitted": 2 * own + verifier, "three-verifiers": 2 * own + 3 * verifier,
                   "single-unet": own + verifier, "public_ensemble": 5 * (own + public),
                   "public-ensemble": 2 * own + 5 * (own + public) + 3 * verifier})
    return counts


def calibration_bins(cache):
    """Rebuild annotation-count-weighted calibration bins from the trusted candidate cache."""
    from tools.experiments.cached_selector import load_blend, load_training

    _, _, trees = load_training(cache)
    rows = []
    for source, name in (("pubown", "public-four-verifiers"), ("own", "three-verifiers")):
        blend, _, _ = load_blend(cache, source, trees[source])
        stems = sorted(trees[source])
        scores = np.concatenate([blend[s] for s in stems])
        gains = np.concatenate([trees[source][s]["g"] for s in stems])
        weights = np.concatenate([np.full(len(blend[s]), trees[source][s]["R"]) for s in stems])
        for lower in np.arange(20) * .04:
            keep = (scores >= lower) & (scores < lower + .04)
            rows.append(dict(final=name, lo=float(lower), hi=float(lower + .04), n=int(keep.sum()),
                             mean_score=float(np.average(scores[keep], weights=weights[keep])),
                             mean_gain=float(np.average(gains[keep], weights=weights[keep]))))
    return rows


def selector_control_rows(cache, gt):
    """Recheck historical public-four-verifiers/three-verifiers controls against the official training drawings."""
    from tools.experiments.cached_selector import SOURCES, load_blend, load_training
    from tools.experiments.selector_ablation import audit, summary_rows

    _, folds, trees = load_training(cache)
    if set(gt) != set(folds):
        raise ValueError("Selector cache and official training cohort must agree.")
    result = {"public_archive": "previous_ens4"}
    for source in SOURCES:
        result[source] = audit(source, gt, folds, trees[source],
                               load_blend(cache, source, trees[source]), None)
    return summary_rows(result, len(gt))


def replay_evidence(directory, references, stems):
    """Recover CPU cell timings and mask parity from executed replay notebooks."""
    from tools.timing_replay import read_rows, replay_summary

    stems = set(stems)
    if not stems:
        raise ValueError("Declare every replay observation, including zero-mask images.")
    rows = []
    for token, name in ((name.replace("-", "_"), name) for name in ("submitted", "three-verifiers", "single-unet", "public-ensemble")):
        path = directory / f"pipeline-{token}.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cell = next(c for c in notebook["cells"] if c.get("id") == "inference")
        timing = cell["metadata"]["execution"]
        start, finish = timing["iopub.execute_input"], timing["shell.execute_reply"]
        actual = read_rows(directory / f"final-{token}.csv", stems)
        expected = read_rows(references / f"{token}_test.csv", stems, subset=True)
        parity = replay_summary(expected, actual, stems)
        rows.append(dict(recipe=name, stem=";".join(sorted(stems)), started_utc=start, finished_utc=finish,
                         seconds=(datetime.fromisoformat(finish) - datetime.fromisoformat(start)).total_seconds(),
                         archive_masks=len(expected), actual_masks=len(actual), shared_ids=parity['shared_ids'],
                         exact_rles=parity['exact_rles'], strict_matches=parity['strict_matches'],
                         minimum_iou=parity['minimum_iou'],
                         notebook_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return rows


def main():
    """Write recomputed numerical evidence without changing any released artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True, help="Official training annotation JSON only.")
    parser.add_argument("--data", type=Path, default=Path("report/data"))
    parser.add_argument("--references", type=Path, default=Path("docs/reference"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--parameters", action="store_true", help="Also instantiate the released architectures and count parameters.")
    parser.add_argument("--cache", type=Path, help="Optional trusted candidate cache for calibration and historical selector controls.")
    parser.add_argument("--replay-dir", type=Path, help="Optional executed pipeline notebooks and their CSVs.")
    parser.add_argument("--replay-stems", nargs="+", help="All observations processed by the replay notebooks, including empty predictions.")
    args = parser.parse_args()
    if bool(args.replay_dir) != bool(args.replay_stems):
        parser.error("Use --replay-dir and --replay-stems together.")
    if args.bootstrap < 2:
        parser.error("Use at least two bootstrap draws.")
    gt = gt_rles_by_image(load_records(args.gt))
    predictions = load_submissions([args.data / name for name in RECIPES.values()], gt)
    result, draws = dict(annotations=annotation_summary(gt), bootstrap_draws=args.bootstrap), {}
    result["human_reference"] = human_reference(gt)
    for name, prediction in zip(RECIPES, predictions):
        result[name], draws[name] = score_recipe(gt, prediction, args.bootstrap)
    from tools.figures import rubric_table as rubric
    named = dict(zip(("submitted", "three_verifiers", "public_ensemble"), predictions))
    rubric_rows = []
    for name, prediction in named.items():
        stems = sorted(gt)
        scored = [rubric.image_terms(gt[stem], [counts_to_rle(c) for c in prediction.get(stem, [])])
                  for stem in stems]
        terms = np.array([row for row, _ in scored])
        pairs = [pairs for _, pairs in scored]
        rubric_rows.extend(rubric.rubric_rows(name, stems, terms, pairs, args.bootstrap))
    result["rubric_rows"] = rubric_rows
    distributions, summaries = nd.tables(gt, named, args.bootstrap)
    result["notebook_tables"] = dict(distributions=distributions, summary=summaries)
    result["submitted_minus_three_verifiers"] = paired_difference(result["submitted"], result["three-verifiers"], draws["submitted"], draws["three-verifiers"])
    result["dice_ablation"] = dice_ablation(args.data / "dice_ablation_fold0_per_image.csv", args.bootstrap)
    from tools.experiments.submitted_verifier_ablation import summarize as verifier_summary
    with (args.data / "submitted_verifier_per_image.csv").open(newline="", encoding="utf-8") as stream:
        verifier_rows = list(csv.DictReader(stream))
    summary, folds, blocks = verifier_summary(verifier_rows, args.bootstrap)
    result["submitted_verifier_ablation"] = dict(summary=summary, folds=folds, blocks=blocks)
    from tools.experiments.submitted_rule_ablation import summarize as rule_summary
    with (args.data / "submitted_rule_per_image.csv").open(newline="", encoding="utf-8") as stream:
        rule_rows = list(csv.DictReader(stream))
    summary, folds = rule_summary(rule_rows, args.bootstrap)
    result["submitted_rule_ablation"] = dict(summary=summary, folds=folds)
    from tools.experiments.pixel_controls import summarize as pixel_summary
    result["historical_pixel_controls"] = {}
    for name, control in (("capacity", "k06"), ("multiscale", "scale1")):
        with (args.data / f"pixel_{name}_per_image.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        result["historical_pixel_controls"][name] = pixel_summary(rows, control, args.bootstrap)
    paths = [args.references / f"{name}_test.csv" for name in ("submitted", "three_verifiers", "public_ensemble", "single_unet")]
    reference_stems = set()
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reference_stems.update(row["filament_id"].rsplit("_", 1)[0] for row in csv.DictReader(stream))
    references = load_submissions(paths, reference_stems)
    result["test_mask_counts"] = dict(zip(("submitted", "three-verifiers", "public-ensemble", "single-unet"), [sum(map(len, p.values())) for p in references]))
    result["submitted_vs_three_verifiers_test"] = archive_agreement(references[0], references[1])
    result["submitted_vs_public_ensemble_test"] = archive_agreement(references[0], references[2])
    if args.parameters:
        result["parameters"] = parameter_counts()
    if args.cache:
        result["calibration_bins"] = calibration_bins(args.cache)
        result["historical_selector_rows"] = selector_control_rows(args.cache, gt)
    if args.replay_dir:
        result["replay_evidence"] = replay_evidence(args.replay_dir, args.references, args.replay_stems)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=lambda value: value.item()) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, default=lambda value: value.item()))


if __name__ == "__main__":
    main()
