"""Check that the numerical audit pools records and retains host multi-hit behavior."""

import numpy as np
import pytest
import csv
import json

from filseg.metric import encode
from filseg.rle import mask_to_counts
from tools.figures.claim_audit import annotation_summary, paired_difference, replay_evidence, score_recipe


def test_audit_counts_missing_drawings_and_pairs_bootstrap():
    stem = "20260101000000Ah"
    first = np.zeros((2048, 2048), dtype=np.uint8)
    second = first.copy()
    first[20:30, 20:30] = 1
    second[50:60, 50:60] = 1
    gt = {stem: {f"a-{stem}": [encode(first)], f"b-{stem}": [encode(second)]}}
    result, draws = score_recipe(gt, {stem: [mask_to_counts(first)]}, 20)
    assert result["pq"]["value"] == .5
    assert result["mean_iou"]["value"] == result["mean_dice"]["value"] == 1
    assert result["gt_missed"]["value"] == result["pred_spurious"]["value"] == .5
    assert result["multi_hit_records"] == 0
    assert annotation_summary(gt) == dict(images=1, records=2, instances=2, annotators=2,
                                          observing_days=1, multi_drawing_images=1,
                                          minimum_area=100, below_30=0, below_100=0)
    assert all(row == dict(delta=0, lo=0, hi=0)
               for row in paired_difference(result, result, draws, draws).values())


def test_audit_detects_multiple_strict_hits_in_one_drawing():
    first = np.zeros((2048, 2048), dtype=np.uint8)
    second = first.copy()
    first[0, :2] = 1
    second[0, 1:3] = 1
    gt = {"20260101x": {"a-20260101x": [encode(first), encode(second)]}}
    result, _ = score_recipe(gt, {"20260101x": [mask_to_counts(first | second)]}, 20)
    assert result["pq"]["value"] == pytest.approx(2 / 3)
    assert result["tp"] == 2 and result["multi_hit_records"] == 1


def test_replay_evidence_uses_recorded_execution_times_and_decoded_masks(tmp_path):
    runs, references = tmp_path / "runs", tmp_path / "references"
    runs.mkdir()
    references.mkdir()
    mask = np.zeros((2048, 2048), dtype=np.uint8)
    mask[10:20, 30:40] = 1
    for token in ("submitted", "three_verifiers", "single_unet", "public_ensemble"):
        for path in (runs / f"final-{token}.csv", references / f"{token}_test.csv"):
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("filament_id", "segmentation_rle"))
                writer.writerow(("observation_1", mask_to_counts(mask)))
        cell = dict(id="inference", metadata=dict(execution={
            "iopub.execute_input": "2026-11-10T12:00:00Z", "shell.execute_reply": "2026-11-10T12:00:02.25Z"}))
        (runs / f"pipeline-{token}.ipynb").write_text(json.dumps(dict(cells=[cell])), encoding="utf-8")
    rows = replay_evidence(runs, references, {"observation"})
    assert [row["recipe"] for row in rows] == ["submitted", "three-verifiers", "single-unet", "public-ensemble"]
    assert all(row["seconds"] == 2.25 and row["minimum_iou"] == 1 and row["exact_rles"] == 1
               and row["strict_matches"] == 1 for row in rows)


def test_human_reference_pools_pairs_instead_of_averaging_pair_scores():
    from tools.figures.claim_audit import human_reference
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 1
    rle = encode(mask)
    result = human_reference({"day": {"a": [rle], "b": [rle], "c": []}})
    assert result == dict(images=1, ordered_pairs=6, pq=.5)


def test_replay_evidence_keeps_empty_images_and_never_matches_between_observations(tmp_path):
    runs, references = tmp_path / "runs", tmp_path / "refs"
    runs.mkdir(); references.mkdir()
    mask = np.zeros((2048, 2048), dtype=np.uint8)
    mask[20:30, 20:30] = 1
    counts = mask_to_counts(mask)
    for token in ("submitted", "three_verifiers", "single_unet", "public_ensemble"):
        with (runs / f"final-{token}.csv").open("w", newline="") as stream:
            csv.writer(stream).writerows([("filament_id", "segmentation_rle"), ("other_1", counts)])
        with (references / f"{token}_test.csv").open("w", newline="") as stream:
            csv.writer(stream).writerows([("filament_id", "segmentation_rle"), ("empty_1", counts)])
        cell = dict(id="inference", metadata=dict(execution={
            "iopub.execute_input": "2026-11-13T12:00:00Z", "shell.execute_reply": "2026-11-13T12:00:02Z"}))
        (runs / f"pipeline-{token}.ipynb").write_text(json.dumps(dict(cells=[cell])), encoding="utf-8")
    rows = replay_evidence(runs, references, {"empty", "other"})
    assert all(row["archive_masks"] == row["actual_masks"] == 1 for row in rows)
    assert all(row["strict_matches"] == 0 and row["minimum_iou"] is None for row in rows)
    assert all(row["stem"] == "empty;other" for row in rows)
    with pytest.raises(ValueError, match="every replay"):
        replay_evidence(runs, references, set())
    with pytest.raises(ValueError, match="unexpected image"):
        replay_evidence(runs, references, {"empty"})
