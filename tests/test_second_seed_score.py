"""The seed-study audit must use the host scorer, not the simplified candidate-gain identity."""

import csv
import sys

import numpy as np
import pytest

from filseg.metric import encode
from tools.experiments.second_seed.score import main, read_terms, selection_terms, summary_rows


def test_seed_study_preserves_multiple_strict_hits():
    first = np.array([[1, 1, 0]], dtype=np.uint8)
    second = np.array([[0, 1, 1]], dtype=np.uint8)
    tree = dict(parent=[-1], children=[[]], level=[0], rle=[encode(first | second)], g=[0])
    result = selection_terms(tree, [.7], {"drawing": [encode(first), encode(second)]}, .25)
    assert result == pytest.approx((4 / 3, 2, 0, 0))
    with pytest.raises(ValueError, match="aligned"):
        selection_terms(tree, [], {"drawing": []}, .25)


def test_seed_contrast_averages_two_pooled_single_seed_scores():
    """The reference is the mean of two PQs, not the PQ of their combined terms."""
    terms = np.zeros((1, 3, 4, 4))
    terms[0, :, :, 1] = 1
    terms[0, 0, :, 2] = 4
    terms[0, :, :, 0] = np.array([.6, .8, .7])[:, None]
    for row in summary_rows(["19700101000000Ah"], terms):
        assert [row[k] for k in ("delta_vs_single_avg", "lo", "hi")] == pytest.approx([.2] * 3)



def test_public_terms_replay_and_missing_pair_detection(tmp_path, monkeypatch):
    """The public CSV alone must reproduce the paired result, regardless of row order."""
    rows = [dict(stem=f"1970010{fold + 1}000000Ah", fold=fold, recipe=name, penalty=penalty,
                 sum_iou=iou, tp=1, fp=4 if name == "s0" else 0, fn=0)
            for fold in range(4) for name, iou in (("s0", .6), ("s1", .8), ("mean2", .7))
            for penalty in (.21, .23, .25, .27)]
    source, summary = tmp_path / "terms.csv", tmp_path / "summary.csv"

    def write(values):
        with source.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(values)
    write(rows[::-1])
    monkeypatch.setattr(sys, "argv", ["score", "--terms", str(source), "--summary", str(summary)])
    main()
    with summary.open(newline="") as stream:
        values = list(csv.DictReader(stream))
    assert len(values) == 4
    assert all(float(row["delta_vs_single_avg"]) == pytest.approx(.2) for row in values)
    with (tmp_path / "summary_folds.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 48
    write(rows[:-1])
    with pytest.raises(ValueError, match="complete paired"):
        read_terms(source)
