"""Exercise the sparse CV CLI with synthetic masks and the existing pickle formats."""

import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from filseg.metric import encode, score_record
from tools.cv_sparse import dp_record_totals, record_weights


def test_record_weights_preserve_image_weight_and_handle_ties():
    np.testing.assert_array_equal(record_weights([1, 1, 2, 4]), [
        [1, 1, 1, 1], [2, 2, 0, 0], [0, 0, 0, 4],
    ])
    np.testing.assert_array_equal(record_weights([0, 0]), np.ones((3, 2)))
    np.testing.assert_array_equal(record_weights([4]), np.ones((3, 1)))


def test_cli_scores_both_pickle_formats_on_the_same_records(tmp_path):
    a = encode(np.eye(4, dtype=np.uint8))
    b = encode(np.fliplr(np.eye(4, dtype=np.uint8)))
    first, second = "20260101000000Bh", "20260102000000Bh"
    gt = {first: {"sparse": [a], "dense": [a, b]}, second: {"solo": [a]}}
    rule = {record: score_record(masks, [a]) for image in gt.values() for record, masks in image.items()}
    candidates = {
        first: dict(rle=[a, b], parent=[-1, -1], children=[[], []], level=[0, 0]),
        second: dict(rle=[a], parent=[-1], children=[[]], level=[0]),
    }
    predictions = {"oof_pred": {first: [0.8, 0.4], second: [0.8]}}
    for name, data in (("gt", gt), ("rule", rule), ("candidates", candidates), ("oof", predictions)):
        (tmp_path / f"{name}.pkl").write_bytes(pickle.dumps(data))

    script = Path(__file__).resolve().parents[1] / "tools" / "cv_sparse.py"
    cmd = [sys.executable, str(script), "gt.pkl", "--records", "rule", "rule.pkl",
           "--dp", "blend", "candidates.pkl", "oof.pkl", "0.27", "--bootstrap", "50"]
    result = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, check=True)
    lines = [line.split()[:4] for line in result.stdout.splitlines()]
    assert ["rule", "full", "PQ", "0.8571"] in lines
    assert ["rule", "sparse", "PQ", "1.0000"] in lines
    assert ["rule", "dense", "PQ", "0.7500"] in lines
    assert ["blend", "full", "PQ", "0.8889"] in lines
    assert ["blend", "sparse", "PQ", "0.7500"] in lines
    assert ["blend", "dense", "PQ", "1.0000"] in lines
    assert "paired day-bootstrap intervals" in result.stdout
    assert "Shortlist: blend, rule" in result.stdout

    (tmp_path / "rule.pkl").write_bytes(pickle.dumps({"solo": rule["solo"]}))
    result = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 2
    assert "rule is missing record sparse" in result.stderr


def test_dp_rejects_unaligned_scores_and_allows_empty_candidate_lists():
    gt = {"stem": {"record": []}}
    candidates = {"stem": dict(rle=[], parent=[], children=[], level=[])}
    assert dp_record_totals(gt, candidates, {}, 0.27)["record"].pq == 0
    with pytest.raises(ValueError, match="one finite score per candidate"):
        dp_record_totals(gt, candidates, {"stem": [0.9]}, 0.27)
