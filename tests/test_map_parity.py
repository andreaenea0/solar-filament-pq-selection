"""Check paired map comparisons with known masks and both input formats."""

import csv
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

from filseg.metric import encode
from filseg.rle import write_submission


def test_paired_map_cli_tracks_crossings_and_pools_pq(tmp_path):
    first, blank = "20260101000000Bh", "20260102000000Bh"
    a = np.zeros((16, 16), dtype=np.uint8)
    a[1:5, 1:5] = 255
    a[10:14, 10:14] = 252
    b = a.copy()
    b[1:5, 1:5] = 250
    b[10:14, 10:14] = 255
    # An unmatched low-threshold component in the full average only.
    b[1:4, 12:15] = 255
    empty = np.zeros_like(a)
    gt = {first: {"r1": [encode(a == 255)]}, blank: {"r2": []}}
    (tmp_path / "gt.pkl").write_bytes(pickle.dumps(gt))
    for name, image in (("a", a), ("b", b), ("empty", empty)):
        np.save(tmp_path / f"{name}.npy", image)
    manifest = {first: [["a.npy", "b.npy"]], blank: [["empty.npy", "empty.npy"]]}
    (tmp_path / "members.json").write_text(json.dumps(manifest))

    for dirname, values in (("matched", a), ("full", np.mean([a, b], axis=0).round().astype(np.uint8))):
        directory = tmp_path / dirname
        directory.mkdir()
        np.savez(directory / f"{first}.npz", p=values)
        np.savez(directory / f"{blank}.npz", p=empty)

    script = Path(__file__).resolve().parents[1] / "tools" / "cv_map_parity.py"
    command = [sys.executable, str(script), "gt.pkl", "--min-area", "4", "--out", "pair.csv"]
    sources = (["--members", "members.json", "--counts", "1"], ["--matched", "matched", "--full", "full"])
    for source in sources:
        run = subprocess.run(command + source, cwd=tmp_path, capture_output=True, text=True, check=True)
        assert "2 paired images. PQ matched 1.0000, full 0.0000, delta -1.0000" in run.stdout
        assert "paired day-bootstrap interval for delta [-1.0000, +0.0000]" in run.stdout
        with (tmp_path / "pair.csv").open(newline="") as stream:
            event, zero = list(csv.DictReader(stream))
        assert (event["matched_components"], event["full_components"], event["shared_components"]) == ("2", "3", "2")
        assert (event["peak_pass_to_fail"], event["peak_fail_to_pass"]) == ("1", "1")
        assert (event["matched_masks"], event["full_masks"], event["shared_masks"]) == ("1", "1", "0")
        assert zero["stem"] == blank and zero["matched_components"] == "0"

    run = subprocess.run(command + sources[0] + ["--unrounded"], cwd=tmp_path, capture_output=True, text=True, check=True)
    assert "PQ matched 1.0000, full 0.6667, delta -0.3333" in run.stdout
    assert "Peak pass to fail / fail to pass: 0 1" in run.stdout

    for name in ("selected_a.csv", "selected_b.csv"):
        write_submission(tmp_path / name, [(first, gt[first]["r1"])])
    run = subprocess.run(command + sources[1] + ["--submissions", "selected_a.csv", "selected_b.csv"],
                         cwd=tmp_path, capture_output=True, text=True, check=True)
    assert "PQ matched 1.0000, full 1.0000, delta +0.0000" in run.stdout
    assert "Selected masks matched/full/shared: 1 1 1" in run.stdout

    (tmp_path / "full" / f"{blank}.npz").unlink()
    run = subprocess.run(command + sources[1], cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 2 and blank in run.stderr
