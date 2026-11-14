"""Public scoring CLIs must work when workers start with fresh module state."""

import csv
import json
import multiprocessing
import sys

import pytest

from filseg.data import gt_rles_by_image, load_records
from filseg.rle import write_submission
from tools.figures import rubric_table
from tools.experiments import rubric_paired


@pytest.mark.parametrize("paired", (False, True))
def test_public_scoring_cli_works_with_spawn(tmp_path, monkeypatch, paired):
    """Exercise actual child processes, file loading and complete CLI output."""
    stem = "20000101000000Ah"
    record = f"010101-{stem}"
    source = tmp_path / "training.json"
    source.write_text(json.dumps(dict(
        images=[dict(id=record, file_name=f"{stem}.jpeg")],
        annotations=[dict(image_id=record, segmentation=[polygon]) for polygon in
                     ([10, 10, 20, 10, 20, 20, 10, 20], [30, 30, 40, 30, 40, 40, 30, 40])],
    )))
    masks = gt_rles_by_image(load_records(source))[stem][record]
    reference, alternative, out = (tmp_path / name for name in ("ref.csv", "alt.csv", "out.csv"))
    write_submission(reference, [(stem, masks)])
    write_submission(alternative, [(stem, masks[:1])])
    named = [f"ref={reference}", f"alt={alternative}"]
    module = rubric_paired if paired else rubric_table
    monkeypatch.setattr(module, "Pool", multiprocessing.get_context("spawn").Pool)
    rubric_table._SHARED.clear()
    args = ([str(source), str(out), *named] if paired else
            ["--gt", str(source), "--out", str(out), "--pred", *named, "--bootstrap", "20"])
    monkeypatch.setattr(sys, "argv", ["score", *args, "--workers", "1"])
    module.main()
    with out.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    row = next(row for row in rows if row["model"] == "alt" and row["metric"] == "pq")
    expected, key = (-1 / 3, "delta") if paired else (2 / 3, "value")
    assert [float(row[k]) for k in (key, "lo", "hi")] == pytest.approx([expected] * 3)
