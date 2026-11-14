"""Check nested fallback evaluation with synthetic, aligned candidate proposals."""

import csv
import json
import sys

from filseg.data import gt_rles_by_image, load_records
from tools.cv_empty_fallback import load_proposals, main


def test_archive_loader_uses_predicted_gain_only(tmp_path):
    path = tmp_path / "top.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("stem", "n_cands", "top_blend_gain", "true_g", "rle_json"))
        writer.writerow(("empty", 0, "", "", ""))
        writer.writerow(("one", 1, 0.23, 0.99, json.dumps(dict(size=[2048, 2048], counts="example"))))
    assert load_proposals(path) == {"empty": None, "one": {"counts": "example", "gain": 0.23}}


def test_aligned_fallback_scores_on_held_out_folds(tmp_path, monkeypatch, capsys):
    stems = [f"2020010{day}000000Ch" for day in range(1, 6)]
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps({
        "images": [{"id": f"01-{stem}", "file_name": f"{stem}.jpeg"} for stem in stems],
        "annotations": [{"image_id": f"01-{stem}", "segmentation": [[20, 20, 40, 20, 40, 40, 20, 40]]}
                        for stem in stems],
    }), encoding="utf-8")
    labels = gt_rles_by_image(load_records(gt))
    proposals = {stem: {"counts": next(iter(records.values()))[0]["counts"].decode("ascii"), "gain": .23}
                 for stem, records in labels.items()}
    proposal_file = tmp_path / "top.json"
    proposal_file.write_text(json.dumps(proposals), encoding="utf-8")
    pred = tmp_path / "frozen.csv"
    pred.write_text("filament_id,segmentation_rle\n", encoding="utf-8")
    folds = tmp_path / "folds.csv"
    folds.write_text("stem,fold\n" + "".join(f"{stem},{i}\n" for i, stem in enumerate(stems)), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["cv_empty_fallback", "--gt", str(gt), "--pred", str(pred),
                                     "--folds", str(folds), "--proposals", str(proposal_file)])

    main()

    output = capsys.readouterr().out
    assert "0 predictions: 5 images, PQ 0.000000" in output
    assert "Nested: delta +1.000000 [+1.000000, +1.000000], 5 additions" in output
