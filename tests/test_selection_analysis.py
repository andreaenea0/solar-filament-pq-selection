"""Portable selection analyses must preserve host scoring and expose failed fixed points."""

from pathlib import Path
import csv
import pickle
import subprocess
import sys

import numpy as np
import pytest

from filseg.metric import encode, score_image
from tools.experiments.selection.calib import fixed_point
from tools.experiments.selection.common import build_terms, evaluate, load_inputs, pq


def test_cached_contributions_preserve_multiple_strict_hits():
    """The historical gain identity undercounts two overlapping GT masks hit by one prediction."""
    truth = [encode(np.array([[1, 1, 0, 0, 0]])), encode(np.array([[0, 1, 1, 0, 0]]))]
    masks = [encode(np.array([[1, 1, 1, 0, 0]])), encode(np.array([[0, 0, 0, 1, 1]]))]
    tree = dict(parent=[-1, -1], children=[[], []], level=[0, 0], rle=masks, g=[2 / 3, 0], R=1)
    trees, gt, scores = {"s": tree}, {"s": {"drawing": truth}}, {"s": np.array([.8, .6])}
    for penalty, selected in ((.1, masks), (.7, masks[:1]), (1., [])):
        values, counts = evaluate(trees, scores, build_terms(trees, gt), ["s"], penalty)
        assert pq(values.sum(0)) == pytest.approx(score_image([truth], selected).pq)
        assert counts.sum() == len(selected)
    approximate, _ = evaluate(trees, scores, build_terms(trees, gt, "one-hit"), ["s"], .1)
    assert pq(approximate.sum(0)) != pytest.approx(score_image([truth], masks).pq)


def test_fixed_point_checks_convergence_and_cycles():
    """Learned-score iteration is empirical; a cycle must not be labelled converged."""
    result = fixed_point(lambda penalty: .8)
    assert result["converged"] and result["penalty"] == pytest.approx(.4)
    cycle = fixed_point(lambda penalty: 1 if penalty < .4 else .2, max_iterations=6)
    assert not cycle["converged"] and cycle["iterations"] == 6
    assert abs(cycle["residual"]) == pytest.approx(.4)


def test_score_precision_survives_input_validation(tmp_path):
    """Casting archived float32 scores changes isotonic near-tie handling."""
    paths = [tmp_path / name for name in ("trees.pkl", "scores.pkl", "gt.pkl")]
    scores = np.array([.25, np.nextafter(np.float32(.25), np.float32(1))], dtype=np.float32)
    for path, value in zip(paths, ({"s": dict(rle=[None, None], R=1)}, {"s": scores}, {"s": {"drawing": []}})):
        with path.open("wb") as stream:
            pickle.dump(value, stream)
    _, loaded, _ = load_inputs(*paths)
    assert loaded["s"].dtype == np.float32
    np.testing.assert_array_equal(loaded["s"], scores)


@pytest.mark.parametrize("module", ("calib", "calib2", "sweep", "lambda_sweep_finals"))
def test_ported_cli_help_does_not_read_private_files(module):
    """Importing a ported command must not load a machine-specific cache or launch work."""
    result = subprocess.run([sys.executable, "-m", f"tools.experiments.selection.{module}", "--help"],
                            cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True, text=True)
    assert "--gt" in result.stdout and "--out" in result.stdout


@pytest.mark.parametrize("module", ("calib", "calib2", "sweep", "lambda_sweep_finals"))
def test_commands_run_with_explicit_inputs(tmp_path, module):
    """Run each complete CLI on five tiny held-out observations, including path parsing."""
    if module.startswith("calib"):
        pytest.importorskip("sklearn")
    mask = encode(np.ones((1, 3)))
    stems = [f"2000010{i + 1}000000Ah" for i in range(5)]
    trees = {s: dict(parent=[-1], children=[[]], level=[0], rle=[mask], g=[1.], R=1) for s in stems}
    gt = {s: {"drawing": [mask]} for s in stems}
    paths = [tmp_path / name for name in ("trees.pkl", "scores.pkl", "gt.pkl")]
    for path, value in zip(paths, (trees, {s: np.array([.25]) for s in stems}, gt)):
        with path.open("wb") as stream:
            pickle.dump(value, stream)
    folds, out = tmp_path / "folds.csv", tmp_path / "out.csv"
    folds.write_text("stem,fold\n" + "".join(f"{s},{i}\n" for i, s in enumerate(stems)))
    args = ["--gt", str(paths[2]), "--out", str(out)]
    if module == "lambda_sweep_finals":
        args += ["--model", "demo", str(paths[0]), str(paths[1]), "--bootstrap", "20"]
    else:
        args += ["--name", "demo", "--candidates", str(paths[0]), "--scores", str(paths[1])]
        args += ["--folds", str(folds), "--score-mode", "host"] if module == "calib" else ["--bootstrap", "20"]
    subprocess.run([sys.executable, "-m", f"tools.experiments.selection.{module}", *args],
                   cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True, text=True)
    with out.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    if module == "calib":
        assert float(next(row["pq"] for row in rows if row["method"] == "tuned")) == 1


def test_report_routes_each_recipe_to_its_own_current_curve(tmp_path, monkeypatch):
    """Exercise the report CLI so the two model panels cannot silently reuse the final's curve."""
    pytest.importorskip("matplotlib")
    from report import make_figures

    data = tmp_path / "data"
    data.mkdir()
    names = ("public-ensemble", "three-verifiers", "submitted")
    (data / "lambda_sweep_finals.csv").write_text("model,lambda,pq\n" +
            "".join(f"{name},0.25,{.46 - i / 100}\n" for i, name in enumerate(names)))
    for name in names:
        (data / name).mkdir()
    routed, primary = [], []
    monkeypatch.setattr(make_figures, "fig_distributions", lambda iou, dice, sweep, out:
                        routed.append(tuple(sweep["model"].unique())))
    monkeypatch.setattr(make_figures, "fig_lambda", lambda sweep, out: primary.append(tuple(sweep["model"].unique())))
    monkeypatch.setattr(sys, "argv", ["figures", "--data", str(data), "--out", str(tmp_path / "out")])
    make_figures.main()
    assert routed == [("submitted",), ("public-ensemble",), ("three-verifiers",), ("submitted",)]
    assert primary == [("submitted",)]
