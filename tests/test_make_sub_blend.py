"""The released training CLI must import safely and reject bad verifier evidence early."""
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from training import make_sub_blend


def test_import_without_training_arguments():
    """Documentation and test discovery can import the trainer without opening input files."""
    subprocess.run([sys.executable, "-c", "import training.make_sub_blend"],
                   cwd=Path(__file__).resolve().parents[1], check=True)


@pytest.mark.parametrize("scores", [{}, {"image": np.array([])}, {"image": np.array([np.nan])}])
def test_bad_verifier_rejected_before_fitting(tmp_path, monkeypatch, scores):
    """Missing, misaligned or nonfinite verifier rows must not produce a fitted model."""
    def unexpected_fit(_):
        pytest.fail("Training started before verifier validation.")

    monkeypatch.setattr(make_sub_blend, "fit_gain_model", unexpected_fit)
    paths = [tmp_path / name for name in ("train.pkl", "test.pkl", "verifier.pkl")]
    for path, value in zip(paths, ({}, {"image": {"rle": [None]}}, scores), strict=True):
        path.write_bytes(pickle.dumps(value))
    output = tmp_path / "submission.csv"
    with pytest.raises(ValueError, match="Verifier"):
        make_sub_blend.main([*map(str, paths), str(output), "0.5", "0.25"])
    assert not output.exists()
    assert not (tmp_path / "submission_lgb.txt").exists()
