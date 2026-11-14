"""Verify paired term alignment and raw fold coverage for the one-verifier public ensemble."""

import pickle

import pandas as pd
import pytest

from tools.experiments.single_verifier import merge_exports, summarize


def test_repeated_export_images_fail(tmp_path):
    """A repeated fold cannot silently replace the first held-out prediction."""
    paths = [tmp_path / f"{i}.pkl" for i in range(2)]
    for path in paths:
        path.write_bytes(pickle.dumps({"same": []}))
    with pytest.raises(ValueError, match="repeated"):
        merge_exports(paths)


def test_pairing_survives_row_shuffle_and_reports_known_fold_delta():
    """Two perfect matches versus one of two drawn instances has delta -1/3."""
    rows = [dict(stem=f"2026010{i + 1}000000Ah", fold=i, recipe=recipe, numerator=n, denominator=d)
            for i in range(2) for recipe, n, d in (("public-ensemble", 2, 2), ("public-one-verifier", 1, 1.5))]
    result = summarize(pd.DataFrame(rows).sample(frac=1, random_state=4))
    assert result.delta.to_numpy() == pytest.approx([-1/3] * 3)
    assert result.iloc[0][["lo", "hi"]].to_numpy(dtype=float) == pytest.approx([-1/3] * 2)
