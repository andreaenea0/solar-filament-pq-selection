"""A single-seed diagnostic must preserve sampling and the full selection path."""
import json
from pathlib import Path

import numpy as np
import pytest

from tools.compare_retrained_seed import cached_submission, map_metrics


def test_probability_errors_are_on_probability_scale():
    released = np.zeros((2048, 2048), dtype=np.uint8)
    retrained = np.full_like(released, 255)
    result = map_metrics(released, retrained)
    assert result == dict(mae=1., rmse=1., max_absolute_error=1., exact_pixel_fraction=0.)
    assert map_metrics(released, released)["exact_pixel_fraction"] == 1.
    with pytest.raises(ValueError, match="uint8"):
        map_metrics(released, retrained.astype(np.float32))
    with pytest.raises(ValueError, match="2048"):
        map_metrics(released, retrained[:10])


def test_cached_replay_supplies_seed_maps_in_original_entrypoint(tmp_path, monkeypatch):
    from filseg import inference
    import submit
    maps = [tmp_path / str(seed) for seed in range(2)]
    for seed, directory in enumerate(maps):
        directory.mkdir()
        np.savez(directory / "observation.npz", p=np.full((2048, 2048), seed, dtype=np.uint8))
    original_load, original_predict = inference.load_model, inference.predict_map
    monkeypatch.setattr(inference, "read_jpeg", lambda path: path.stem)

    def entrypoint():
        assert inference.read_jpeg(Path("observation.jpeg")) == "observation"
        for seed in range(2):
            checkpoint = tmp_path / f"effv2s_s{seed}_fold-1.pt"
            model = inference.load_model(checkpoint, "cpu")
            probability = inference.predict_map(model, None, "cpu", 1)
            assert np.all(probability == seed)
        raise RuntimeError("restore hooks")

    monkeypatch.setattr(submit, "main", entrypoint)
    with pytest.raises(RuntimeError, match="restore hooks"):
        cached_submission(tmp_path, maps, tmp_path, tmp_path/"out.csv", "cpu", 1)
    assert inference.load_model is original_load
    assert inference.predict_map is original_predict


def test_training_notebook_is_prepare_only_and_compiles():
    path = Path(__file__).resolve().parents[1] / "kernels/release/retrain_seed0.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    compile(code, str(path), "exec")
    assert "training.retrain_seed0" in code
    assert "tools.compare_retrained_seed" in code
    assert "CUDA_VISIBLE_DEVICES" in code
    assert all(not cell.get("outputs") for cell in notebook["cells"])
