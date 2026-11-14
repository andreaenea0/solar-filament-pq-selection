"""Training must keep each annotated frame at index zero or fail before training."""

import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

pytest.importorskip("segmentation_models_pytorch")

RECIPES = (
    "kernels/k03-train/train.py", "kernels/k05-recipe/train.py",
    "kernels/k06-dice/train.py", "kernels/k07-final/final.py",
    "training/k09-oof12/train.py", "training/k11-oof34/train.py",
)


@pytest.fixture(params=RECIPES)
def recipe(request, tmp_path, monkeypatch):
    """Import each historical recipe without its guarded GPU launcher."""
    source = Path(__file__).resolve().parents[1] / request.param
    incoming = tmp_path / "input"
    incoming.mkdir()
    monkeypatch.setenv("FILSEG_INPUT", str(incoming))
    monkeypatch.setenv("FILSEG_TMP", str(tmp_path / "prepared"))
    spec = importlib.util.spec_from_file_location("recipe_under_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "INPUT", str(incoming))
    monkeypatch.setattr(module, "TMP", str(tmp_path / "prepared"))
    monkeypatch.setattr(module, "H", 16)
    monkeypatch.setattr(module, "W", 16)
    monkeypatch.setattr(module, "DISK", np.ones((16, 16), dtype=bool))
    stem = "20000101000000Ah"
    images = incoming / "train_images"
    images.mkdir()
    neighbors = incoming / "neighbors" / stem
    neighbors.mkdir(parents=True)
    Image.fromarray(np.full((16, 16), 200, dtype=np.uint8)).save(neighbors / "+2_valid.jpg")
    (neighbors / "-2_bad.jpg").write_bytes(b"not a JPEG")
    data = dict(images=[dict(id="drawing", file_name=f"{stem}.jpeg")],
                annotations=[dict(image_id="drawing", segmentation=[[2, 2, 12, 2, 12, 12, 2, 12]])])
    (incoming / "MAGFiLO_1.0_Annotations_kaggle2026_train.json").write_text(json.dumps(data))
    (incoming / "folds.csv").write_text(f"stem,fold\n{stem},0\n")
    return module, images / f"{stem}.jpeg"


def test_original_stays_first_when_an_optional_neighbor_is_corrupt(recipe):
    """The non-temporal control must use the annotated pixels, including its normalization."""
    module, original = recipe
    Image.fromarray(np.arange(256, dtype=np.uint8).reshape(16, 16)).save(original)
    module.prepare()
    prepared = Path(module.TMP)
    meta = json.loads((prepared / "meta.json").read_text())
    assert meta["index"] == [[0, 1]]
    assert meta["n"] == 2
    expected = module.load(original)
    np.testing.assert_array_equal(np.load(prepared / "img.npy")[0], expected)
    stats = np.load(prepared / "stats.npy")
    assert stats[0, 0] == pytest.approx((expected.astype(np.float32) / 255).mean())


@pytest.mark.parametrize("failure", ("missing", "wrong_shape"))
def test_bad_original_cannot_be_replaced_by_a_valid_neighbor(recipe, failure):
    """Missing or malformed annotated pixels must stop, rather than become a mislabeled neighbor."""
    module, original = recipe
    if failure == "wrong_shape":
        Image.fromarray(np.zeros((8, 16), dtype=np.uint8)).save(original)
    with pytest.raises(RuntimeError, match="annotated frame"):
        module.prepare()
    assert not (Path(module.TMP) / "meta.json").exists()



def test_k07_original_only_preparation_preserves_non_temporal_samples(recipe, tmp_path, monkeypatch):
    """Unused neighbours must not affect annotated pixels, targets, crops or RNG."""
    import inspect
    module, original = recipe
    if "include_neighbors" not in inspect.signature(module.prepare).parameters:
        pytest.skip("single-seed option belongs to k07")
    Image.fromarray(np.arange(256, dtype=np.uint8).reshape(16, 16)).save(original)
    monkeypatch.setattr(module, "CROP", 8)
    module.prepare()
    prepared = Path(module.TMP)

    def read(folder):
        return (np.load(folder/"img.npy"), np.load(folder/"tgt.npy"),
                np.load(folder/"stats.npy"), json.loads((folder/"meta.json").read_text()))

    full = read(prepared)
    original_only = tmp_path / "original-only"
    monkeypatch.setattr(module, "TMP", str(original_only))
    for path in Path(module.INPUT).rglob("+2_*.jpg"):
        path.unlink()
    module.prepare(include_neighbors=False)
    single = read(original_only)
    assert full[3]["n"] == 2
    assert single[3]["n"] == 1
    for index in range(3):
        np.testing.assert_array_equal(full[index][0], single[index][0])
    for seed in range(20):
        rng_full, rng_single = np.random.default_rng(seed), np.random.default_rng(seed)
        expected = module.sample_crop(*full, [0], rng_full, False)
        actual = module.sample_crop(*single, [0], rng_single, False)
        for a, b in zip(expected, actual):
            np.testing.assert_array_equal(a, b)
        assert rng_full.bit_generator.state == rng_single.bit_generator.state
