"""Check the public ensemble receipt and map-channel recipe."""

import json

import numpy as np
import pytest

pytest.importorskip("torch")
from filseg.public_maps import public_checkpoints, public_map


def test_receipt_requires_five_distinct_folds_per_family(tmp_path):
    models = []
    for encoder in ("resnet34", "tu-tf_efficientnetv2_s.in21k_ft_in1k"):
        for fold in reversed(range(5)):
            file = tmp_path / f"{encoder}-{fold}.pt"
            file.touch()
            models.append(dict(encoder=encoder, fold=fold, file=file.name))
    receipt = tmp_path / "solar_unet_receipt.json"
    receipt.write_text(json.dumps({"ensemble": {"models": models}}), encoding="utf-8")
    families = public_checkpoints(receipt)
    assert [path.stem.split("-")[-1] for path, _ in families[0]] == list("01234")
    assert len(families[1]) == 5
    models[0]["fold"] = 3
    receipt.write_text(json.dumps({"ensemble": {"models": models}}), encoding="utf-8")
    with pytest.raises(ValueError, match="repeated fold"):
        public_checkpoints(receipt)
    models[0]["fold"] = 4
    models[0]["file"] = models[1]["file"]
    receipt.write_text(json.dumps({"ensemble": {"models": models}}), encoding="utf-8")
    with pytest.raises(ValueError, match="Repeated public checkpoint"):
        public_checkpoints(receipt)


@pytest.mark.parametrize(("models", "expected"), [
    ([[0, 1, 1, 1, 0], [2, 3, 2, 2, 2]], 2),
    ([[0, 0, 1, 0, 1], [1, 2, 1, 1, 1]], 0),
    ([[255] * 5, [254] * 5], 254),
])
def test_public_map_rounds_each_family_then_half_to_even(monkeypatch, models, expected):
    def predict(model, pixels, device, batch, *, channels_last):
        assert channels_last
        return np.full(pixels.shape, model, dtype=np.uint8)

    monkeypatch.setattr("filseg.public_maps.predict_map", predict)
    probability = public_map(models, np.zeros((2, 3)), "cpu", 1)
    np.testing.assert_array_equal(probability, np.full((2, 3), expected, dtype=np.uint8))
