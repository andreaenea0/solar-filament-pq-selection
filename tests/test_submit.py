"""Check inference alignment and the jury CLI against the cached-map workflow."""

import csv
import json
import sys

import numpy as np
import pytest
from PIL import Image

from filseg.rle import counts_to_mask


def test_d4_predictions_invert_every_transform():
    torch = pytest.importorskip("torch")
    from filseg.inference import predict_map

    x = np.arange(64, dtype=np.float32).reshape(8, 8) / 20 - 2
    expected = (torch.sigmoid(torch.from_numpy(x)).numpy() * 255).round().astype(np.uint8)
    for batch in (1, 4):
        np.testing.assert_array_equal(predict_map(torch.nn.Identity(), x, torch.device("cpu"), batch), expected)


def test_submit_rule_and_finalist_selectors(tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    lgb = pytest.importorskip("lightgbm")
    import submit
    from filseg import inference

    stem = "20260101000000Bh"
    images = tmp_path / "test"
    images.mkdir()
    Image.fromarray(np.full((2048, 2048), 127, dtype=np.uint8)).save(images / f"{stem}.jpeg")
    p0, p1 = (np.zeros((2048, 2048), dtype=np.uint8) for _ in range(2))
    p0[10:25, 10:25] = 255
    p1[10:25, 10:25] = 254
    p0[30:45, 30:45] = 255
    p1[30:45, 30:45] = 249

    def predict(model, normalized, device, batch_size):
        assert normalized.dtype == np.float32
        np.testing.assert_allclose(normalized, 0, atol=1e-3)
        return p0 if model.name == "seed0.pt" else p1

    monkeypatch.setattr(inference, "load_model", lambda path, device: path)
    monkeypatch.setattr(inference, "predict_map", predict)
    out = tmp_path / "submission.csv"
    command = ["submit.py", "--images", str(images), "--weights", "seed0.pt", "seed1.pt",
               "--device", "cpu", "--out", str(out)]

    def run(extra, unets=("seed0.pt", "seed1.pt")):
        weights_at = command.index("--weights") + 1
        monkeypatch.setattr(sys, "argv", command[:weights_at] + list(unets) + command[weights_at + 2:] + extra)
        submit.main()
        with out.open(newline="") as stream:
            return list(csv.DictReader(stream))

    baseline = run([])
    assert [row["filament_id"] for row in baseline] == [f"{stem}_1"]
    np.testing.assert_array_equal(counts_to_mask(baseline[0]["segmentation_rle"]), p1 == 254)

    data = lgb.Dataset(np.arange(6, dtype=float).reshape(-1, 1), label=np.ones(6), feature_name=["level"])
    model = lgb.train({"verbosity": -1, "num_threads": 1}, data, num_boost_round=1)
    weights = tmp_path / "selector.txt"
    model.save_model(str(weights))
    learned = run(["--selector-model", str(weights)])
    assert [row["filament_id"] for row in learned] == [f"{stem}_1", f"{stem}_2"]
    expected = [p1 == 254, p1 == 249]
    for row, mask in zip(learned, expected):
        np.testing.assert_array_equal(counts_to_mask(row["segmentation_rle"]), mask)

    # Seed mean = -0.5. The blend with LGB = 1 lands exactly on the
    # penalty, so neither candidate should be selected.
    from filseg import public_maps, select

    columns = select.FEATS + select.EXTRA
    values = np.tile(np.arange(6, dtype=float)[:, None], (1, len(columns)))
    data = lgb.Dataset(values, label=np.ones(6), feature_name=columns)
    model = lgb.train({"verbosity": -1, "num_threads": 1}, data, num_boost_round=1)
    final_weights = tmp_path / "finalist_selector.txt"
    model.save_model(str(final_weights))
    select_masks = select.learned_masks

    def check_final_schema(candidates, booster, penalty, verifier=None, **options):
        frame = select.feature_frame(candidates)
        assert list(frame.columns) == booster.feature_name()
        if options.get("alpha") == .35:
            assert (frame["m0"] > frame["m1"]).all()
        else:
            for name in ("m0", "m1", "min_fam"):
                np.testing.assert_array_equal(frame[name], frame["meanp"])
            np.testing.assert_array_equal(frame["dfam"], 0)
            np.testing.assert_array_equal(frame["hi0"], frame["hi1"])
        return select_masks(candidates, booster, penalty, verifier, **options)

    monkeypatch.setattr(select, "learned_masks", check_final_schema)
    torch = pytest.importorskip("torch")
    timm = pytest.importorskip("timm")
    verifier = timm.create_model("resnet18.a1_in1k", pretrained=False, in_chans=4, num_classes=1)
    with torch.no_grad():
        for param in verifier.parameters():
            param.zero_()
        checkpoints = []
        for seed, bias in enumerate((-0.75, -0.5, -0.25)):
            verifier.fc.bias.fill_(bias)
            checkpoint = tmp_path / f"verifier_{seed}.pt"
            torch.save(verifier.state_dict(), checkpoint)
            checkpoints.append(str(checkpoint))
    trace = tmp_path / "trace.json"
    assert run(["--final", "three-verifiers", "--selector", str(final_weights), "--verifier-weights", *checkpoints,
                "--trace", str(trace)]) == []
    row = json.loads(trace.read_text(encoding="utf-8"))[stem]
    assert row["candidates"] > 0 and row["blended_max"] == row["penalty"] == 0.25
    assert row["above_penalty"] == row["selected"] == 0

    monkeypatch.setattr(public_maps, "load_public_models", lambda receipt, device: receipt)
    monkeypatch.setattr(public_maps, "public_map", lambda models, normalized, device, batch: p0)
    public_ensemble_args = ["--final", "public-ensemble", "--public-receipt", "receipt.json",
                    "--selector", str(final_weights), "--trace", str(trace),
                    "--verifier-weights", *checkpoints]
    assert run(public_ensemble_args) == []
    row = json.loads(trace.read_text(encoding="utf-8"))[stem]
    assert row["penalty"] == .27 and row["blended_max"] == pytest.approx(.025)
    with pytest.raises(SystemExit) as error:
        run(public_ensemble_args + ["legacy-effnet.pt"])
    assert error.value.code == 2
    assert "Recipe public-ensemble requires three verifier weights." in capsys.readouterr().err

    lite_args = ["--final", "submitted", "--selector", str(final_weights), "--trace", str(trace)]
    assert run(lite_args + ["--verifier-weights", checkpoints[0]]) == []
    row = json.loads(trace.read_text(encoding="utf-8"))[stem]
    assert row["penalty"] == .25
    with pytest.raises(SystemExit) as error:
        run(lite_args + ["--verifier-weights", *checkpoints])
    assert error.value.code == 2
    assert "Recipe submitted requires one verifier weight." in capsys.readouterr().err
    assert run(["--selector", str(final_weights), "--verifier-weights", checkpoints[0]]) == []  # default submitted

    single_args = ["--final", "single-unet", "--selector", str(final_weights), "--trace", str(trace),
                   "--verifier-weights", checkpoints[0]]
    assert run(single_args, unets=("seed0.pt",)) == []
    row = json.loads(trace.read_text(encoding="utf-8"))[stem]
    assert row["penalty"] == .25 and len(row["own_seed_max_probabilities"]) == 1
    with pytest.raises(SystemExit) as error:
        run(single_args)
    assert error.value.code == 2
    assert "Recipe single-unet requires one own U-Net weight and penalty 0.25." in capsys.readouterr().err


def test_verifier_crop_channels_at_edge():
    pytest.importorskip("torch")
    from filseg.metric import encode
    from filseg.verifier import own_crops

    image = np.full((32, 32), 17, dtype=np.uint8)
    probability = np.full_like(image, 29)
    mask = np.zeros_like(image)
    mask[:4, :4] = 1
    crop, = own_crops(image, probability, [encode(mask)])
    assert crop.shape == (4, 96, 96)
    assert np.all(crop[0] == 17) and np.all(crop[1] == 29)
    np.testing.assert_array_equal(crop[1], crop[2])
    assert crop[3, 0, 0] == 255 and crop[3, -1, -1] == 0


def test_verifier_averages_raw_d4_scores_and_seeds():
    torch = pytest.importorskip("torch")
    from filseg.verifier import verifier_gains

    crops = np.zeros((2, 4, 4, 4), dtype=np.uint8)
    crops[0, 0, 0, 0] = 255
    crops[1, 0, -1, -1] = 255
    crops[1, 0, -1, 0] = 255

    def model(bias):
        return lambda x: x[:, 0, 0, 0, None] + bias

    gains = verifier_gains([model(-1), model(0), model(4)], crops, torch.device("cpu"))
    np.testing.assert_array_equal(gains, [1.25, 1.5])
