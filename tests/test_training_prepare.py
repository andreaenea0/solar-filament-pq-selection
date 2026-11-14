"""Verify the three-verifiers preparation interfaces without training models."""

import numpy as np
import pytest

from filseg.maps import read_map
from training.prepare import mean_gains, write_maps, write_targets


def test_test_maps_round_before_duplicating(tmp_path):
    sources = [tmp_path / "seed0", tmp_path / "seed1"]
    for source, pixels in zip(sources, ([[0, 1, 253]], [[1, 2, 255]])):
        source.mkdir()
        np.savez(source / "image.npz", p=np.array(pixels, dtype=np.uint8))
    out = tmp_path / "maps"
    write_maps(sources, out)
    np.testing.assert_array_equal(read_map(out / "image.npz"), [[[0, 2, 254]], [[0, 2, 254]]])


def test_oof_maps_reject_stems_from_other_folds(tmp_path):
    sources = [tmp_path / name for name in ("k06", "k09", "k11")]
    for source in sources:
        source.mkdir()
    folds = {str(i): i for i in range(5)}
    for source, stems in zip(sources, ((0,), (1, 2), (3, 4))):
        for stem in stems:
            np.savez(source / f"{stem}.npz", p=np.array([[stem]], dtype=np.uint8))
    write_maps(sources, tmp_path / "ok", folds)
    np.testing.assert_array_equal(read_map(tmp_path / "ok" / "3.npz"), [[[3]], [[3]]])
    (sources[0] / "0.npz").rename(sources[1] / "0.npz")
    with pytest.raises(ValueError, match="OOF map stems"):
        write_maps(sources, tmp_path / "bad", folds)


def test_targets_follow_candidate_rows(tmp_path):
    crops, targets = tmp_path / "crops.npz", tmp_path / "targets.npz"
    candidates = {
        "second": {"rle": [None], "g": [0.4], "R": 1},
        "first": {"rle": [None, None], "g": [0.2, 0.8], "R": 3},
    }
    folds = {"first": 2, "second": 0}
    stems = ["first", "first", "second"]
    np.savez(crops, stem=stems, idx=[0, 1, 0])
    write_targets(candidates, crops, folds, targets)
    with np.load(targets) as archive:
        np.testing.assert_array_equal(archive["g"], [0.2, 0.8, 0.4])
        np.testing.assert_array_equal(archive["w"], [3, 3, 1])
        np.testing.assert_array_equal(archive["fold"], [2, 2, 0])
    np.savez(crops, stem=stems, idx=[1, 0, 0])
    with pytest.raises(ValueError, match="alignment for first"):
        write_targets(candidates, crops, folds, targets)


def test_seed_average_rejects_misaligned_lengths():
    predictions = [{"image": np.array([i, i + 1], dtype=np.float32)} for i in range(3)]
    np.testing.assert_array_equal(mean_gains(predictions)["image"], [1, 2])
    predictions[1]["image"] = np.array([1])
    with pytest.raises(ValueError, match="gains for image"):
        mean_gains(predictions)
