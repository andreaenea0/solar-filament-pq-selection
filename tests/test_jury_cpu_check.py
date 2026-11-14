"""Check row and geometry parity independently of the team checkpoints."""

import numpy as np
import pytest

from filseg.rle import mask_to_counts
from tools.jury_cpu_check import compare, select_images


def test_compare_detects_id_reordering():
    mask = np.zeros((2048, 2048), dtype=bool)
    mask[2, 3] = True
    first = mask_to_counts(mask)
    second = mask_to_counts(mask.T)
    summary = compare({"x_1": first, "x_2": second}, {"x_1": second, "x_2": first})
    assert "2 shared IDs, 0 byte-identical, 2 strict IoU matches" in summary
    assert "min matched IoU 1.000000" in summary


def test_explicit_replay_supports_one_image_and_rejects_ambiguous_inputs(tmp_path):
    for name in ("a.jpeg", "b.JPG", "c.jpeg", "d.jpeg", "ignored.txt"):
        (tmp_path / name).touch()
    assert [p.stem for p in select_images(tmp_path)] == ["a", "c", "d"]
    assert [p.stem for p in select_images(tmp_path, ["b"])] == ["b"]
    with pytest.raises(ValueError, match="not found"):
        select_images(tmp_path, ["missing"])
    with pytest.raises(ValueError, match="distinct"):
        select_images(tmp_path, ["a", "a"])
    (tmp_path / "a.jpg").touch()
    with pytest.raises(ValueError, match="unique"):
        select_images(tmp_path, ["b"])


def test_default_replay_requires_three_images(tmp_path):
    (tmp_path / "one.jpeg").touch()
    with pytest.raises(ValueError, match="--stems"):
        select_images(tmp_path)
    assert select_images(tmp_path, ["one"]) == [tmp_path / "one.jpeg"]


def test_compare_cannot_match_same_coordinates_in_another_observation():
    mask = np.zeros((2048, 2048), dtype=np.uint8)
    mask[20:30, 20:30] = 1
    counts = mask_to_counts(mask)
    result = compare({"first_1": counts}, {"second_1": counts})
    assert "0 strict IoU matches, min matched IoU n/a" in result
