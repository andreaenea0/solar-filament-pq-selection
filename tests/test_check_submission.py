"""Exercise submission format, coverage and mask failures."""

import csv

import numpy as np
import pytest

from filseg.rle import mask_to_counts
from tools.check_submission import check_submission

SHAPE = (8, 10)


def counts(y, x):
    """Encode a horizontal run at an asymmetric position."""
    mask = np.zeros(SHAPE, dtype=bool)
    mask[y, x:x + 3] = True
    return mask_to_counts(mask)


def write_csv(path, rows):
    """Write the host's two columns."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("filament_id", "segmentation_rle"))
        writer.writerows(rows)


def test_full_coverage_with_disjoint_column_major_masks(tmp_path):
    path = tmp_path / "valid.csv"
    write_csv(path, [("a_1", counts(1, 3)), ("a_3", counts(4, 1)), ("b_2", counts(5, 7))])
    assert check_submission(path, {"a", "b"}, SHAPE) == (3, 2)


def test_all_empty_images_can_be_valid_in_host_mode(tmp_path):
    path = tmp_path / "empty.csv"
    write_csv(path, [])
    assert check_submission(path, {"a", "b"}, SHAPE, allow_empty_images=True) == (0, 0)


@pytest.mark.parametrize(("rows", "stems", "message"), [
    ([("a_0", counts(1, 3))], {"a"}, "unexpected ID"),
    ([("x_1", counts(1, 3))], {"a"}, "unexpected ID"),
    ([("a_1", counts(1, 3)), ("a_1", counts(4, 1))], {"a"}, "Duplicate filament ID"),
    ([("a_1", counts(1, 3)), ("a_2", counts(1, 3))], {"a"}, "Duplicate masks"),
    ([("a_1", counts(1, 3)), ("a_2", counts(1, 4))], {"a"}, "Overlapping masks"),
    ([("a_1", counts(1, 3))], {"a", "b"}, "no rows"),
    ([("a_1", mask_to_counts(np.zeros(SHAPE)))], {"a"}, "Empty mask"),
    ([("a_1", "0")], {"a"}, "expected 80"),
    ([("a_1", "P")], {"a"}, "Truncated compressed RLE"),
    ([("a_1", "a" * 7)], {"a"}, "Overlong compressed RLE"),
    ([("a_1", "'")], {"a"}, "Invalid compressed RLE character"),
])
def test_rejects_invalid_submissions(tmp_path, rows, stems, message):
    path = tmp_path / "invalid.csv"
    write_csv(path, rows)
    with pytest.raises(ValueError, match=message):
        check_submission(path, stems, SHAPE)
