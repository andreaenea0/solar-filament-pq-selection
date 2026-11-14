"""Protocol checks must detect shared days, cross-site proximity and renamed pixel duplicates."""

import pytest

from filseg.data import Record
from tools.experiments.protocol_audit import pixel_duplicates, split_summary, timestamp_summary


def test_image_folds_do_not_imply_day_folds():
    first, second = "20200101000000Ah", "20200101010000Ah"
    records = {stem: [Record(f"01-x-{stem}", stem)] for stem in (first, second)}
    result = split_summary(records, {first: 0, second: 1})
    assert result["days_across_folds"] == {"20200101": [0, 1]}
    assert result["images_on_cross_fold_days"] == 2
    with pytest.raises(ValueError, match="match exactly"):
        split_summary(records, {first: 0})


def test_site_proximity_and_proposed_neighbour_times_are_separate():
    result = timestamp_summary(["20200101000000Ah"], ["20200101000200Ah", "20200101000000Bh"])
    assert result["all_sites"]["within_minutes"]["2"] == 2
    assert result["same_site"]["eligible_test_images"] == 1
    assert result["same_site"]["minimum_minutes"] == 2
    assert result["test_exposures_equal_proposed_training_neighbour"] == 1
    assert pixel_duplicates({"old_name": "digest"}, {"new_name": "digest"}) == {"new_name": ["old_name"]}
