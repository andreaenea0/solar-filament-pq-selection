"""Check pooled paired statistics and reject unmatched control cohorts."""
import pytest

from tools.experiments.submitted_rule_ablation import summarize


def paired_rows():
    """A small cohort where pooled PQ differs materially from mean image PQ."""
    rows = []
    for fold, stem, terms in (
        (0, "20200101000000Ch", ((.6, 1, 0, 0), (.8, 1, 0, 0))),
        (1, "20200102000000Ch", ((.6, 1, 0, 9), (1.2, 2, 0, 8))),
    ):
        for recipe, values in zip(("fixed-own-map-rule", "submitted"), terms):
            rows.append(dict(stem=stem, fold=fold, recipe=recipe, records=1,
                             **dict(zip(("sum_iou", "tp", "fp", "fn"), values))))
    return rows


def test_pool_before_computing_pq():
    summary, folds = summarize(paired_rows())
    assert {row["block_days"] for row in summary} == {1, 7, 28}
    for row in summary:
        assert row["rule_pq"] == pytest.approx(2 * 1.2 / 13)
        assert row["submitted_pq"] == pytest.approx(2 * 2 / 14)
        assert row["delta"] == pytest.approx(2 * 2 / 14 - 2 * 1.2 / 13)
        assert row["positive_folds"] == 2
        assert row["lo"] <= row["delta"] <= row["hi"]
    assert all(row["delta"] > 0 for row in folds)


@pytest.mark.parametrize("change", ["missing", "duplicate", "fold", "gt", "nan"])
def test_reject_unpaired_or_invalid_evidence(change):
    rows = paired_rows()
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(rows[0].copy())
    elif change == "fold":
        rows[1]["fold"] = 2
    elif change == "gt":
        rows[1]["fn"] = 1
    else:
        rows[1]["sum_iou"] = float("nan")
    with pytest.raises(ValueError):
        summarize(rows)
