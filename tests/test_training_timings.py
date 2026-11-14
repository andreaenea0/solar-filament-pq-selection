"""Training timing evidence must not conflate a final epoch with full export."""

import pytest

from tools.figures.training_timings import timing_rows


def test_pixel_worker_timers_exclude_shared_preparation():
    rows = timing_rows("k06", [dict(data="prep 438 s\n"), dict(data="[temporal] done 6520s\n"),
                               dict(data="[control] done 6923s\n")])
    assert [row["seconds"] for row in rows] == [6520, 6923]
    assert all(row["preparation_seconds"] == 438 and row["scope"] == "training_and_map_export" for row in rows)


def test_verifier_timers_ignore_earlier_epochs_and_retain_seed_ids():
    events = [dict(data="[-1 resnet18.a1_in1k 0] epoch 10 loss 0.1 110s\n")]
    events += [dict(data=f"[-1 resnet18.a1_in1k {seed}] epoch 11 loss 0.1 {seconds}s\n")
               for seed, seconds in ((2, 131), (0, 138), (1, 139))]
    rows = timing_rows("k17", events)
    assert [row["seconds"] for row in rows] == [138, 139, 131]
    assert all(row["scope"] == "through_final_training_epoch" for row in rows)
    with pytest.raises(ValueError, match="each verifier seed"):
        timing_rows("k17", events[:-1])
