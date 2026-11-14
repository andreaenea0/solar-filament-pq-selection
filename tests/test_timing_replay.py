"""Timing must preserve calls, count work once, and never match across images."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from filseg.rle import mask_to_counts
from tools import timing_replay as timing


def test_nested_exclusive_time_and_exception():
    ticks = iter((0., 1., 3., 6.))
    timer = timing.StageTimer(clock=lambda: next(ticks))
    with pytest.raises(RuntimeError):
        with timer.measure("outer"):
            with timer.measure("inner"):
                raise RuntimeError("stop")
    assert timer.seconds == {"inner": 2., "outer": 4.}
    assert not timer.stack
    assert sum(timer.seconds.values()) == 6.


def test_wrapped_call_and_gpu_sync():
    events = []
    timer = timing.StageTimer(synchronize=lambda: events.append("sync"))
    wrapped = timer.wrap("gpu", lambda x, *, offset: x + offset, gpu=True)
    assert wrapped(3, offset=4) == 7
    assert events == ["sync", "sync"]
    assert timer.calls["gpu"] == 1


def test_instrument_restores_hooks_on_failure():
    def identity(*args, **kwargs):
        return args[0] if args else None

    def module(*names):
        return SimpleNamespace(**dict.fromkeys(names, identity))

    inference = module("load_model", "predict_map", "read_jpeg", "normalize")
    candidates = module("_feats")
    oof = module("build_tree", "candidate_record")
    select = module("feature_frame", "learned_masks", "dp_select")
    verifier = module("load_verifier", "candidate_crops", "verifier_gains")
    class Booster:
        def __init__(self):
            pass
        predict = identity
    lgb = SimpleNamespace(Booster=Booster)
    submit = module("write_submission")
    timer = timing.StageTimer()
    with pytest.raises(RuntimeError):
        with timing.instrument(timer, inference, candidates, oof, select, verifier, lgb, submit):
            model = object()
            assert inference.load_model(model) is model
            assert inference.predict_map(model) is model
            assert timer.calls["unet_seed0_d4"] == 1
            raise RuntimeError("stop")
    assert inference.load_model is identity
    assert oof.build_tree is identity
    assert select.dp_select is identity
    assert submit.write_submission is identity


@pytest.fixture
def counts():
    mask = np.zeros((2048, 2048), dtype=np.uint8)
    mask[10:20, 10:20] = 1
    return mask_to_counts(mask)


def test_geometry_matches_renumbered_ids(counts):
    result = timing.replay_summary({"a_1": counts}, {"a_2": counts}, {"a"})
    assert result["shared_ids"] == 0
    assert result["strict_matches"] == 1
    assert result["mean_matched_iou"] == result["prediction_pq_agreement"] == 1.


def test_identical_pixels_in_other_image_cannot_match(counts):
    result = timing.replay_summary({"a_1": counts}, {"b_1": counts}, {"a", "b"})
    assert result["strict_matches"] == 0
    assert result["unmatched_actual"] == result["unmatched_reference"] == 1
    assert result["minimum_iou"] is None
    assert result["prediction_pq_agreement"] == 0.


def test_read_prediction_schema_and_duplicate_guard(tmp_path, counts):
    path = tmp_path / "predictions.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerows((("filament_id", "segmentation_rle"), ("a_1", counts), ("b_1", counts)))
    assert timing.read_rows(path, {"a"}, subset=True) == {"a_1": counts}
    with pytest.raises(ValueError, match="unexpected image"):
        timing.read_rows(path, {"a"})
    with path.open("a", newline="") as stream:
        csv.writer(stream).writerow(("a_1", counts))
    with pytest.raises(ValueError, match="duplicate"):
        timing.read_rows(path, {"a", "b"})


def test_peak_memory_is_available():
    assert timing.peak_cpu_mib() > 0


def test_notebook_code_compiles_and_uses_public_entrypoint():
    path = Path(__file__).resolve().parents[1] / "kernels/release/submitted_timing.ipynb"
    notebook = json.loads(path.read_text())
    code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    compile(code, str(path), "exec")
    assert "tools.timing_replay" in code
    assert '"180"' in code
    assert 'CUDA_VISIBLE_DEVICES' in code
