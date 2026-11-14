"""Protect import safety, original quantization and output-cache separation."""

import importlib
from pathlib import Path
import numpy as np


def test_study_modules_import_without_environment_or_data(monkeypatch,tmp_path):
    for key in ("FILSEG_WORK","FILSEG_CACHE","FILSEG_TEST_IMAGES"):
        monkeypatch.delenv(key,raising=False)
    monkeypatch.chdir(tmp_path)
    for name in ("mk_maps","eval_lgb","build"):
        module=importlib.import_module(f"tools.experiments.second_seed.{name}")
        assert callable(module.main)
    assert not list(tmp_path.iterdir())


def test_mean_maps_preserve_original_rounding_and_both_channels(tmp_path):
    from tools.experiments.second_seed.mk_maps import build_maps
    first=tmp_path/"s0";second=tmp_path/"s1";first.mkdir();second.mkdir()
    a=np.array([[0,1,2,3],[255,254,253,252]],dtype=np.uint8)
    b=np.array([[1,2,3,4],[254,253,252,251]],dtype=np.uint8)
    np.savez_compressed(first/"frame.npz",p=a);np.savez_compressed(second/"frame.npz",p=b)
    assert build_maps(first,[second],tmp_path/"out")==1
    with np.load(tmp_path/"out/maps_mean2/frame.npz") as data:
        expected=np.array([[0,2,2,4],[254,254,252,252]],dtype=np.uint8)
        np.testing.assert_array_equal(data["p"],np.stack((expected,expected)))
    with np.load(tmp_path/"out/maps_s1/frame.npz") as data:
        np.testing.assert_array_equal(data["p"],np.stack((b,b)))


def test_verifier_cache_cannot_overwrite_submission_or_rewrite_parent():
    from tools.experiments.second_seed.build import verifier_path
    for output in (Path("run"),Path("inputs.csv/run.csv")):
        cache=verifier_path(output)
        assert cache!=output and cache.parent==output.parent and cache.name.endswith("_verif.pkl")
