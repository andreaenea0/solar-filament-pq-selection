"""Inference features must not depend on the file name (host rule, Kaggle discussion 753665)."""
import numpy as np
from pathlib import Path
import pytest

from filseg import select
from filseg.oof import candidate_record
from filseg.select import feature_frame


def test_no_site_or_year_feature_names():
    assert not {"site", "year"} & set(select.FEATS + select.EXTRA)


def test_features_do_not_depend_on_stem():
    p = np.zeros((2048, 2048), np.uint8)
    p[500:540, 600:800] = 200
    p[505:535, 620:780] = 250
    image = np.random.default_rng(0).integers(0, 255, (2048, 2048)).astype(np.uint8)
    maps = np.stack([p, p])
    a = feature_frame(candidate_record("20110109104734Ch", maps, image))
    b = feature_frame(candidate_record("20220101000000Uh", maps, image))
    assert list(a.columns) == list(b.columns)
    assert np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True)


def test_released_boosters_have_the_image_only_feature_schema():
    lgb = pytest.importorskip("lightgbm")
    root = Path(__file__).resolve().parents[1]
    for name in ("own_ver3_a05_l025_nometa_lgb.txt", "public_ensemble/lgb_pubown_nometa.txt"):
        booster = lgb.Booster(model_file=str(root / "weights" / name))
        assert booster.feature_name() == select.FEATS + select.EXTRA
