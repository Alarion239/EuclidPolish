"""EnsembleModel: the explicit member-label subset and the gain definitions.

``Model`` is stubbed so nothing loads TensorFlow checkpoints."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from euclid_polish import ensemble as ens
from euclid_polish.ensemble import EnsembleModel
from euclid_polish.image import Image


class _FakeModel:
    def __init__(self, directory, **_kwargs):
        self.directory = directory
        self.id = None

    def upsample_array(self, arr):
        return np.asarray(arr, np.float32)


def _mk(base, i, *, starless=False):
    d = os.path.join(base, f"member_{i:02d}")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "checkpoint"), "w") as f:
        f.write("x")
    with open(os.path.join(d, "origin.json"), "w") as f:
        json.dump({"starless": starless}, f)


@pytest.fixture
def base(tmp_path, monkeypatch):
    monkeypatch.setattr(ens, "Model", _FakeModel)
    b = str(tmp_path / "ensemble")
    for i in (1, 2, 3):
        _mk(b, i)
    _mk(b, 7, starless=True)
    return b


def test_labels_select_exactly_those_members_in_the_given_order(base):
    e = EnsembleModel(base, labels=["03·psnr", "member_01", "2"])
    assert e.member_labels == ["03·psnr", "01·psnr", "02·psnr"]
    assert [os.path.basename(m.directory) for m in e.members] == [
        "member_03", "member_01", "member_02"]


def test_labels_refuse_inactive_duplicate_or_other_regime_members(base):
    with pytest.raises(ValueError, match="member_09"):
        EnsembleModel(base, labels=["09·psnr"])
    with pytest.raises(ValueError, match="twice"):
        EnsembleModel(base, labels=["01·psnr", "member_01"])
    with pytest.raises(ValueError, match="starless"):
        EnsembleModel(base, labels=["07·psnr"], starless=False)
    # without a regime filter a starless member is a valid explicit pick
    assert EnsembleModel(base, labels=["07·psnr"]).member_labels == ["07·psnr"]


def test_no_labels_keeps_the_registry_order_and_regime_filter(base):
    assert EnsembleModel(base, starless=False).member_labels == [
        "01·psnr", "02·psnr", "03·psnr"]
    assert EnsembleModel(base, starless=False, n_members=2).member_labels == [
        "01·psnr", "02·psnr"]


class _Stub:
    def __init__(self, fn):
        self._fn = fn
        self.id = None

    def upsample_array(self, arr):
        return np.asarray(self._fn(arr), np.float32)


def test_gain_keys_have_one_meaning_each():
    rng = np.random.default_rng(1)
    hr = (rng.random((8, 8, 1)) * 100.0).astype(np.float32)
    noise = [rng.normal(0, s, hr.shape).astype(np.float32) for s in (1.0, 2.0, 4.0)]
    e = EnsembleModel("x", _models=[_Stub(lambda a, p=p: hr + p) for p in noise])
    img = Image(data=hr, pixel_scale_arcsec=0.05, band_names=("VIS",), is_clean=False, index=0)
    out = e.evaluate([img], [img])
    per = out["per_member_psnr"]
    assert out["best_member_psnr"] == pytest.approx(max(per))
    assert out["best_member_label"] == out["per_member_labels"][int(np.argmax(per))]
    assert out["ensemble_vs_mean_member_db"] == pytest.approx(
        out["ensemble_psnr"] - np.mean(per))
    assert out["ensemble_vs_best_member_db"] == pytest.approx(
        out["ensemble_psnr"] - max(per))
    # ensemble_gain_db is the "vs mean member" number everywhere (the eval
    # summary rebuilt from cached cubes uses the same definition)
    assert out["ensemble_gain_db"] == pytest.approx(out["ensemble_vs_mean_member_db"])
