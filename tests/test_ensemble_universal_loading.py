"""EnsembleModel(include_universal=True): a registered universal (standalone)
model loads as an ensemble of one, but only by explicit label — the default
ensemble (mean, Evaluate, gate fits, production) never sees it.

``Model`` is stubbed so nothing loads TensorFlow checkpoints."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from euclid_polish import ensemble as ens
from euclid_polish import ensemble_registry as er
from euclid_polish.ensemble import EnsembleModel


class _FakeModel:
    def __init__(self, directory, **_kwargs):
        self.directory = directory
        self.id = None

    def upsample_array(self, arr):
        return np.asarray(arr, np.float32)


def _mk(base, i, *, starless=False, checkpoint=True):
    d = os.path.join(base, f"member_{i:02d}")
    os.makedirs(d, exist_ok=True)
    if checkpoint:
        with open(os.path.join(d, "checkpoint"), "w") as f:
            f.write("x")
    with open(os.path.join(d, "origin.json"), "w") as f:
        json.dump({"starless": starless, "asinh_knees": [0.1, 1, 10], "output_knee": 10.0}, f)
    return d


@pytest.fixture
def base(tmp_path, monkeypatch):
    monkeypatch.setattr(ens, "Model", _FakeModel)
    b = str(tmp_path / "ensemble")
    _mk(b, 1)
    _mk(b, 2)
    er.load_registry(b)                       # 01, 02 bootstrap as ensemble members
    for i in (211, 212):                      # registered BEFORE their dirs land
        er.add_universal_entry(b, f"member_{i}")
        _mk(b, i)
    er.add_universal_entry(b, "member_213")
    _mk(b, 213, checkpoint=False)             # registered, still mid-pull
    return b


def _names(e):
    return [os.path.basename(m.directory) for m in e.members]


def test_a_universal_model_loads_by_explicit_label(base):
    e = EnsembleModel(base, labels=["211·psnr"], include_universal=True)
    assert e.member_labels == ["211·psnr"]
    assert _names(e) == ["member_211"]


def test_universal_and_active_labels_mix_in_the_given_order(base):
    e = EnsembleModel(base, labels=["212", "member_01", "211·psnr"],
                      include_universal=True, starless=False)
    assert e.member_labels == ["212·psnr", "01·psnr", "211·psnr"]
    assert _names(e) == ["member_212", "member_01", "member_211"]


def test_include_universal_needs_explicit_labels(base):
    with pytest.raises(ValueError, match="include_universal needs explicit labels"):
        EnsembleModel(base, include_universal=True)
    with pytest.raises(ValueError, match="include_universal needs explicit labels"):
        EnsembleModel(base, include_universal=True, starless=False)


def test_the_default_ensemble_never_sees_universal_models(base):
    assert EnsembleModel(base).member_labels == ["01·psnr", "02·psnr"]
    assert EnsembleModel(base, starless=False).member_labels == ["01·psnr", "02·psnr"]
    assert er.regime_labels(base, False) == ["01·psnr", "02·psnr"]
    with pytest.raises(ValueError, match="is not an active ensemble member with a checkpoint"):
        EnsembleModel(base, labels=["211·psnr"])


def test_unknown_or_unloadable_labels_name_both_classes(base):
    msg = "is not an active ensemble member or a universal model with a checkpoint"
    with pytest.raises(ValueError, match=f"member_09 {msg}"):
        EnsembleModel(base, labels=["09"], include_universal=True)
    with pytest.raises(ValueError, match=f"member_213 {msg}"):  # no checkpoint yet
        EnsembleModel(base, labels=["213"], include_universal=True)
    with pytest.raises(ValueError, match="twice"):
        EnsembleModel(base, labels=["211", "member_211"], include_universal=True)


def test_regime_filter_applies_to_universal_models(base):
    er.add_universal_entry(base, "member_214")
    _mk(base, 214, starless=True)
    with pytest.raises(ValueError, match="member_214 is a starless member"):
        EnsembleModel(base, labels=["214"], include_universal=True, starless=False)
    assert EnsembleModel(base, labels=["214"], include_universal=True).member_labels == ["214·psnr"]


def test_an_archived_universal_model_no_longer_loads(base):
    er.archive_member_entry(base, "member_212", zip_path="models/u.zip", commit=None)
    with pytest.raises(ValueError, match="member_212"):
        EnsembleModel(base, labels=["212"], include_universal=True)


def test_ensemble_available_ignores_universal_models(tmp_path, monkeypatch):
    monkeypatch.setattr(ens, "Model", _FakeModel)
    b = str(tmp_path / "ensemble")
    er.add_universal_entry(b, "member_211")
    _mk(b, 211)
    assert ens.ensemble_available(b) is False
    assert EnsembleModel(b, labels=["211"], include_universal=True).n_members == 1
