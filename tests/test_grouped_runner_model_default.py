"""load_eval_ensemble(): the STARFULL ensemble + the production combiner.

Evaluators (grouped run, catalog runner, the records' "Generate SR") used to
average every registry-active member of BOTH star regimes. They now load only
STARFULL members and reconstruct through the production combiner
(``ACTIVE_COMBINER_KINDS[0]``, the spatial gate), falling back to the plain
mean when no current combiner loads for that membership.
"""
from __future__ import annotations

import numpy as np
import pytest

from euclid_polish.eval import ensemble_infer as ei
from euclid_polish.eval.combiner import ACTIVE_COMBINER_KINDS, COMBINER_MODELS
from euclid_polish.image import Image


class _FakeEns:
    def __init__(self, n, stack=None):
        self.n_members = n
        self.member_labels = [f"{i:02d}·psnr" for i in range(n)]
        self._stack = stack

    def member_arrays(self, lr):
        return self._stack


class _Gate:
    use_lr = True

    def __init__(self):
        self.calls = []

    def apply_field(self, members, lr=None):
        self.calls.append((members.shape, None if lr is None else lr.shape))
        return np.full(members.shape[1:], 7.0, np.float32)


@pytest.fixture
def built(monkeypatch):
    seen = {}

    def fake_ensemble(base_dir, **kwargs):
        seen.update(base_dir=base_dir, **kwargs)
        return seen.get("ensemble") or _FakeEns(3)

    monkeypatch.setattr(ei, "EnsembleModel", fake_ensemble)
    return seen


def test_loads_only_starfull_members(built, monkeypatch):
    monkeypatch.setattr(ei, "load_combiner", lambda *a, **k: None)
    logged = []
    out = ei.load_eval_ensemble(log=logged.append)
    assert built["starless"] is False
    assert out.n_members == 3
    assert out.combiner is None
    assert any("3 STARFULL models" in m and "mean" in m for m in logged)


def test_zero_members_raises(built, monkeypatch):
    built["ensemble"] = _FakeEns(0)
    monkeypatch.setattr(ei, "load_combiner", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="no active STARFULL ensemble members"):
        ei.load_eval_ensemble(log=lambda m: None)


def test_production_combiner_is_loaded_for_the_membership(built, monkeypatch, tmp_path):
    gate = _Gate()
    requests = []

    def fake_load(base, *, member_labels, artifact_dir):
        requests.append((base, list(member_labels), artifact_dir))
        return gate

    monkeypatch.setattr(ei, "load_combiner", fake_load)
    out = ei.load_eval_ensemble(combiner_dir=str(tmp_path), log=lambda m: None)
    production = ACTIVE_COMBINER_KINDS[0]
    assert requests == [(str(tmp_path), ["00·psnr", "01·psnr", "02·psnr"],
                         COMBINER_MODELS[production].artifact_dir)]
    assert out.combiner is gate and out.combiner_kind == production


def test_default_combiner_dir_is_the_starfull_regime(monkeypatch, tmp_path):
    monkeypatch.setattr(ei.Config, "VIS_DIR", str(tmp_path / "vis"))
    assert ei.starfull_regime_dir() == str((tmp_path / "vis" / "ensemble" / "starfull").resolve())


def test_sr_from_model_uses_the_combiner_with_the_lr_input():
    stack = np.stack([np.full((4, 4, 4), v, np.float32) for v in (1.0, 3.0)])
    gate = _Gate()
    model = ei.EvalEnsemble(_FakeEns(2, stack), gate, ACTIVE_COMBINER_KINDS[0])
    lr = np.zeros((2, 2, 4), np.float32)
    lr_vis, sr, members = ei.sr_from_model(model, lr)
    assert np.allclose(sr, 7.0)
    assert gate.calls == [((2, 4, 4, 4), (2, 2, 4))]
    assert members.shape == (2, 4, 4, 4) and lr_vis.shape == (2, 2)


def test_sr_from_model_falls_back_to_the_mean():
    stack = np.stack([np.full((4, 4, 4), v, np.float32) for v in (1.0, 3.0)])
    model = ei.EvalEnsemble(_FakeEns(2, stack), None, None)
    _lr_vis, sr, _members = ei.sr_from_model(model, np.zeros((2, 2, 4), np.float32))
    assert np.allclose(sr, 2.0)


def test_upsample_batch_uses_the_combined_prediction():
    stack = np.stack([np.full((4, 4, 4), v, np.float32) for v in (1.0, 3.0)])
    model = ei.EvalEnsemble(_FakeEns(2, stack), _Gate(), ACTIVE_COMBINER_KINDS[0])
    lr = Image(data=np.zeros((2, 2, 4), np.float32), pixel_scale_arcsec=0.1,
               band_names=("VIS", "Y_E", "J_E", "H_E"), is_clean=False, index=5)
    progress = []
    (sr,) = model.upsample_batch([lr], on_progress=lambda *a: progress.append(a))
    assert np.allclose(sr.data, 7.0)
    assert progress == [(1, 1, "field 5")]


def test_sr_from_model_hides_members_for_singleton():
    class _One:
        n_members = 1

        def member_arrays(self, lr):
            return np.zeros((1, 4, 4, 1), np.float32)

    _lr_vis, sr, members = ei.sr_from_model(_One(), np.zeros((2, 2, 1)))
    assert members is None                      # single member → no std cubes
    assert sr.shape == (4, 4, 1)
