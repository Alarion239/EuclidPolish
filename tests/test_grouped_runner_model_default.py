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

LABELS = ["00·psnr", "01·psnr", "02·psnr"]


class _FakeEns:
    def __init__(self, n, stack=None, labels=None):
        self.member_labels = list(labels) if labels is not None else [
            f"{i:02d}·psnr" for i in range(n)]
        self.n_members = len(self.member_labels)
        self._stack = stack

    def member_arrays(self, lr):
        return self._stack


class _Gate:
    use_lr = True

    def __init__(self, member_labels=None, active=None):
        self.calls = []
        if member_labels is not None:
            self.member_labels = list(member_labels)
            self._active = list(range(len(member_labels))) if active is None else list(active)

    def needed_member_indices(self):
        return list(self._active)

    def apply_field(self, members, lr=None):
        self.calls.append((members.shape, None if lr is None else lr.shape))
        return np.full(members.shape[1:], 7.0, np.float32)


@pytest.fixture
def built(monkeypatch):
    """A fake registry (``seen["active"]``, never the real one) and a fake
    EnsembleModel that records how it was built and restores exactly the
    ``labels=`` it is given (every active member without them)."""
    seen = {"active": list(LABELS), "builds": []}

    def fake_ensemble(base_dir, **kwargs):
        seen.update(base_dir=base_dir, **kwargs)
        seen["builds"].append(dict(kwargs))
        if seen.get("ensemble") is not None:
            return seen["ensemble"]
        return _FakeEns(0, labels=kwargs.get("labels") or seen["active"])

    monkeypatch.setattr(ei, "EnsembleModel", fake_ensemble)
    monkeypatch.setattr(ei, "regime_labels", lambda _base, starless: (
        [] if starless else list(seen["active"])))
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
    built["active"] = []
    monkeypatch.setattr(ei, "load_combiner", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="no active STARFULL ensemble members"):
        ei.load_eval_ensemble(log=lambda m: None)


def test_production_combiner_is_loaded_for_the_membership(built, monkeypatch, tmp_path):
    gate = _Gate(LABELS)
    requests = []

    def fake_load(base, *, available_labels, artifact_dir):
        requests.append((base, list(available_labels), artifact_dir))
        return gate

    monkeypatch.setattr(ei, "load_combiner", fake_load)
    out = ei.load_eval_ensemble(combiner_dir=str(tmp_path), log=lambda m: None)
    production = ACTIVE_COMBINER_KINDS[0]
    assert requests == [(str(tmp_path), LABELS,
                         COMBINER_MODELS[production].artifact_dir)]
    assert out.combiner is gate and out.combiner_kind == production
    assert built["labels"] == LABELS                 # exactly the members it reads


def test_pruned_gate_restores_and_runs_only_the_members_it_reads(built, monkeypatch):
    """A gate reading members 00 and 02 of the 3 it was fitted for restores
    only those two checkpoints; the identity keeps the full fitted list."""
    gate = _Gate(LABELS, active=[0, 2])
    monkeypatch.setattr(ei, "load_combiner", lambda *a, **k: gate)
    logged = []
    out = ei.load_eval_ensemble(log=logged.append)
    assert built["builds"] == [{"num_res_blocks": built["num_res_blocks"],
                                "starless": False, "labels": ["00·psnr", "02·psnr"]}]
    assert out.member_labels == LABELS and out.n_members == 3
    assert out.run_labels == ["00·psnr", "02·psnr"] and out.n_run == 2
    assert "2 of 3" in out.label
    assert any("2 of 3" in m for m in logged)


def test_members_that_joined_after_the_fit_are_a_note_not_a_fallback(built, monkeypatch):
    built["active"] = [*LABELS, "03·psnr"]           # a member registered since the fit
    gate = _Gate(LABELS, active=[1])
    monkeypatch.setattr(ei, "load_combiner", lambda *a, **k: gate)
    logged = []
    out = ei.load_eval_ensemble(log=logged.append)
    assert out.combiner is gate and out.joined == ["03·psnr"]
    assert out.member_labels == LABELS and out.run_labels == ["01·psnr"]
    assert any("joined after this fit" in m for m in logged)


def test_mean_fallback_runs_every_member_and_warns(built, monkeypatch, caplog):
    monkeypatch.setattr(ei, "load_combiner", lambda *a, **k: None)
    logged = []
    with caplog.at_level("WARNING", logger=ei.__name__):
        out = ei.load_eval_ensemble(log=logged.append)
    assert out.combiner is None and "labels" not in built["builds"][0]
    assert out.run_labels == out.member_labels == LABELS
    assert any(m.startswith("WARNING") and "mean" in m for m in logged)
    assert any("member mean" in r.getMessage() for r in caplog.records)


def test_a_read_member_without_a_checkpoint_falls_back_to_the_mean(built, monkeypatch):
    gate = _Gate(LABELS)
    monkeypatch.setattr(ei, "load_combiner", lambda *a, **k: gate)
    real = ei.EnsembleModel

    def picky(base_dir, **kwargs):
        if kwargs.get("labels"):
            raise ValueError("member_01 is not an active ensemble member with a checkpoint")
        return real(base_dir, **kwargs)

    monkeypatch.setattr(ei, "EnsembleModel", picky)
    logged = []
    out = ei.load_eval_ensemble(log=logged.append)
    assert out.combiner is None and out.n_run == 3
    assert any("member_01" in m and "WARNING" in m for m in logged)


def test_production_plan_resolves_reads_without_networks(monkeypatch):
    gate = _Gate(LABELS, active=[2])
    monkeypatch.setattr(ei, "load_combiner", lambda *a, **k: gate)
    plan = ei.production_plan([*LABELS, "09·psnr"])
    assert plan.member_labels == tuple(LABELS)
    assert plan.run_labels == ("02·psnr",) and plan.joined == ("09·psnr",)
    assert plan.combiner_kind == ACTIVE_COMBINER_KINDS[0]
    monkeypatch.setattr(ei, "load_combiner", lambda *a, **k: None)
    mean = ei.production_plan(LABELS)
    assert mean.run_labels == mean.member_labels == tuple(LABELS)
    assert mean.combiner_kind is None


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


def test_sr_from_model_returns_the_members_that_ran():
    """Disagreement is over the members that ran: a gate that reads one of
    its three members returns no member stack (no all-zero std cubes)."""
    stack = np.full((1, 4, 4, 4), 2.0, np.float32)
    gate = _Gate(LABELS, active=[1])
    model = ei.EvalEnsemble(_FakeEns(0, stack, labels=["01·psnr"]), gate,
                            ACTIVE_COMBINER_KINDS[0], member_labels=LABELS)
    _lr_vis, sr, members = ei.sr_from_model(model, np.zeros((2, 2, 4), np.float32))
    assert members is None and model.n_members == 3 and model.n_run == 1
    assert gate.calls == [((1, 4, 4, 4), (2, 2, 4))]
    assert np.allclose(sr, 7.0)


def test_sr_from_model_hides_members_for_singleton():
    class _One:
        n_members = 1

        def member_arrays(self, lr):
            return np.zeros((1, 4, 4, 1), np.float32)

    _lr_vis, sr, members = ei.sr_from_model(_One(), np.zeros((2, 2, 1)))
    assert members is None                      # single member → no std cubes
    assert sr.shape == (4, 4, 1)
