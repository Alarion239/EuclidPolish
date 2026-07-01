from __future__ import annotations

import euclid_polish.eval.ensemble_infer as ei


def test_loader_falls_back_to_single_when_no_ensemble(monkeypatch):
    calls = {}

    class _Empty:
        n_members = 0

    # load_ensemble is imported function-locally from euclid_polish.ensemble,
    # so patch the true source symbol there.
    monkeypatch.setattr("euclid_polish.ensemble.load_ensemble", lambda **k: _Empty())
    def _fake_single(c, n):
        calls["single"] = (c, n)
        return "SINGLE_MODEL"

    monkeypatch.setattr(
        "euclid_polish.eval.catalog_runner.load_eval_model",
        _fake_single)
    out = ei.load_eval_ensemble_or_single("ckpt/x", 8, log=lambda m: None)
    assert out == "SINGLE_MODEL"
    assert "single" in calls


def test_loader_uses_ensemble_when_present(monkeypatch):
    class _Ens:
        n_members = 5

    monkeypatch.setattr("euclid_polish.ensemble.load_ensemble", lambda **k: _Ens())
    out = ei.load_eval_ensemble_or_single(log=lambda m: None)
    assert isinstance(out, _Ens)


def test_loader_falls_back_on_exception(monkeypatch):
    def _boom(**k):
        raise RuntimeError("no ensemble dir")

    monkeypatch.setattr("euclid_polish.ensemble.load_ensemble", _boom)
    monkeypatch.setattr(
        "euclid_polish.eval.catalog_runner.load_eval_model",
        lambda c, n: "SINGLE_MODEL")
    out = ei.load_eval_ensemble_or_single(log=lambda m: None)
    assert out == "SINGLE_MODEL"
