"""load_eval_ensemble(): always an EnsembleModel; clear error when empty."""
from __future__ import annotations

import numpy as np
import pytest

from euclid_polish.eval import ensemble_infer as ei


class _FakeEns:
    def __init__(self, n):
        self.n_members = n


def test_returns_ensemble(monkeypatch):
    monkeypatch.setattr(ei, "load_ensemble", lambda *a, **kw: _FakeEns(3))
    logged = []
    out = ei.load_eval_ensemble(log=logged.append)
    assert out.n_members == 3
    assert any("3 models" in m for m in logged)


def test_zero_members_raises(monkeypatch):
    monkeypatch.setattr(ei, "load_ensemble", lambda *a, **kw: _FakeEns(0))
    with pytest.raises(RuntimeError, match="no active ensemble members"):
        ei.load_eval_ensemble(log=lambda m: None)


def test_sr_from_model_hides_members_for_singleton():
    class _One:
        n_members = 1

        def member_arrays(self, lr):
            return np.zeros((1, 4, 4, 1), np.float32)

    _lr_vis, sr, members = ei.sr_from_model(_One(), np.zeros((2, 2, 1)))
    assert members is None                      # single member → no std cubes
    assert sr.shape == (4, 4, 1)
