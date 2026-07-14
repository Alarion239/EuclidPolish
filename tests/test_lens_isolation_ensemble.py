from __future__ import annotations

import os

import numpy as np
import pytest

from euclid_polish.experiments.lens_isolation.ensemble import (
    LensIsolationEnsemble,
    detection_score,
)


class FakeModel:
    def __init__(self, path):
        self.value = int(os.path.basename(path).split("_")[-1]) + 1

    def upsample_array(self, lr):
        return np.ones((4, 4, 1), np.float32) * self.value


def test_explicit_discovery_mean_and_disagreement(tmp_path):
    for name in ("member_00", "member_01"):
        path = tmp_path / name
        path.mkdir()
        (path / "checkpoint").write_text("x")
    (tmp_path / "not_a_member").mkdir()
    ensemble = LensIsolationEnsemble(str(tmp_path), model_factory=FakeModel)
    mean, std = ensemble.predict(np.zeros((2, 2, 1), np.float32))
    assert len(ensemble.members) == 2
    assert np.all(mean == 1.5)
    assert np.all(std == 0.5)
    assert ensemble.member_labels == ["00·psnr", "01·psnr"]
    assert ensemble.n_members == 2


def test_evaluate_delegates_to_production_metric_path(tmp_path, monkeypatch):
    for name in ("member_00", "member_01"):
        path = tmp_path / name
        path.mkdir()
        (path / "checkpoint").write_text("x")
    captured = {}

    class FakeProductionEnsemble:
        def __init__(self, base_dir, _models):
            captured["base_dir"] = base_dir
            captured["models"] = _models

        def evaluate(self, lr, hr, **kwargs):
            captured.update(lr=lr, hr=hr, labels=self._member_labels, kwargs=kwargs)
            return {"ensemble_psnr": 12.5}

    monkeypatch.setattr("euclid_polish.ensemble.EnsembleModel", FakeProductionEnsemble)
    ensemble = LensIsolationEnsemble(str(tmp_path), model_factory=FakeModel)
    result = ensemble.evaluate("dirty", "lens", on_field="field", on_progress="progress")

    assert result == {"ensemble_psnr": 12.5}
    assert captured["labels"] == ["00·psnr", "01·psnr"]
    assert captured["lr"] == "dirty"
    assert captured["hr"] == "lens"
    assert captured["kwargs"] == {"on_field": "field", "on_progress": "progress"}


def test_detection_score_clips_negative_flux_and_supports_aperture():
    image = np.zeros((5, 5, 1), np.float32)
    image[0, 0] = 10
    image[2, 2] = 2
    image[1, 1] = -100
    assert detection_score(image) == 12
    assert detection_score(image, aperture=1) == 2


def test_empty_ensemble_fails_loudly(tmp_path):
    with pytest.raises(RuntimeError, match="no lens-isolation members"):
        LensIsolationEnsemble(str(tmp_path), model_factory=FakeModel)
