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
