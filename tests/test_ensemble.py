"""EnsembleModel: mean prediction, per-pixel disagreement, and evaluation.

Members are stubbed (``upsample_array`` injected) so the maths is pinned without
training real checkpoints.
"""

from __future__ import annotations

import numpy as np

from euclid_polish.ensemble import EnsembleModel, member_dir
from euclid_polish.image import Image


class _Stub:
    """A stand-in member exposing the bits EnsembleModel uses."""

    def __init__(self, fn, mid=None):
        self._fn = fn
        self.id = mid

    def upsample_array(self, arr):
        return np.asarray(self._fn(arr), np.float32)


def _img(arr, index=0):
    return Image(data=np.asarray(arr, np.float32), pixel_scale_arcsec=0.05,
                 band_names=("VIS",), is_clean=False, index=index)


def test_member_dir_naming():
    assert member_dir("/base", 3).endswith("member_03")


def test_predict_mean_is_average_and_std_is_disagreement():
    base = np.ones((8, 8, 1), np.float32)
    # Members differ by a per-member constant → known mean (0 offset) + spread.
    ens = EnsembleModel("x", _models=[
        _Stub(lambda a, c=c: a + c) for c in (-1.0, 0.0, 1.0)])
    mean, std = ens.predict(base)
    assert ens.n_members == 3
    assert np.allclose(mean, base)                       # mean of {-1,0,+1} = 0
    assert np.allclose(std, np.std([-1.0, 0.0, 1.0]))    # disagreement = spread
    assert np.allclose(ens.disagreement(base), std)


def test_single_member_has_zero_disagreement():
    ens = EnsembleModel("x", _models=[_Stub(lambda a: a)])
    _mean, std = ens.predict(np.ones((4, 4, 1), np.float32))
    assert np.allclose(std, 0.0)


def test_evaluate_psnr_disagreement_and_hallucination():
    rng = np.random.default_rng(0)
    hr_arr = (rng.random((8, 8, 1)) * 100.0).astype(np.float32)
    lr, hr = _img(hr_arr, index=0), _img(hr_arr, index=0)
    # Each member ≈ HR + distinct zero-mean noise → finite PSNR, real disagreement.
    perturb = [rng.normal(0, 2.0, hr_arr.shape).astype(np.float32) for _ in range(3)]
    ens = EnsembleModel("x", _models=[_Stub(lambda a, p=p: hr_arr + p)
                                      for p in perturb])

    out = ens.evaluate([lr], [hr])
    assert out["n_members"] == 3 and out["n_fields"] == 1 and out["n_scored"] == 1
    assert len(out["per_member_psnr"]) == 3
    assert np.isfinite(out["ensemble_psnr"])
    # Averaging members cancels zero-mean noise ⇒ ensemble ≥ the mean member.
    assert out["ensemble_psnr"] >= out["mean_member_psnr"] - 1e-6
    d = out["disagreement"]
    assert d["mean_std_e"] > 0.0
    assert 0.0 <= d["frac_flux_hallucinated"] <= 1.0


def test_evaluate_without_hr_still_reports_disagreement():
    # No HR → no PSNR, but the hallucination cross-check still runs.
    ens = EnsembleModel("x", _models=[
        _Stub(lambda a, c=c: a + c) for c in (0.0, 5.0)])
    out = ens.evaluate([_img(np.ones((4, 4, 1), np.float32))])
    assert out["n_scored"] == 0
    assert out["per_member_psnr"] == []
    assert out["disagreement"]["mean_std_e"] > 0.0


def test_predict_images_shapes_and_roles():
    from euclid_polish.image import Role
    ens = EnsembleModel("x", _models=[_Stub(lambda a: a), _Stub(lambda a: a + 1)])
    lr = _img(np.ones((6, 6, 1), np.float32))
    mean_img, std_img = ens.predict_images(lr)
    assert mean_img.role is Role.SR
    assert mean_img.data.shape == (6, 6, 1) and std_img.data.shape == (6, 6, 1)
