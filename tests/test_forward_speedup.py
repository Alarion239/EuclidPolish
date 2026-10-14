"""Forward-model speed-ups (spec 2026-10-14-forward-model-speedup-design.md):
each fast path against the scipy original it replaces."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.signal import fftconvolve

from euclid_polish.config import Config
from euclid_polish.image import Image
from euclid_polish.psf import PSF
from euclid_polish.psf.fast_ops import bilinear_warp_reflect, circular_same_length, same_convolve
from euclid_polish.psf.psf_set import PSFSet
from euclid_polish.sky.observation.observation_simulator import (
    ObservationSimulator,
    ObservationSimulatorConfig,
)


@pytest.mark.parametrize("img_shape,ker_shape", [
    ((510, 510), (1023, 1023)),  # the training geometry itself
    ((64, 64), (127, 127)),     # kernel wider than the image (the training case, scaled)
    ((64, 48), (101, 77)),      # non-square, odd
    ((31, 40), (64, 64)),       # even kernel larger than the image
    ((64, 64), (10, 10)),       # even kernel smaller than the image
    ((50, 50), (7, 7)),         # small odd kernel
])
def test_same_convolve_matches_fftconvolve(img_shape, ker_shape):
    rng = np.random.default_rng(0)
    img = rng.normal(10, 3, img_shape).astype(np.float32)
    img[5, 7] += 1e4
    ker = np.abs(rng.normal(0, 1, ker_shape)).astype(np.float32)
    ker /= ker.sum()
    ref = fftconvolve(img, ker, mode="same")
    out = same_convolve(img, ker)
    assert out.shape == img.shape
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-5 * np.abs(ref).max())


def test_same_convolve_keeps_float64():
    rng = np.random.default_rng(2)
    img = rng.normal(0, 1, (40, 40))
    ker = rng.random((81, 81))
    out = same_convolve(img, ker)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, fftconvolve(img, ker, mode="same"), rtol=0, atol=1e-10)


def test_training_geometry_needs_one_1024_period():
    # 510² field, 1023² kernel: 1024² instead of the 1536² linear length.
    assert circular_same_length(510, 1023) == 1024
    # never longer than the full linear convolution's fast length
    for n_img, n_ker in [(50, 7), (64, 10), (31, 64), (510, 1023)]:
        assert circular_same_length(n_img, n_ker) <= n_img + n_ker - 1 + 8


# ---------------------------------------------------------------------------
# bilinear warp (map_coordinates(order=1, mode="reflect") replacement)
# ---------------------------------------------------------------------------

def _reference_warp(data, dy, dx):
    yy, xx = np.indices(data.shape, dtype=np.float64)
    return map_coordinates(np.asarray(data, np.float64), (yy + dy, xx + dx),
                           order=1, mode="reflect")


@pytest.mark.parametrize("shape,alpha", [((41, 41), 5.0), ((33, 50), 20.0), ((64, 64), 3.0)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_bilinear_warp_matches_map_coordinates(shape, alpha, dtype):
    rng = np.random.default_rng(1)
    data = rng.random(shape).astype(dtype)
    dy, dx = PSF.elastic_displacement(shape, alpha, 3.0, seed=4)
    ref = _reference_warp(data, dy, dx)
    out = bilinear_warp_reflect(data, dy, dx)
    assert out.dtype == dtype
    tol = 1e-6 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(out, ref, rtol=0, atol=tol * ref.max())


def test_bilinear_warp_reflects_far_beyond_the_edge():
    data = np.arange(12, dtype=np.float64).reshape(3, 4)
    for shift_y, shift_x in [(-7.25, 9.5), (2.75, -0.4), (-0.5, 3.5), (11.0, -13.3)]:
        dy = np.full((3, 4), shift_y)
        dx = np.full((3, 4), shift_x)
        np.testing.assert_allclose(bilinear_warp_reflect(data, dy, dx),
                                   _reference_warp(data, dy, dx), atol=1e-12)


def test_elastic_warp_equals_the_scipy_path_after_normalisation():
    g = np.exp(-((np.indices((51, 51)) - 25.0) ** 2).sum(0) / 30.0).astype(np.float32)
    psf = PSF(data=g / g.sum(), pixel_scale=0.05)
    dy, dx = PSF.elastic_displacement((51, 51), 5.0, 3.0, seed=9)
    ref = np.clip(_reference_warp(psf.data, dy, dx), 0, None)
    ref /= ref.sum()
    out = psf.elastic_warp(5.0, 3.0, seed=9, displacement=(dy, dx)).data
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-6 * ref.max())


# ---------------------------------------------------------------------------
# float32 displacement smoothing
# ---------------------------------------------------------------------------

def test_displacement_is_float32_and_keeps_the_random_stream():
    dy, dx = PSF.elastic_displacement((40, 30), 5.0, 3.0, seed=11)
    assert dy.dtype == np.float32 and dx.dtype == np.float32
    rng = np.random.default_rng(11)
    ref_dy = gaussian_filter(rng.uniform(-1.0, 1.0, size=(40, 30)), 3.0, mode="reflect") * 5.0
    ref_dx = gaussian_filter(rng.uniform(-1.0, 1.0, size=(40, 30)), 3.0, mode="reflect") * 5.0
    np.testing.assert_allclose(dy, ref_dy, rtol=0, atol=1e-5)
    np.testing.assert_allclose(dx, ref_dx, rtol=0, atol=1e-5)


# ---------------------------------------------------------------------------
# the star plane reuses the scene's realised PSF
# ---------------------------------------------------------------------------

def _gauss_psf(n, s):
    yy, xx = np.indices((n, n)) - (n - 1) / 2
    g = np.exp(-(yy ** 2 + xx ** 2) / (2 * s * s)).astype(np.float32)
    return PSF(data=g / g.sum(), pixel_scale=Config.DEFAULT_PIXEL_SCALE)


def test_star_plane_reuses_the_scene_psf(monkeypatch):
    """One realised (warped) PSF per band, shared by the scene and its stars."""
    sets = {b.name: PSFSet.from_psfs([_gauss_psf(31, 2.0), _gauss_psf(31, 3.0)])
            for b in Config.BANDS}
    calls = []
    original = PSFSet.apply_sample

    def counted(self, *a, **k):
        calls.append(1)
        return original(self, *a, **k)

    monkeypatch.setattr(PSFSet, "apply_sample", counted)
    sim = ObservationSimulator(psf_sets_by_band=sets, config=ObservationSimulatorConfig(
        add_noise=False, add_saturation=False, randomize_psf=True, psf_unrotated_prob=1.0,
        psf_warp_prob=1.0, psf_warp_alpha_max=5.0, psf_warp_sigma=3.0))
    field = np.full((96, 96, 4), 5.0, np.float32)
    stars = np.zeros_like(field)
    stars[40, 50, :] = 1e4
    hr = Image(data=field, pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
               band_names=Config.LR_INPUT_BAND_NAMES, is_clean=True)
    sim.process(hr, np.random.default_rng(3), star_hr_4ch=stars)
    assert len(calls) == len(Config.LR_INPUT_BAND_NAMES)
