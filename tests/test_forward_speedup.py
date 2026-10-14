"""Forward-model speed-ups (spec 2026-10-14-forward-model-speedup-design.md):
each fast path against the scipy original it replaces."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve

from euclid_polish.psf import PSF
from euclid_polish.psf.fast_ops import bilinear_warp_reflect, circular_same_length, same_convolve


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
