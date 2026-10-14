"""Forward-model speed-ups (spec 2026-10-14-forward-model-speedup-design.md):
each fast path against the scipy original it replaces."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import fftconvolve

from euclid_polish.psf.fast_ops import circular_same_length, same_convolve


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
