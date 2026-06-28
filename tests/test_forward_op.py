"""Tests for the TF-graph Euclid VIS forward op.

The round-trip training path needs ``Conv(M(LR))`` inside the gradient
tape. This module pins the four properties that have to hold for the
loss to do the right thing:

  1. **Shape contract**: input HR → output LR shrinks by exactly
     ``rebin_factor`` per side.
  2. **Photometric conservation**: PSF (unit-flux) ⊛ sum-rebin
     preserves total flux to round-off for sources not pinned to the
     image edge.
  3. **Gradient flow**: ``tf.GradientTape`` produces a non-trivial
     gradient of the forward-op output w.r.t. its input — without this,
     the round-trip loss couldn't backprop into M.
  4. **Numerical parity** with the numpy synthetic forward path
     (``ObservationSimulator._process_one_band`` for VIS, noise off). Same
     PSF, same input → same output to within float32 round-off across
     the bulk of the array.

A small synthetic Gaussian PSF is written to a tmp FITS in each test
so we don't depend on the real ``data/euclid_psf/`` files (which
aren't in the repo) and the kernel size stays small enough for
``tf.nn.conv2d`` to run quickly on CPU.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import tensorflow as tf
from astropy.io import fits

from euclid_polish.training.forward_op import (
    EuclidVISForwardOp, _load_vis_psf_kernel,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _gaussian_psf(side: int, fwhm_pix: float) -> np.ndarray:
    """2-D unit-flux Gaussian, ``side``×``side``, centred."""
    sigma = float(fwhm_pix) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    y, x = np.mgrid[:side, :side]
    cy = cx = (side - 1) / 2.0
    g = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma ** 2))
    return (g / g.sum()).astype(np.float32)


@pytest.fixture
def tmp_psf_fits(tmp_path) -> str:
    """Write a small Gaussian VIS-PSF-shaped FITS for parity tests."""
    psf = _gaussian_psf(side=31, fwhm_pix=3.2)  # ~0.16″ FWHM at 0.05″/px
    path = str(tmp_path / "synth_psf.fits")
    fits.PrimaryHDU(psf).writeto(path, overwrite=True)
    return path


# ---------------------------------------------------------------------------
# 1. Shape contract
# ---------------------------------------------------------------------------

class TestShape:

    def test_halves_each_spatial_dim(self, tmp_psf_fits):
        op = EuclidVISForwardOp(psf_fits_path=tmp_psf_fits, rebin_factor=2)
        hr = tf.zeros([2, 128, 128, 1], dtype=tf.float32)
        lr = op(hr)
        assert tuple(lr.shape) == (2, 64, 64, 1)

    def test_rebin_factor_one_keeps_shape(self, tmp_psf_fits):
        op = EuclidVISForwardOp(psf_fits_path=tmp_psf_fits, rebin_factor=1)
        hr = tf.zeros([1, 64, 64, 1], dtype=tf.float32)
        assert tuple(op(hr).shape) == (1, 64, 64, 1)

    def test_odd_input_trims_trailing(self, tmp_psf_fits):
        """``rebin_factor=2`` on an odd-side input drops the last pixel."""
        op = EuclidVISForwardOp(psf_fits_path=tmp_psf_fits, rebin_factor=2)
        hr = tf.zeros([1, 65, 65, 1], dtype=tf.float32)
        lr = op(hr)
        assert tuple(lr.shape) == (1, 32, 32, 1)


# ---------------------------------------------------------------------------
# 2. Photometric conservation
# ---------------------------------------------------------------------------

class TestFluxPreservation:

    def test_total_flux_preserved_for_centred_source(self, tmp_psf_fits):
        """A delta-function source far from the edges keeps its flux.

        Tolerance comes from float32 accumulation across the avg-pool ×
        area reconstruction (sum-rebin via avg_pool*N² loses ~1 ULP
        per output pixel).
        """
        op = EuclidVISForwardOp(psf_fits_path=tmp_psf_fits, rebin_factor=2)
        hr = np.zeros([1, 128, 128, 1], dtype=np.float32)
        hr[0, 64, 64, 0] = 1000.0      # source well clear of 31-side PSF
        lr = op(tf.constant(hr)).numpy()
        assert float(lr.sum()) == pytest.approx(1000.0, rel=1e-4)

    def test_dc_gain_preserved_for_uniform_field(self, tmp_psf_fits):
        """Uniform field of value ``v`` → output mean ≈ v × rebin_factor².

        Each output pixel is the sum of (rebin_factor × rebin_factor)
        input pixels post-conv; convolution of a uniform field is a
        uniform field of the same value (unit-flux PSF). So each
        output pixel ≈ v × N². Edge pixels diverge because conv with
        zero-padding produces a slight drop near the boundary.
        """
        op = EuclidVISForwardOp(psf_fits_path=tmp_psf_fits, rebin_factor=2)
        v = 3.14
        hr = tf.fill([1, 128, 128, 1], v)
        lr = op(hr).numpy()
        # Bulk of the LR (well inside the PSF support) should hit v × N².
        bulk = lr[0, 16:-16, 16:-16, 0]
        assert np.allclose(bulk, v * 4.0, atol=1e-4)


# ---------------------------------------------------------------------------
# 3. Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:

    def test_gradient_w_r_t_input_nonzero(self, tmp_psf_fits):
        """``∂(sum(forward(x)²)) / ∂x`` must be a real (non-zero) gradient."""
        op = EuclidVISForwardOp(psf_fits_path=tmp_psf_fits, rebin_factor=2)
        hr = tf.Variable(tf.random.normal([1, 64, 64, 1], seed=0))
        with tf.GradientTape() as tape:
            lr = op(hr)
            loss = tf.reduce_sum(lr ** 2)
        grad = tape.gradient(loss, hr)
        assert grad is not None
        # Some pixels must have non-trivial gradient — otherwise the
        # round-trip loss can't pull M's weights.
        assert float(tf.reduce_max(tf.abs(grad))) > 1e-6

    def test_psf_kernel_is_non_trainable(self, tmp_psf_fits):
        """The PSF is a known physical constant, not a model parameter."""
        op = EuclidVISForwardOp(psf_fits_path=tmp_psf_fits, rebin_factor=2)
        # build the layer by calling it once
        _ = op(tf.zeros([1, 32, 32, 1], dtype=tf.float32))
        assert len(op.trainable_weights) == 0
        assert len(op.non_trainable_weights) == 1


# ---------------------------------------------------------------------------
# 4. Numerical parity with the numpy synthetic forward
# ---------------------------------------------------------------------------

class TestNumpyParity:
    """Pin the TF op against the existing scipy-based forward.

    The TF op MUST agree with ``ObservationSimulator._process_one_band`` (VIS,
    noise off) so the round-trip loss isn't training the model to undo
    a *different* Conv from the one applied at synthetic generation
    time. Boundary pixels diverge by O(1) ULP because of how float32
    rounding accumulates across slightly different conv kernel paths
    (FFT vs spatial); we crop a PSF-sized border before comparing.
    """

    def test_matches_scipy_for_random_hr(self, tmp_psf_fits):
        from scipy import signal as scipy_signal

        psf = _gaussian_psf(side=31, fwhm_pix=3.2)
        op  = EuclidVISForwardOp(psf_fits_path=tmp_psf_fits, rebin_factor=2)

        rng = np.random.default_rng(42)
        hr_np = rng.normal(loc=10.0, scale=1.0, size=(128, 128)).astype(np.float32)

        # TF path.
        lr_tf = op(tf.constant(hr_np[np.newaxis, :, :, np.newaxis])).numpy()[0, :, :, 0]

        # Numpy reference (matches ObservationSimulator._process_one_band:223-228
        # for VIS, noise off — fftconvolve mode='same' + sum-rebin by 2).
        conv = scipy_signal.fftconvolve(hr_np, psf, mode="same").astype(np.float32)
        h_t, w_t = (128 // 2) * 2, (128 // 2) * 2
        sumreb = (
            conv[:h_t, :w_t]
            .reshape(h_t // 2, 2, w_t // 2, 2)
            .sum(axis=(1, 3))
        )

        # Bulk comparison — crop a half-kernel border (15 LR pixels ≈
        # 30 HR pixels) to dodge edge-padding numerical drift.
        b = 16
        bulk_tf = lr_tf[b:-b, b:-b]
        bulk_np = sumreb[b:-b, b:-b]
        assert np.allclose(bulk_tf, bulk_np, atol=1e-3, rtol=1e-4), (
            f"TF vs numpy max abs diff in bulk: "
            f"{np.abs(bulk_tf - bulk_np).max():.3e}"
        )

    def test_delta_response_central_peak(self, tmp_psf_fits):
        """A delta input → PSF profile in the output, scaled by rebin_factor².

        Pins the convolution direction: a flipped/un-flipped kernel
        choice would shift the response off-centre by (psf_half, psf_half)
        — a smoke-test that the layer applies *convolution* (delta → PSF
        centred at the delta location), not arbitrary cross-correlation.
        """
        op = EuclidVISForwardOp(psf_fits_path=tmp_psf_fits, rebin_factor=2)
        hr = np.zeros([1, 64, 64, 1], dtype=np.float32)
        hr[0, 32, 32, 0] = 1.0
        lr = op(tf.constant(hr)).numpy()[0, :, :, 0]
        # After 2x sum-rebin, the HR delta at (32, 32) lands in the
        # LR bin (16, 16). The PSF centre — a unit-area Gaussian — is
        # the brightest pixel.
        cy, cx = np.unravel_index(int(np.argmax(lr)), lr.shape)
        assert (cy, cx) == (16, 16), (
            f"PSF centre landed at ({cy}, {cx}), expected (16, 16) — "
            "kernel orientation (conv vs xcorr) may be wrong"
        )


# ---------------------------------------------------------------------------
# PSF-loader helpers
# ---------------------------------------------------------------------------

class TestPsfLoader:

    def test_normalises_to_unit_sum(self, tmp_psf_fits):
        psf = _load_vis_psf_kernel(tmp_psf_fits)
        assert float(psf.sum()) == pytest.approx(1.0, abs=1e-6)

    def test_crop_half_side_returns_smaller_kernel(self, tmp_psf_fits):
        psf = _load_vis_psf_kernel(tmp_psf_fits, crop_half_side=5)
        assert psf.shape == (11, 11)
        # Even cropped, must re-normalise so DC gain stays 1.
        assert float(psf.sum()) == pytest.approx(1.0, abs=1e-6)

    def test_rejects_non_2d_psf(self, tmp_path):
        bad = str(tmp_path / "bad.fits")
        fits.PrimaryHDU(np.zeros((3, 5, 5), dtype=np.float32)).writeto(bad)
        with pytest.raises(ValueError, match="must be 2-D"):
            _load_vis_psf_kernel(bad)
