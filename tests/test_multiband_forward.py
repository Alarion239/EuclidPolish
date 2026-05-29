"""Tests for the multi-band forward model."""

from __future__ import annotations

import numpy as np
import pytest

from euclid_polish.config import Config
from euclid_polish.sky.multiband_forward import (
    MultiBandForward,
    MultiBandForwardConfig,
    default_psf_for_band,
)
from euclid_polish.sky.types import MultiBandSkyImage


@pytest.fixture
def hr_field():
    """Construct a synthetic HR field with a single bright source per band."""
    H = W = 96
    data = np.zeros((H, W, 4), dtype=np.float32)
    # 1e6 e⁻ delta at the centre of each band's channel.
    data[H // 2, W // 2, :] = 1.0e6
    return MultiBandSkyImage(
        data=data,
        pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
        band_names=Config.LR_INPUT_BAND_NAMES,
        is_clean=True,
    )


@pytest.fixture
def forward():
    return MultiBandForward(config=MultiBandForwardConfig(add_noise=False))


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

def test_process_returns_correct_shapes(forward: MultiBandForward, hr_field):
    lr, hr = forward.process(hr_field, rng=np.random.default_rng(0))
    assert lr.shape[-1] == 4
    assert hr.shape[-1] == 1
    # LR all bands on VIS grid (0.10″/pix); HR is 0.05″/pix.
    assert lr.pixel_scale_arcsec == pytest.approx(Config.BAND_VIS.pixel_scale_lr_arcsec)
    assert hr.pixel_scale_arcsec == pytest.approx(Config.DEFAULT_PIXEL_SCALE)
    # All four LR channels share the same grid.
    expected_lr_side = hr_field.shape[0] // (
        Config.BAND_VIS.pixel_scale_lr_arcsec / Config.DEFAULT_PIXEL_SCALE
    )
    assert lr.shape[0] == int(expected_lr_side)
    assert lr.shape[1] == int(expected_lr_side)


def test_hr_target_is_vis_only(forward: MultiBandForward, hr_field):
    _, hr = forward.process(hr_field, rng=np.random.default_rng(0))
    assert hr.band_names == ("VIS",)
    # Bit-identical to channel 0 of the HR input (no noise on HR target).
    np.testing.assert_array_equal(hr.data[..., 0], hr_field.data[..., 0])


def test_lr_band_names_in_canonical_order(forward: MultiBandForward, hr_field):
    lr, _ = forward.process(hr_field, rng=np.random.default_rng(0))
    assert lr.band_names == Config.LR_INPUT_BAND_NAMES


# ---------------------------------------------------------------------------
# Photometric integrity (noise off)
# ---------------------------------------------------------------------------

def test_noise_off_preserves_total_flux_per_band(forward: MultiBandForward, hr_field):
    """Sum over each LR channel ≈ sum over the corresponding HR channel.

    Convolution + sum-rebin conserve total electrons exactly, so VIS (2×
    rebin, no upsample) matches to floating-point. NISP samples at its native
    0.30″ (6× rebin) and is then Lanczos3-upsampled back to 0.10″ with
    ``conserve_flux=True``; that resample splits each native pixel's flux
    across the finer grid, conserving the total only approximately (~%) due to
    Lanczos side lobes — hence the looser tolerance for the NISP bands.
    """
    lr, _ = forward.process(hr_field, rng=np.random.default_rng(0))
    for k, name in enumerate(Config.LR_INPUT_BAND_NAMES):
        hr_total = hr_field.data[..., k].sum()
        lr_total = lr.data[..., k].sum()
        rel = 1e-3 if name == "VIS" else 1e-2
        assert lr_total == pytest.approx(hr_total, rel=rel), name


# ---------------------------------------------------------------------------
# Noise model (noise on)
# ---------------------------------------------------------------------------

def test_noise_on_yields_negative_pixels():
    """With noise on, sky-subtracted output should have some negative pixels."""
    forward = MultiBandForward(config=MultiBandForwardConfig(add_noise=True))
    H = W = 64
    blank = MultiBandSkyImage(
        data=np.zeros((H, W, 4), dtype=np.float32),
        pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
        band_names=Config.LR_INPUT_BAND_NAMES, is_clean=True,
    )
    lr, _ = forward.process(blank, rng=np.random.default_rng(0))
    # Each LR channel should have many negative pixels (sky-subtracted + read noise).
    for k in range(4):
        assert (lr.data[..., k] < 0).sum() > 0


def test_noise_off_yields_zero_blank_image(forward: MultiBandForward):
    """A blank HR scene with noise off produces an all-zero LR (no sky added)."""
    H = W = 64
    blank = MultiBandSkyImage(
        data=np.zeros((H, W, 4), dtype=np.float32),
        pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
        band_names=Config.LR_INPUT_BAND_NAMES, is_clean=True,
    )
    lr, _ = forward.process(blank, rng=np.random.default_rng(0))
    np.testing.assert_allclose(lr.data, 0.0, atol=1e-3)


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------

def test_band_count_mismatch_rejected(forward: MultiBandForward):
    bad = MultiBandSkyImage(
        data=np.zeros((32, 32, 1), dtype=np.float32),
        pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
        band_names=("VIS",),
        is_clean=True,
    )
    with pytest.raises(ValueError):
        forward.process(bad, rng=np.random.default_rng(0))


def test_hr_scale_mismatch_rejected(forward: MultiBandForward):
    bad = MultiBandSkyImage(
        data=np.zeros((32, 32, 4), dtype=np.float32),
        pixel_scale_arcsec=0.99,
        band_names=Config.LR_INPUT_BAND_NAMES,
        is_clean=True,
    )
    with pytest.raises(ValueError):
        forward.process(bad, rng=np.random.default_rng(0))


def test_default_psf_pixel_scale():
    psf = default_psf_for_band(Config.BAND_VIS, Config.DEFAULT_PIXEL_SCALE)
    assert psf.pixel_scale == pytest.approx(Config.DEFAULT_PIXEL_SCALE)
    assert psf.data.sum() == pytest.approx(1.0)


def test_invalid_kernel_raises():
    with pytest.raises(ValueError):
        MultiBandForwardConfig(nisp_resample_kernel="bogus")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_noise_reproducible_with_same_rng(hr_field):
    forward = MultiBandForward(config=MultiBandForwardConfig(add_noise=True))
    a, _ = forward.process(hr_field, rng=np.random.default_rng(123))
    b, _ = forward.process(hr_field, rng=np.random.default_rng(123))
    np.testing.assert_array_equal(a.data, b.data)
