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
    # 4-band training: the HR target keeps every band.
    assert hr.shape[-1] == 4
    # LR all bands on VIS grid (0.10″/pix); HR is 0.05″/pix.
    assert lr.pixel_scale_arcsec == pytest.approx(Config.BAND_VIS.pixel_scale_lr_arcsec)
    assert hr.pixel_scale_arcsec == pytest.approx(Config.DEFAULT_PIXEL_SCALE)
    # All four LR channels share the same grid.
    expected_lr_side = hr_field.shape[0] // (
        Config.BAND_VIS.pixel_scale_lr_arcsec / Config.DEFAULT_PIXEL_SCALE
    )
    assert lr.shape[0] == int(expected_lr_side)
    assert lr.shape[1] == int(expected_lr_side)


def test_hr_target_keeps_all_bands_clean(forward: MultiBandForward, hr_field):
    _, hr = forward.process(hr_field, rng=np.random.default_rng(0))
    assert hr.band_names == Config.HR_TARGET_BAND_NAMES
    # Bit-identical to the HR input (no noise on the clean target).
    np.testing.assert_array_equal(hr.data, hr_field.data)


def test_lr_band_names_in_canonical_order(forward: MultiBandForward, hr_field):
    lr, _ = forward.process(hr_field, rng=np.random.default_rng(0))
    assert lr.band_names == Config.LR_INPUT_BAND_NAMES


# ---------------------------------------------------------------------------
# Photometric integrity (noise off)
# ---------------------------------------------------------------------------

def test_noise_off_preserves_total_flux_per_band(forward: MultiBandForward, hr_field):
    """Sum over each LR channel ≈ sum over the corresponding HR channel.

    Convolution + sum-rebin conserve total electrons; with every band at
    0.10″ LR the rebin factor is 2 and there is no NISP upsample, so flux is
    conserved across all channels.
    """
    lr, _ = forward.process(hr_field, rng=np.random.default_rng(0))
    for k, name in enumerate(Config.LR_INPUT_BAND_NAMES):
        hr_total = hr_field.data[..., k].sum()
        lr_total = lr.data[..., k].sum()
        assert lr_total == pytest.approx(hr_total, rel=1e-3), name


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
# Position-dependent PSF sets
# ---------------------------------------------------------------------------

def _vis_psf_set(n, fwhms):
    """A VIS PSFSet of ``n`` Gaussian kernels at the HR grid."""
    from euclid_polish.psf import PSF, PSFSet
    members = []
    for fwhm in fwhms[:n]:
        side = 31
        x = np.arange(side) - side // 2
        X, Y = np.meshgrid(x, x)
        s = fwhm / 2.355
        g = np.exp(-(X * X + Y * Y) / (2 * s * s)).astype(np.float32)
        members.append(PSF(data=g / g.sum(),
                           pixel_scale=Config.DEFAULT_PIXEL_SCALE))
    return PSFSet.from_psfs(members)


def test_single_psf_set_matches_old_psf_dict(hr_field):
    """Wrapping a single PSF as a 1-element set (new psf_sets_by_band API)
    reproduces the old psfs_by_band path — with randomisation off both use the
    deterministic mean, which for K=1 is just that PSF."""
    from euclid_polish.psf import PSFSet
    psfs = {b.name: default_psf_for_band(b, Config.DEFAULT_PIXEL_SCALE)
            for b in Config.BANDS}
    sets = {name: PSFSet.from_psfs([p]) for name, p in psfs.items()}
    cfg = MultiBandForwardConfig(add_noise=False, randomize_psf=False)
    old = MultiBandForward(psfs_by_band=psfs, config=cfg)
    new = MultiBandForward(psf_sets_by_band=sets, config=cfg)
    lr_o, _ = old.process(hr_field, rng=np.random.default_rng(3))
    lr_n, _ = new.process(hr_field, rng=np.random.default_rng(3))
    np.testing.assert_allclose(lr_o.data, lr_n.data, atol=1e-5)


def test_randomized_psf_varies_scene_to_scene(hr_field):
    """With randomisation on, two scenes draw different PSFs (different pick
    and/or roll), so their VIS LR channels differ. Force always-rotate so the
    test is robust to the 30% unrotated draws."""
    sets = {b.name: _vis_psf_set(1, [0.16]) for b in Config.BANDS}
    sets[Config.BAND_VIS.name] = _vis_psf_set(2, [1.0, 5.0])
    fwd = MultiBandForward(
        psf_sets_by_band=sets,
        config=MultiBandForwardConfig(add_noise=False, randomize_psf=True,
                                      psf_unrotated_prob=0.0),
    )
    a, _ = fwd.process(hr_field, rng=np.random.default_rng(1))
    b, _ = fwd.process(hr_field, rng=np.random.default_rng(2))
    assert not np.allclose(a.data[..., 0], b.data[..., 0])
    # Rotation preserves total flux (the kernel stays sum=1).
    assert a.data[..., 0].sum() == pytest.approx(hr_field.data[..., 0].sum(),
                                                 rel=1e-3)


def test_forward_shares_one_psf_sample_across_bands(hr_field, monkeypatch):
    """One PSFSample (cluster index + roll) is drawn per scene and applied to
    all four bands — physically a single pointing, one telescope roll."""
    from euclid_polish.psf import PSFSet
    seen = []
    orig = PSFSet.apply_sample
    monkeypatch.setattr(
        PSFSet, "apply_sample",
        lambda self, sample, **k: seen.append(sample) or orig(self, sample, **k))
    sets = {b.name: _vis_psf_set(3, [1.0, 2.0, 3.0]) for b in Config.BANDS}
    fwd = MultiBandForward(
        psf_sets_by_band=sets,
        config=MultiBandForwardConfig(add_noise=False, randomize_psf=True))
    fwd.process(hr_field, rng=np.random.default_rng(1))
    assert len(seen) == 4                      # one apply per band
    assert all(s == seen[0] for s in seen)     # the SAME shared sample


def test_randomize_off_uses_mean_deterministically(hr_field):
    """randomize_psf=False → the field-mean PSF every scene (deterministic)."""
    sets = {b.name: _vis_psf_set(1, [0.16]) for b in Config.BANDS}
    sets[Config.BAND_VIS.name] = _vis_psf_set(3, [1.0, 3.0, 5.0])
    fwd = MultiBandForward(
        psf_sets_by_band=sets,
        config=MultiBandForwardConfig(add_noise=False, randomize_psf=False),
    )
    a, _ = fwd.process(hr_field, rng=np.random.default_rng(1))
    b, _ = fwd.process(hr_field, rng=np.random.default_rng(2))
    np.testing.assert_allclose(a.data[..., 0], b.data[..., 0], atol=1e-5)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_noise_reproducible_with_same_rng(hr_field):
    forward = MultiBandForward(config=MultiBandForwardConfig(add_noise=True))
    a, _ = forward.process(hr_field, rng=np.random.default_rng(123))
    b, _ = forward.process(hr_field, rng=np.random.default_rng(123))
    np.testing.assert_array_equal(a.data, b.data)
