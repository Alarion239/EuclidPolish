"""Tests for the multi-band forward model."""

from __future__ import annotations

import numpy as np
import pytest

from euclid_polish.config import Config
from euclid_polish.image import Image
from euclid_polish.sky.observation.noise import (
    apply_archive_noise,
    apply_band_noise,
)
from euclid_polish.sky.observation.observation_simulator import (
    ObservationSimulator,
    ObservationSimulatorConfig,
    default_psf_for_band,
)


@pytest.fixture
def hr_field():
    """Construct a synthetic HR field with a single bright source per band."""
    H = W = 96
    data = np.zeros((H, W, 4), dtype=np.float32)
    # 1e6 e⁻ delta at the centre of each band's channel.
    data[H // 2, W // 2, :] = 1.0e6
    return Image(
        data=data,
        pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
        band_names=Config.LR_INPUT_BAND_NAMES,
        is_clean=True,
    )


@pytest.fixture
def forward():
    # These tests probe the deterministic PSF/resample/flux path; the bright
    # 1e6 delta would otherwise trip detector-saturation masking (covered in
    # test_saturation.py), so disable it here.
    return ObservationSimulator(config=ObservationSimulatorConfig(
        add_noise=False, add_saturation=False))


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

def test_process_returns_correct_shapes(forward: ObservationSimulator, hr_field):
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


def test_hr_target_keeps_all_bands_clean(forward: ObservationSimulator, hr_field):
    _, hr = forward.process(hr_field, rng=np.random.default_rng(0))
    assert hr.band_names == Config.HR_TARGET_BAND_NAMES
    # Bit-identical to the HR input (no noise on the clean target).
    np.testing.assert_array_equal(hr.data, hr_field.data)


def test_separate_star_plane_matches_combined_scene_without_warp(
    forward: ObservationSimulator,
):
    base = np.zeros((96, 96, 4), dtype=np.float32)
    base[24:28, 20:30, :] = 100.0
    stars = np.zeros_like(base)
    stars[70, 68, :] = 2.0e4
    base_image = Image(
        data=base,
        pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
        band_names=Config.LR_INPUT_BAND_NAMES,
        is_clean=True,
    )
    combined_image = Image(
        data=base + stars,
        pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
        band_names=Config.LR_INPUT_BAND_NAMES,
        is_clean=True,
    )

    lr_separate, hr_separate = forward.process(
        base_image, np.random.default_rng(3), star_hr_4ch=stars,
    )
    lr_combined, _ = forward.process(
        combined_image, np.random.default_rng(3),
    )

    np.testing.assert_allclose(
        lr_separate.data, lr_combined.data, rtol=2e-5, atol=3e-4,
    )
    np.testing.assert_array_equal(hr_separate.data, base + stars)


def test_sparse_star_convolution_matches_fft(forward: ObservationSimulator):
    psf = default_psf_for_band(Config.BAND_VIS, Config.DEFAULT_PIXEL_SCALE)
    stars = np.zeros((72, 72), dtype=np.float32)
    stars[2, 3] = 7.0
    stars[35, 40] = 11.0
    stars[70, 68] = 5.0
    sparse = forward._convolve_sparse_deltas(stars, psf)
    fft = psf.convolved_with(stars)
    np.testing.assert_allclose(sparse, fft, rtol=2e-5, atol=3e-5)


def test_separate_star_plane_shares_warped_observation_psf():
    base = np.zeros((96, 96, 4), dtype=np.float32)
    base[20, 20, :] = 1.0e5
    stars = np.zeros_like(base)
    stars[72, 72, :] = 1.0e5
    image = Image(
        data=base,
        pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
        band_names=Config.LR_INPUT_BAND_NAMES,
        is_clean=True,
    )
    common = {
        "add_noise": False,
        "add_artifacts": False,
        "add_saturation": False,
        "randomize_psf": True,
        "psf_unrotated_prob": 1.0,
        "psf_warp_alpha_max": 20.0,
        "psf_warp_sigma": 3.0,
    }
    plain = ObservationSimulator(config=ObservationSimulatorConfig(
        **common, psf_warp_prob=0.0,
    ))
    warped = ObservationSimulator(config=ObservationSimulatorConfig(
        **common, psf_warp_prob=1.0,
    ))
    combined_image = Image(
        data=base + stars,
        pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
        band_names=Config.LR_INPUT_BAND_NAMES,
        is_clean=True,
    )

    lr_plain, _ = plain.process(
        image, np.random.default_rng(4), star_hr_4ch=stars,
    )
    lr_warped, _ = warped.process(
        image, np.random.default_rng(4), star_hr_4ch=stars,
    )
    lr_warped_combined, _ = warped.process(
        combined_image, np.random.default_rng(4),
    )
    difference = np.abs(lr_plain.data - lr_warped.data)

    # HR (20,20) lands at LR (10,10), HR (72,72) at LR (36,36): both the
    # ordinary scene and star plane receive the same observation-level warp.
    assert difference[4:17, 4:17].max() > 100.0
    assert difference[29:44, 29:44].max() > 100.0
    np.testing.assert_allclose(
        lr_warped.data, lr_warped_combined.data, rtol=2e-5, atol=1e-3,
    )


def test_lr_band_names_in_canonical_order(forward: ObservationSimulator, hr_field):
    lr, _ = forward.process(hr_field, rng=np.random.default_rng(0))
    assert lr.band_names == Config.LR_INPUT_BAND_NAMES


# ---------------------------------------------------------------------------
# Photometric integrity (noise off)
# ---------------------------------------------------------------------------

def test_noise_off_preserves_total_flux_per_band(forward: ObservationSimulator, hr_field):
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
    forward = ObservationSimulator(config=ObservationSimulatorConfig(add_noise=True))
    H = W = 64
    blank = Image(
        data=np.zeros((H, W, 4), dtype=np.float32),
        pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
        band_names=Config.LR_INPUT_BAND_NAMES, is_clean=True,
    )
    lr, _ = forward.process(blank, rng=np.random.default_rng(0))
    # Each LR channel should have many negative pixels (sky-subtracted + read noise).
    for k in range(4):
        assert (lr.data[..., k] < 0).sum() > 0


def test_vis_archive_noise_is_unchanged_native_noise():
    """VIS is already native at 0.10", so the new MER path is identical."""
    signal = np.full((48, 48), 20.0, dtype=np.float32)
    direct = apply_band_noise(
        signal, Config.BAND_VIS, np.random.default_rng(21),
        add_artifacts=False,
    )
    archive = apply_archive_noise(
        signal, Config.BAND_VIS, np.random.default_rng(21),
        add_artifacts=False,
    )
    np.testing.assert_array_equal(archive, direct)


def test_nisp_archive_noise_has_mer_covariance_and_scale():
    """NISP native 0.30" noise becomes correlated and faint after 3x MER."""
    noise = apply_archive_noise(
        np.zeros((192, 192), dtype=np.float32),
        Config.BAND_Y_E,
        np.random.default_rng(7),
        add_artifacts=False,
    )
    core = noise[12:-12, 12:-12].astype(np.float64)
    centered = core - core.mean()
    variance = float(np.mean(centered * centered))
    lag1 = float(np.mean(centered[:, :-1] * centered[:, 1:]) / variance)
    lag2 = float(np.mean(centered[:, :-2] * centered[:, 2:]) / variance)
    sigma = 1.4826 * float(np.median(np.abs(core - np.median(core))))
    phase_sigma = np.asarray([
        core[y::3, x::3].std()
        for y in range(3)
        for x in range(3)
    ])

    # Real MER Y/J/H fields measured ~0.78 and ~0.44 at lags 1 and 2;
    # tolerate field/realisation differences while rejecting white noise.
    assert 0.70 < lag1 < 0.93
    assert 0.30 < lag2 < 0.65
    # Real blank-sky RMS is ~2 e-/0.10" pixel, not the old ~17 e- white RMS.
    assert 1.2 < sigma < 4.0
    # Four balanced dither phases must not create a modulo-3 checkerboard.
    assert float(phase_sigma.max() / phase_sigma.min()) < 1.12


def test_nisp_archive_noise_preserves_non_multiple_of_three_shape():
    """Native-grid edge padding must never change the network-facing shape."""
    signal = np.zeros((32, 35), dtype=np.float32)
    observed = apply_archive_noise(
        signal,
        Config.BAND_J_E,
        np.random.default_rng(9),
        add_artifacts=False,
    )
    assert observed.shape == signal.shape


def test_noise_off_yields_zero_blank_image(forward: ObservationSimulator):
    """A blank HR scene with noise off produces an all-zero LR (no sky added)."""
    H = W = 64
    blank = Image(
        data=np.zeros((H, W, 4), dtype=np.float32),
        pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
        band_names=Config.LR_INPUT_BAND_NAMES, is_clean=True,
    )
    lr, _ = forward.process(blank, rng=np.random.default_rng(0))
    np.testing.assert_allclose(lr.data, 0.0, atol=1e-3)


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------

def test_band_count_mismatch_rejected(forward: ObservationSimulator):
    bad = Image(
        data=np.zeros((32, 32, 1), dtype=np.float32),
        pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
        band_names=("VIS",),
        is_clean=True,
    )
    with pytest.raises(ValueError):
        forward.process(bad, rng=np.random.default_rng(0))


def test_hr_scale_mismatch_rejected(forward: ObservationSimulator):
    bad = Image(
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
        ObservationSimulatorConfig(nisp_resample_kernel="bogus")


@pytest.mark.parametrize("kwargs", [
    {"psf_warp_prob": -0.1},
    {"psf_warp_prob": 1.1},
    {"psf_warp_alpha_max": -1.0},
    {"psf_warp_sigma": 0.0},
    {"saturation_mask_prob": -0.1},
    {"saturation_mask_prob": 1.1},
])
def test_invalid_psf_warp_config_raises(kwargs):
    with pytest.raises(ValueError):
        ObservationSimulatorConfig(**kwargs)


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
    cfg = ObservationSimulatorConfig(add_noise=False, randomize_psf=False)
    old = ObservationSimulator(psfs_by_band=psfs, config=cfg)
    new = ObservationSimulator(psf_sets_by_band=sets, config=cfg)
    lr_o, _ = old.process(hr_field, rng=np.random.default_rng(3))
    lr_n, _ = new.process(hr_field, rng=np.random.default_rng(3))
    np.testing.assert_allclose(lr_o.data, lr_n.data, atol=1e-5)


def test_randomized_psf_varies_scene_to_scene(hr_field):
    """With randomisation on, two scenes draw different PSFs (different pick
    and/or roll), so their VIS LR channels differ. Force always-rotate so the
    test is robust to the 30% unrotated draws."""
    sets = {b.name: _vis_psf_set(1, [0.16]) for b in Config.BANDS}
    sets[Config.BAND_VIS.name] = _vis_psf_set(2, [1.0, 5.0])
    fwd = ObservationSimulator(
        psf_sets_by_band=sets,
        config=ObservationSimulatorConfig(add_noise=False, add_saturation=False,
                                      randomize_psf=True, psf_unrotated_prob=0.0),
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
    fwd = ObservationSimulator(
        psf_sets_by_band=sets,
        config=ObservationSimulatorConfig(add_noise=False, randomize_psf=True))
    fwd.process(hr_field, rng=np.random.default_rng(1))
    assert len(seen) == 4                      # one apply per band
    assert all(s == seen[0] for s in seen)     # the SAME shared sample


def test_forward_psf_warp_changes_dirty_not_target(hr_field):
    sets = {b.name: _vis_psf_set(1, [4.0]) for b in Config.BANDS}
    nominal = ObservationSimulator(
        psf_sets_by_band=sets,
        config=ObservationSimulatorConfig(
            add_noise=False, add_saturation=False, randomize_psf=True,
            psf_unrotated_prob=1.0, psf_warp_prob=0.0),
    )
    warped = ObservationSimulator(
        psf_sets_by_band=sets,
        config=ObservationSimulatorConfig(
            add_noise=False, add_saturation=False, randomize_psf=True,
            psf_unrotated_prob=1.0, psf_warp_prob=1.0,
            psf_warp_alpha_max=20.0, psf_warp_sigma=3.0),
    )
    lr_nominal, hr_nominal = nominal.process(
        hr_field, rng=np.random.default_rng(4))
    lr_warped, hr_warped = warped.process(
        hr_field, rng=np.random.default_rng(4))
    assert not np.array_equal(lr_nominal.data, lr_warped.data)
    np.testing.assert_array_equal(hr_nominal.data, hr_warped.data)


def test_forward_reuses_one_warp_field_for_same_shape_bands(
    hr_field, monkeypatch,
):
    """A four-band exposure filters one displacement field, then reuses it."""
    from euclid_polish.psf import PSF

    calls = []
    original = PSF.elastic_displacement

    def counted(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(PSF, "elastic_displacement", staticmethod(counted))
    sets = {b.name: _vis_psf_set(1, [4.0]) for b in Config.BANDS}
    fwd = ObservationSimulator(
        psf_sets_by_band=sets,
        config=ObservationSimulatorConfig(
            add_noise=False, add_saturation=False, randomize_psf=True,
            psf_unrotated_prob=1.0, psf_warp_prob=1.0,
            psf_warp_alpha_max=20.0, psf_warp_sigma=3.0,
        ),
    )
    fwd.process(hr_field, rng=np.random.default_rng(4))
    assert calls == [(31, 31)]


def test_randomize_off_uses_mean_deterministically(hr_field):
    """randomize_psf=False → the field-mean PSF every scene (deterministic)."""
    sets = {b.name: _vis_psf_set(1, [0.16]) for b in Config.BANDS}
    sets[Config.BAND_VIS.name] = _vis_psf_set(3, [1.0, 3.0, 5.0])
    fwd = ObservationSimulator(
        psf_sets_by_band=sets,
        config=ObservationSimulatorConfig(add_noise=False, add_saturation=False,
                                      randomize_psf=False),
    )
    a, _ = fwd.process(hr_field, rng=np.random.default_rng(1))
    b, _ = fwd.process(hr_field, rng=np.random.default_rng(2))
    np.testing.assert_allclose(a.data[..., 0], b.data[..., 0], atol=1e-5)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_noise_reproducible_with_same_rng(hr_field):
    forward = ObservationSimulator(config=ObservationSimulatorConfig(add_noise=True))
    a, _ = forward.process(hr_field, rng=np.random.default_rng(123))
    b, _ = forward.process(hr_field, rng=np.random.default_rng(123))
    np.testing.assert_array_equal(a.data, b.data)
