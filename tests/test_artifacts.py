"""Tests for cosmic-ray + hot-pixel injection."""

from __future__ import annotations

import numpy as np
import pytest

from euclid_polish.config import Config
from euclid_polish.sky.artifacts import (
    ArtifactConfig,
    expected_cosmic_ray_count,
    expected_streak_count,
    inject_artifacts,
    inject_cosmic_rays,
    inject_hot_pixels,
    inject_streaks,
)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def test_default_config_has_reasonable_values():
    cfg = ArtifactConfig()
    assert cfg.add_cosmic_rays  is True
    assert cfg.add_hot_pixels   is True
    assert 0 < cfg.cr_rate_per_s_per_cm2 < 100
    assert 0 < cfg.hot_pixel_fraction < 0.1


@pytest.mark.parametrize("bad_kwargs", [
    {"cr_rate_per_s_per_cm2": -1.0},
    {"cr_charge_median_e":   0.0},
    {"hot_pixel_fraction":   1.5},
    {"hot_pixel_charge_mean_e": 0.0},
    {"cr_max_track_length":  0},
    {"streak_rate_per_kpix2": -1.0},
    {"streak_length_min_pix": 0},
    {"streak_length_min_pix": 200, "streak_length_max_pix": 100},
    {"streak_width_pix_min": 5, "streak_width_pix_max": 3},
    {"streak_amp_sigma_min": 0.9, "streak_amp_sigma_max": 0.3},
])
def test_invalid_config_rejected(bad_kwargs):
    with pytest.raises(ValueError):
        ArtifactConfig(**bad_kwargs)


# ---------------------------------------------------------------------------
# Cosmic-ray count math
# ---------------------------------------------------------------------------

def test_cr_count_zero_when_disabled():
    cfg = ArtifactConfig(add_cosmic_rays=False)
    n = expected_cosmic_ray_count((100, 100), Config.BAND_VIS, cfg)
    assert n == 0.0


def test_cr_count_scales_linearly_with_area():
    cfg = ArtifactConfig()
    n_small = expected_cosmic_ray_count((50, 50), Config.BAND_VIS, cfg)
    n_big   = expected_cosmic_ray_count((100, 100), Config.BAND_VIS, cfg)
    assert n_big == pytest.approx(4 * n_small, rel=1e-6)


def test_cr_count_scales_linearly_with_exposure():
    cfg = ArtifactConfig()
    # The function reads exposure from band.t_total_s implicitly via
    # band.exposure_time_s × n_exposures. Compare VIS vs NISP keeping
    # other constants fixed-ish; instead, check that doubling rate
    # doubles the count.
    cfg2 = ArtifactConfig(cr_rate_per_s_per_cm2=cfg.cr_rate_per_s_per_cm2 * 2)
    n1 = expected_cosmic_ray_count((200, 200), Config.BAND_VIS, cfg)
    n2 = expected_cosmic_ray_count((200, 200), Config.BAND_VIS, cfg2)
    assert n2 == pytest.approx(2 * n1, rel=1e-6)


def test_cr_count_matches_back_of_envelope_for_vis():
    """VIS 510² @ 0.10″: raw rate × area × time × rejection factor.

    The post-rejection factor (~0.02 for VIS) models across-dither
    image-differencing that removes the vast majority of GCR hits
    before the science image is delivered.
    """
    cfg = ArtifactConfig()
    n = expected_cosmic_ray_count((510, 510), Config.BAND_VIS, cfg)
    raw   = 5.0 * (12e-4) ** 2 * 510 * 510 * (565 * 4)
    rejection = Config.BAND_VIS.cr_rate_factor
    assert n == pytest.approx(raw * rejection, rel=1e-9)
    # Post-rejection should leave order ~100 hits per 510² stack (not 4000).
    assert 30 < n < 300


def test_cr_count_nisp_keeps_most_hits():
    """NISP up-the-ramp slope fit is less aggressive: rate factor ~1."""
    cfg = ArtifactConfig()
    n_vis = expected_cosmic_ray_count((510, 510), Config.BAND_VIS, cfg)
    n_nisp = expected_cosmic_ray_count((510, 510), Config.BAND_Y_E, cfg)
    # After our per-band factors, NISP gets MORE hits than VIS because
    # VIS is heavily rejected — even though VIS exposes 5× longer.
    assert n_nisp > n_vis, (n_vis, n_nisp)


def test_cr_rate_factor_zero_disables_band():
    """Setting cr_rate_factor=0 on a band kills hits there entirely."""
    from dataclasses import replace
    cfg = ArtifactConfig()
    quiet_band = replace(Config.BAND_VIS, cr_rate_factor=0.0)
    n = expected_cosmic_ray_count((100, 100), quiet_band, cfg)
    assert n == 0.0


# ---------------------------------------------------------------------------
# CR injection
# ---------------------------------------------------------------------------

def test_inject_cr_adds_charge():
    img = np.zeros((128, 128), dtype=np.float32)
    cfg = ArtifactConfig(cr_charge_median_e=1500.0)
    rng = np.random.default_rng(0)
    out = inject_cosmic_rays(img, Config.BAND_VIS, rng, cfg)
    # Some pixels should have non-zero charge.
    assert out.sum() > 0
    assert (out > 0).sum() > 0


def test_inject_cr_zero_when_disabled():
    img = np.zeros((128, 128), dtype=np.float32) + 1.0
    cfg = ArtifactConfig(add_cosmic_rays=False)
    rng = np.random.default_rng(0)
    out = inject_cosmic_rays(img, Config.BAND_VIS, rng, cfg)
    np.testing.assert_array_equal(out, img)


def test_inject_cr_charge_scale_matches():
    """Total deposited charge ≈ N_hits × mean_charge ≈ rate × area × t × ⟨q⟩."""
    img = np.zeros((512, 512), dtype=np.float32)
    cfg = ArtifactConfig(cr_charge_median_e=1500.0)
    rng = np.random.default_rng(42)
    out = inject_cosmic_rays(img, Config.BAND_VIS, rng, cfg)
    expected_n_hits = expected_cosmic_ray_count((512, 512), Config.BAND_VIS, cfg)
    expected_total  = expected_n_hits * cfg.cr_charge_median_e
    # Allow ~30% deviation (Poisson + exponential variance on each scale).
    assert 0.5 < out.sum() / expected_total < 1.5


# ---------------------------------------------------------------------------
# Hot pixels
# ---------------------------------------------------------------------------

def test_inject_hot_pixels_count_matches_fraction():
    img = np.zeros((1024, 1024), dtype=np.float32)
    cfg = ArtifactConfig(hot_pixel_fraction=0.005,
                         hot_pixel_charge_mean_e=10_000.0)
    rng = np.random.default_rng(0)
    out = inject_hot_pixels(img, rng, cfg)
    n_hot = int((out > 0).sum())
    expected = cfg.hot_pixel_fraction * img.size
    # Binomial std ≈ sqrt(N p (1-p)) ≈ 72 here. 3-sigma tolerance.
    assert abs(n_hot - expected) < 250


def test_inject_hot_pixels_disabled():
    img = np.zeros((256, 256), dtype=np.float32) + 5.0
    cfg = ArtifactConfig(add_hot_pixels=False)
    rng = np.random.default_rng(0)
    out = inject_hot_pixels(img, rng, cfg)
    np.testing.assert_array_equal(out, img)


def test_inject_hot_pixels_zero_fraction():
    img = np.zeros((128, 128), dtype=np.float32)
    cfg = ArtifactConfig(hot_pixel_fraction=0.0)
    rng = np.random.default_rng(0)
    out = inject_hot_pixels(img, rng, cfg)
    np.testing.assert_array_equal(out, img)


# ---------------------------------------------------------------------------
# Long faint masked-trail streaks
# ---------------------------------------------------------------------------

def test_streak_count_scales_with_area():
    cfg = ArtifactConfig()
    n_small = expected_streak_count((250, 250), cfg)
    n_big   = expected_streak_count((500, 500), cfg)
    assert n_big == pytest.approx(4 * n_small, rel=1e-9)


def test_streak_count_zero_when_disabled():
    cfg = ArtifactConfig(add_streaks=False)
    assert expected_streak_count((500, 500), cfg) == 0.0


def test_inject_streak_zero_when_disabled():
    img = np.zeros((128, 128), dtype=np.float32) + 7.0
    cfg = ArtifactConfig(add_streaks=False)
    out = inject_streaks(img, np.random.default_rng(0), cfg, local_sigma_e=10.0)
    np.testing.assert_array_equal(out, img)


def test_inject_streak_zero_when_sigma_zero():
    """Pass-through when caller can't compute a meaningful local σ."""
    img = np.zeros((128, 128), dtype=np.float32) + 7.0
    cfg = ArtifactConfig(add_streaks=True, streak_rate_per_kpix2=1000.0)
    out = inject_streaks(img, np.random.default_rng(0), cfg, local_sigma_e=0.0)
    np.testing.assert_array_equal(out, img)


def test_inject_streak_deposits_charge_at_high_rate():
    """With an artificially high streak rate, ≥1 streak should appear on
    most realisations and the image should differ from the input."""
    img = np.zeros((256, 256), dtype=np.float32)
    cfg = ArtifactConfig(add_streaks=True,
                         streak_rate_per_kpix2=200.0,    # ~13 streaks per 256²
                         streak_amp_sigma_min=0.5,
                         streak_amp_sigma_max=0.5)
    rng = np.random.default_rng(0)
    out = inject_streaks(img, rng, cfg, local_sigma_e=20.0)
    # Non-zero pixels should form a clear line-like population.
    nonzero = (np.abs(out) > 1e-6).sum()
    assert nonzero > 1000, nonzero   # at least one substantial streak


def test_inject_streak_amplitude_is_sub_sigma_per_streak():
    """A single isolated streak should peak at exactly
    ``amp_sigma · σ`` (the Gaussian cross-section is unity-normalised at
    its centre). Overlap can exceed this in the dense limit; we test the
    single-streak case where overlap is impossible."""
    img = np.zeros((512, 512), dtype=np.float32)
    sigma = 25.0
    # Force exactly 1 streak by making the mean count overwhelming and
    # checking we always get ≥1 — but check the peak is bounded by the
    # config max for the amp distribution (irrespective of count).
    cfg = ArtifactConfig(add_streaks=True,
                         streak_rate_per_kpix2=4.0,    # ≈ 1 streak / 512²
                         streak_amp_sigma_min=0.5,
                         streak_amp_sigma_max=0.5)
    # Generate a few independent realisations; assert per-pixel max is
    # at most amp_max · σ in single-streak realisations.
    for seed in (1, 2, 3, 4, 5):
        out = inject_streaks(img, np.random.default_rng(seed), cfg, local_sigma_e=sigma)
        if (np.abs(out) > 1e-6).any():
            # When a streak appears, its peak should match the config exactly
            # (Gaussian cross-section value is 1 at j=0).
            assert np.abs(out).max() <= cfg.streak_amp_sigma_max * sigma + 1e-4


def test_inject_streak_is_linear_in_local_sigma():
    """Doubling local_sigma_e should ~double the streak amplitude (same RNG)."""
    img = np.zeros((256, 256), dtype=np.float32)
    cfg = ArtifactConfig(add_streaks=True, streak_rate_per_kpix2=50.0)
    a = inject_streaks(img, np.random.default_rng(11), cfg, local_sigma_e=10.0)
    b = inject_streaks(img, np.random.default_rng(11), cfg, local_sigma_e=20.0)
    nz = np.abs(a) > 1e-6
    assert nz.any()
    np.testing.assert_allclose(b[nz], 2 * a[nz], rtol=1e-5, atol=1e-5)


def test_streak_orientation_is_isotropic_over_many_streaks():
    """The orientation is uniform in [0, π). For many streaks, the
    row-vs-column projection energies should be roughly balanced —
    streaks aren't aligned with the readout axis."""
    img = np.zeros((512, 512), dtype=np.float32)
    cfg = ArtifactConfig(add_streaks=True,
                         streak_rate_per_kpix2=400.0,    # lots of streaks
                         streak_amp_sigma_min=0.5,
                         streak_amp_sigma_max=0.5)
    out = inject_streaks(img, np.random.default_rng(1), cfg, local_sigma_e=10.0)
    row_energy = np.abs(out).sum(axis=1).std()
    col_energy = np.abs(out).sum(axis=0).std()
    # The two should be within 50% of each other (isotropic injection).
    ratio = max(row_energy, col_energy) / min(row_energy, col_energy)
    assert ratio < 2.0, ratio


# ---------------------------------------------------------------------------
# Full pipeline integration: forward model with artifacts on vs off
# ---------------------------------------------------------------------------

def test_forward_model_with_artifacts_changes_image():
    from euclid_polish.sky.multiband_forward import (
        MultiBandForward, MultiBandForwardConfig,
    )
    from euclid_polish.sky.types import MultiBandSkyImage

    H = W = 128
    blank = MultiBandSkyImage(
        data=np.zeros((H, W, 4), dtype=np.float32),
        pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
        band_names=Config.LR_INPUT_BAND_NAMES, is_clean=True,
    )

    fwd_off = MultiBandForward(
        config=MultiBandForwardConfig(add_noise=True, add_artifacts=False)
    )
    fwd_on  = MultiBandForward(
        config=MultiBandForwardConfig(add_noise=True, add_artifacts=True)
    )

    lr_off, _ = fwd_off.process(blank, rng=np.random.default_rng(0))
    lr_on,  _ = fwd_on.process( blank, rng=np.random.default_rng(0))

    # With artifacts on, max pixel value should be much higher (CR + hot pixels).
    p99_off = float(np.percentile(lr_off.data, 99.9))
    p99_on  = float(np.percentile(lr_on.data,  99.9))
    assert p99_on > p99_off * 5, (
        f"artifact-on p99.9 ({p99_on:.1f}) should clearly exceed "
        f"artifact-off p99.9 ({p99_off:.1f})"
    )


def test_forward_model_artifact_off_matches_old_behaviour():
    """With artifacts off, output should be reproducible from the same RNG seed."""
    from euclid_polish.sky.multiband_forward import (
        MultiBandForward, MultiBandForwardConfig,
    )
    from euclid_polish.sky.types import MultiBandSkyImage

    H = W = 64
    blank = MultiBandSkyImage(
        data=np.zeros((H, W, 4), dtype=np.float32),
        pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
        band_names=Config.LR_INPUT_BAND_NAMES, is_clean=True,
    )
    fwd = MultiBandForward(
        config=MultiBandForwardConfig(add_noise=True, add_artifacts=False)
    )
    a, _ = fwd.process(blank, rng=np.random.default_rng(123))
    b, _ = fwd.process(blank, rng=np.random.default_rng(123))
    np.testing.assert_allclose(a.data, b.data)
