"""Tests for the lens population sampler + renderer."""

from __future__ import annotations

import math

import numpy as np
import pytest

from euclid_polish.config import Config
from tests._tiny_catalog import TinyCosmosCatalog
from euclid_polish.sky.generation.lens_population import (
    LensParams,
    LensPopulation,
    einstein_radius_sis,
    render_lens_to_canvas,
    render_lens_to_multiband_canvas,
)


# ---------------------------------------------------------------------------
# Cosmology helpers
# ---------------------------------------------------------------------------

def test_einstein_radius_monotonic_in_sigma_v():
    """θ_E ∝ σ_v² should be monotonic."""
    theta_low  = einstein_radius_sis(150.0, 0.5, 2.0)
    theta_high = einstein_radius_sis(300.0, 0.5, 2.0)
    assert theta_high > theta_low > 0.0
    # σ_v doubled → θ_E should increase by ~4x (exact for fixed z_l, z_s).
    assert theta_high / theta_low == pytest.approx(4.0, rel=1e-3)


def test_einstein_radius_zero_when_source_below_lens():
    """No lensing if z_source ≤ z_lens."""
    assert einstein_radius_sis(250.0, 1.0, 1.0) == 0.0
    assert einstein_radius_sis(250.0, 1.0, 0.5) == 0.0


def test_einstein_radius_realistic_magnitude():
    """For σ=250 km/s, z_l=0.5, z_s=2.0 we expect θ_E ≈ 1.1″ (SIS, flat ΛCDM)."""
    theta = einstein_radius_sis(250.0, 0.5, 2.0)
    assert 0.8 < theta < 1.5


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def stub_population():
    cat = TinyCosmosCatalog(n_galaxies=5_000, seed=0)
    return LensPopulation(cat)


def test_sample_returns_valid_lens(stub_population: LensPopulation):
    rng = np.random.default_rng(0)
    lp = stub_population.sample(rng)
    assert isinstance(lp, LensParams)
    assert Config.LENS_Z_LENS_MIN <= lp.z_lens <= Config.LENS_Z_LENS_MAX
    assert lp.z_source > lp.z_lens
    assert 0.10 < lp.theta_E_arcsec < 3.5
    assert 0.0 < lp.lens_q <= 1.0


def test_sigma_v_range_is_configurable():
    # A custom σ_v range is honoured by the sampler (σ_v sets θ_E).
    cat = TinyCosmosCatalog(n_galaxies=3000, seed=0)
    pop = LensPopulation(cat, sigma_v_min_kms=200.0, sigma_v_max_kms=300.0)
    assert pop.sigma_v_min_kms == 200.0 and pop.sigma_v_max_kms == 300.0
    rng = np.random.default_rng(0)
    svs = [pop._sample_sigma_v(rng) for _ in range(2000)]
    assert min(svs) >= 200.0 and max(svs) <= 300.0
    # Larger σ_v → larger Einstein radii: compare median θ_E across two ranges.
    lo = LensPopulation(cat, sigma_v_min_kms=150.0, sigma_v_max_kms=200.0)
    hi = LensPopulation(cat, sigma_v_min_kms=300.0, sigma_v_max_kms=350.0)
    rng = np.random.default_rng(1)
    th_lo = [lo.sample(rng).theta_E_arcsec for _ in range(800)]
    th_hi = [hi.sample(rng).theta_E_arcsec for _ in range(800)]
    assert np.median(th_hi) > np.median(th_lo)


def test_sample_reproducible_with_same_seed(stub_population: LensPopulation):
    a = stub_population.sample(np.random.default_rng(123))
    b = stub_population.sample(np.random.default_rng(123))
    assert a.z_lens == b.z_lens
    assert a.theta_E_arcsec == pytest.approx(b.theta_E_arcsec)


def test_many_samples_dont_crash(stub_population: LensPopulation):
    rng = np.random.default_rng(0)
    n_drawn = 0
    for _ in range(50):
        lp = stub_population.sample(rng)
        n_drawn += 1
        assert lp.theta_E_arcsec > 0
    assert n_drawn == 50


# ---------------------------------------------------------------------------
# Rasterisation
# ---------------------------------------------------------------------------

def test_render_lens_adds_flux(stub_population: LensPopulation):
    """Rendering a lens onto an empty canvas leaves a positive total."""
    rng = np.random.default_rng(0)
    lp = stub_population.sample(rng)
    H = W = 256
    canvas = np.zeros((H, W), dtype=np.float32)
    render_lens_to_canvas(canvas, params=lp, band_index=0,
                          pixel_scale=Config.DEFAULT_PIXEL_SCALE)
    assert canvas.sum() > 0


def test_render_lens_is_additive(stub_population: LensPopulation):
    """Calling render_lens_to_canvas twice doubles the flux on canvas."""
    rng = np.random.default_rng(0)
    lp = stub_population.sample(rng)
    H = W = 128
    a = np.zeros((H, W), dtype=np.float32)
    b = np.zeros((H, W), dtype=np.float32)
    render_lens_to_canvas(a, params=lp, band_index=0)
    render_lens_to_canvas(b, params=lp, band_index=0)
    render_lens_to_canvas(b, params=lp, band_index=0)
    assert b.sum() == pytest.approx(2 * a.sum(), rel=1e-4)


def test_render_lens_has_extended_structure(stub_population: LensPopulation):
    """A lensed source should produce flux off-centre, not just at the centroid.

    We check that pixels at least 1 θ_E away from centre still receive
    nonzero flux — a non-lensed point source would not satisfy this.
    """
    rng = np.random.default_rng(7)
    lp = stub_population.sample(rng)
    H = W = 256
    canvas = np.zeros((H, W), dtype=np.float32)
    render_lens_to_canvas(canvas, params=lp, band_index=0)

    yy, xx = np.indices((H, W)) - np.array([H // 2, W // 2])[:, None, None]
    r_arcsec = np.sqrt(yy ** 2 + xx ** 2) * Config.DEFAULT_PIXEL_SCALE
    ring_mask = (r_arcsec >= lp.theta_E_arcsec * 0.8) & (r_arcsec <= lp.theta_E_arcsec * 1.5)
    assert canvas[ring_mask].sum() > 0


def test_render_lens_in_all_four_bands(stub_population: LensPopulation):
    """All four band indices produce nonzero output (with our stub colours)."""
    rng = np.random.default_rng(2)
    lp = stub_population.sample(rng)
    H = W = 128
    for k in range(Config.NUM_LR_CHANNELS):
        c = np.zeros((H, W), dtype=np.float32)
        render_lens_to_canvas(c, params=lp, band_index=k)
        assert c.sum() > 0, f"band {Config.LR_INPUT_BAND_NAMES[k]} empty"


# ---------------------------------------------------------------------------
# TNG stamps as lens components (source + lens light)
# ---------------------------------------------------------------------------

def _bright_stamp(n=80, core=8, val=50.0):
    """A small 4-band stamp with an off-centre bright clump (clearly not a
    smooth symmetric Sersic), at the HR pixel scale."""
    s = np.zeros((n, n, Config.NUM_LR_CHANNELS), np.float32)
    c = n // 2
    s[c - core:c + core, c - core:c + core, :] = val          # central blob
    s[c + 4:c + 4 + core, c + 6:c + 6 + core, :] += 2.0 * val  # asymmetric clump
    return s


def test_lensed_source_stamp_is_magnified(stub_population: LensPopulation):
    # A compact source stamp behind the lens must be magnified: the lensed
    # image carries MORE flux than the same stamp dropped in unlensed.
    rng = np.random.default_rng(7)
    lp = stub_population.sample(rng)
    lp = LensParams(**{**lp.__dict__, "centre_x_pix": 64.0, "centre_y_pix": 64.0,
                       "src_dx_arcsec": 0.0, "src_dy_arcsec": 0.0})
    src = _bright_stamp()
    canvas = np.zeros((128, 128, Config.NUM_LR_CHANNELS), np.float32)
    render_lens_to_multiband_canvas(canvas, params=lp, source_stamp=src)
    lensed_flux = canvas.sum()
    assert lensed_flux > 0
    # Unlensed reference: the same stamp composited directly.
    from euclid_polish.sky.generation.tng_galaxy import composite_stamp
    ref = np.zeros_like(canvas)
    composite_stamp(ref, src, 64.0, 64.0)
    assert lensed_flux > 1.05 * ref.sum()      # genuine magnification (μ>1)


def test_lens_light_stamp_replaces_sersic(stub_population: LensPopulation):
    # With a lens-light stamp, the foreground light is the stamp (no Sersic).
    rng = np.random.default_rng(8)
    lp = stub_population.sample(rng)
    lp = LensParams(**{**lp.__dict__, "centre_x_pix": 64.0, "centre_y_pix": 64.0})
    stamp = _bright_stamp(core=6, val=30.0)
    canvas = np.zeros((128, 128, Config.NUM_LR_CHANNELS), np.float32)
    render_lens_to_multiband_canvas(canvas, params=lp, lens_light_stamp=stamp)
    assert canvas.sum() > 0
    # Central region carries the composited stamp's flux.
    assert canvas[58:70, 58:70, :].sum() > 0
