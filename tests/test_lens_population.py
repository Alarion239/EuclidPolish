"""Tests for the lens population sampler + renderer."""

from __future__ import annotations

import math

import numpy as np
import pytest

from euclid_polish.config import Config
from tests._tiny_catalog import TinyCosmosCatalog
from euclid_polish.sky.lens_population import (
    LensParams,
    LensPopulation,
    einstein_radius_sis,
    render_lens_to_canvas,
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
