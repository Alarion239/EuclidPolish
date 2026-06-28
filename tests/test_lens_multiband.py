"""Tests for the multi-band lens renderer (the optimised fast path)."""

from __future__ import annotations

import numpy as np
import pytest

from euclid_polish.config import Config
from tests._tiny_catalog import TinyCosmosCatalog
from euclid_polish.sky.generation.lens_population import (
    LensPopulation,
    render_lens_to_canvas,
    render_lens_to_multiband_canvas,
)


@pytest.fixture(scope="module")
def stub_population():
    cat = TinyCosmosCatalog(n_galaxies=5_000, seed=0)
    return LensPopulation(cat)


def test_multiband_render_adds_flux_to_every_channel(stub_population):
    rng = np.random.default_rng(0)
    lp = stub_population.sample(rng)
    H = W = 128
    canvas = np.zeros((H, W, Config.NUM_LR_CHANNELS), dtype=np.float32)
    render_lens_to_multiband_canvas(canvas, params=lp)
    for k in range(Config.NUM_LR_CHANNELS):
        assert canvas[..., k].sum() > 0


def test_multiband_render_agrees_with_per_band_loop(stub_population):
    """Multi-band fast path must give the same image as the single-band
    loop, channel by channel.
    """
    rng = np.random.default_rng(0)
    lp = stub_population.sample(rng)
    H = W = 96

    fast = np.zeros((H, W, Config.NUM_LR_CHANNELS), dtype=np.float32)
    render_lens_to_multiband_canvas(fast, params=lp)

    slow = np.zeros((H, W, Config.NUM_LR_CHANNELS), dtype=np.float32)
    for k in range(Config.NUM_LR_CHANNELS):
        render_lens_to_canvas(slow[..., k], params=lp, band_index=k)

    # The single-band wrapper internally calls the multi-band path, so
    # results should be identical to float32 precision.
    np.testing.assert_allclose(fast, slow, rtol=1e-5, atol=1e-3)


def test_multiband_render_is_additive(stub_population):
    rng = np.random.default_rng(0)
    lp = stub_population.sample(rng)
    H = W = 96
    a = np.zeros((H, W, Config.NUM_LR_CHANNELS), dtype=np.float32)
    b = np.zeros((H, W, Config.NUM_LR_CHANNELS), dtype=np.float32)
    render_lens_to_multiband_canvas(a, params=lp)
    render_lens_to_multiband_canvas(b, params=lp)
    render_lens_to_multiband_canvas(b, params=lp)
    np.testing.assert_allclose(b.sum(), 2 * a.sum(), rtol=1e-4)


def test_multiband_render_extended_ring_structure(stub_population):
    """A lensed system should put flux at radii near θ_E in all channels."""
    rng = np.random.default_rng(7)
    lp = stub_population.sample(rng)
    H = W = 256
    canvas = np.zeros((H, W, Config.NUM_LR_CHANNELS), dtype=np.float32)
    render_lens_to_multiband_canvas(canvas, params=lp)

    yy, xx = np.indices((H, W)) - np.array([H // 2, W // 2])[:, None, None]
    r_arcsec = np.sqrt(yy**2 + xx**2) * Config.DEFAULT_PIXEL_SCALE
    ring_mask = (r_arcsec >= 0.8 * lp.theta_E_arcsec) & (r_arcsec <= 1.5 * lp.theta_E_arcsec)
    for k in range(Config.NUM_LR_CHANNELS):
        assert canvas[..., k][ring_mask].sum() > 0
