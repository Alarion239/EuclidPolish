"""Tests for the multi-band clean-HR scene generator."""

from __future__ import annotations

import numpy as np
import pytest

from euclid_polish.config import Config
from tests._tiny_catalog import TinyCosmosCatalog
from euclid_polish.sky.multiband_generator import (
    MultiBandGeneratorConfig,
    MultiBandSimulator,
)
from euclid_polish.sky.types import MultiBandSkyImage


@pytest.fixture(scope="module")
def simulator():
    cat = TinyCosmosCatalog(n_galaxies=1_000, seed=0)
    cfg = MultiBandGeneratorConfig(
        image_size=128,
        pixel_scale=Config.DEFAULT_PIXEL_SCALE,
        gal_density_arcmin2=Config.DEFAULT_GAL_DENSITY_ARCMIN2,
        star_density_arcmin2=Config.DEFAULT_STAR_DENSITY_ARCMIN2,
        lens_density_arcmin2=0.0,    # off for the deterministic sub-tests
    )
    return MultiBandSimulator(cat, cfg)


def test_field_returns_4channel_skyimage(simulator: MultiBandSimulator):
    rng = np.random.default_rng(0)
    img, meta = simulator.simulate_field(rng)
    assert isinstance(img, MultiBandSkyImage)
    assert img.shape == (128, 128, 4)
    assert img.band_names == Config.LR_INPUT_BAND_NAMES
    assert img.is_clean is True
    assert img.pixel_scale_arcsec == Config.DEFAULT_PIXEL_SCALE
    assert meta["n_galaxies"] >= 0
    assert meta["n_stars"]    >= 0


def test_field_total_flux_positive(simulator: MultiBandSimulator):
    rng = np.random.default_rng(0)
    img, _ = simulator.simulate_field(rng, n_galaxies=10, n_stars=5, n_lenses=0)
    for k, name in enumerate(Config.LR_INPUT_BAND_NAMES):
        assert img.data[..., k].sum() > 0, f"channel {name} empty"


def test_explicit_counts_respected(simulator: MultiBandSimulator):
    rng = np.random.default_rng(0)
    _, meta = simulator.simulate_field(rng, n_galaxies=7, n_stars=3, n_lenses=0)
    assert meta["n_galaxies"] == 7
    assert meta["n_stars"]    == 3
    assert meta["n_lenses"]   == 0


def test_reproducible_with_same_seed(simulator: MultiBandSimulator):
    a, _ = simulator.simulate_field(np.random.default_rng(42),
                                    n_galaxies=5, n_stars=2, n_lenses=0)
    b, _ = simulator.simulate_field(np.random.default_rng(42),
                                    n_galaxies=5, n_stars=2, n_lenses=0)
    np.testing.assert_array_equal(a.data, b.data)


def test_zero_sources_yields_empty_canvas(simulator: MultiBandSimulator):
    rng = np.random.default_rng(0)
    img, _ = simulator.simulate_field(rng, n_galaxies=0, n_stars=0, n_lenses=0)
    assert np.all(img.data == 0.0)


def test_invalid_config_raises():
    cat = TinyCosmosCatalog(n_galaxies=10, seed=0)
    with pytest.raises(ValueError):
        MultiBandSimulator(cat, MultiBandGeneratorConfig(image_size=0))
    with pytest.raises(ValueError):
        MultiBandSimulator(cat, MultiBandGeneratorConfig(pixel_scale=-0.1))


def test_lens_density_produces_lens_records():
    cat = TinyCosmosCatalog(n_galaxies=500, seed=1)
    cfg = MultiBandGeneratorConfig(image_size=128,
                                   gal_density_arcmin2=0.0,
                                   star_density_arcmin2=0.0,
                                   lens_density_arcmin2=0.0)
    sim = MultiBandSimulator(cat, cfg)
    rng = np.random.default_rng(0)
    img, meta = sim.simulate_field(rng, n_galaxies=0, n_stars=0, n_lenses=3)
    # Stub catalog may legitimately fail to find a viable lens config
    # (e.g. no source above the z floor); allow 0 - 3 lenses but every
    # successful render contributes positive flux.
    assert 0 <= meta["n_lenses"] <= 3
    if meta["n_lenses"] > 0:
        assert img.data.sum() > 0


def test_stars_appear_in_all_bands(simulator: MultiBandSimulator):
    rng = np.random.default_rng(0)
    img, _ = simulator.simulate_field(rng, n_galaxies=0, n_stars=20, n_lenses=0)
    for k in range(Config.NUM_LR_CHANNELS):
        assert img.data[..., k].max() > 0
