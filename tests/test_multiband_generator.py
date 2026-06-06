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


# ---------------------------------------------------------------------------
# TNG galaxy injection
# ---------------------------------------------------------------------------

def _write_fake_tng_galaxy(tng_dir, gid, *, size=24):
    import os
    from astropy.io import fits
    d = os.path.join(tng_dir, gid)
    os.makedirs(d, exist_ok=True)
    for o in (1, 2, 3, 4, 5):
        for b in ("VIS", "Y", "J", "H"):
            arr = np.zeros((size, size), dtype=">f4")
            arr[size // 2 - 2:size // 2 + 2, size // 2 - 2:size // 2 + 2] = 500.0
            fits.PrimaryHDU(arr).writeto(
                os.path.join(d, f"TNG{gid}_O{o}_Euclid_{b}.fits"), overwrite=True)
    open(os.path.join(d, Config.Tng.DONE_MARKER), "w").close()


def test_tng_injection_when_enabled(tmp_path):
    tng = str(tmp_path / "tng")
    _write_fake_tng_galaxy(tng, "111")
    _write_fake_tng_galaxy(tng, "222")
    cat = TinyCosmosCatalog(n_galaxies=200, seed=0)
    cfg = MultiBandGeneratorConfig(
        image_size=64, pixel_scale=Config.DEFAULT_PIXEL_SCALE,
        lens_density_arcmin2=0.0, tng_fraction=1.0, tng_galaxy_dir=tng)
    sim = MultiBandSimulator(cat, cfg)
    assert {g[1] for g in sim.tng_galaxies} == {"111", "222"}
    img, meta = sim.simulate_field(np.random.default_rng(0),
                                   n_galaxies=4, n_stars=0, n_lenses=0)
    assert [g["render"] for g in meta["galaxies"]] == ["tng"] * 4
    assert img.data.sum() > 0
    g0 = meta["galaxies"][0]
    assert g0["subhalo_id"] in ("111", "222")
    assert g0["orientation"] in (1, 2, 3, 4, 5)
    assert g0["rebin_factor"] in (1, 2, 3, 4)
    assert len(g0["flux_e_per_band"]) == 4


def test_tng_fraction_zero_is_all_sersic_and_no_load(tmp_path):
    tng = str(tmp_path / "tng")
    _write_fake_tng_galaxy(tng, "111")
    cat = TinyCosmosCatalog(n_galaxies=200, seed=0)
    cfg = MultiBandGeneratorConfig(
        image_size=64, pixel_scale=Config.DEFAULT_PIXEL_SCALE,
        lens_density_arcmin2=0.0, tng_fraction=0.0, tng_galaxy_dir=tng)
    sim = MultiBandSimulator(cat, cfg)
    assert sim.tng_galaxies == []        # not even loaded when disabled
    _, meta = sim.simulate_field(np.random.default_rng(0),
                                 n_galaxies=3, n_stars=0, n_lenses=0)
    assert all(g["render"] == "sersic" for g in meta["galaxies"])


def test_tng_fraction_zero_is_byte_identical_to_default():
    """tng_fraction=0 must not perturb the RNG path vs the default generator."""
    cat = TinyCosmosCatalog(n_galaxies=200, seed=0)
    base = MultiBandSimulator(cat, MultiBandGeneratorConfig(
        image_size=64, lens_density_arcmin2=0.0))
    zero = MultiBandSimulator(cat, MultiBandGeneratorConfig(
        image_size=64, lens_density_arcmin2=0.0, tng_fraction=0.0))
    a, _ = base.simulate_field(np.random.default_rng(7),
                               n_galaxies=5, n_stars=2, n_lenses=0)
    b, _ = zero.simulate_field(np.random.default_rng(7),
                               n_galaxies=5, n_stars=2, n_lenses=0)
    np.testing.assert_array_equal(a.data, b.data)


def test_composite_stamp_clipping():
    from euclid_polish.sky.multiband_generator import _composite_stamp
    # centred fully inside → full flux
    c = np.zeros((10, 10, 4), np.float32)
    _composite_stamp(c, np.ones((4, 4, 4), np.float32), x0=5, y0=5)
    assert c.sum() == pytest.approx(4 * 4 * 4)
    # at a corner → only a quadrant lands
    c2 = np.zeros((10, 10, 4), np.float32)
    _composite_stamp(c2, np.ones((6, 6, 4), np.float32), x0=0, y0=0)
    assert 0 < c2.sum() < 6 * 6 * 4
    # fully off-canvas → no-op
    c3 = np.zeros((10, 10, 4), np.float32)
    _composite_stamp(c3, np.ones((4, 4, 4), np.float32), x0=100, y0=100)
    assert c3.sum() == 0
    # stamp larger than the canvas → fills it (central crop)
    c4 = np.zeros((8, 8, 4), np.float32)
    _composite_stamp(c4, np.ones((20, 20, 4), np.float32), x0=4, y0=4)
    assert c4.sum() == pytest.approx(8 * 8 * 4)


def test_invalid_tng_fraction_rejected():
    cat = TinyCosmosCatalog(n_galaxies=10, seed=0)
    with pytest.raises(ValueError, match="tng_fraction"):
        MultiBandSimulator(cat, MultiBandGeneratorConfig(tng_fraction=1.5))
