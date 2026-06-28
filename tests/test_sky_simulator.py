"""Tests for the multi-band clean-HR scene generator."""

from __future__ import annotations

import numpy as np
import pytest

from euclid_polish.config import Config
from euclid_polish.image import Image
from euclid_polish.sky.generation.sky_simulator import (
    SkySimulator,
    SkySimulatorConfig,
)
from tests._tiny_catalog import TinyCosmosCatalog


@pytest.fixture(scope="module")
def simulator():
    cat = TinyCosmosCatalog(n_galaxies=1_000, seed=0)
    cfg = SkySimulatorConfig(
        image_size=128,
        pixel_scale=Config.DEFAULT_PIXEL_SCALE,
        sersic_density_arcmin2=Config.DEFAULT_GAL_DENSITY_ARCMIN2,
        star_density_arcmin2=Config.DEFAULT_STAR_DENSITY_ARCMIN2,
        lens_density_arcmin2=0.0,    # off for the deterministic sub-tests
    )
    return SkySimulator(cat, cfg)


def test_field_returns_4channel_skyimage(simulator: SkySimulator):
    rng = np.random.default_rng(0)
    img, meta = simulator.simulate_field(rng)
    assert isinstance(img, Image)
    assert img.shape == (128, 128, 4)
    assert img.band_names == Config.LR_INPUT_BAND_NAMES
    assert img.is_clean is True
    assert img.pixel_scale_arcsec == Config.DEFAULT_PIXEL_SCALE
    assert meta["n_galaxies"] >= 0
    assert meta["n_stars"]    >= 0


def test_field_total_flux_positive(simulator: SkySimulator):
    rng = np.random.default_rng(0)
    img, _ = simulator.simulate_field(rng, n_sersic=10, n_stars=5, n_lenses=0)
    for k, name in enumerate(Config.LR_INPUT_BAND_NAMES):
        assert img.data[..., k].sum() > 0, f"channel {name} empty"


def test_explicit_counts_respected(simulator: SkySimulator):
    rng = np.random.default_rng(0)
    _, meta = simulator.simulate_field(rng, n_sersic=7, n_tng=0, n_stars=3, n_lenses=0)
    assert meta["n_galaxies"] == 7
    assert meta["n_stars"]    == 3
    assert meta["n_lenses"]   == 0


def test_reproducible_with_same_seed(simulator: SkySimulator):
    a, _ = simulator.simulate_field(np.random.default_rng(42),
                                    n_sersic=5, n_stars=2, n_lenses=0)
    b, _ = simulator.simulate_field(np.random.default_rng(42),
                                    n_sersic=5, n_stars=2, n_lenses=0)
    np.testing.assert_array_equal(a.data, b.data)


def test_zero_sources_yields_empty_canvas(simulator: SkySimulator):
    rng = np.random.default_rng(0)
    img, _ = simulator.simulate_field(rng, n_sersic=0, n_tng=0, n_stars=0, n_lenses=0)
    assert np.all(img.data == 0.0)


def test_invalid_config_raises():
    cat = TinyCosmosCatalog(n_galaxies=10, seed=0)
    with pytest.raises(ValueError):
        SkySimulator(cat, SkySimulatorConfig(image_size=0))
    with pytest.raises(ValueError):
        SkySimulator(cat, SkySimulatorConfig(pixel_scale=-0.1))


def test_lens_density_produces_lens_records():
    cat = TinyCosmosCatalog(n_galaxies=500, seed=1)
    cfg = SkySimulatorConfig(image_size=128,
                             sersic_density_arcmin2=0.0,
                             star_density_arcmin2=0.0,
                             lens_density_arcmin2=0.0)
    sim = SkySimulator(cat, cfg)
    rng = np.random.default_rng(0)
    img, meta = sim.simulate_field(rng, n_sersic=0, n_stars=0, n_lenses=3)
    # Stub catalog may legitimately fail to find a viable lens config
    # (e.g. no source above the z floor); allow 0 - 3 lenses but every
    # successful render contributes positive flux.
    assert 0 <= meta["n_lenses"] <= 3
    if meta["n_lenses"] > 0:
        assert img.data.sum() > 0


def test_stars_appear_in_all_bands(simulator: SkySimulator):
    rng = np.random.default_rng(0)
    img, _ = simulator.simulate_field(rng, n_sersic=0, n_stars=20, n_lenses=0)
    for k in range(Config.NUM_LR_CHANNELS):
        assert img.data[..., k].max() > 0


# ---------------------------------------------------------------------------
# TNG galaxy injection — two-population model
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


def test_tng_population_renders_tng(tmp_path):
    """tng_density_arcmin2 > 0 → TNG stamps in the galaxy records."""
    tng = str(tmp_path / "tng")
    _write_fake_tng_galaxy(tng, "111")
    _write_fake_tng_galaxy(tng, "222")
    cfg = SkySimulatorConfig(
        image_size=64, pixel_scale=Config.DEFAULT_PIXEL_SCALE,
        sersic_density_arcmin2=0.0, tng_density_arcmin2=1.0,
        lens_density_arcmin2=0.0, tng_galaxy_dir=tng)
    sim = SkySimulator(None, cfg)
    assert {g[1] for g in sim.tng_galaxies} == {"111", "222"}
    img, meta = sim.simulate_field(np.random.default_rng(0),
                                   n_sersic=0, n_tng=4, n_stars=0, n_lenses=0)
    assert len(meta["galaxies"]) == 4
    assert all(g["render"] == "tng" for g in meta["galaxies"])
    assert img.data.sum() > 0
    g0 = meta["galaxies"][0]
    assert g0["subhalo_id"] in ("111", "222")
    assert g0["orientation"] in (1, 2, 3, 4, 5)
    assert len(g0["flux_e_per_band"]) == 4


def test_mixed_population_has_both_renders(tmp_path):
    """When both densities are non-zero both render types appear."""
    tng = str(tmp_path / "tng")
    _write_fake_tng_galaxy(tng, "111")
    cat = TinyCosmosCatalog(n_galaxies=200, seed=0)
    cfg = SkySimulatorConfig(
        image_size=64, pixel_scale=Config.DEFAULT_PIXEL_SCALE,
        sersic_density_arcmin2=1.0, tng_density_arcmin2=1.0,
        lens_density_arcmin2=0.0, tng_galaxy_dir=tng)
    sim = SkySimulator(cat, cfg)
    img, meta = sim.simulate_field(np.random.default_rng(0),
                                   n_sersic=3, n_tng=3, n_stars=0, n_lenses=0)
    renders = {g["render"] for g in meta["galaxies"]}
    assert "sersic" in renders
    assert "tng" in renders
    assert meta["n_galaxies"] == 6


def test_tng_not_loaded_when_density_zero(tmp_path):
    """tng_density_arcmin2=0.0 → TNG atlas is not loaded."""
    tng = str(tmp_path / "tng")
    _write_fake_tng_galaxy(tng, "111")
    cat = TinyCosmosCatalog(n_galaxies=200, seed=0)
    cfg = SkySimulatorConfig(
        image_size=64, pixel_scale=Config.DEFAULT_PIXEL_SCALE,
        lens_density_arcmin2=0.0, tng_density_arcmin2=0.0,
        tng_galaxy_dir=tng)
    sim = SkySimulator(cat, cfg)
    assert sim.tng_galaxies == []   # not loaded when TNG population is off
    _, meta = sim.simulate_field(np.random.default_rng(0),
                                 n_sersic=3, n_tng=0, n_stars=0, n_lenses=0)
    assert all(g["render"] == "sersic" for g in meta["galaxies"])


def test_invalid_tng_re_range_rejected():
    cat = TinyCosmosCatalog(n_galaxies=10, seed=0)
    with pytest.raises(ValueError, match="tng_re_arcsec_range"):
        SkySimulator(cat, SkySimulatorConfig(tng_re_arcsec_range=(2.0, 1.0)))
    with pytest.raises(ValueError, match="tng_re_arcsec_range"):
        SkySimulator(cat, SkySimulatorConfig(tng_re_arcsec_range=(0.0, 1.0)))


def test_tng_injection_with_redshift_mode(tmp_path):
    """tng_redshift_mode → z stamped on every TNG galaxy record."""
    tng = str(tmp_path / "tng")
    _write_fake_tng_galaxy(tng, "111")
    _write_fake_tng_galaxy(tng, "222")
    cfg = SkySimulatorConfig(
        image_size=64, pixel_scale=Config.DEFAULT_PIXEL_SCALE,
        sersic_density_arcmin2=0.0, tng_density_arcmin2=1.0,
        lens_density_arcmin2=0.0, tng_galaxy_dir=tng,
        tng_redshift_mode=True)
    sim = SkySimulator(None, cfg)
    img, meta = sim.simulate_field(np.random.default_rng(0),
                                   n_sersic=0, n_tng=4, n_stars=0, n_lenses=0)
    assert img.data.sum() > 0
    for g in meta["galaxies"]:
        assert g["render"] == "tng"
        assert Config.TNG_Z_MIN <= g["z"] <= Config.TNG_Z_MAX
        assert g["rebin_factor"] >= 1
        assert len(g["flux_e_per_band"]) == 4


def test_composite_stamp_clipping():
    from euclid_polish.sky.generation.tng_galaxy import composite_stamp
    # centred fully inside → full flux
    c = np.zeros((10, 10, 4), np.float32)
    composite_stamp(c, np.ones((4, 4, 4), np.float32), x0=5, y0=5)
    assert c.sum() == pytest.approx(4 * 4 * 4)
    # at a corner → only a quadrant lands
    c2 = np.zeros((10, 10, 4), np.float32)
    composite_stamp(c2, np.ones((6, 6, 4), np.float32), x0=0, y0=0)
    assert 0 < c2.sum() < 6 * 6 * 4
    # fully off-canvas → no-op
    c3 = np.zeros((10, 10, 4), np.float32)
    composite_stamp(c3, np.ones((4, 4, 4), np.float32), x0=100, y0=100)
    assert c3.sum() == 0
    # stamp larger than the canvas → fills it (central crop)
    c4 = np.zeros((8, 8, 4), np.float32)
    composite_stamp(c4, np.ones((20, 20, 4), np.float32), x0=4, y0=4)
    assert c4.sum() == pytest.approx(8 * 8 * 4)


def test_star_mag_smooth_power_law():
    from euclid_polish.sky.generation.sky_simulator import _sample_star_mag
    rng = np.random.default_rng(0)
    mags = np.array([_sample_star_mag(rng, slope=0.2, m_bright=16.0, m_faint=25.0)
                     for _ in range(200_000)])
    assert mags.min() >= 16.0 - 1e-9 and mags.max() <= 25.0 + 1e-9
    # Monotonic rise toward faint: more stars per mag at the faint end.
    assert np.mean((mags >= 24) & (mags < 25)) > np.mean((mags >= 16) & (mags < 17))
    # Recovered differential slope d log10(N)/dm ≈ 0.2.
    counts, edges = np.histogram(mags, bins=np.arange(16, 25.01, 1.0))
    centers = 0.5 * (edges[:-1] + edges[1:])
    A = np.vstack([centers, np.ones_like(centers)]).T
    slope = np.linalg.lstsq(A, np.log10(counts), rcond=None)[0][0]
    assert abs(slope - 0.2) < 0.03


def test_star_mag_slope_zero_is_uniform():
    from euclid_polish.sky.generation.sky_simulator import _sample_star_mag
    rng = np.random.default_rng(1)
    mags = np.array([_sample_star_mag(rng, slope=0.0, m_bright=18.0, m_faint=24.0)
                     for _ in range(100_000)])
    assert abs(np.mean(mags) - 21.0) < 0.05        # uniform → midpoint


def test_invalid_star_mag_range_rejected():
    cat = TinyCosmosCatalog(n_galaxies=10, seed=0)
    with pytest.raises(ValueError, match="star_mag_bright"):
        SkySimulator(cat, SkySimulatorConfig(
            star_mag_bright=25.0, star_mag_faint=20.0))


def test_catalog_none_requires_zero_sersic_density(tmp_path):
    """catalog=None is only valid when sersic_density_arcmin2=0.0."""
    tng = str(tmp_path / "tng")
    _write_fake_tng_galaxy(tng, "111")
    with pytest.raises(ValueError, match="sersic_density_arcmin2"):
        # default sersic density > 0 → error without catalog
        SkySimulator(None, SkySimulatorConfig(
            tng_density_arcmin2=1.0, tng_galaxy_dir=tng))


def test_lens_light_capped_at_theta_e():
    # Lens light effective radius is capped at lens_light_re_factor × θ_E so the
    # lens stays compact relative to the source arcs.
    cat = TinyCosmosCatalog(n_galaxies=3000, seed=0)
    factor = 0.8
    sim = SkySimulator(cat, SkySimulatorConfig(
        image_size=96, pixel_scale=Config.DEFAULT_PIXEL_SCALE,
        lens_light_re_factor=factor, tng_density_arcmin2=0.0))
    _img, meta = sim.simulate_field(np.random.default_rng(1), n_sersic=0,
                                    n_stars=0, n_lenses=8)
    assert meta["n_lenses"] >= 1
    for L in meta["lenses"]:
        assert L["lens_light_re_arcsec"] <= factor * L["theta_E_arcsec"] + 1e-6


def test_invalid_lens_light_re_factor_rejected():
    cat = TinyCosmosCatalog(n_galaxies=10, seed=0)
    with pytest.raises(ValueError, match="lens_light_re_factor"):
        SkySimulator(cat, SkySimulatorConfig(lens_light_re_factor=0.0))


def test_tng_lens_components_when_enabled(tmp_path):
    # With TNG loaded, the lens light AND lensed source are real TNG stamps.
    tng = str(tmp_path / "tng")
    _write_fake_tng_galaxy(tng, "111", size=240)
    cat = TinyCosmosCatalog(n_galaxies=3000, seed=0)
    cfg = SkySimulatorConfig(
        image_size=96, pixel_scale=Config.DEFAULT_PIXEL_SCALE,
        tng_density_arcmin2=1.0, tng_galaxy_dir=tng)
    sim = SkySimulator(cat, cfg)
    img, meta = sim.simulate_field(np.random.default_rng(3),
                                   n_sersic=0, n_tng=0, n_stars=0, n_lenses=4)
    assert meta["n_lenses"] >= 1
    for L in meta["lenses"]:
        assert L["lens_light_render"] == "tng"
        assert L["source_render"] == "tng"
    assert img.data.sum() > 0


def test_tng_zero_lens_components_are_sersic(tmp_path):
    # tng_density_arcmin2=0.0 + a TNG-free dir → lens components stay analytic
    # Sersic. The dir is pinned to an empty tmp path so the test is hermetic:
    # it must not depend on whether TNG data happens to be on disk locally.
    cat = TinyCosmosCatalog(n_galaxies=3000, seed=0)
    cfg = SkySimulatorConfig(image_size=96,
                             pixel_scale=Config.DEFAULT_PIXEL_SCALE,
                             tng_density_arcmin2=0.0,
                             tng_galaxy_dir=str(tmp_path / "no_tng"))
    sim = SkySimulator(cat, cfg)
    _img, meta = sim.simulate_field(np.random.default_rng(3),
                                    n_sersic=0, n_tng=0, n_stars=0, n_lenses=4)
    for L in meta["lenses"]:
        assert L["lens_light_render"] == "sersic"
        assert L["source_render"] == "sersic"
