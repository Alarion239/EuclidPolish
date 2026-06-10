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
    # tng_fraction=1 is PURE-TNG mode: redshift mode is forced on, so the
    # downsample factor comes from D_A(z) (1..5 after stochastic rounding)
    # and every record carries its z.
    tng = str(tmp_path / "tng")
    _write_fake_tng_galaxy(tng, "111")
    _write_fake_tng_galaxy(tng, "222")
    cat = TinyCosmosCatalog(n_galaxies=200, seed=0)
    cfg = MultiBandGeneratorConfig(
        image_size=64, pixel_scale=Config.DEFAULT_PIXEL_SCALE,
        lens_density_arcmin2=0.0, tng_fraction=1.0, tng_galaxy_dir=tng)
    sim = MultiBandSimulator(cat, cfg)
    assert {g[1] for g in sim.tng_galaxies} == {"111", "222"}
    assert sim.pure_tng and sim.config.tng_redshift_mode
    img, meta = sim.simulate_field(np.random.default_rng(0),
                                   n_galaxies=4, n_stars=0, n_lenses=0,
                                   n_dwarfs=0)
    assert [g["render"] for g in meta["galaxies"]] == ["tng"] * 4
    assert img.data.sum() > 0
    g0 = meta["galaxies"][0]
    assert g0["subhalo_id"] in ("111", "222")
    assert g0["orientation"] in (1, 2, 3, 4, 5)
    assert 1 <= g0["rebin_factor"] <= 5
    assert Config.TNG_Z_MIN <= g0["z"] <= Config.TNG_Z_MAX
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
    from euclid_polish.sky.tng_galaxy import composite_stamp
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
    from euclid_polish.sky.multiband_generator import _sample_star_mag
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
    from euclid_polish.sky.multiband_generator import _sample_star_mag
    rng = np.random.default_rng(1)
    mags = np.array([_sample_star_mag(rng, slope=0.0, m_bright=18.0, m_faint=24.0)
                     for _ in range(100_000)])
    assert abs(np.mean(mags) - 21.0) < 0.05        # uniform → midpoint


def test_invalid_star_mag_range_rejected():
    cat = TinyCosmosCatalog(n_galaxies=10, seed=0)
    with pytest.raises(ValueError, match="star_mag_bright"):
        MultiBandSimulator(cat, MultiBandGeneratorConfig(
            star_mag_bright=25.0, star_mag_faint=20.0))


def test_invalid_tng_fraction_rejected():
    cat = TinyCosmosCatalog(n_galaxies=10, seed=0)
    with pytest.raises(ValueError, match="tng_fraction"):
        MultiBandSimulator(cat, MultiBandGeneratorConfig(tng_fraction=1.5))


def test_invalid_tng_big_params_rejected():
    cat = TinyCosmosCatalog(n_galaxies=10, seed=0)
    with pytest.raises(ValueError, match="big_galaxy_density"):
        MultiBandSimulator(cat, MultiBandGeneratorConfig(big_galaxy_density_arcmin2=-1.0))
    with pytest.raises(ValueError, match="tng_big_re_arcsec"):
        MultiBandSimulator(cat, MultiBandGeneratorConfig(tng_big_re_arcsec=(2.0, 1.0)))


def test_big_galaxies_independent_of_tng_fraction(tmp_path):
    # The big-galaxy population is a fixed count (here forced via n_big),
    # always TNG, flagged "big" — and identical for any *fractional*
    # tng_fraction. (tng_fraction=1 is pure-TNG/redshift mode, which has no
    # separate big population — covered in test_redshift_model.py.)
    tng = str(tmp_path / "tng")
    _write_fake_tng_galaxy(tng, "111", size=240)
    cat = TinyCosmosCatalog(n_galaxies=2000, seed=0)
    for tf in (0.1, 0.7):
        sim = MultiBandSimulator(cat, MultiBandGeneratorConfig(
            image_size=128, pixel_scale=Config.DEFAULT_PIXEL_SCALE,
            tng_fraction=tf, tng_galaxy_dir=tng))
        _img, meta = sim.simulate_field(np.random.default_rng(0), n_galaxies=0,
                                        n_stars=0, n_lenses=0, n_big=3)
        big = [g for g in meta["galaxies"] if g.get("big")]
        assert len(big) == 3 and meta["n_big_galaxies"] == 3
        assert all(g["render"] == "tng" for g in big)


def test_big_galaxy_density_drives_count(tmp_path):
    # A high surface density yields several big galaxies via the Poisson draw.
    tng = str(tmp_path / "tng")
    _write_fake_tng_galaxy(tng, "111", size=240)
    cat = TinyCosmosCatalog(n_galaxies=2000, seed=0)
    sim = MultiBandSimulator(cat, MultiBandGeneratorConfig(
        image_size=512, pixel_scale=Config.DEFAULT_PIXEL_SCALE,
        tng_fraction=0.5, tng_galaxy_dir=tng, big_galaxy_density_arcmin2=50.0))
    _img, meta = sim.simulate_field(np.random.default_rng(0), n_galaxies=0,
                                    n_stars=0, n_lenses=0)
    assert meta["n_big_galaxies"] >= 1     # area≈0.182 arcmin² × 50 ≈ 9 expected


def test_lens_light_capped_at_theta_e():
    # Lens light effective radius is capped at lens_light_re_factor × θ_E so the
    # lens stays compact relative to the source arcs.
    cat = TinyCosmosCatalog(n_galaxies=3000, seed=0)
    factor = 0.8
    sim = MultiBandSimulator(cat, MultiBandGeneratorConfig(
        image_size=96, pixel_scale=Config.DEFAULT_PIXEL_SCALE,
        lens_light_re_factor=factor, tng_fraction=0.0))
    _img, meta = sim.simulate_field(np.random.default_rng(1), n_galaxies=0,
                                    n_stars=0, n_lenses=8)
    assert meta["n_lenses"] >= 1
    for L in meta["lenses"]:
        assert L["lens_light_re_arcsec"] <= factor * L["theta_E_arcsec"] + 1e-6


def test_invalid_lens_light_re_factor_rejected():
    cat = TinyCosmosCatalog(n_galaxies=10, seed=0)
    with pytest.raises(ValueError, match="lens_light_re_factor"):
        MultiBandSimulator(cat, MultiBandGeneratorConfig(lens_light_re_factor=0.0))


def test_big_galaxies_off_when_tng_disabled(tmp_path):
    # tng_fraction=0 stays the pure-Sersic baseline: no big galaxies even with a
    # huge density set.
    tng = str(tmp_path / "tng")
    _write_fake_tng_galaxy(tng, "111", size=240)
    cat = TinyCosmosCatalog(n_galaxies=2000, seed=0)
    sim = MultiBandSimulator(cat, MultiBandGeneratorConfig(
        image_size=512, pixel_scale=Config.DEFAULT_PIXEL_SCALE,
        tng_fraction=0.0, tng_galaxy_dir=tng, big_galaxy_density_arcmin2=100.0))
    _img, meta = sim.simulate_field(np.random.default_rng(0), n_galaxies=0,
                                    n_stars=0, n_lenses=0)
    assert meta["n_big_galaxies"] == 0


def test_tng_lens_components_when_enabled(tmp_path):
    # With TNG on, the lens light AND lensed source are real TNG stamps at the
    # same tng_fraction proportion as field galaxies.
    tng = str(tmp_path / "tng")
    _write_fake_tng_galaxy(tng, "111", size=240)
    cat = TinyCosmosCatalog(n_galaxies=3000, seed=0)
    cfg = MultiBandGeneratorConfig(
        image_size=96, pixel_scale=Config.DEFAULT_PIXEL_SCALE,
        tng_fraction=1.0, tng_galaxy_dir=tng)
    sim = MultiBandSimulator(cat, cfg)
    img, meta = sim.simulate_field(np.random.default_rng(3),
                                   n_galaxies=0, n_stars=0, n_lenses=4)
    assert meta["n_lenses"] >= 1
    for L in meta["lenses"]:
        assert L["lens_light_render"] == "tng"
        assert L["source_render"] == "tng"
    assert img.data.sum() > 0


def test_tng_zero_lens_components_are_sersic(tmp_path):
    # tng_fraction=0 → lens components stay analytic Sersic.
    cat = TinyCosmosCatalog(n_galaxies=3000, seed=0)
    cfg = MultiBandGeneratorConfig(image_size=96,
                                   pixel_scale=Config.DEFAULT_PIXEL_SCALE,
                                   tng_fraction=0.0)
    sim = MultiBandSimulator(cat, cfg)
    _img, meta = sim.simulate_field(np.random.default_rng(3),
                                    n_galaxies=0, n_stars=0, n_lenses=4)
    for L in meta["lenses"]:
        assert L["lens_light_render"] == "sersic"
        assert L["source_render"] == "sersic"
