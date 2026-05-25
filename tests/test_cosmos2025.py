"""Tests for the COSMOS2025 catalog wrapper.

The real 10-GB FITS catalog is required for production. These tests
exercise the public API on (a) a tiny synthetic test-only catalog
(:class:`TinyCosmosCatalog`) and (b) the real catalog when the FITS
file is present (otherwise those tests are skipped).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from euclid_polish.config import Config
from euclid_polish.sky.cosmos2025 import (
    Cosmos2025Catalog,
    CosmosCatalog,
    GalaxyParams,
    open_cosmos2025,
)
from tests._tiny_catalog import TinyCosmosCatalog


# ---------------------------------------------------------------------------
# API surface against the tiny test-only catalog
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cat() -> TinyCosmosCatalog:
    return TinyCosmosCatalog(n_galaxies=2_000, seed=42)


def test_tiny_length_matches_request(cat: TinyCosmosCatalog):
    assert len(cat) == 2_000


def test_tiny_sample_galaxy_returns_params(cat: TinyCosmosCatalog):
    rng = np.random.default_rng(0)
    g = cat.sample_galaxy(rng)
    assert isinstance(g, GalaxyParams)
    assert len(g.bulge_flux_e) == Config.NUM_LR_CHANNELS
    assert len(g.disk_flux_e)  == Config.NUM_LR_CHANNELS


def test_tiny_galaxy_geometry_valid(cat: TinyCosmosCatalog):
    rng = np.random.default_rng(1)
    for _ in range(50):
        g = cat.sample_galaxy(rng)
        assert g.bulge_r_e_arcsec > 0
        assert g.disk_r_e_arcsec > 0
        assert 0.0 < g.bulge_axis_ratio <= 1.0
        assert 0.0 < g.disk_axis_ratio <= 1.0
        assert g.z_phot > 0
        for flux_tuple in (g.bulge_flux_e, g.disk_flux_e):
            for f in flux_tuple:
                assert np.isfinite(f)
                assert f >= 0


def test_tiny_lens_galaxy_in_redshift_range(cat: TinyCosmosCatalog):
    rng = np.random.default_rng(0)
    z_lo, z_hi = Config.LENS_Z_LENS_MIN, Config.LENS_Z_LENS_MAX
    for _ in range(20):
        g = cat.sample_lens_galaxy(rng, (z_lo, z_hi))
        assert z_lo <= g.z_phot <= z_hi


def test_tiny_source_galaxy_beyond_lens(cat: TinyCosmosCatalog):
    rng = np.random.default_rng(0)
    z_lens = 0.5
    for _ in range(20):
        s = cat.sample_source_galaxy(rng, z_lens)
        assert s.z_phot >= z_lens + Config.LENS_Z_SOURCE_OFFSET


def test_tiny_total_flux_matches_components(cat: TinyCosmosCatalog):
    rng = np.random.default_rng(0)
    g = cat.sample_galaxy(rng)
    for k in range(Config.NUM_LR_CHANNELS):
        assert g.total_flux_e(k) == pytest.approx(
            g.bulge_flux_e[k] + g.disk_flux_e[k]
        )


def test_typical_band_electron_ratios(cat: TinyCosmosCatalog):
    """``typical_band_electron_ratios`` returns a length-4 vector with
    VIS exactly 1.0 and NISP entries in a sensible range.

    Used by the HST→Euclid TFRecord generator to scale a single-band
    HST cutout into all four NISP channels via a per-pixel global
    colour. A regression here would silently shift NISP brightness
    by orders of magnitude in HST-derived training data.
    """
    ratios = cat.typical_band_electron_ratios()
    assert ratios.shape == (Config.NUM_LR_CHANNELS,)
    assert ratios.dtype == np.float32
    # VIS / VIS is exactly 1 by construction.
    assert ratios[0] == pytest.approx(1.0)
    # NISP/VIS ratios should be positive (no sign flip) and finite,
    # and within a few orders of magnitude of VIS (typical Euclid
    # galaxy colours give per-band stack electron counts inside ~0.01
    # to ~10× VIS even at the catalog tails).
    for k in (1, 2, 3):
        assert np.isfinite(ratios[k])
        assert ratios[k] > 0
        assert 0.001 < ratios[k] < 10.0, (
            f"band {k}: ratio {ratios[k]:.4g} outside plausible range"
        )


def test_tiny_reproducible_with_same_seed():
    a = TinyCosmosCatalog(n_galaxies=100, seed=7)
    b = TinyCosmosCatalog(n_galaxies=100, seed=7)
    np.testing.assert_array_equal(a.z_phot, b.z_phot)
    np.testing.assert_array_equal(a.bulge_flux_e, b.bulge_flux_e)


def test_tiny_empty_lens_range_raises():
    c = TinyCosmosCatalog(n_galaxies=10, seed=0)
    with pytest.raises(RuntimeError):
        c.sample_lens_galaxy(np.random.default_rng(0), (100.0, 200.0))


# ---------------------------------------------------------------------------
# open_cosmos2025 factory (no fallback — catalog file is mandatory)
# ---------------------------------------------------------------------------

def test_open_cosmos2025_raises_on_missing(tmp_path):
    bogus = str(tmp_path / "nonexistent.fits")
    with pytest.raises(FileNotFoundError, match="COSMOS2025 catalog not found"):
        open_cosmos2025(path=bogus)


def test_concrete_classes_implement_cosmoscatalog_protocol():
    assert issubclass(Cosmos2025Catalog, CosmosCatalog)
    assert issubclass(TinyCosmosCatalog, CosmosCatalog)


# ---------------------------------------------------------------------------
# Real catalog (skipped if the FITS file is missing — keeps CI fast)
# ---------------------------------------------------------------------------

_REAL_PATH = Config.COSMOS2025_CATALOG_PATH
_HAVE_REAL = os.path.isfile(_REAL_PATH)


@pytest.fixture(scope="module")
def real_catalog():
    if not _HAVE_REAL:
        pytest.skip(f"Real catalog not present at {_REAL_PATH}")
    return Cosmos2025Catalog(verbose=False)


@pytest.mark.skipif(not _HAVE_REAL, reason="real catalog FITS not on disk")
def test_real_catalog_nonempty(real_catalog: Cosmos2025Catalog):
    assert len(real_catalog) > 10_000   # > 10k galaxies after quality cuts


@pytest.mark.skipif(not _HAVE_REAL, reason="real catalog FITS not on disk")
def test_real_catalog_sample_galaxy(real_catalog: Cosmos2025Catalog):
    rng = np.random.default_rng(0)
    g = real_catalog.sample_galaxy(rng)
    assert g.catalog_id is not None
    assert g.ra_deg is not None
    assert 140.0 < g.ra_deg < 160.0       # COSMOS field
    assert -5.0  < g.dec_deg < 10.0
    for k in range(Config.NUM_LR_CHANNELS):
        assert g.bulge_flux_e[k] >= 0
        assert g.disk_flux_e[k]  >= 0


@pytest.mark.skipif(not _HAVE_REAL, reason="real catalog FITS not on disk")
def test_real_catalog_lens_z_range(real_catalog: Cosmos2025Catalog):
    rng = np.random.default_rng(0)
    g = real_catalog.sample_lens_galaxy(
        rng, (Config.LENS_Z_LENS_MIN, Config.LENS_Z_LENS_MAX)
    )
    assert Config.LENS_Z_LENS_MIN <= g.z_phot <= Config.LENS_Z_LENS_MAX


@pytest.mark.skipif(not _HAVE_REAL, reason="real catalog FITS not on disk")
def test_real_catalog_source_beyond_lens(real_catalog: Cosmos2025Catalog):
    rng = np.random.default_rng(0)
    s = real_catalog.sample_source_galaxy(rng, z_lens=0.5)
    assert s.z_phot >= 0.5 + Config.LENS_Z_SOURCE_OFFSET
