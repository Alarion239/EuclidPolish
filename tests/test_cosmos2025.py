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

import time

from euclid_polish.config import Config
from euclid_polish.sky import cosmos2025 as P
from euclid_polish.sky.cosmos2025 import (
    Cosmos2025Catalog,
    CosmosCatalog,
    GalaxyParams,
    circularized_effective_radius_arcsec,
    ensure_prefiltered_catalog,
    open_cosmos2025,
)
from tests._tiny_catalog import TinyCosmosCatalog


def _tiny_real_catalog(n: int = 6) -> Cosmos2025Catalog:
    """A real Cosmos2025Catalog with hand-set filtered arrays (no FITS read),
    via ``__new__`` — just enough to exercise (de)serialisation + sampling."""
    rng = np.random.default_rng(0)
    obj = Cosmos2025Catalog.__new__(Cosmos2025Catalog)
    obj.catalog_id      = np.arange(n, dtype=np.int64)
    obj.ra_deg          = rng.uniform(150.0, 151.0, n)
    obj.dec_deg         = rng.uniform(2.0, 3.0, n)
    obj.z_phot          = rng.uniform(0.2, 2.0, n)
    obj.angle_rad       = rng.uniform(0.0, np.pi, n)
    obj.disk_re_arcsec  = rng.uniform(0.1, 0.5, n)
    obj.bulge_re_arcsec = rng.uniform(0.05, 0.3, n)
    obj.disk_q          = rng.uniform(0.3, 1.0, n)
    obj.bulge_q         = rng.uniform(0.3, 1.0, n)
    obj.bulge_flux_e    = rng.uniform(1.0, 100.0, (n, Config.NUM_LR_CHANNELS))
    obj.disk_flux_e     = rng.uniform(1.0, 100.0, (n, Config.NUM_LR_CHANNELS))
    obj.max_mag = 25.0
    obj.max_bd_chi2 = 10.0
    obj.path = "<in-memory>"
    return obj


# ---------------------------------------------------------------------------
# API surface against the tiny test-only catalog
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cat() -> TinyCosmosCatalog:
    return TinyCosmosCatalog(n_galaxies=2_000, seed=42)


def test_tiny_length_matches_request(cat: TinyCosmosCatalog):
    assert len(cat) == 2_000


def test_effective_radius_single_component_limit():
    # Disk-only (zero bulge flux) → combined R_e = circularized disk radius
    # = disk_re·√q.  (P(2, b1·R/re_circ)=0.5 ⇒ R=re_circ for n=1.)
    re_d = np.array([0.30, 0.50]); q_d = np.array([0.25, 1.0])
    out = circularized_effective_radius_arcsec(
        np.array([0.1, 0.1]), np.array([0.8, 0.8]), np.array([0.0, 0.0]),  # bulge flux 0
        re_d, q_d, np.array([10.0, 10.0]))
    assert np.allclose(out, re_d * np.sqrt(q_d), rtol=2e-3)


def test_effective_radius_combined_below_disk():
    # A bright compact bulge pulls the combined half-light radius below the disk.
    out = circularized_effective_radius_arcsec(
        np.array([0.05]), np.array([1.0]), np.array([100.0]),   # dominant bulge
        np.array([0.5]),  np.array([1.0]), np.array([10.0]))    # faint extended disk
    assert out[0] < 0.5            # well below the disk major-axis radius
    assert out[0] > 0.05 * 0.9     # but at least near the bulge scale


def test_catalog_effective_re_property(cat: TinyCosmosCatalog):
    re = cat.effective_re_arcsec
    assert re.shape == (len(cat),)
    assert np.all(np.isfinite(re)) and np.all(re > 0)
    assert cat.effective_re_arcsec is cat.effective_re_arcsec   # cached


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


# ---------------------------------------------------------------------------
# Pre-filtered .npz round-trip + cache (the worker fast-load path)
# ---------------------------------------------------------------------------

def test_filtered_npz_round_trip(tmp_path):
    cat = _tiny_real_catalog(7)
    p = str(tmp_path / "filtered.npz")
    cat.save_filtered_npz(p)
    back = Cosmos2025Catalog.from_filtered_npz(p, verbose=False)
    assert len(back) == 7
    for k in P._FILTERED_ARRAYS:
        assert np.allclose(getattr(back, k), getattr(cat, k))
    assert back.max_mag == 25.0 and back.max_bd_chi2 == 10.0
    # Full sampling API works off the reloaded arrays.
    g = back.sample_galaxy(np.random.default_rng(0))
    assert isinstance(g, GalaxyParams)
    assert np.isfinite(back.typical_band_electron_ratios()).all()


def test_open_cosmos2025_reads_npz(tmp_path):
    cat = _tiny_real_catalog(4)
    p = str(tmp_path / "filtered.npz")
    cat.save_filtered_npz(p)
    back = open_cosmos2025(path=p, verbose=False)
    assert isinstance(back, Cosmos2025Catalog) and len(back) == 4


def test_ensure_prefiltered_passthrough_npz(tmp_path):
    p = str(tmp_path / "already.npz")
    open(p, "w").close()
    assert ensure_prefiltered_catalog(p) == p           # no rebuild


def test_ensure_prefiltered_builds_caches_and_invalidates(tmp_path, monkeypatch):
    fits_path = str(tmp_path / "cosmos2025.fits")
    open(fits_path, "w").close()                         # dummy source (mtime only)
    real = _tiny_real_catalog(6)
    calls = {"n": 0}

    def fake_ctor(path, max_mag=25.0, max_bd_chi2=10.0, verbose=True):
        calls["n"] += 1
        return real
    monkeypatch.setattr(P, "Cosmos2025Catalog", fake_ctor)

    npz1 = ensure_prefiltered_catalog(fits_path, verbose=False)
    assert npz1.endswith(".npz") and os.path.isfile(npz1) and calls["n"] == 1
    with np.load(npz1) as d:
        assert int(d["catalog_id"].size) == 6 and "bulge_flux_e" in d

    # Fresh cache → no rebuild.
    assert ensure_prefiltered_catalog(fits_path, verbose=False) == npz1
    assert calls["n"] == 1

    # Source FITS newer than the cache → rebuild.
    os.utime(fits_path, (time.time() + 10, time.time() + 10))
    ensure_prefiltered_catalog(fits_path, verbose=False)
    assert calls["n"] == 2
