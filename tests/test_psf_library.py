"""Tests for the per-band PSF loader."""

from __future__ import annotations

import os

import numpy as np
import pytest

from euclid_polish.config import Config
from euclid_polish.euclid.psf_library import (
    load_all_band_psf_sets,
    load_all_band_psfs,
    load_band_psf,
    load_band_psf_set,
    psf_inventory,
    psf_path_for_band,
)
from euclid_polish.euclid.types import PSF
from euclid_polish.psf import PSFSet


def test_psf_path_for_band_uses_config_filename(tmp_path):
    path = psf_path_for_band(Config.BAND_VIS, psf_dir=str(tmp_path))
    assert path == os.path.join(str(tmp_path), Config.BAND_VIS.psf_fits_filename)


def test_load_band_psf_falls_back_to_gaussian(tmp_path):
    psf = load_band_psf(
        Config.BAND_Y_E, target_pixel_scale=Config.DEFAULT_PIXEL_SCALE,
        psf_dir=str(tmp_path), require_empirical=False,
    )
    assert isinstance(psf, PSF)
    assert psf.pixel_scale == pytest.approx(Config.DEFAULT_PIXEL_SCALE)
    assert psf.data.sum() == pytest.approx(1.0, rel=1e-6)


def test_load_band_psf_require_empirical_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_band_psf(
            Config.BAND_Y_E, psf_dir=str(tmp_path), require_empirical=True,
        )


def test_load_band_psf_reads_empirical_when_present(tmp_path):
    """Save a Gaussian as an "empirical" PSF, then load it back."""
    # Synthesise a Gaussian PSF at NISP native scale (0.30") to simulate an
    # empirical kernel and save it under the canonical filename for Y_E.
    from euclid_polish.euclid.psf_library import _gaussian_psf
    src = _gaussian_psf(0.45, 0.30, size=63)
    saved_path = src.save(str(tmp_path), filename=Config.BAND_Y_E.psf_fits_filename)
    assert os.path.exists(saved_path)

    # Now load at the HR pixel scale (0.05") — implicit resampling kicks in.
    psf = load_band_psf(
        Config.BAND_Y_E, target_pixel_scale=Config.DEFAULT_PIXEL_SCALE,
        psf_dir=str(tmp_path), require_empirical=True,
    )
    assert psf.pixel_scale == pytest.approx(Config.DEFAULT_PIXEL_SCALE)
    assert psf.data.sum() == pytest.approx(1.0, rel=1e-3)


def test_load_all_band_psfs_returns_one_per_band(tmp_path):
    psfs = load_all_band_psfs(psf_dir=str(tmp_path))
    assert set(psfs.keys()) == {b.name for b in Config.BANDS}
    for name, psf in psfs.items():
        assert psf.data.sum() == pytest.approx(1.0, rel=1e-6)
        # All at the HR grid by default
        assert psf.pixel_scale == pytest.approx(Config.DEFAULT_PIXEL_SCALE)


def test_psf_inventory_reports_missing_bands(tmp_path):
    inv = psf_inventory(psf_dir=str(tmp_path))
    assert set(inv.keys()) == {b.name for b in Config.BANDS}
    for v in inv.values():
        assert v is None  # nothing on disk in the temp dir


def test_psf_inventory_reports_present_band(tmp_path):
    from euclid_polish.euclid.psf_library import _gaussian_psf
    src = _gaussian_psf(0.16, 0.05, size=31)
    src.save(str(tmp_path), filename=Config.BAND_VIS.psf_fits_filename)
    inv = psf_inventory(psf_dir=str(tmp_path))
    assert inv["VIS"] is not None
    assert inv["Y_E"] is None


# ---------------------------------------------------------------------------
# PSF *sets* (position-dependent ensembles)
# ---------------------------------------------------------------------------

def _save_band_psf_set(tmp_path, band, n, *, scale=0.05):
    from euclid_polish.euclid.psf_library import _gaussian_psf
    members = [_gaussian_psf(0.10 + 0.02 * i, scale, size=31) for i in range(n)]
    pset = PSFSet.from_psfs(members)
    return pset.save(str(tmp_path), filename=band.psf_fits_filename)


def test_load_band_psf_set_falls_back_to_one_element(tmp_path):
    pset = load_band_psf_set(Config.BAND_Y_E, psf_dir=str(tmp_path))
    assert isinstance(pset, PSFSet)
    assert pset.n == 1                         # Gaussian fallback → single
    assert pset.pixel_scale == pytest.approx(Config.DEFAULT_PIXEL_SCALE)


def test_load_band_psf_set_reads_all_clusters(tmp_path):
    _save_band_psf_set(tmp_path, Config.BAND_VIS, n=4)
    pset = load_band_psf_set(
        Config.BAND_VIS, target_pixel_scale=Config.DEFAULT_PIXEL_SCALE,
        psf_dir=str(tmp_path),
    )
    assert pset.n == 4
    for p in pset.psfs:
        assert p.pixel_scale == pytest.approx(Config.DEFAULT_PIXEL_SCALE)
        assert p.data.sum() == pytest.approx(1.0, rel=1e-3)


def test_load_band_psf_still_reads_mean_from_set_file(tmp_path):
    """Back-compat: the single-PSF loader keeps working on a multi-PSF file —
    it reads HDU[0] (the mean)."""
    _save_band_psf_set(tmp_path, Config.BAND_VIS, n=3)
    psf = load_band_psf(Config.BAND_VIS, psf_dir=str(tmp_path))
    assert isinstance(psf, PSF)
    assert psf.data.sum() == pytest.approx(1.0, rel=1e-3)


def test_load_all_band_psf_sets_returns_one_set_per_band(tmp_path):
    _save_band_psf_set(tmp_path, Config.BAND_VIS, n=3)
    sets = load_all_band_psf_sets(psf_dir=str(tmp_path))
    assert set(sets.keys()) == {b.name for b in Config.BANDS}
    assert sets["VIS"].n == 3
    assert sets["Y_E"].n == 1                  # fallback


def test_load_band_psf_set_require_empirical_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_band_psf_set(
            Config.BAND_Y_E, psf_dir=str(tmp_path), require_empirical=True,
        )
