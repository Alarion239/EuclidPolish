"""Tests for :class:`euclid_polish.psf.PSFSet` — the position-dependent PSF
ensemble: mean, Dirichlet sampling, and the multi-extension FITS round-trip
(PrimaryHDU = mean so legacy single-PSF readers keep working)."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from euclid_polish.psf import PSF, PSFSet


def _delta(side: int, dy: int, dx: int, scale: float = 0.05) -> PSF:
    a = np.zeros((side, side), dtype=np.float32)
    a[side // 2 + dy, side // 2 + dx] = 1.0
    return PSF(data=a, pixel_scale=scale)


def _gauss(side: int, fwhm: float, scale: float = 0.05) -> PSF:
    x = np.arange(side) - side // 2
    X, Y = np.meshgrid(x, x)
    s = fwhm / 2.355
    g = np.exp(-(X * X + Y * Y) / (2 * s * s)).astype(np.float32)
    return PSF(data=g / g.sum(), pixel_scale=scale)


def test_from_psfs_normalises_and_keeps_metadata():
    pset = PSFSet.from_psfs([_delta(11, -1, 0), _delta(11, 1, 0)],
                            centroids=[(10.0, 2.0), (10.2, 2.0)])
    assert pset.n == 2
    assert pset.shape == (11, 11)
    assert pset.pixel_scale == pytest.approx(0.05)
    for p in pset.psfs:
        assert p.total_flux == pytest.approx(1.0)


def test_from_psfs_rejects_mismatched_grid():
    with pytest.raises(ValueError, match="pixel_scale"):
        PSFSet.from_psfs([_delta(11, 0, 0, 0.05), _delta(11, 0, 0, 0.10)])
    with pytest.raises(ValueError, match="shape"):
        PSFSet.from_psfs([_delta(11, 0, 0), _delta(13, 0, 0)])


def test_mean_is_unit_sum_average():
    pset = PSFSet.from_psfs([_delta(11, -2, 0), _delta(11, 2, 0)])
    m = pset.mean()
    assert m.total_flux == pytest.approx(1.0)
    # The mean of two disjoint deltas has 0.5 at each location.
    assert m.data[3, 5] == pytest.approx(0.5)
    assert m.data[7, 5] == pytest.approx(0.5)


def test_sample_is_convex_unit_sum_and_varies():
    pset = PSFSet.from_psfs([_gauss(31, 2.0), _gauss(31, 5.0)])
    rng = np.random.default_rng(0)
    s1 = pset.sample(rng)
    s2 = pset.sample(rng)
    assert s1.total_flux == pytest.approx(1.0, abs=1e-5)
    # Convex blend of two non-negative PSFs stays non-negative.
    assert float(s1.data.min()) >= -1e-7
    assert not np.allclose(s1.data, s2.data)


def test_sample_is_reproducible_under_seed():
    pset = PSFSet.from_psfs([_gauss(31, 2.0), _gauss(31, 5.0), _gauss(31, 8.0)])
    a = pset.sample(np.random.default_rng(42))
    b = pset.sample(np.random.default_rng(42))
    np.testing.assert_allclose(a.data, b.data)


def test_single_element_sample_is_the_member():
    pset = PSFSet.from_psfs([_gauss(21, 3.0)])
    assert pset.n == 1
    s = pset.sample(np.random.default_rng(1))
    np.testing.assert_allclose(s.data, pset.psfs[0].data)


def test_fits_roundtrip_primary_is_mean(tmp_path):
    pset = PSFSet.from_psfs([_gauss(21, 2.0), _gauss(21, 4.0), _gauss(21, 6.0)],
                            centroids=[(10.0, 2.0), (10.1, 2.1), (10.2, 2.2)])
    path = pset.save(str(tmp_path), "euclid_psf_TEST.fits")

    # Multi-extension: 1 primary + K image HDUs, NPSF header records K.
    with fits.open(path) as hdul:
        assert hdul[0].header["NPSF"] == 3
        data_hdus = [h for h in hdul if h.data is not None]
        assert len(data_hdus) == 4

    loaded = PSFSet.from_fits(path)
    assert loaded.n == 3
    assert loaded.centroids is not None
    for orig, got in zip(pset.psfs, loaded.psfs):
        np.testing.assert_allclose(orig.data, got.data, atol=1e-6)

    # HDU[0] is the mean → legacy PSF.from_fits gets a sensible single PSF.
    legacy = PSF.from_fits(path)
    np.testing.assert_allclose(legacy.data, pset.mean().data, atol=1e-6)


def test_from_fits_legacy_single_hdu_loads_as_one_element(tmp_path):
    # A pre-existing single-PSF FITS (only a PrimaryHDU) must load as a
    # 1-element set so the new loader works on both formats.
    legacy = _gauss(21, 3.0)
    path = legacy.save(str(tmp_path), "euclid_psf_LEGACY.fits")
    pset = PSFSet.from_fits(path)
    assert pset.n == 1
    np.testing.assert_allclose(pset.psfs[0].data, legacy.data, atol=1e-6)


def test_grid_ops_map_over_members():
    pset = PSFSet.from_psfs([_gauss(31, 3.0), _gauss(31, 5.0)])
    cropped = pset.centre_cropped_to(21)
    assert cropped.n == 2
    assert cropped.shape == (21, 21)
    for p in cropped.psfs:
        assert p.total_flux == pytest.approx(1.0)
