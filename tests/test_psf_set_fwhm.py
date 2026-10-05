"""An empirical :class:`PSFSet` exposes the FWHM of its field-mean kernel.

The mean (HDU0 on disk) is measured the way the extractor measures each
cluster kernel (radial-profile FWHM × pixel scale), saved as HDU0's ``FWHM``
card, and measured again on load — so a file written without the card works
too.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from euclid_polish.psf import PSF, PSFSet

_SCALE = 0.05


def _gauss(side: int, fwhm_pix: float) -> PSF:
    x = np.arange(side) - side // 2
    xx, yy = np.meshgrid(x, x)
    s = fwhm_pix / 2.355
    g = np.exp(-(xx * xx + yy * yy) / (2 * s * s)).astype(np.float32)
    return PSF(data=g / g.sum(), pixel_scale=_SCALE)


def _set() -> PSFSet:
    return PSFSet.from_psfs([_gauss(41, 4.0), _gauss(41, 4.0)], n_stars=[10, 20])


def test_mean_carries_a_measured_fwhm():
    mean = _set().mean()
    assert mean.fwhm_arcsec is not None
    assert mean.fwhm_arcsec == pytest.approx(mean.fwhm_pixels() * _SCALE)
    assert mean.fwhm_arcsec == pytest.approx(4.0 * _SCALE, rel=0.1)


def test_save_writes_the_mean_fwhm_card(tmp_path):
    pset = _set()
    path = pset.save(str(tmp_path), "euclid_psf_TEST.fits")
    with fits.open(path) as hdul:
        assert hdul[0].header["FWHM"] == pytest.approx(pset.mean().fwhm_arcsec, rel=1e-6)
    # The legacy single-PSF reader of HDU0 picks the card up.
    assert PSF.from_fits(path).fwhm_arcsec == pytest.approx(pset.mean().fwhm_arcsec, rel=1e-6)


def test_file_without_the_card_still_yields_a_mean_fwhm(tmp_path):
    path = _set().save(str(tmp_path), "euclid_psf_OLD.fits")
    with fits.open(path, mode="update") as hdul:
        del hdul[0].header["FWHM"]
    loaded = PSFSet.from_fits(path)
    assert loaded.mean().fwhm_arcsec == pytest.approx(4.0 * _SCALE, rel=0.1)


def test_measure_fwhm_is_none_when_unmeasurable():
    assert PSF(data=np.zeros((9, 9), np.float32), pixel_scale=_SCALE).measure_fwhm_arcsec() is None
    assert PSF(data=_gauss(21, 3.0).data, pixel_scale=0.0).measure_fwhm_arcsec() is None
