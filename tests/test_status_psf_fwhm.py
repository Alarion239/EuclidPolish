"""Synthetic › PSF's per-band "ePSF FWHM" comes from the cached file's HDU0:
its ``FWHM`` card, or a measurement of the mean kernel when the card is
absent."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from euclid_polish.psf import PSF, PSFSet
from euclid_polish.web.helpers.status import psf_file_summary


def _gauss(side: int, fwhm_pix: float) -> PSF:
    x = np.arange(side) - side // 2
    xx, yy = np.meshgrid(x, x)
    s = fwhm_pix / 2.355
    g = np.exp(-(xx * xx + yy * yy) / (2 * s * s)).astype(np.float32)
    return PSF(data=g / g.sum(), pixel_scale=0.05)


def _saved(tmp_path, name: str) -> tuple[str, float]:
    pset = PSFSet.from_psfs([_gauss(41, 4.0), _gauss(41, 4.0)])
    path = pset.save(str(tmp_path), name)
    return path, float(pset.mean().fwhm_pixels() * 0.05)


def test_summary_reports_the_mean_fwhm_card(tmp_path):
    path, expected = _saved(tmp_path, "euclid_psf_CARD.fits")
    assert psf_file_summary(path)["fwhm_arcsec"] == pytest.approx(expected, rel=1e-5)


def test_summary_measures_the_mean_when_the_card_is_absent(tmp_path):
    path, expected = _saved(tmp_path, "euclid_psf_NOCARD.fits")
    with fits.open(path, mode="update") as hdul:
        if "FWHM" in hdul[0].header:
            del hdul[0].header["FWHM"]
    assert psf_file_summary(path)["fwhm_arcsec"] == pytest.approx(expected, rel=1e-5)
