"""PSF / PSFSet carry a provenance stamp through their FITS headers, and the
net-new FITS-card helper round-trips a stamp.
"""

from __future__ import annotations

import numpy as np

from euclid_polish.provenance.ids import ProvId
from euclid_polish.provenance.records import Stamp


def _gauss(n=9):
    x = np.arange(n) - n // 2
    g = np.exp(-(x[:, None] ** 2 + x[None, :] ** 2) / 4.0).astype(np.float32)
    return g / g.sum()


# --------------------------------------------------------------------------- #
# FITS card helper (net-new)
# --------------------------------------------------------------------------- #

def test_fits_stamp_cards_round_trip():
    from astropy.io import fits
    from euclid_polish.provenance.fits import read_stamp_cards, write_stamp_cards
    hdr = fits.Header()
    stamp = Stamp(id=ProvId("4b1e7a90"), produced_by=ProvId("7f3a9c21"),
                  parents=(ProvId("2b8e44d1"),), schema_version=3)
    write_stamp_cards(hdr, stamp)
    assert read_stamp_cards(hdr) == stamp


def test_fits_no_cards_reads_none():
    from astropy.io import fits
    from euclid_polish.provenance.fits import read_stamp_cards
    assert read_stamp_cards(fits.Header()) is None


# --------------------------------------------------------------------------- #
# PSF / PSFSet
# --------------------------------------------------------------------------- #

def test_psf_stamp_survives_fits_round_trip(tmp_path):
    from euclid_polish.psf.core import PSF
    psf = PSF(data=_gauss(), pixel_scale=0.05, fwhm_arcsec=0.16)
    stamp = Stamp(id=ProvId("a1b2c3d4"), produced_by=ProvId("0f0f0f0f"))
    psf = psf.with_stamp(stamp)
    path = psf.save(str(tmp_path), "vis_epsf.fits")
    back = PSF.from_fits(path)
    assert back.prov_stamp() == stamp


def test_psf_without_stamp_still_loads(tmp_path):
    from euclid_polish.psf.core import PSF
    psf = PSF(data=_gauss(), pixel_scale=0.05)
    path = psf.save(str(tmp_path), "vis_epsf.fits")
    back = PSF.from_fits(path)
    assert back.prov_stamp() is None


def test_psfset_stamp_survives_fits_round_trip(tmp_path):
    from euclid_polish.psf.core import PSF
    from euclid_polish.psf.psf_set import PSFSet
    pset = PSFSet.from_psfs([PSF(data=_gauss(), pixel_scale=0.05, fwhm_arcsec=0.16)])
    stamp = Stamp(id=ProvId("cafebabe"), produced_by=ProvId("12341234"))
    pset = pset.with_stamp(stamp)
    path = pset.save(str(tmp_path), "vis_psfset.fits")
    back = PSFSet.from_fits(path)
    assert back.prov_stamp() == stamp
