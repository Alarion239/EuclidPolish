"""Smoke tests for the star-anchor record generator. The real star cutouts
live on FASRC, so these drive the generator's helpers against synthetic
FITS to pin the WCS → pixel → single-pixel-delta placement and the
ADU/s→e⁻ conversion."""

import importlib
import os

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from euclid_polish.config import Config
from euclid_polish.euclid.photometry import ab_mag_to_electrons

gen = importlib.import_module("scripts.fasrc_generate_star_anchor_tfrecords")


def _tan_header(naxis: int, ra: float, dec: float, scale_arcsec: float = 0.10):
    w = WCS(naxis=2)
    w.wcs.crpix = [naxis / 2 + 0.5, naxis / 2 + 0.5]   # 1-based centre
    w.wcs.crval = [ra, dec]
    w.wcs.cdelt = [-scale_arcsec / 3600.0, scale_arcsec / 3600.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    h = w.to_header()
    h["MAGZERO"] = Config.BAND_VIS.sim_zeropoint_e   # factor 1 → cube == input
    return h


def test_make_anchor_example_places_single_delta():
    stamp = 64
    cube = np.zeros((128, 128, 4), dtype=np.float32)
    star_xy = (64.3, 63.7)                            # near centre, sub-pixel
    flux = 5.0e6
    lr, hr = gen.make_anchor_example(cube, star_xy, flux, stamp)
    assert lr.shape == (stamp, stamp, 4)
    assert hr.shape == (gen.SCALE * stamp, gen.SCALE * stamp, 1)
    # Exactly one non-zero pixel, carrying the full flux.
    nz = np.argwhere(hr[..., 0] > 0)
    assert nz.shape == (1, 1 * 2) and len(nz) == 1
    assert hr[..., 0].sum() == pytest.approx(flux)
    # Pixel = round(2 * star-in-stamp). Star is centred → stamp offset 32.
    sx_s, sy_s = 64.3 - 32, 63.7 - 32
    assert tuple(nz[0]) == (round(gen.SCALE * sy_s), round(gen.SCALE * sx_s))


def test_make_anchor_example_rejects_edge_star():
    cube = np.zeros((64, 64, 4), dtype=np.float32)
    assert gen.make_anchor_example(cube, (10, 10), 1.0, 128) is None  # stamp > cutout


def test_star_pixel_and_load_cube_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "STAR_CUTOUTS_ROOT", str(tmp_path))
    naxis, ra, dec = 64, 150.0, 2.0
    for band in Config.LR_INPUT_BAND_NAMES:
        d = os.path.join(str(tmp_path), band)
        os.makedirs(d, exist_ok=True)
        data = np.zeros((naxis, naxis), dtype=np.float32)
        data[31, 33] = 1234.0                         # arbitrary content
        hdr = _tan_header(naxis, ra, dec)
        fits.PrimaryHDU(data=data, header=hdr).writeto(
            os.path.join(d, f"star_0000_{naxis}.fits"), overwrite=True)

    loaded = gen.load_star_cube(0, naxis)
    assert loaded is not None
    cube, vis_header = loaded
    assert cube.shape == (naxis, naxis, 4)
    # MAGZERO == stack ZP ⇒ conversion factor 1 ⇒ values unchanged.
    assert cube[31, 33, 0] == pytest.approx(1234.0)

    # The star sits at the WCS reference (cutout centre).
    sx, sy = gen.star_pixel(vis_header, ra, dec)
    assert sx == pytest.approx(naxis / 2 - 0.5, abs=1e-3)
    assert sy == pytest.approx(naxis / 2 - 0.5, abs=1e-3)


def test_anchor_flux_matches_catalog_magnitude():
    # The delta value is the catalog VIS flux in electrons.
    mag = 17.0
    flux = ab_mag_to_electrons(mag, Config.BAND_VIS)
    _, hr = gen.make_anchor_example(
        np.zeros((128, 128, 4), np.float32), (64.0, 64.0), flux, 64)
    assert hr[..., 0].max() == pytest.approx(flux)
