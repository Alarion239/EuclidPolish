"""Bounded-memory reads of one large 2-D FITS plane (``helpers/fits_plane``).

The NEXUS QDR mosaic is a 4.46 GB float32 plane shipped gzip-compressed;
reading it whole (and ``astype``-copying it) peaked near 9 GB per job. The
plane reader streams only the rows a crop needs and must give exactly the
pixels, header and WCS the whole-array path gave.
"""

from __future__ import annotations

import gzip
import shutil
import tracemalloc
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS

from euclid_polish.web.helpers import fits_plane


def _wcs() -> WCS:
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [1.0, 1.0]
    wcs.wcs.crval = [268.4625, 65.19917]
    wcs.wcs.cdelt = [-0.03 / 3600.0, 0.03 / 3600.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


def _plane(ny: int = 700, nx: int = 900) -> np.ndarray:
    data = np.random.default_rng(3).normal(size=(ny, nx)).astype(">f4")
    data[:40, :] = np.nan                      # a masked border, like the mosaic
    return data


def _write(tmp_path: Path, data: np.ndarray, *, gz: bool, extension: bool = False) -> Path:
    header = _wcs().to_header()
    header["BUNIT"] = "MJy/sr"
    if extension:
        hdul = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data, header=header, name="SCI")])
    else:
        hdul = fits.HDUList([fits.PrimaryHDU(data, header=header)])
    plain = tmp_path / ("mosaic_ext.fits" if extension else "mosaic.fits")
    hdul.writeto(plain, overwrite=True)
    if not gz:
        return plain
    packed = plain.with_name(plain.name + ".gz")
    with plain.open("rb") as source, gzip.open(packed, "wb") as target:
        shutil.copyfileobj(source, target)
    plain.unlink()
    return packed


@pytest.mark.parametrize("gz", [False, True])
def test_header_wcs_and_shape_match_astropy(tmp_path, gz):
    data = _plane()
    path = _write(tmp_path, data, gz=gz)
    with fits_plane.open_plane(path) as plane:
        assert plane.shape == data.shape
        assert plane.hdu_name == "PRIMARY"
        assert plane.header == fits.getheader(path, 0)
        assert plane.header["BUNIT"] == "MJy/sr"
        assert plane.wcs.wcs.compare(WCS(fits.getheader(path, 0)).celestial.wcs)


@pytest.mark.parametrize("gz", [False, True])
def test_slices_equal_the_whole_array_in_any_order(tmp_path, gz):
    data = _plane()
    path = _write(tmp_path, data, gz=gz)
    expected = data.astype(np.float32)
    with fits_plane.open_plane(path) as plane:
        for ys, xs in [(slice(500, 650), slice(10, 300)),      # forward
                       (slice(0, 50), slice(0, 900)),          # back to the top
                       (slice(640, 700), slice(850, 900)),     # the far corner
                       (slice(100, 101), slice(3, 4)),         # a single pixel
                       (slice(-60, None), slice(None, 20))]:   # numpy-style bounds
            got = plane[ys, xs]
            assert got.dtype == np.float32 and got.flags.c_contiguous
            np.testing.assert_array_equal(got, expected[ys, xs])
        np.testing.assert_array_equal(plane.rows(200, 260), data[200:260])


@pytest.mark.parametrize("gz", [False, True])
@pytest.mark.parametrize("pixel", [(450, 350), (5, 690)])      # interior, partial edge
def test_cutout2d_on_the_plane_is_identical_to_the_in_memory_cutout(tmp_path, gz, pixel):
    data = _plane()
    path = _write(tmp_path, data, gz=gz)
    # The whole-array path (``jwst_euclid._find_image``) took its WCS from the
    # written header, whose CDELT is rounded to the header's precision.
    wcs = WCS(fits.getheader(path, 0)).celestial
    position = wcs.pixel_to_world(*pixel)
    reference = Cutout2D(np.asarray(data, np.float32), position=position,
                         size=4 * u.arcsec, wcs=wcs, mode="partial", fill_value=np.nan)
    with fits_plane.open_plane(path) as plane:
        cutout = Cutout2D(plane, position=position, size=4 * u.arcsec, wcs=plane.wcs,
                          mode="partial", fill_value=np.nan)
    np.testing.assert_array_equal(cutout.data, reference.data)
    assert cutout.data.dtype == np.float32
    assert cutout.wcs.wcs.compare(reference.wcs.wcs)
    assert cutout.bbox_original == reference.bbox_original


def test_gzip_crop_never_holds_the_whole_plane(tmp_path, monkeypatch):
    data = np.random.default_rng(0).normal(size=(2000, 2000)).astype(">f4")   # 16 MB
    path = _write(tmp_path, data, gz=True)
    monkeypatch.setattr(fits_plane, "CHUNK_BYTES", 1 << 20)
    tracemalloc.start()
    try:
        with fits_plane.open_plane(path) as plane:
            crop = plane[1500:1700, 900:1100]
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    np.testing.assert_array_equal(crop, data[1500:1700, 900:1100].astype(np.float32))
    assert peak < data.nbytes / 4, f"peak {peak / 1e6:.1f} MB for a {data.nbytes / 1e6:.0f} MB plane"


def test_the_image_may_live_in_an_extension(tmp_path):
    data = _plane(120, 90)
    path = _write(tmp_path, data, gz=True, extension=True)
    with fits_plane.open_plane(path) as plane:
        assert plane.hdu_name == "SCI" and plane.shape == (120, 90)
        np.testing.assert_array_equal(plane[60:80, 10:20], data[60:80, 10:20].astype(np.float32))
        assert plane.wcs.has_celestial


def test_a_file_without_a_celestial_image_is_refused(tmp_path):
    path = tmp_path / "table.fits"
    fits.HDUList([fits.PrimaryHDU(np.zeros((4, 4), np.float32))]).writeto(path)
    with pytest.raises(ValueError, match="no 2-D image with celestial WCS"):
        fits_plane.open_plane(path)


def test_sky_cutout_through_skycoord_matches(tmp_path):
    data = _plane()
    path = _write(tmp_path, data, gz=True)
    coordinate = SkyCoord(ra=268.4625 - 0.002, dec=65.19917 + 0.004, unit="deg")
    reference = Cutout2D(np.asarray(data, np.float32), position=coordinate, size=3 * u.arcsec,
                         wcs=_wcs(), mode="partial", fill_value=np.nan)
    with fits_plane.open_plane(path) as plane:
        cutout = Cutout2D(plane, position=coordinate, size=3 * u.arcsec, wcs=plane.wcs,
                          mode="partial", fill_value=np.nan)
    np.testing.assert_array_equal(cutout.data, reference.data)
