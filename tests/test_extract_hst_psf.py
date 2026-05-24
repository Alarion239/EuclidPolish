"""Integration tests for ``scripts/fasrc_extract_hst_psf.py``.

The on-FASRC run that this script supports is expensive (sbatch, tile
downloads, etc.). The two tests here build a tiny synthetic HLSP-like
tile and walk it through the same DAOStarFinder → extract_stars →
EPSFBuilder chain, catching API breakage (e.g. the recent xcentroid vs.
x column-name issue) before submitting an sbatch.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest
from astropy.io import fits


_SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts", "fasrc_extract_hst_psf.py",
)


@pytest.fixture(scope="module")
def script_module():
    """Import the script as a module so we can call its helpers in-process."""
    spec = importlib.util.spec_from_file_location(
        "fasrc_extract_hst_psf", _SCRIPT_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_synth_tile(side: int = 800, n_stars: int = 4, seed: int = 0) -> np.ndarray:
    """Build a synthetic HLSP-shaped float32 tile with N bright Gaussians.

    Each star is a small Gaussian (FWHM 4 px) on top of a low-amplitude
    sky-noise background. Star peaks are well above the noise so
    DAOStarFinder reliably picks them up.
    """
    rng = np.random.default_rng(seed)
    sky = rng.normal(0.0, 0.001, size=(side, side)).astype(np.float32)
    sigma = 4.0 / 2.355
    border = 100
    centres = rng.integers(border, side - border, size=(n_stars, 2))
    yy, xx = np.mgrid[:side, :side]
    for (sy, sx) in centres:
        g = np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2.0 * sigma ** 2))
        sky += (5.0 * g).astype(np.float32)         # peak ~5 e/s vs σ=0.001
    return sky


def _write_synth_hlsp_tile(path: str, *, side: int = 800,
                            pix_scale_arcsec: float = 0.05) -> None:
    """Write a synthetic HLSP-shaped FITS at ``path`` with a real WCS."""
    data = _make_synth_tile(side=side, n_stars=4, seed=0)
    hdu = fits.PrimaryHDU(data)
    # Minimum WCS the pixel-scale reader can parse.
    hdu.header["CTYPE1"] = "RA---TAN"
    hdu.header["CTYPE2"] = "DEC--TAN"
    hdu.header["CRVAL1"] = 150.0
    hdu.header["CRVAL2"] = 2.0
    hdu.header["CRPIX1"] = side / 2.0
    hdu.header["CRPIX2"] = side / 2.0
    hdu.header["CDELT1"] = -pix_scale_arcsec / 3600.0
    hdu.header["CDELT2"] =  pix_scale_arcsec / 3600.0
    hdu.writeto(path, overwrite=True)


# ---------------------------------------------------------------------------
# Pixel scale parsing
# ---------------------------------------------------------------------------

class TestPixelScaleReader:

    def test_reads_cdelt(self, script_module, tmp_path):
        path = str(tmp_path / "fake.fits")
        _write_synth_hlsp_tile(path, pix_scale_arcsec=0.05)
        with fits.open(path) as hdul:
            scale = script_module._pixel_scale_from_header(hdul[0].header)
        assert scale == pytest.approx(0.05, abs=1e-6)

    def test_reads_30mas(self, script_module, tmp_path):
        path = str(tmp_path / "fake.fits")
        _write_synth_hlsp_tile(path, pix_scale_arcsec=0.03)
        with fits.open(path) as hdul:
            scale = script_module._pixel_scale_from_header(hdul[0].header)
        assert scale == pytest.approx(0.03, abs=1e-6)

    def test_reads_cd_matrix(self, script_module):
        """Falls through CDELT-absent header to the CD matrix instead."""
        h = fits.Header()
        h["CD1_1"] = -1.3889e-05
        h["CD1_2"] =  0.0
        scale = script_module._pixel_scale_from_header(h)
        assert scale == pytest.approx(0.05, abs=1e-4)

    def test_falls_back_when_no_wcs(self, script_module):
        h = fits.Header()
        scale = script_module._pixel_scale_from_header(h)
        assert scale == script_module.FALLBACK_PIX_SCALE_ARCSEC


# ---------------------------------------------------------------------------
# The bug that crashed the sbatch — column rename + extract_stars roundtrip
# ---------------------------------------------------------------------------

class TestExtractStarsColumnNames:
    """Regression for the photutils extract_stars 'x/y vs xcentroid/ycentroid'
    error that crashed a real FASRC run.

    The misleading photutils error message ("When inputting multiple
    catalogs, each one must have a 'x' and 'y' column") triggers even
    for a single input — we need x/y columns whether we pass one table
    or many.
    """

    # Synthetic tiles must be ≥ 2·(PSF_HALF_SIDE_PIX + 5) per axis or the
    # find-stars edge-border filter rejects every detection. The script
    # uses PSF_HALF_SIDE_PIX=255 → border 260 → minimum useful tile ≈ 600.
    _TILE_SIDE = 800

    def test_find_stars_renames_columns(self, script_module):
        data = _make_synth_tile(side=self._TILE_SIDE, n_stars=5, seed=1)
        sources = script_module._find_stars_in_tile(data, max_n=10)
        assert sources is not None
        assert len(sources) >= 1, "synthetic stars should be detectable"
        assert "x" in sources.colnames
        assert "y" in sources.colnames
        # Sanity: the renamed columns hold the right numeric values.
        assert (sources["x"] == sources["xcentroid"]).all()

    def test_extract_stamps_succeeds_with_renamed_columns(self, script_module):
        data = _make_synth_tile(side=self._TILE_SIDE, n_stars=5, seed=2)
        sources = script_module._find_stars_in_tile(data, max_n=10)
        assert sources is not None and len(sources) >= 1
        stamps = script_module._extract_stamps_from_tile(
            data, sources, half_side=50,
        )
        assert len(stamps) >= 1
        # Each stamp must have the requested side; 50 × 2 + 1 = 101.
        for s in stamps:
            assert s.data.shape == (101, 101)

    def test_extract_stamps_rejects_table_missing_xy(self, script_module):
        """If a future caller forgets the rename, the helper fails loud."""
        from astropy.table import Table
        sources = Table({
            "xcentroid": [100.0, 200.0],
            "ycentroid": [100.0, 200.0],
            "peak":      [1.0, 1.0],
        })
        data = _make_synth_tile(side=self._TILE_SIDE, n_stars=1, seed=0)
        with pytest.raises(ValueError, match="x.*y.*columns"):
            script_module._extract_stamps_from_tile(data, sources)
