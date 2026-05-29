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

from euclid_polish.config import Config


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
        assert scale == Config.HST.FALLBACK_PIX_SCALE_ARCSEC


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


# ---------------------------------------------------------------------------
# Cache fast path — reuse per-star FITS from a prior run instead of re-scanning
# ---------------------------------------------------------------------------

def _write_synth_star_fits(
    path: str, *, side: int, pix_scale: float = 0.05, n_tiles_src: int = 7,
    star_idx: int = 0,
) -> None:
    """Write a synthetic star FITS matching the layout the script saves.

    Mirrors the header keys the slow path writes (PIXSCALE, NTILESRC,
    STARIDX) so the cache loader has the same provenance to read back.
    """
    rng = np.random.default_rng(seed=star_idx)
    sigma = 4.0 / 2.355
    yy, xx = np.mgrid[:side, :side]
    cy = cx = (side - 1) / 2.0
    data = (5.0 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2)
                          / (2.0 * sigma ** 2))).astype(np.float32)
    data += rng.normal(0.0, 0.001, size=(side, side)).astype(np.float32)
    hdu = fits.PrimaryHDU(data)
    h = hdu.header
    h["FILTER"]   = "F814W"
    h["INSTRUME"] = "ACS/WFC"
    h["PIXSCALE"] = pix_scale
    h["STARIDX"]  = star_idx
    h["NTILESRC"] = n_tiles_src
    hdu.writeto(path, overwrite=True)


class TestCachedStarLoader:
    """``_load_cached_star_stamps`` is the cache fast path for ePSF rebuilds.

    On FASRC the per-tile DAOStarFinder + extract_stars pass is what
    actually takes hours; loading cached stamps reduces a full re-extract
    to a few seconds of EPSFBuilder. These tests pin the loader's
    behaviour on the three states the production directory can be in:
    missing, populated, and partially corrupted."""

    _HALF_SIDE = 50              # → 101² stamps, tiny but exercises the
                                  # suffix-matching logic that filters on
                                  # ``_{full_side}.fits``
    _FULL_SIDE = 2 * _HALF_SIDE + 1

    def test_missing_dir_returns_empty(self, script_module, tmp_path):
        stamps, scale, n = script_module._load_cached_star_stamps(
            str(tmp_path / "does_not_exist"), half_side=self._HALF_SIDE,
        )
        assert stamps == []
        assert scale == Config.HST.FALLBACK_PIX_SCALE_ARCSEC
        assert n == 0

    def test_empty_dir_returns_empty(self, script_module, tmp_path):
        stamps, scale, n = script_module._load_cached_star_stamps(
            str(tmp_path), half_side=self._HALF_SIDE,
        )
        assert stamps == []
        assert scale == Config.HST.FALLBACK_PIX_SCALE_ARCSEC
        assert n == 0

    def test_loads_compatible_stamps_with_metadata(self, script_module, tmp_path):
        """Happy path: writes a few stamps then reads them back as EPSFStars.

        The pix-scale and n_tiles fields come from the first stamp's
        header; both must round-trip exactly so the ePSF FITS we write
        downstream keeps the right PIXSCALE / NTILES provenance.
        """
        for i in range(3):
            _write_synth_star_fits(
                str(tmp_path / f"star_{i:04d}_{self._FULL_SIDE}.fits"),
                side=self._FULL_SIDE, pix_scale=0.03, n_tiles_src=5,
                star_idx=i,
            )
        stamps, scale, n = script_module._load_cached_star_stamps(
            str(tmp_path), half_side=self._HALF_SIDE,
        )
        assert len(stamps) == 3
        assert scale == pytest.approx(0.03, abs=1e-9)
        assert n == 5
        # Each stamp must arrive as an EPSFStar of the right shape so
        # EPSFBuilder can consume it without further wrapping.
        from photutils.psf import EPSFStar
        for s in stamps:
            assert isinstance(s, EPSFStar)
            assert s.data.shape == (self._FULL_SIDE, self._FULL_SIDE)

    def test_skips_wrong_size_stamps(self, script_module, tmp_path):
        """A stale cache mixing sides must not poison the ePSF.

        Loader filters on the ``_{full_side}.fits`` suffix in the
        filename — anything else is silently ignored. Mixing sides would
        otherwise mean EPSFBuilder sees a shape mismatch mid-iteration.
        """
        # Correct size — should be picked up.
        _write_synth_star_fits(
            str(tmp_path / f"star_0000_{self._FULL_SIDE}.fits"),
            side=self._FULL_SIDE, star_idx=0,
        )
        # Different naming convention from an older script version —
        # written as a sibling file the loader must skip.
        _write_synth_star_fits(
            str(tmp_path / "star_0001_77.fits"),
            side=77, star_idx=1,
        )
        # Junk that happens to start with "star_" but isn't a stamp.
        (tmp_path / "star_notes.txt").write_text("ignore me")
        stamps, scale, n = script_module._load_cached_star_stamps(
            str(tmp_path), half_side=self._HALF_SIDE,
        )
        assert len(stamps) == 1
        assert stamps[0].data.shape == (self._FULL_SIDE, self._FULL_SIDE)


class TestCleanStarStamp:
    """Per-stamp acceptance cuts. Large stamps (e.g. 1023²) straddle tile
    seams / coverage holes and a coverage-biased background lets noise
    through; both must be rejected so they don't poison the ePSF. This
    pins the regression that produced "border" and "pure noise" cutouts.
    """

    @staticmethod
    def _star(side: int, amp: float, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        a = rng.normal(0.0, 1.0, (side, side)).astype(np.float32)
        y, x = np.mgrid[:side, :side]
        c = (side - 1) / 2.0
        a += (amp * np.exp(-(((x - c) ** 2 + (y - c) ** 2) / (2.0 * 3.0 ** 2)))
              ).astype(np.float32)
        return a

    def test_clean_star_accepted(self, script_module):
        assert script_module._is_clean_star_stamp(self._star(101, 200.0)) is True

    def test_pure_noise_rejected(self, script_module):
        rng = np.random.default_rng(1)
        noise = rng.normal(0.0, 1.0, (101, 101)).astype(np.float32)
        assert script_module._is_clean_star_stamp(noise) is False

    def test_coverage_hole_rejected(self, script_module):
        # A block of exact zeros = an uncovered tile seam / chip gap.
        s = self._star(101, 200.0)
        s[:, :40] = 0.0
        assert script_module._is_clean_star_stamp(s) is False

    def test_nan_patch_rejected(self, script_module):
        s = self._star(101, 200.0)
        s[10:20, 10:20] = np.nan
        assert script_module._is_clean_star_stamp(s) is False

    def test_off_center_brighter_neighbor_rejected(self, script_module):
        """Crowding: a neighbour brighter than the centred star pulls
        EPSFBuilder's recentring off-target — reject (the actual cause of
        the noisy/asymmetric ePSF at large half-side)."""
        s = self._star(201, 50.0)                      # centred star, peak ~50
        y, x = np.mgrid[:201, :201]
        s += (120.0 * np.exp(-(((x - 160) ** 2 + (y - 40) ** 2)
                               / (2.0 * 3.0 ** 2)))).astype(np.float32)  # brighter, off-centre
        assert script_module._is_clean_star_stamp(s) is False
