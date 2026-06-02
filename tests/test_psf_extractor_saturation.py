"""Saturation filter for PSF extraction.

The Euclid MER pipeline zeros out detector pixels above the well
capacity (most apparent in NISP cutouts for bright targets). When those
"clipped cores" are fed to EPSFBuilder, the iterative shift-stack
averages a zero plateau into the centre, producing a PSF with a flat
core. The extractor's ``saturation_core_size`` filter drops any star
whose central N×N region contains any non-positive or NaN pixels.
"""

from __future__ import annotations

import numpy as np

from euclid_polish.euclid.psf_extractor import (
    PSFExtractionConfig, PSFExtractor,
)


def _make_clean_star(side: int = 33, peak: float = 1000.0) -> np.ndarray:
    """A simple Gaussian star centred in a square array."""
    y, x = np.mgrid[:side, :side]
    cy = cx = side // 2
    g = np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * 1.5 ** 2))
    return (peak * g).astype(np.float32)


def _make_clipped_star(side: int = 33, peak: float = 1000.0,
                       core: int = 5) -> np.ndarray:
    """Same star with the central ``core × core`` region zeroed out
    (the MER-clipping signature)."""
    img = _make_clean_star(side, peak)
    cy = cx = side // 2
    sh = core // 2
    img[cy - sh: cy + sh + 1, cx - sh: cx + sh + 1] = 0.0
    return img


def test_clean_star_passes_filter():
    cfg = PSFExtractionConfig(psf_size=15, saturation_core_size=7,
                              progress_bar=False)
    ext = PSFExtractor(cfg)
    star = ext.extract_psf_star_from_cutout(_make_clean_star())
    assert star is not None


def test_clipped_core_star_is_rejected():
    cfg = PSFExtractionConfig(psf_size=15, saturation_core_size=7,
                              progress_bar=False)
    ext = PSFExtractor(cfg)
    star = ext.extract_psf_star_from_cutout(_make_clipped_star(core=5))
    assert star is None


def test_nan_core_star_is_rejected():
    cfg = PSFExtractionConfig(psf_size=15, saturation_core_size=7,
                              progress_bar=False)
    img = _make_clean_star()
    cy = cx = img.shape[0] // 2
    img[cy, cx] = np.nan
    ext = PSFExtractor(cfg)
    assert ext.extract_psf_star_from_cutout(img) is None


def test_saturation_filter_disabled():
    """``saturation_core_size = 0`` turns the filter off; clipped stars
    are kept (callers may want pre-filtered data)."""
    cfg = PSFExtractionConfig(psf_size=15, saturation_core_size=0,
                              progress_bar=False)
    ext = PSFExtractor(cfg)
    assert ext.extract_psf_star_from_cutout(_make_clipped_star(core=5)) is not None


def test_filter_only_inspects_central_region():
    """Pixels OUTSIDE the configured core size do not trigger rejection
    even if they are zero — that is just background sky."""
    cfg = PSFExtractionConfig(psf_size=15, saturation_core_size=5,
                              progress_bar=False)
    img = _make_clean_star(side=33)
    # Zero a ring far from the centre — must not trigger rejection.
    img[0, 0] = 0.0
    img[-1, -1] = 0.0
    ext = PSFExtractor(cfg)
    assert ext.extract_psf_star_from_cutout(img) is not None


def test_rejection_counters_attribute_cause(tmp_path):
    """End-to-end: feed a mix of clean + clipped + un-loadable cutouts
    through ``extract_psf_stars_from_files`` and check the counters."""
    from astropy.io import fits
    files = []
    # 3 clean, 2 clipped, 1 broken file
    for i, arr in enumerate([
        _make_clean_star(), _make_clean_star(), _make_clean_star(),
        _make_clipped_star(), _make_clipped_star(),
    ]):
        path = tmp_path / f"star_{i:04d}_33.fits"
        fits.PrimaryHDU(arr).writeto(path, overwrite=True)
        files.append((i, str(path)))
    # Broken file → triggers load error
    broken = tmp_path / "star_9999_33.fits"
    broken.write_bytes(b"not a fits file at all")
    files.append((9999, str(broken)))

    cfg = PSFExtractionConfig(psf_size=15, saturation_core_size=7,
                              progress_bar=False)
    ext = PSFExtractor(cfg)
    stars = ext.extract_psf_stars_from_files(files)

    assert len(stars)                == 3
    assert ext.n_rejected_saturated  == 2
    assert ext.n_rejected_load       == 1
    assert ext.n_rejected_edge       == 0


def test_extract_accepted_stars_keeps_indices(tmp_path):
    """``extract_accepted_stars`` returns (id, EPSFStar) for the accepted
    stars only — the id (from the filename) lets the caller join each star
    to its catalog position for spatial clustering."""
    from astropy.io import fits
    files = []
    # ids 0,2,4 clean; ids 1,3 clipped (rejected).
    for i, arr in enumerate([
        _make_clean_star(), _make_clipped_star(), _make_clean_star(),
        _make_clipped_star(), _make_clean_star(),
    ]):
        path = tmp_path / f"star_{i:04d}_33.fits"
        fits.PrimaryHDU(arr).writeto(path, overwrite=True)
        files.append((i, str(path)))

    cfg = PSFExtractionConfig(psf_size=15, saturation_core_size=7,
                              progress_bar=False)
    ext = PSFExtractor(cfg)
    accepted = ext.extract_accepted_stars(files)

    assert [sid for sid, _ in accepted] == [0, 2, 4]
    assert ext.n_rejected_saturated == 2
    # The legacy id-free wrapper returns the same stars, sans ids.
    assert len(ext.extract_psf_stars_from_files(files)) == 3
