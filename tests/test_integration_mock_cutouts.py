"""End-to-end integration tests with mock cutouts for every band.

These tests synthesise small star-cutout FITS files for VIS + the three
NISP bands at their native pixel scales, then push them through the
full downstream chain:

    mock cutouts on disk
        → CatalogObject flags per (band, size)
        → PSFExtractor builds the ePSF (per-band oversampling)
        → ePSF saved at 0.05"/pix
        → load_band_psf reads the FITS back
        → ObservationSimulator convolves a synthetic clean HR field

If any link in this chain breaks (file layout, catalog schema, PSF
oversampling, scale resampling, kernel sizing), one of these tests
fails with a precise message about which stage is wrong.

The mock cutouts use a Gaussian star — the photutils EPSFBuilder
converges in ~3 iterations and the resulting kernel is a Gaussian of
the same FWHM, which is easy to verify.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from euclid_polish.config import BandConfig, Config
from euclid_polish.catalog.catalog_object import CatalogObject, summarize
from euclid_polish.catalog.downloader import DownloadConfig
from euclid_polish.psf.psf_extractor import (
    PSFExtractionConfig, PSFExtractor,
)
from euclid_polish.psf.psf_library import (
    load_all_band_psfs, load_band_psf, psf_path_for_band,
)
from euclid_polish.psf import PSF
from euclid_polish.sky.observation.observation_simulator import (
    ObservationSimulator, ObservationSimulatorConfig,
)
from euclid_polish.image import Image


# ---------------------------------------------------------------------------
# Mock-cutout factory
# ---------------------------------------------------------------------------

def _gaussian_2d(
    shape: tuple[int, int], fwhm_pix: float,
    *, x0: float | None = None, y0: float | None = None,
    total_counts: float = 1.0e5,
) -> np.ndarray:
    """Render a 2-D Gaussian star centred on (x0, y0) with the given FWHM."""
    H, W = shape
    if x0 is None: x0 = (W - 1) / 2.0
    if y0 is None: y0 = (H - 1) / 2.0
    sigma = fwhm_pix / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    yy, xx = np.indices(shape, dtype=np.float64)
    g = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2.0 * sigma * sigma))
    g /= g.sum()
    return (g * total_counts).astype(np.float64)


def _write_mock_cutout(
    path: str,
    *,
    band: BandConfig,
    cutout_size_pix: int,
    ra_deg: float = 150.0,
    dec_deg: float = 2.5,
    star_pix_offset: tuple[float, float] = (0.0, 0.0),
    total_counts: float = 1.0e5,
) -> None:
    """Write a single mock star cutout FITS with sensible WCS + a Gaussian PSF.

    The Gaussian FWHM matches ``band.psf_fwhm_arcsec / band.pixel_scale_lr_arcsec``
    so the cutout looks like a real star at the band's native resolution.
    The header carries CRVAL1/CRVAL2 so the downloader's WCS check and
    the existing-cutouts scanner both accept the file.
    """
    fwhm_pix = band.psf_fwhm_arcsec / band.pixel_scale_lr_arcsec
    cy = (cutout_size_pix - 1) / 2.0 + star_pix_offset[1]
    cx = (cutout_size_pix - 1) / 2.0 + star_pix_offset[0]
    img = _gaussian_2d(
        (cutout_size_pix, cutout_size_pix), fwhm_pix=fwhm_pix,
        x0=cx, y0=cy, total_counts=total_counts,
    )
    hdu = fits.PrimaryHDU(data=img.astype(np.float32))
    h = hdu.header
    h["CRVAL1"]  = ra_deg
    h["CRVAL2"]  = dec_deg
    h["CRPIX1"]  = (cutout_size_pix + 1) / 2.0
    h["CRPIX2"]  = (cutout_size_pix + 1) / 2.0
    h["CDELT1"]  = -band.pixel_scale_lr_arcsec / 3600.0   # degrees / pix
    h["CDELT2"]  =  band.pixel_scale_lr_arcsec / 3600.0
    h["CTYPE1"]  = "RA---TAN"
    h["CTYPE2"]  = "DEC--TAN"
    h["BAND"]    = band.name
    os.makedirs(os.path.dirname(path), exist_ok=True)
    hdu.writeto(path, overwrite=True)


def _write_cutout_set(
    output_dir: str,
    band: BandConfig,
    *,
    cutout_size_pix: int,
    n_stars: int = 10,
    base_ra: float = 150.0,
    base_dec: float = 2.5,
) -> list[str]:
    """Write ``n_stars`` cutouts of ``band`` into ``output_dir`` and
    return their file paths in star-id order.
    """
    paths = []
    for star_id in range(n_stars):
        # Small jitter in RA/Dec keeps the scanner's dedup logic happy
        # without breaking the WCS-based position match.
        ra  = base_ra  + 0.001 * star_id
        dec = base_dec + 0.001 * star_id
        path = os.path.join(
            output_dir, f"star_{star_id:04d}_{cutout_size_pix}.fits",
        )
        _write_mock_cutout(
            path, band=band, cutout_size_pix=cutout_size_pix,
            ra_deg=ra, dec_deg=dec,
        )
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Fixtures: a stars.csv + per-band cutout subdirs under one tmp_path root
# ---------------------------------------------------------------------------

CUTOUT_SIZE_VIS = 65   # 65 VIS px = 6.5" — small enough for a fast test


@pytest.fixture
def stars_root(tmp_path: Path) -> Path:
    """A tmp_path/euclid_stars with per-band cutout subdirs + a stars.csv."""
    root = tmp_path / "euclid_stars"
    cutouts_root = root / "cutouts"
    cutouts_root.mkdir(parents=True)
    return root


@pytest.fixture
def four_band_cutouts(stars_root: Path) -> dict[str, list[str]]:
    """Write 8 mock cutouts per band at the VIS-equivalent angular extent.

    Returns ``{band_name: [paths]}``. Cutouts go into
    ``stars_root/cutouts/<band>/``.
    """
    arcsec_side = CUTOUT_SIZE_VIS * Config.BAND_VIS.pixel_scale_lr_arcsec
    paths_by_band: dict[str, list[str]] = {}
    cutouts_root = str(stars_root / "cutouts")
    for band in Config.BANDS:
        native_size = band.cutout_size_for_arcsec(arcsec_side)
        band_dir = Config.cutout_dir_for_band(band.name, root=cutouts_root)
        paths = _write_cutout_set(
            band_dir, band, cutout_size_pix=native_size, n_stars=8,
        )
        paths_by_band[band.name] = paths
    return paths_by_band


# ---------------------------------------------------------------------------
# Stage 1: mock cutouts exist in the right per-band layout
# ---------------------------------------------------------------------------

def test_mock_cutouts_land_in_per_band_subdirs(stars_root, four_band_cutouts):
    """Every band has cutouts under its own ``cutouts/<band>/`` subdir."""
    for band in Config.BANDS:
        d = Config.cutout_dir_for_band(band.name,
                                       root=str(stars_root / "cutouts"))
        assert os.path.isdir(d), f"missing dir {d}"
        files = sorted(os.listdir(d))
        assert len(files) == 8, f"{band.name}: expected 8 cutouts, got {len(files)}"


def test_per_band_native_sizes_match_angular_field(stars_root, four_band_cutouts):
    """Each band's cutout side covers the same arcsec patch of sky."""
    arcsec_side = CUTOUT_SIZE_VIS * Config.BAND_VIS.pixel_scale_lr_arcsec
    for band in Config.BANDS:
        expected = band.cutout_size_for_arcsec(arcsec_side)
        # Read one of the band's cutouts; its NAXIS should match.
        d = Config.cutout_dir_for_band(band.name,
                                       root=str(stars_root / "cutouts"))
        sample = sorted(os.listdir(d))[0]
        with fits.open(os.path.join(d, sample)) as hdul:
            data_shape = hdul[0].data.shape
        assert data_shape == (expected, expected), (
            f"{band.name}: expected {expected}², got {data_shape}"
        )


# ---------------------------------------------------------------------------
# Stage 2: CatalogObject tracks per-band, per-size flags from mock downloads
# ---------------------------------------------------------------------------

def test_catalog_records_per_band_validity(stars_root, four_band_cutouts):
    """Mark each band's cutouts valid via CatalogObject and read them back."""
    arcsec_side = CUTOUT_SIZE_VIS * Config.BAND_VIS.pixel_scale_lr_arcsec
    path = os.path.join(str(stars_root), Config.CATALOG_FILE)
    objects = [CatalogObject(ra=150.0 + 0.001 * i, dec=2.5 + 0.001 * i, id=i)
               for i in range(8)]

    for band in Config.BANDS:
        size = band.cutout_size_for_arcsec(arcsec_side)
        for o in objects:
            o.set_valid(size, band=band.name)

    CatalogObject.write(objects, path)
    loaded = CatalogObject.read(path)
    for o in loaded:
        for band in Config.BANDS:
            size = band.cutout_size_for_arcsec(arcsec_side)
            assert o.is_valid(size, band=band.name), (
                f"star {o.id} not valid for {band.name} at size={size}"
            )

    summary = summarize(loaded)
    for band in Config.BANDS:
        assert summary["valid_by_band"][band.name] == 8


# ---------------------------------------------------------------------------
# Stage 3: PSF extraction for every band → ePSF lands at 0.05"/pix
# ---------------------------------------------------------------------------

@pytest.fixture
def extracted_psfs(tmp_path, four_band_cutouts) -> Path:
    """Run PSFExtractor for every band; return the ePSF output directory."""
    psf_dir = tmp_path / "psfs"
    psf_dir.mkdir()
    for band in Config.BANDS:
        d = Config.cutout_dir_for_band(
            band.name, root=str(tmp_path / "euclid_stars" / "cutouts"),
        )
        # Pick a psf_size that fits inside the cutout (and stays odd).
        # Each band's cutout is band.cutout_size_for_arcsec(arcsec_side) px.
        files = sorted(os.listdir(d))
        with fits.open(os.path.join(d, files[0])) as hdul:
            cutout_side = hdul[0].data.shape[0]
        # Largest odd value ≤ cutout_side - 2
        psf_size = (cutout_side - 1) if (cutout_side - 1) % 2 == 1 else (cutout_side - 2)
        cfg = PSFExtractionConfig(
            psf_size=psf_size,
            oversampling=band.epsf_oversampling,
            max_iters=3,             # mock star converges fast
            progress_bar=False,
        )
        extractor = PSFExtractor(cfg)
        cutout_files = extractor.get_cutout_files(d, cutout_size=cutout_side)
        assert cutout_files, f"no cutouts visible for {band.name} at size {cutout_side}"
        extractor.build_epsf(cutout_files)
        psf = extractor.to_psf(band.epsf_pixel_scale_arcsec)
        psf.save(str(psf_dir), filename=band.psf_fits_filename)
    return psf_dir


def test_extracted_psfs_all_land_at_target_resolution(extracted_psfs):
    """Every band's saved ePSF must report pixel_scale ≈ 0.05"/pix."""
    for band in Config.BANDS:
        p = PSF.from_fits(psf_path_for_band(band, str(extracted_psfs)))
        assert p.pixel_scale == pytest.approx(0.05, abs=1e-4), (
            f"{band.name}: pixel_scale = {p.pixel_scale}"
        )


def test_extracted_psfs_are_normalised(extracted_psfs):
    for band in Config.BANDS:
        p = PSF.from_fits(psf_path_for_band(band, str(extracted_psfs)))
        assert p.data.sum() == pytest.approx(1.0, rel=1e-3)


def test_extracted_psfs_peak_at_centre(extracted_psfs):
    """The synthetic Gaussian star → ePSF peaked at the centre pixel."""
    for band in Config.BANDS:
        p = PSF.from_fits(psf_path_for_band(band, str(extracted_psfs)))
        H, W = p.data.shape
        peak_yx = np.unravel_index(p.data.argmax(), p.data.shape)
        # photutils centres on the geometric centre; allow ±1 pixel.
        assert abs(peak_yx[0] - H // 2) <= 1
        assert abs(peak_yx[1] - W // 2) <= 1


def test_extracted_psfs_cover_same_angular_field(extracted_psfs):
    """Cutouts cover the same arcsec patch at 0.05"/pix → ePSF sides match.

    This is the *correct* invariant once the user picks a single
    cutout_size_vis_pixels: each band's native cutout is sized to the
    same angular field, and saving at 0.05"/pix (via the band-specific
    oversampling) gives all four ePSFs the same dimensions.
    """
    sides = {}
    for band in Config.BANDS:
        p = PSF.from_fits(psf_path_for_band(band, str(extracted_psfs)))
        sides[band.name] = p.data.shape
    unique_sides = {s for s in sides.values()}
    assert len(unique_sides) == 1, (
        f"VIS-equivalent cutouts produced inconsistent ePSF shapes: {sides}"
    )
    side = next(iter(unique_sides))[0]
    # All saved at 0.05"/pix → arcsec extent = side × 0.05
    arcsec_extent = side * 0.05
    expected = CUTOUT_SIZE_VIS * Config.BAND_VIS.pixel_scale_lr_arcsec
    # ±1 native-pixel rounding from the cutout_size_for_arcsec helper.
    assert abs(arcsec_extent - expected) <= 1.0, (
        f"ePSF side {side} px → {arcsec_extent}\" not within 1\" of {expected}\""
    )


# ---------------------------------------------------------------------------
# Stage 4: PSF library loads every band → returns kernels at the HR grid
# ---------------------------------------------------------------------------

def test_load_all_band_psfs_after_extraction(extracted_psfs):
    """``load_all_band_psfs`` returns a normalised PSF at 0.05"/pix per band."""
    psfs = load_all_band_psfs(
        target_pixel_scale=Config.DEFAULT_PIXEL_SCALE,
        psf_dir=str(extracted_psfs),
        require_empirical=True,    # extraction succeeded → no fallback needed
    )
    assert set(psfs.keys()) == {b.name for b in Config.BANDS}
    for name, psf in psfs.items():
        assert psf.pixel_scale == pytest.approx(Config.DEFAULT_PIXEL_SCALE, abs=1e-4)
        assert psf.data.sum() == pytest.approx(1.0, rel=1e-3)


# ---------------------------------------------------------------------------
# Stage 5: ObservationSimulator consumes the per-band PSFs end-to-end
# ---------------------------------------------------------------------------

def test_forward_model_runs_with_extracted_psfs(extracted_psfs):
    """The forward model successfully consumes the per-band empirical PSFs."""
    psfs = load_all_band_psfs(
        target_pixel_scale=Config.DEFAULT_PIXEL_SCALE,
        psf_dir=str(extracted_psfs),
    )
    fwd = ObservationSimulator(
        psfs_by_band=psfs,
        config=ObservationSimulatorConfig(add_noise=False),
    )

    # Tiny synthetic HR scene: a single 1e6-electron source per band at centre.
    H = W = 96
    data = np.zeros((H, W, 4), dtype=np.float32)
    data[H // 2, W // 2, :] = 1.0e6
    hr = Image(
        data=data, pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
        band_names=Config.LR_INPUT_BAND_NAMES, is_clean=True,
    )

    lr, hr_target = fwd.process(hr, rng=np.random.default_rng(0))
    assert lr.shape[-1] == Config.NUM_LR_CHANNELS
    assert hr_target.shape[-1] == Config.NUM_HR_CHANNELS

    # Flux conservation (noise off) — each channel's total e⁻ should match
    # the HR input total, modulo the NISP Lanczos upsample which preserves
    # mean rather than sum.
    vis_in  = data[..., 0].sum()
    vis_out = lr.data[..., 0].sum()
    assert vis_out == pytest.approx(vis_in, rel=2e-2)


def test_forward_model_lr_grid_is_vis_aligned(extracted_psfs):
    """All four LR channels share the VIS LR grid (0.10"/pix) after forward."""
    psfs = load_all_band_psfs(
        target_pixel_scale=Config.DEFAULT_PIXEL_SCALE,
        psf_dir=str(extracted_psfs),
    )
    fwd = ObservationSimulator(psfs_by_band=psfs,
                           config=ObservationSimulatorConfig(add_noise=True))
    H = W = 96
    hr = Image(
        data=np.random.default_rng(0).normal(size=(H, W, 4)).astype(np.float32),
        pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
        band_names=Config.LR_INPUT_BAND_NAMES, is_clean=True,
    )
    lr, _ = fwd.process(hr, rng=np.random.default_rng(0))
    assert lr.pixel_scale_arcsec == pytest.approx(Config.BAND_VIS.pixel_scale_lr_arcsec)
    # All four channels are the same spatial size after NISP upsample.
    assert lr.data.shape[:2] == (H // 2, W // 2)


# ---------------------------------------------------------------------------
# End-to-end "no empirical PSFs" path: Gaussian fallback for every band
# ---------------------------------------------------------------------------

def test_forward_model_runs_with_only_gaussian_fallbacks(tmp_path):
    """Fresh empty PSF dir → every band uses a Gaussian fallback; pipeline runs."""
    psfs = load_all_band_psfs(
        target_pixel_scale=Config.DEFAULT_PIXEL_SCALE,
        psf_dir=str(tmp_path),
    )
    fwd = ObservationSimulator(psfs_by_band=psfs,
                           config=ObservationSimulatorConfig(add_noise=False))
    hr = Image(
        data=np.zeros((96, 96, 4), dtype=np.float32),
        pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
        band_names=Config.LR_INPUT_BAND_NAMES, is_clean=True,
    )
    lr, _ = fwd.process(hr, rng=np.random.default_rng(0))
    assert lr.shape == (48, 48, 4)


# ---------------------------------------------------------------------------
# Sanity: the DownloadConfig.for_band paths match the mock layout
# ---------------------------------------------------------------------------

def test_download_config_layout_matches_mock(stars_root, four_band_cutouts):
    """DownloadConfig + Config.cutout_dir_for_band point at the same
    directories where the mock cutouts live.
    """
    for band in Config.BANDS:
        dl = DownloadConfig.for_band(
            band.name, cutout_size_vis_pixels=CUTOUT_SIZE_VIS,
        )
        expected_dir = Config.cutout_dir_for_band(
            band.name, root=str(stars_root / "cutouts"),
        )
        assert os.path.isdir(expected_dir)
        # The download size in native pixels equals what we wrote on disk.
        files = sorted(os.listdir(expected_dir))
        with fits.open(os.path.join(expected_dir, files[0])) as hdul:
            native_size = hdul[0].data.shape[0]
        assert dl.cutout_size == native_size
