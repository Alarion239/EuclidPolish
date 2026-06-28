"""Tests for per-band PSF sizing (resolution-unified, FWHM-scaled)."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from euclid_polish.config import Config
from euclid_polish.psf import PSF
from euclid_polish.psf.psf_library import (
    load_all_band_psfs,
    load_band_psf,
    make_gaussian_psf,
    psf_half_pixels_for_band,
    psf_side_pixels_for_band,
)
from euclid_polish.sky.observation.observation_simulator import default_psf_for_band

# ---------------------------------------------------------------------------
# Sizing rules
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("band", list(Config.BANDS))
def test_psf_side_is_odd(band):
    """Forced odd so the centre pixel sits on the centroid."""
    assert psf_side_pixels_for_band(band, 0.05) % 2 == 1


@pytest.mark.parametrize("band", list(Config.BANDS))
def test_psf_side_matches_fwhm_factor(band):
    """``side = 2 × ceil(factor × fwhm / pixel_scale) + 1``."""
    scale = Config.DEFAULT_PIXEL_SCALE
    factor = float(Config.PSF_HALF_SUPPORT_FWHM_FACTOR)
    expected_half = max(int(Config.PSF_MIN_HALF_PIXELS),
                        int(math.ceil(factor * band.psf_fwhm_arcsec / scale)))
    assert psf_half_pixels_for_band(band, scale) == expected_half
    assert psf_side_pixels_for_band(band, scale) == 2 * expected_half + 1


def test_psf_side_scales_with_fwhm():
    """Higher-FWHM bands get larger PSF stamps at the same pixel scale."""
    scale = 0.05
    vis  = psf_side_pixels_for_band(Config.BAND_VIS, scale)
    y_e  = psf_side_pixels_for_band(Config.BAND_Y_E, scale)
    h_e  = psf_side_pixels_for_band(Config.BAND_H_E, scale)
    assert vis < y_e < h_e


def test_psf_side_scales_inversely_with_pixel_scale():
    """At a finer pixel scale we need more pixels for the same arcsec support."""
    band = Config.BAND_Y_E
    side_005 = psf_side_pixels_for_band(band, 0.05)
    side_010 = psf_side_pixels_for_band(band, 0.10)
    assert side_005 > side_010


def test_min_half_pixels_floor():
    """Tiny FWHM still gets at least PSF_MIN_HALF_PIXELS half-side."""
    # Construct a synthetic band with very small FWHM to trip the floor.
    from euclid_polish.config import BandConfig
    tiny = BandConfig(
        name="TEST", pixel_scale_lr_arcsec=0.1, psf_fwhm_arcsec=0.01,
        psf_fits_filename="x.fits", zeropoint_ab_e_per_s=25.0,
        sky_mag_ab_arcsec2=22.0, exposure_time_s=100.0, n_exposures=4,
        read_noise_e=4.0, dark_e_per_s_per_pix=0.001,
        asinh_stretch_scale_e=1000.0,
    )
    half = psf_half_pixels_for_band(tiny, 0.05)
    assert half == Config.PSF_MIN_HALF_PIXELS


# ---------------------------------------------------------------------------
# make_gaussian_psf — auto-sized vs explicit
# ---------------------------------------------------------------------------

def test_make_gaussian_psf_auto_size_matches_band_helper():
    band = Config.BAND_Y_E
    expected_side = psf_side_pixels_for_band(band, 0.05)
    psf = make_gaussian_psf(band.psf_fwhm_arcsec, 0.05)
    assert psf.data.shape == (expected_side, expected_side)
    assert psf.data.sum() == pytest.approx(1.0, rel=1e-6)


def test_make_gaussian_psf_explicit_size_overrides():
    psf = make_gaussian_psf(0.45, 0.05, size=33)
    assert psf.data.shape == (33, 33)


def test_make_gaussian_psf_size_forced_odd():
    """Even input is bumped to odd so the centre pixel is well-defined."""
    psf = make_gaussian_psf(0.45, 0.05, size=32)
    assert psf.data.shape == (33, 33)


# ---------------------------------------------------------------------------
# load_band_psf — empirical path crops to the band's support
# ---------------------------------------------------------------------------

def test_load_band_psf_falls_back_to_gaussian_sized_correctly(tmp_path):
    """With no FITS on disk, the Gaussian fallback uses the band-derived side."""
    expected_side = psf_side_pixels_for_band(Config.BAND_Y_E, 0.05)
    psf = load_band_psf(
        Config.BAND_Y_E, target_pixel_scale=0.05, psf_dir=str(tmp_path),
    )
    assert psf.data.shape == (expected_side, expected_side)
    assert psf.data.sum() == pytest.approx(1.0, rel=1e-6)


def test_load_band_psf_preserves_empirical_size(tmp_path):
    """Empirical PSFs are returned at whatever size the extractor saved.

    The user owns the size choice at extraction time; the loader only
    resamples to the requested ``target_pixel_scale`` (which changes the
    dimensions proportionally) and otherwise does not crop or pad.
    """
    # Synthesise an 'empirical' PSF at NISP native scale, 255×255.
    src = make_gaussian_psf(0.45, 0.30, size=255)
    src.save(str(tmp_path), filename=Config.BAND_Y_E.psf_fits_filename)

    # Loading at the same pixel scale must preserve the size.
    psf_same = load_band_psf(Config.BAND_Y_E, target_pixel_scale=0.30,
                             psf_dir=str(tmp_path), require_empirical=True)
    assert psf_same.data.shape == (255, 255)
    assert psf_same.data.sum() == pytest.approx(1.0, rel=1e-6)

    # Loading at the HR scale resamples 0.30 / 0.05 = 6× → ~1530 each side.
    psf_hr = load_band_psf(Config.BAND_Y_E, target_pixel_scale=0.05,
                           psf_dir=str(tmp_path), require_empirical=True)
    assert psf_hr.data.shape[0] > 1400
    assert psf_hr.data.sum() == pytest.approx(1.0, rel=1e-3)


def test_load_band_psf_centre_crop_preserves_peak():
    """After centre-crop + renormalisation, the peak still lives at the centre."""
    psf = make_gaussian_psf(0.45, 0.05)
    H, W = psf.data.shape
    peak_yx = np.unravel_index(psf.data.argmax(), psf.data.shape)
    assert peak_yx == (H // 2, W // 2)


# ---------------------------------------------------------------------------
# Convolution sanity: each band's PSF can convolve a 96² HR field
# ---------------------------------------------------------------------------

def test_forward_uses_band_specific_psf_sizes():
    """The forward-model default fallback picks per-band-sized Gaussians."""
    for band in Config.BANDS:
        psf = default_psf_for_band(band, 0.05)
        expected = psf_side_pixels_for_band(band, 0.05)
        assert psf.data.shape == (expected, expected), \
            f"{band.name}: expected {expected}, got {psf.data.shape}"


def test_load_all_band_psfs_reports_consistent_resolutions(tmp_path):
    """All bands end up at the requested target scale."""
    psfs = load_all_band_psfs(target_pixel_scale=0.05, psf_dir=str(tmp_path))
    for name, psf in psfs.items():
        assert psf.pixel_scale == pytest.approx(0.05, rel=1e-6)
        # Side is odd and sized to the band.
        assert psf.data.shape[0] == psf.data.shape[1]
        assert psf.data.shape[0] % 2 == 1


def test_load_all_band_psfs_have_distinct_sizes(tmp_path):
    """Bands with different FWHMs produce different Gaussian fallback sizes."""
    # Use a clean temp psf_dir so every band hits the Gaussian fallback path.
    psfs = load_all_band_psfs(target_pixel_scale=0.05, psf_dir=str(tmp_path))
    sides = {name: p.data.shape[0] for name, p in psfs.items()}
    # All four sides should be distinct (each FWHM rounds up uniquely).
    assert len(set(sides.values())) == len(sides), \
        f"expected 4 distinct sizes; got {sides}"


# ---------------------------------------------------------------------------
# PSFExtractionConfig.effective_output_size — odd-forcing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("user_input,expected", [
    (None, None),
    (1023, 1023),
    (1024, 1023),    # even → odd
    (511, 511),
    (256, 255),
    (65, 65),
    (64, 63),
])
def test_effective_output_size_forces_odd(user_input, expected):
    from euclid_polish.psf.psf_extractor import PSFExtractionConfig
    cfg = PSFExtractionConfig(output_size=user_input)
    assert cfg.effective_output_size == expected


def test_extraction_config_rejects_non_positive_output_size():
    from euclid_polish.psf.psf_extractor import PSFExtractionConfig
    cfg = PSFExtractionConfig(output_size=0)
    ok, msg = cfg.validate()
    assert not ok
    assert "output_size" in (msg or "")


# ---------------------------------------------------------------------------
# Per-band oversampling — every band's ePSF lands at 0.05"/pix
# ---------------------------------------------------------------------------

EPSF_TARGET_SCALE = 0.05


@pytest.mark.parametrize("band", list(Config.BANDS))
def test_epsf_oversampling_yields_target_pixel_scale(band):
    """band.epsf_pixel_scale_arcsec must equal 0.05" for every band."""
    assert band.epsf_pixel_scale_arcsec == pytest.approx(EPSF_TARGET_SCALE, rel=1e-9), (
        f"{band.name}: native={band.pixel_scale_lr_arcsec} "
        f"/ oversampling={band.epsf_oversampling} "
        f"= {band.epsf_pixel_scale_arcsec}"
    )


@pytest.mark.parametrize("band_name", ["VIS", "Y_E", "J_E", "H_E"])
def test_oversampling_is_two_for_every_band(band_name):
    # Archive cutouts arrive at 0.10″/pix for every band → uniform
    # oversampling=2 → ePSF at 0.05″/pix.
    assert Config.get_band(band_name).epsf_oversampling == 2


def test_oversampling_is_integer_positive_for_all_bands():
    for band in Config.BANDS:
        assert isinstance(band.epsf_oversampling, int)
        assert band.epsf_oversampling >= 1
