"""Pin the magnitude ↔ electrons ↔ ADU/s conversions (Step 0 of the
star-anchor work). These are the formulas the anchor delta-targets (from
catalog magnitude) and the model input (from archive ADU/s) must share so
the two are on one electron-over-the-stack scale."""

import numpy as np
import pytest

import math

from euclid_polish.photometry import (
    ab_mag_to_electrons,
    ab_mag_to_uJy,
    adu_per_s_to_electrons,
    adu_per_s_to_electrons_factor,
    electrons_to_ab_mag,
    mjy_per_sr_to_electrons_factor,
    pixel_solid_angle_sr,
    uJy_to_ab_mag,
    uJy_to_electrons,
)
from euclid_polish.config import Config

BAND = Config.BAND_VIS


def test_zeropoint_magnitude_is_one_electron():
    # A source AT the stack zero-point contributes exactly 1 e⁻.
    assert ab_mag_to_electrons(BAND.sim_zeropoint_e, BAND) == pytest.approx(1.0)


def test_2p5_mag_brighter_is_ten_times_flux():
    m = 18.0
    assert ab_mag_to_electrons(m - 2.5, BAND) == pytest.approx(
        10.0 * ab_mag_to_electrons(m, BAND), rel=1e-6)


def test_mag17_vis_flux_matches_closed_form():
    # 10 ** (-0.4 * (17 - sim_zp)); guards against an accidental sign flip.
    expected = 10.0 ** (-0.4 * (17.0 - BAND.sim_zeropoint_e))
    assert ab_mag_to_electrons(17.0, BAND) == pytest.approx(expected, rel=1e-9)
    assert ab_mag_to_electrons(17.0, BAND) > 1e5  # a mag-17 star is bright


def test_adu_factor_unity_when_magzero_equals_stack_zp():
    f = adu_per_s_to_electrons_factor(BAND.sim_zeropoint_e, BAND)
    assert f == pytest.approx(1.0)
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    out = adu_per_s_to_electrons(arr, BAND.sim_zeropoint_e, BAND)
    np.testing.assert_allclose(out, arr, rtol=1e-6)


def test_uJy_ab_mag_zeropoint():
    # A 1-µJy source sits at the µJy AB zeropoint; 10× brighter = 2.5 mag less.
    assert uJy_to_ab_mag(1.0) == pytest.approx(Config.AB_ZP_UJY, abs=1e-9)
    assert uJy_to_ab_mag(10.0) == pytest.approx(Config.AB_ZP_UJY - 2.5, abs=1e-9)


def test_uJy_to_electrons_matches_mag_path():
    # Direct µJy→e⁻ must equal routing through AB mag → electrons.
    for f in (0.5, 12.0, 3.4e3):
        assert uJy_to_electrons(f, BAND) == pytest.approx(
            ab_mag_to_electrons(uJy_to_ab_mag(f), BAND), rel=1e-9)


def test_uJy_to_electrons_scales_linearly():
    assert uJy_to_electrons(20.0, BAND) == pytest.approx(
        2.0 * uJy_to_electrons(10.0, BAND), rel=1e-9)


def test_adu_conversion_consistent_with_mag_chain():
    # A star measured at mag m in an ADU/s image with keyword MAGZERO: its
    # ADU/s total maps to the SAME electrons as ab_mag_to_electrons(m).
    magzero = 24.6           # the Q1 VIS mosaics' actual MAGZERO (ADU/s)
    m = 18.3
    total_adu_per_s = 10.0 ** (-0.4 * (m - magzero))     # archive scale
    e_from_adu = float(adu_per_s_to_electrons(
        np.array([total_adu_per_s], dtype=np.float32), magzero, BAND)[0])
    e_from_mag = ab_mag_to_electrons(m, BAND)
    assert e_from_adu == pytest.approx(e_from_mag, rel=1e-4)


# --------------------------------------------------------------------------- #
# Inverses + cross-consistency (the "no duplicated methods" contract): every
# conversion pair must round-trip exactly, and every composition must equal
# routing through its parts — so there is only ONE way any unit maps to
# electrons or magnitudes.
# --------------------------------------------------------------------------- #

def test_ab_zp_ujy_is_exact_ab_definition():
    # AB is DEFINED by m = 8.90 − 2.5·log10(F[Jy]); in µJy the constant is
    # 8.90 + 2.5·log10(1e6) = 23.90 exactly (2.5·log10(1e6) = 15 exactly).
    assert Config.AB_ZP_UJY == 8.90 + 2.5 * math.log10(1e6)
    assert Config.AB_ZP_UJY == 23.90


def test_uJy_mag_roundtrip():
    for f in (0.13, 1.0, 47.0, 8.8e5):
        assert ab_mag_to_uJy(uJy_to_ab_mag(f)) == pytest.approx(f, rel=1e-12)
    for m in (14.0, 22.65, 27.3):
        assert uJy_to_ab_mag(ab_mag_to_uJy(m)) == pytest.approx(m, abs=1e-12)


def test_electrons_mag_roundtrip_scalar_and_array():
    for m in (12.0, 19.43, 30.0):
        assert electrons_to_ab_mag(
            ab_mag_to_electrons(m, BAND), BAND) == pytest.approx(m, abs=1e-12)
    mags = np.array([17.0, 21.5, 25.0])
    flux = ab_mag_to_electrons(mags, BAND)
    assert isinstance(flux, np.ndarray) and flux.shape == mags.shape
    np.testing.assert_allclose(electrons_to_ab_mag(flux, BAND), mags,
                               rtol=0, atol=1e-12)


def test_electrons_to_ab_mag_nonpositive_is_nan():
    # No magnitude exists for ≤0 flux — nan, never an exception (the viewer
    # sums background-subtracted maps that can legitimately go negative).
    assert math.isnan(electrons_to_ab_mag(0.0, BAND))
    assert math.isnan(electrons_to_ab_mag(-3.0, BAND))
    out = electrons_to_ab_mag(np.array([1.0, 0.0, -5.0]), BAND)
    assert np.isfinite(out[0]) and np.isnan(out[1]) and np.isnan(out[2])


def test_array_matches_scalar_path():
    m = 20.25
    assert ab_mag_to_electrons(np.array([m]), BAND)[0] == pytest.approx(
        ab_mag_to_electrons(m, BAND), rel=1e-15)


def test_catalog_flux_threshold_matches_uJy_inverse():
    # The archive-query magnitude limit becomes a µJy flux bound via
    # ab_mag_to_uJy; it must invert uJy_to_ab_mag exactly (a star exactly AT
    # the magnitude limit sits exactly at the flux threshold).
    limit = 21.0
    thr = ab_mag_to_uJy(limit)
    assert uJy_to_ab_mag(thr) == pytest.approx(limit, abs=1e-12)


def test_mjy_per_sr_factor_composes_from_uJy_path():
    # MJy/sr → e⁻ must equal (MJy/sr → µJy via the pixel solid angle) → e⁻:
    # one intensive-to-flux step (Ω), then the ONE µJy→e⁻ conversion.
    pix = 0.05
    omega = pixel_solid_angle_sr(pix)
    ujy_of_one_mjy_sr = 1.0e12 * omega
    assert mjy_per_sr_to_electrons_factor(BAND, pix) == pytest.approx(
        uJy_to_electrons(ujy_of_one_mjy_sr, BAND), rel=1e-12)


def test_viewer_served_stack_zeropoint_is_the_canonical_anchor():
    # The JS magnitude readout consumes zeropoint_ab_e_total from the meta —
    # it must be BandConfig.sim_zeropoint_e verbatim, and the JS formula
    # zp_total − 2.5·log10(Σe⁻) must equal electrons_to_ab_mag.
    from euclid_polish.web.helpers.viewer_data import color_constants
    consts = color_constants()
    for name, served in consts["bands"].items():
        band = Config.get_band(name)
        assert served["zeropoint_ab_e_total"] == band.sim_zeropoint_e
        flux = 8.84e5
        js_mag = served["zeropoint_ab_e_total"] - 2.5 * math.log10(flux)
        assert js_mag == pytest.approx(
            electrons_to_ab_mag(flux, band), abs=1e-12)


def test_ab_flux_norm_is_inverse_of_ab_zero_electrons():
    # visualization/color.py's display normalisation is the same anchor:
    # 1 / ab_mag_to_electrons(0, band) == 1 / (t_total · 10^(0.4·zp_rate)).
    from euclid_polish.visualization.color import _ab_flux_norm
    for band in Config.BANDS:
        legacy = 1.0 / (band.t_total_s
                        * 10 ** (0.4 * band.zeropoint_ab_e_per_s))
        assert _ab_flux_norm(band.name) == pytest.approx(legacy, rel=1e-12)
