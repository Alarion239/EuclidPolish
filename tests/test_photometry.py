"""Pin the magnitude ↔ electrons ↔ ADU/s conversions (Step 0 of the
star-anchor work). These are the formulas the anchor delta-targets (from
catalog magnitude) and the model input (from archive ADU/s) must share so
the two are on one electron-over-the-stack scale."""

import numpy as np
import pytest

from euclid_polish.catalog.photometry import (
    ab_mag_to_electrons,
    adu_per_s_to_electrons,
    adu_per_s_to_electrons_factor,
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
    magzero = 24.57          # representative Q1 VIS MAGZERO (ADU/s)
    m = 18.3
    total_adu_per_s = 10.0 ** (-0.4 * (m - magzero))     # archive scale
    e_from_adu = float(adu_per_s_to_electrons(
        np.array([total_adu_per_s], dtype=np.float32), magzero, BAND)[0])
    e_from_mag = ab_mag_to_electrons(m, BAND)
    assert e_from_adu == pytest.approx(e_from_mag, rel=1e-4)
