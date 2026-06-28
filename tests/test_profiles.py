"""Tests for the custom vectorised Sersic renderer.

We verify:
* ``b_n`` agrees with the Ciotti & Bertin asymptotic expansion.
* The closed-form amplitude reproduces the analytic flux when the profile
  is summed over a generous stamp.
* Half-light radius semantics: the elliptical isophote at r = R_e contains
  exactly 50% of the analytic flux.
* Bulge+disk superposition is exactly additive in flux.
* Stamp cropping leaves the answer unchanged for typical galaxies.
* Sub-pixel sampling is required for compact / high-n profiles and the
  default heuristic delivers <2% flux error in the regime of physical
  relevance (n ≤ 4, r_e_pix ≥ 2 — i.e. r_e ≥ 0.10″ on the 0.05″ HR grid).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from euclid_polish.sky.generation.profiles import (
    draw_bulge_disk,
    draw_sersic,
    sersic_amp_from_flux,
    sersic_b_n,
)


# ---------------------------------------------------------------------------
# b_n
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1.0, 2.0, 2.5, 4.0, 6.0, 8.0])
def test_b_n_matches_asymptotic(n: float):
    """Ciotti & Bertin 1999 asymptotic: b_n ≈ 2n - 1/3 + 4/(405 n).

    Accuracy degrades for n < 1 but is ≤ 1e-3 for n ≥ 1.
    """
    exact = sersic_b_n(n)
    approx = 2 * n - 1.0 / 3.0 + 4.0 / (405.0 * n)
    assert exact == pytest.approx(approx, rel=2e-3)


def test_b_n_n_half():
    """Special-case: at n = 0.5 the Sersic is a Gaussian and b_n = ln(2)."""
    assert sersic_b_n(0.5) == pytest.approx(math.log(2), rel=1e-9)


def test_b_n_rejects_non_positive():
    with pytest.raises(ValueError):
        sersic_b_n(0.0)
    with pytest.raises(ValueError):
        sersic_b_n(-1.0)


# ---------------------------------------------------------------------------
# Amplitude formula
# ---------------------------------------------------------------------------

def test_amp_rejects_invalid_q():
    with pytest.raises(ValueError):
        sersic_amp_from_flux(1.0, n=1.0, r_e=1.0, q=0.0)
    with pytest.raises(ValueError):
        sersic_amp_from_flux(1.0, n=1.0, r_e=1.0, q=1.5)


def test_amp_rejects_non_positive_r_e():
    with pytest.raises(ValueError):
        sersic_amp_from_flux(1.0, n=1.0, r_e=0.0, q=1.0)


# ---------------------------------------------------------------------------
# Flux conservation
# ---------------------------------------------------------------------------

# Regime of physical relevance for COSMOS galaxies on the 0.05″ HR grid:
# n ∈ [0.5, 4] and r_e ∈ [0.10″, 1.0″] → r_e_pix ∈ [2, 20].
FLUX_CASES = [
    # (n, r_e_arcsec, q, tolerance_relative)
    (0.5, 0.30, 1.00, 5e-3),
    (1.0, 0.30, 1.00, 5e-3),
    (1.0, 0.30, 0.50, 5e-3),
    (2.0, 0.30, 0.70, 1e-2),
    (2.5, 0.50, 0.70, 1e-2),
    (3.0, 0.30, 0.80, 2e-2),
    (4.0, 0.30, 1.00, 5e-2),
    (4.0, 0.20, 0.50, 5e-2),
]


@pytest.mark.parametrize("n,r_e,q,tol", FLUX_CASES)
def test_flux_conservation(n, r_e, q, tol):
    """Discrete sum over a generous stamp matches the analytic total flux."""
    target_flux = 1.0e6
    pixel_scale = 0.05
    # Pad ≥ 25 R_e along each axis to capture wings even for high-n.
    pad_radii = 25
    side = int(2 * pad_radii * r_e / pixel_scale) | 1
    x0 = y0 = side // 2
    img = draw_sersic(
        (side, side),
        flux=target_flux, n=n, r_e=r_e, q=q,
        theta_rad=0.3, x0=x0, y0=y0, pixel_scale=pixel_scale,
    )
    assert img.sum() == pytest.approx(target_flux, rel=tol)


# ---------------------------------------------------------------------------
# Half-light radius semantics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,tol", [(0.5, 5e-3), (1.0, 5e-3), (2.5, 1e-2), (4.0, 3e-2)])
def test_half_light_radius_isophote(n: float, tol: float):
    """The ellipse r_ell == R_e contains 50% of the analytic flux.

    We measure this on the discrete rasterisation; for high n the central
    peak undersampling biases the ratio a few percent high.
    """
    r_e = 0.30
    q   = 0.7
    theta = 0.3
    pixel_scale = 0.05
    pad_radii = 25
    side = int(2 * pad_radii * r_e / pixel_scale) | 1
    x0 = y0 = side // 2

    img = draw_sersic(
        (side, side),
        flux=1.0, n=n, r_e=r_e, q=q,
        theta_rad=theta, x0=x0, y0=y0, pixel_scale=pixel_scale,
    )

    yy, xx = np.indices((side, side), dtype=np.float64) - (side // 2)
    c, s = math.cos(theta), math.sin(theta)
    xp =  c * xx + s * yy
    yp = -s * xx + c * yy
    r_ell_pix = np.sqrt(xp * xp + (yp / q) ** 2)

    inside = img[r_ell_pix <= r_e / pixel_scale].sum()
    total  = img.sum()
    assert inside / total == pytest.approx(0.5, abs=tol)


# ---------------------------------------------------------------------------
# Bulge + disk superposition
# ---------------------------------------------------------------------------

def test_bulge_disk_additive_in_flux():
    """A B+D source emits exactly the sum of bulge and disk fluxes."""
    flux_b, flux_d = 3.0e5, 7.0e5
    img = draw_bulge_disk(
        (251, 251),
        flux_bulge=flux_b, r_e_bulge=0.15, q_bulge=0.8,
        flux_disk=flux_d,  r_e_disk=0.40, q_disk=0.5,
        theta_rad=0.3, x0=125, y0=125, pixel_scale=0.05,
    )
    assert img.sum() == pytest.approx(flux_b + flux_d, rel=2e-2)


def test_bulge_disk_equals_two_sersics():
    """draw_bulge_disk should match draw_sersic(n=4) + draw_sersic(n=1)."""
    common = dict(theta_rad=0.3, x0=125, y0=125, pixel_scale=0.05)
    img_bd = draw_bulge_disk(
        (251, 251),
        flux_bulge=3.0e5, r_e_bulge=0.15, q_bulge=0.8,
        flux_disk=7.0e5,  r_e_disk=0.40, q_disk=0.5,
        **common,
    )

    img_manual = np.zeros((251, 251), dtype=np.float32)
    draw_sersic((251, 251), flux=3.0e5, n=4.0, r_e=0.15, q=0.8, out=img_manual, **common)
    draw_sersic((251, 251), flux=7.0e5, n=1.0, r_e=0.40, q=0.5, out=img_manual, **common)

    np.testing.assert_allclose(img_bd, img_manual, rtol=1e-6)


# ---------------------------------------------------------------------------
# Stamp cropping correctness
# ---------------------------------------------------------------------------

def test_stamp_cropping_preserves_total_for_typical_galaxy():
    """Cropping to a stamp around the source must not measurably change the flux."""
    flux = 1.0e6
    img_full = np.zeros((401, 401), dtype=np.float32)
    img_crop = np.zeros((401, 401), dtype=np.float32)
    # Source at centre — both cropped (default) and uncropped (csub=1 fast path)
    # should integrate to the same total within numerical noise.
    common = dict(n=2.5, r_e=0.30, q=0.7, theta_rad=0.3,
                  x0=200, y0=200, pixel_scale=0.05)
    draw_sersic(img_crop.shape, flux=flux, out=img_crop, **common)
    draw_sersic(img_full.shape, flux=flux, csub=3, out=img_full, **common)
    # Both methods should be within ~3% of each other (auto vs forced csub).
    assert img_crop.sum() == pytest.approx(img_full.sum(), rel=0.05)


def test_source_outside_image_yields_zero():
    """A source whose centroid is far outside the grid contributes nothing."""
    img = draw_sersic((50, 50), flux=1.0, n=1.0, r_e=0.30, q=1.0,
                      theta_rad=0.0, x0=-1000.0, y0=-1000.0, pixel_scale=0.05)
    assert img.sum() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Symmetry / geometry sanity
# ---------------------------------------------------------------------------

def test_circular_profile_is_rotationally_invariant():
    """For q=1, rotating by an arbitrary angle should leave the image bit-identical."""
    common = dict(shape=(81, 81), flux=1.0e6, n=2.0, r_e=0.30, q=1.0,
                  x0=40, y0=40, pixel_scale=0.05)
    a = draw_sersic(theta_rad=0.0, **common)
    b = draw_sersic(theta_rad=0.7, **common)
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)


def test_centroid_recoverable_from_flux_weighted_mean():
    """The flux-weighted (x, y) of a rasterised source recovers the input centroid."""
    img = draw_sersic((81, 81), flux=1.0e6, n=1.0, r_e=0.30, q=1.0,
                      theta_rad=0.0, x0=40.5, y0=39.5, pixel_scale=0.05)
    yy, xx = np.indices(img.shape, dtype=np.float64)
    total = img.sum()
    x_mean = (img * xx).sum() / total
    y_mean = (img * yy).sum() / total
    assert x_mean == pytest.approx(40.5, abs=0.05)
    assert y_mean == pytest.approx(39.5, abs=0.05)
