"""Tests for the public helpers added during the Sersic optimisation pass.

These cover:

* :func:`compute_sersic_stamp` — the unit-flux stamp + bounds primitive
  underlying every other Sersic call.
* :func:`add_sersic_to_bands` — the broadcast-add-into-N-bands helper.
* :func:`evaluate_sersic_at_coords` — arbitrary-coordinate evaluator used
  by the lensed-source path.
* Adaptive-csub correctness: the core+wings two-tier output must agree
  with a uniform high-csub reference on flux conservation.

The basic Sersic correctness (flux conservation, half-light isophote,
symmetry) lives in ``test_profiles.py`` and is not duplicated here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from euclid_polish.config import Config
from euclid_polish.sky.profiles import (
    add_sersic_to_bands,
    compute_sersic_stamp,
    draw_sersic,
    evaluate_sersic_at_coords,
    sersic_amp_from_flux,
    sersic_b_n,
)


# ---------------------------------------------------------------------------
# compute_sersic_stamp
# ---------------------------------------------------------------------------

def test_compute_sersic_stamp_returns_bounds_and_finite():
    stamp, (i_lo, i_hi, j_lo, j_hi) = compute_sersic_stamp(
        (200, 200), n=4.0, r_e=0.20, q=0.7, theta_rad=0.3,
        x0=100, y0=100, pixel_scale=0.05,
    )
    assert i_hi > i_lo and j_hi > j_lo
    assert stamp.shape == (i_hi - i_lo, j_hi - j_lo)
    assert stamp.dtype == np.float32
    assert np.isfinite(stamp).all()


def test_compute_sersic_stamp_unit_flux_integral():
    """The stamp must integrate to ~1.0 — it's the unit-flux primitive."""
    stamp, _ = compute_sersic_stamp(
        (401, 401), n=2.5, r_e=0.30, q=0.7, theta_rad=0.3,
        x0=200, y0=200, pixel_scale=0.05,
    )
    assert stamp.sum() == pytest.approx(1.0, rel=2e-2)


def test_compute_sersic_stamp_out_of_bounds_returns_empty():
    """Source far outside the canvas yields a zero-area stamp."""
    stamp, (i_lo, i_hi, j_lo, j_hi) = compute_sersic_stamp(
        (50, 50), n=1.0, r_e=0.30, q=1.0, theta_rad=0.0,
        x0=-500.0, y0=-500.0, pixel_scale=0.05,
    )
    assert stamp.shape == (0, 0)
    assert i_hi <= i_lo or j_hi <= j_lo


def test_compute_sersic_stamp_bounds_clipped_to_canvas():
    """Stamp bounds must always lie inside ``canvas_shape``."""
    stamp, (i_lo, i_hi, j_lo, j_hi) = compute_sersic_stamp(
        (60, 80), n=4.0, r_e=0.30, q=0.7, theta_rad=0.0,
        x0=10, y0=10, pixel_scale=0.05,
    )
    assert i_lo >= 0 and j_lo >= 0
    assert i_hi <= 60 and j_hi <= 80


# ---------------------------------------------------------------------------
# add_sersic_to_bands — multi-band broadcast helper
# ---------------------------------------------------------------------------

def test_add_sersic_to_bands_matches_per_band_loop():
    """Broadcast-add must produce the same result as N independent draws."""
    H = W = 96
    flux_per_band = np.array([1e5, 5e4, 3e4, 2e4], dtype=np.float32)
    common = dict(n=4.0, r_e=0.10, q=0.8, theta_rad=0.3,
                  x0=W / 2, y0=H / 2, pixel_scale=0.05)

    # Broadcast path
    canvas_a = np.zeros((H, W, 4), dtype=np.float32)
    add_sersic_to_bands(canvas_a, flux_per_band=flux_per_band, **common)

    # Per-band reference: render each band separately
    canvas_b = np.zeros((H, W, 4), dtype=np.float32)
    for k in range(4):
        draw_sersic((H, W), flux=float(flux_per_band[k]), out=canvas_b[..., k],
                    add=True, **common)

    np.testing.assert_allclose(canvas_a, canvas_b, rtol=1e-6, atol=1e-6)


def test_add_sersic_to_bands_per_band_flux_scaling():
    """Channel k's integrated flux must equal ``flux_per_band[k]``."""
    H = W = 401
    flux_per_band = np.array([1e6, 5e5, 2e5, 1e5], dtype=np.float32)
    canvas = np.zeros((H, W, 4), dtype=np.float32)
    add_sersic_to_bands(
        canvas, flux_per_band=flux_per_band,
        n=2.5, r_e=0.30, q=0.7, theta_rad=0.3,
        x0=W / 2, y0=H / 2, pixel_scale=0.05,
    )
    for k in range(4):
        assert canvas[..., k].sum() == pytest.approx(flux_per_band[k], rel=2e-2)


def test_add_sersic_to_bands_rejects_shape_mismatch():
    canvas = np.zeros((32, 32, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="flux_per_band"):
        add_sersic_to_bands(canvas, flux_per_band=[1.0, 2.0],
                            n=1.0, r_e=0.30, q=1.0, theta_rad=0.0,
                            x0=16, y0=16)


def test_add_sersic_to_bands_rejects_non_3d_canvas():
    canvas2d = np.zeros((32, 32), dtype=np.float32)
    with pytest.raises(ValueError, match="canvas_multi"):
        add_sersic_to_bands(canvas2d, flux_per_band=[1.0, 2.0, 3.0, 4.0],
                            n=1.0, r_e=0.30, q=1.0, theta_rad=0.0,
                            x0=16, y0=16)


def test_add_sersic_to_bands_with_out_of_bounds_source_is_noop():
    canvas = np.zeros((32, 32, 4), dtype=np.float32)
    add_sersic_to_bands(canvas, flux_per_band=[1.0, 1.0, 1.0, 1.0],
                        n=1.0, r_e=0.30, q=1.0, theta_rad=0.0,
                        x0=-100.0, y0=-100.0)
    assert canvas.sum() == 0.0


# ---------------------------------------------------------------------------
# evaluate_sersic_at_coords
# ---------------------------------------------------------------------------

def test_evaluate_sersic_at_coords_agrees_with_draw_sersic_csub1():
    """On a regular pixel grid with csub=1, the arbitrary-coord evaluator
    must match draw_sersic to floating-point precision.
    """
    H = W = 81
    pixel_scale = 0.05
    centroid = 40.0

    rendered = draw_sersic(
        (H, W), flux=1.0e6, n=2.0, r_e=0.30, q=0.6,
        theta_rad=0.3, x0=centroid, y0=centroid,
        pixel_scale=pixel_scale, csub=1,
    )

    yy, xx = np.indices((H, W), dtype=np.float64)
    x_arcsec = (xx - centroid) * pixel_scale
    y_arcsec = (yy - centroid) * pixel_scale
    evaluated = evaluate_sersic_at_coords(
        x_arcsec, y_arcsec,
        flux=1.0e6, n=2.0, r_e_arcsec=0.30, q=0.6,
        theta_rad=0.3, pixel_scale=pixel_scale,
    ).astype(np.float32)

    np.testing.assert_allclose(rendered, evaluated, rtol=1e-5, atol=1e-3)


def test_evaluate_sersic_at_coords_central_value():
    """At the centroid, the Sersic value equals amp · exp(b_n)."""
    flux = 1.0e6
    n = 2.5
    r_e_arcsec = 0.20
    q = 1.0
    pixel_scale = 0.05
    r_e_pix = r_e_arcsec / pixel_scale

    expected_peak = sersic_amp_from_flux(flux, n=n, r_e=r_e_pix, q=q) \
                  * math.exp(sersic_b_n(n))
    centre = evaluate_sersic_at_coords(
        np.array([0.0]), np.array([0.0]),
        flux=flux, n=n, r_e_arcsec=r_e_arcsec, q=q,
        theta_rad=0.0, pixel_scale=pixel_scale,
    )[0]
    assert float(centre) == pytest.approx(expected_peak, rel=1e-3)


# ---------------------------------------------------------------------------
# Adaptive csub: two-tier vs uniform high csub
# ---------------------------------------------------------------------------

def test_two_tier_matches_uniform_high_csub_on_total_flux():
    """The adaptive core+wings strategy must integrate to the same total
    flux as a uniform high-csub baseline (within their shared 5% tolerance).
    """
    common = dict(n=4.0, r_e=0.20, q=0.8, theta_rad=0.3,
                  x0=200, y0=200, pixel_scale=0.05)

    adaptive_stamp, _ = compute_sersic_stamp((401, 401), **common)
    uniform_stamp,  _ = compute_sersic_stamp((401, 401), csub=21, **common)

    # Both must conserve flux (1.0 ± small) and agree with each other.
    assert adaptive_stamp.sum() == pytest.approx(1.0, rel=5e-2)
    assert uniform_stamp.sum()  == pytest.approx(1.0, rel=5e-2)
    assert adaptive_stamp.sum() == pytest.approx(uniform_stamp.sum(), rel=2e-2)


def test_two_tier_matches_uniform_on_core_pixels():
    """Inside the core, adaptive must match a uniform pass at the SAME csub.

    The two-tier path uses ``_default_csub(n, r_e_pix)`` inside the core
    and csub=1 outside. So the central pixels of the adaptive stamp
    should match a uniform pass at the same effective csub to float32
    precision.
    """
    from euclid_polish.sky.profiles import _default_csub
    n, r_e, q = 4.0, 0.10, 0.8
    pixel_scale = 0.05
    r_e_pix = r_e / pixel_scale
    csub_core = _default_csub(n, r_e_pix)

    common = dict(n=n, r_e=r_e, q=q, theta_rad=0.0,
                  x0=50, y0=50, pixel_scale=pixel_scale)
    adaptive_stamp, a_bounds = compute_sersic_stamp((101, 101), **common)
    uniform_stamp,  u_bounds = compute_sersic_stamp(
        (101, 101), csub=csub_core, **common,
    )

    # Bounds depend only on (n, r_e), so should be identical.
    assert a_bounds == u_bounds

    # Adaptive uses ``csub_core`` inside ``SERSIC_CORE_RADIUS_R_E × r_e_pix``
    # of the centroid → those pixels must agree to float32 precision.
    half = max(2, int(Config.SERSIC_CORE_RADIUS_R_E * r_e_pix) - 1)
    cy = (a_bounds[1] - a_bounds[0]) // 2
    cx = (a_bounds[3] - a_bounds[2]) // 2
    a_core = adaptive_stamp[cy - half : cy + half, cx - half : cx + half]
    u_core = uniform_stamp [cy - half : cy + half, cx - half : cx + half]
    np.testing.assert_allclose(a_core, u_core, rtol=1e-4, atol=1e-5)
