"""Tests for the HR-vs-SR angular power-spectrum math.

Validates the key diagnostic property: ``r(k)`` separates a *blur* (linear,
fully correlated → r≈1, T<1 at high k) from added *noise* (decorrelated →
r drops at high k), while identical images give r≈T≈1.
"""

from __future__ import annotations

import numpy as np
import pytest

from euclid_polish.eval.power_spectrum import (
    _SpectrumAccumulator as SpectrumAccumulator,
    cross_power_2d,
    k_magnitude_2d,
    log_k_edges,
)

PIXEL_SCALE = 0.05  # arcsec/pixel (HR/SR grid)


def _smooth_field(n: int = 128, sigma: float = 3.0, seed: int = 0) -> np.ndarray:
    """A random field with power across scales (smoothed white noise)."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    return gaussian_filter(rng.standard_normal((n, n)), sigma).astype(np.float64)


def _finite(arr, mask):
    a = arr[mask]
    return a[np.isfinite(a)]


def test_k_grid_reaches_nyquist():
    edges = log_k_edges(PIXEL_SCALE)
    assert edges[0] > 0
    assert np.isclose(edges[-1], 1.0 / (2.0 * PIXEL_SCALE))  # 10 cyc/arcsec
    kmag = k_magnitude_2d(64, PIXEL_SCALE)
    assert kmag.shape == (64, 64)
    assert kmag[0, 0] == 0.0  # DC mode


def test_identical_images_give_unity():
    a = _smooth_field()
    acc = SpectrumAccumulator(log_k_edges(PIXEL_SCALE))
    acc.add(a, a, PIXEL_SCALE)
    res = acc.finalize()
    m = res["count"] > 0
    assert np.allclose(_finite(res["T"], m), 1.0, atol=1e-9)
    assert np.allclose(_finite(res["r"], m), 1.0, atol=1e-9)


def test_blur_lowers_transfer_but_keeps_correlation():
    """A positive-MTF blur is fully correlated: r≈1, T falls at high k."""
    from scipy.ndimage import gaussian_filter

    a = _smooth_field(sigma=1.0)   # broadband power up through mid-k
    b = gaussian_filter(a, 1.5)    # extra positive-MTF blur
    acc = SpectrumAccumulator(log_k_edges(PIXEL_SCALE))
    acc.add(a, b, PIXEL_SCALE)     # default Tukey window suppresses edges
    res = acc.finalize()
    k, T, r = res["k"], res["T"], res["r"]
    m = res["count"] > 0
    lo = m & (k < 1.0)               # large scales
    mid = m & (k > 2.0) & (k < 5.0)  # small scales with real signal
    # correlation stays high (linear, deterministic blur)
    assert np.nanmedian(r[lo]) > 0.97
    assert np.nanmedian(r[mid]) > 0.85
    # transfer function clearly drops toward small scales
    assert np.nanmedian(T[lo]) > 0.9
    assert np.nanmedian(T[mid]) < 0.6 * np.nanmedian(T[lo])


def test_noise_decorrelates_at_high_k():
    """Independent noise added to SR drops r(k) at small scales."""
    a = _smooth_field()
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(a.shape) * 0.5 * a.std()
    b = a + noise
    acc = SpectrumAccumulator(log_k_edges(PIXEL_SCALE))
    acc.add(a, b, PIXEL_SCALE, window=np.ones_like(a))
    res = acc.finalize()
    k, r = res["k"], res["r"]
    m = res["count"] > 0
    lo = m & (k < 1.0)
    hi = m & (k > 5.0)
    assert np.nanmedian(r[lo]) > 0.9          # real structure preserved
    assert np.nanmedian(r[hi]) < np.nanmedian(r[lo])  # high-k decorrelates
    assert np.nanmedian(r[hi]) < 0.7          # clearly invented at small scales


def test_cross_power_shapes_and_dc_removed():
    a = _smooth_field(64)
    b = _smooth_field(64, seed=2)
    p_aa, p_bb, p_ab = cross_power_2d(a, b)
    assert p_aa.shape == p_bb.shape == p_ab.shape == (64, 64)
    # mean subtraction zeroes the DC (0,0) mode
    assert p_aa[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_accumulator_combines_mixed_stamp_sizes():
    """Stamps of different pixel sizes share the physical k-grid."""
    acc = SpectrumAccumulator(log_k_edges(PIXEL_SCALE))
    acc.add(_smooth_field(128), _smooth_field(128, seed=3), PIXEL_SCALE)
    acc.add(_smooth_field(106), _smooth_field(106, seed=4), PIXEL_SCALE)
    res = acc.finalize()
    assert acc.n_obj == 2
    assert np.isfinite(res["r"][res["count"] > 0]).all()
