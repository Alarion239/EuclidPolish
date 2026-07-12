"""FastEPSFBuilder must be a numerically-identical drop-in for photutils
``EPSFBuilder``.

The only change is HOW the ePSF model is sampled: photutils evaluates each
star's separable sample grid with FITPACK's slow scattered ``.ev()`` path;
FastEPSFBuilder evaluates the same points with ``RectBivariateSpline.__call__``
(grid=True), which is ~40x faster and mathematically the same tensor-product
spline. These tests pin that the output matches stock photutils to
floating-point round-off — the safety gate for a correctness-sensitive
subsystem.
"""

from __future__ import annotations

import numpy as np
from photutils.psf import EPSFBuilder, EPSFStar, EPSFStars

from euclid_polish.psf.fast_epsf import FastEPSFBuilder, evaluate_on_grid
from euclid_polish.psf.psf_extractor import PSFExtractionConfig, PSFExtractor

# ---- small, fast fixtures -------------------------------------------------

PSF_SIZE = 21          # odd, small -> 43x43 oversampled grid, sub-second build
OVS = 2
MAXITERS = 3


def _moffat(side, cy, cx, fwhm=2.5, beta=3.5):
    alpha = fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
    y, x = np.mgrid[0:side, 0:side]
    return (1.0 + ((x - cx) ** 2 + (y - cy) ** 2) / alpha ** 2) ** (-beta)


def _make_stars(n, *, nan_star=False, seed=7):
    rng = np.random.default_rng(seed)
    half = PSF_SIZE // 2
    stars = []
    for i in range(n):
        dy, dx = rng.uniform(-1.5, 1.5, size=2)
        img = _moffat(PSF_SIZE, half + dy, half + dx)
        img = img / img.sum()
        if nan_star and i == n - 1:
            # one star with an off-centre bad pixel -> EPSFStar masks it,
            # so its sample points are NOT a full grid -> fallback path.
            img = img.copy()
            img[2, 3] = np.nan
        stars.append(EPSFStar(data=img, cutout_center=(half, half)))
    return EPSFStars(stars)


def _build(builder_cls, stars):
    builder = builder_cls(
        oversampling=OVS, maxiters=MAXITERS, progress_bar=False,
        center_accuracy=0.001, smoothing_kernel=None,
    )
    epsf, _ = builder(stars)
    return epsf


def _capture_legacy_model():
    """Grab the real ``_LegacyEPSFModel`` (and a star) that
    ``_resample_residual`` receives in production — the exact object the
    helper is designed for. The builder *returns* an ``ImagePSF``, so this is
    the only faithful way to unit-test ``evaluate_on_grid``."""
    grabbed: dict = {}

    class _Capture(FastEPSFBuilder):
        def _resample_residual(self, star, epsf):
            grabbed.setdefault("epsf", epsf)
            grabbed.setdefault("star", star)
            return super()._resample_residual(star, epsf)

    _build(_Capture, _make_stars(6))
    return grabbed["epsf"], grabbed["star"]


# ---- unit: the grid helper equals photutils' scattered evaluate -----------

def test_evaluate_on_grid_matches_scattered_evaluate():
    """evaluate_on_grid (grid=True) must equal _LegacyEPSFModel.evaluate
    (scattered .ev) on the same separable grid, to round-off."""
    epsf, star = _capture_legacy_model()
    x = star._xidx_centered
    y = star._yidx_centered

    fast = evaluate_on_grid(epsf, x, y, x_0=0.0, y_0=0.0,
                            shape=star._data.shape)
    ref = epsf.evaluate(x=x, y=y, flux=1.0, x_0=0.0, y_0=0.0)

    assert fast is not None
    assert fast.shape == ref.shape
    np.testing.assert_allclose(fast, ref, rtol=1e-10, atol=1e-12)


def test_evaluate_on_grid_returns_none_for_nongrid():
    """Non-grid (masked / partial) point sets fall back (return None)."""
    epsf, _ = _capture_legacy_model()
    x = np.array([0.0, 1.0, 2.0, 5.0])      # not a full separable grid
    y = np.array([0.0, 0.0, 1.0, 3.0])
    assert evaluate_on_grid(epsf, x, y, 0.0, 0.0, shape=(2, 2)) is None


# ---- integration: full build matches stock photutils ----------------------

def test_fast_builder_matches_stock_unmasked():
    stars = _make_stars(8)
    stock = _build(EPSFBuilder, stars)
    fast = _build(FastEPSFBuilder, _make_stars(8))  # identical inputs

    assert stock.data.max() > 0
    assert fast.data.shape == stock.data.shape
    np.testing.assert_allclose(fast.data, stock.data, rtol=1e-9, atol=1e-12)


def test_fast_builder_matches_stock_with_masked_star():
    """A masked star exercises the fallback path; output still matches."""
    stock = _build(EPSFBuilder, _make_stars(8, nan_star=True))
    fast = _build(FastEPSFBuilder, _make_stars(8, nan_star=True))
    np.testing.assert_allclose(fast.data, stock.data, rtol=1e-9, atol=1e-12)


# ---- wiring: config flag selects the builder ------------------------------

def test_fast_builder_enabled_by_default():
    assert PSFExtractionConfig().fast_builder is True


def test_extractor_selects_fast_builder_when_enabled():
    ext = PSFExtractor(PSFExtractionConfig(fast_builder=True, progress_bar=False))
    assert ext._epsf_builder_cls() is FastEPSFBuilder


def test_extractor_selects_stock_builder_when_disabled():
    ext = PSFExtractor(PSFExtractionConfig(fast_builder=False, progress_bar=False))
    assert ext._epsf_builder_cls() is EPSFBuilder
