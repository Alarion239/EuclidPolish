"""A numerically-identical, faster drop-in for photutils ``EPSFBuilder``.

photutils evaluates the ePSF model at each star's sample points with
``RectBivariateSpline.ev`` — FITPACK's *scattered* point-by-point path — even
though those points are a **separable grid** (the cutout index grid shifted by
a scalar centroid offset). Evaluating the same tensor-product spline with the
gridded ``RectBivariateSpline.__call__`` is ~40x faster and mathematically the
same value (matches to floating-point round-off, ~1e-12).

``FastEPSFBuilder`` overrides only ``_resample_residual`` (the ~57%-of-runtime
hot path); everything else — sigma-clipping, stacking, recentering,
convergence, the ``_LegacyEPSFModel`` plumbing — is inherited unchanged. Any
star whose sample points are not a complete grid (e.g. masked/NaN pixels)
transparently falls back to stock photutils behaviour.

See docs/superpowers/specs/2026-06-29-fast-epsf-builder-design.md.
"""

from __future__ import annotations

import numpy as np
from photutils.psf import EPSFBuilder
from photutils.utils._round import py2intround

__all__ = ["FastEPSFBuilder", "evaluate_on_grid"]


def evaluate_on_grid(epsf, x, y, x_0, y_0, shape):
    """Evaluate ``epsf`` at ``(x, y)`` via the gridded spline path.

    Mirrors ``_LegacyEPSFModel.evaluate`` (with ``flux=1``) but, when the
    raveled point set ``(x, y)`` reconstructs to a complete separable grid of
    ``shape``, uses ``RectBivariateSpline.__call__`` (grid=True) instead of the
    slow scattered ``.ev``. Returns the evaluated model raveled in the same
    C-order as ``.ev`` would, or ``None`` when the points are not a clean grid
    (so the caller can fall back to the stock scattered path).
    """
    ny, nx = shape
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size != ny * nx or y.size != ny * nx:
        return None

    xi = x.ravel() - x_0 + epsf._x_origin
    yi = y.ravel() - y_0 + epsf._y_origin
    xg = xi.reshape(ny, nx)
    yg = yi.reshape(ny, nx)

    # Separable iff every row of x is identical and every column of y is.
    if not (np.array_equal(xg, np.broadcast_to(xg[0], (ny, nx)))
            and np.array_equal(yg, np.broadcast_to(yg[:, :1], (ny, nx)))):
        return None

    x_axis = xg[0, :]   # x varies along columns
    y_axis = yg[:, 0]   # y varies along rows

    # The interpolator is RectBivariateSpline(x_grid, y_grid, data.T), so its
    # first axis is x; __call__(x_axis, y_axis) returns shape (nx, ny).
    values = epsf.interpolator(x_axis, y_axis).T  # -> (ny, nx)

    if epsf._fill_value is not None:
        invalid = (
            (xg < 0) | (xg > (epsf._nx - 1) / epsf.oversampling[1])
            | (yg < 0) | (yg > (epsf._ny - 1) / epsf.oversampling[0])
        )
        values = values.copy()
        values[invalid] = epsf._fill_value

    return values.ravel()


class FastEPSFBuilder(EPSFBuilder):
    """``EPSFBuilder`` with a gridded-spline residual resampling hot path."""

    def _resample_residual(self, star, epsf):
        # Identical to photutils' implementation except the model evaluation
        # uses the gridded spline path when the star's sample points form a
        # complete separable grid (the common, fully-unmasked case).
        x = star._xidx_centered
        y = star._yidx_centered

        model = evaluate_on_grid(
            epsf, x, y, x_0=0.0, y_0=0.0, shape=star._data.shape
        )
        if model is None:
            model = epsf.evaluate(x=x, y=y, flux=1.0, x_0=0.0, y_0=0.0)

        stardata = star._data_values_normalized - model

        x = epsf.oversampling[1] * star._xidx_centered
        y = epsf.oversampling[0] * star._yidx_centered

        epsf_xcenter, epsf_ycenter = (
            int((epsf.data.shape[1] - 1) / 2),
            int((epsf.data.shape[0] - 1) / 2),
        )
        xidx = py2intround(x + epsf_xcenter)
        yidx = py2intround(y + epsf_ycenter)

        resampled_img = np.full(epsf.shape, np.nan)

        mask = np.logical_and(
            np.logical_and(xidx >= 0, xidx < epsf.shape[1]),
            np.logical_and(yidx >= 0, yidx < epsf.shape[0]),
        )
        xidx_ = xidx[mask]
        yidx_ = yidx[mask]

        resampled_img[yidx_, xidx_] = stardata[mask]

        return resampled_img
