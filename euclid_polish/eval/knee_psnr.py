"""Knee-independent PSNR: PSNR as a function of the asinh knee, and its mean.

The stretched PSNR the ensemble is scored with compares ``asinh(x / knee)``
images, and the knee decides which brightnesses count: below the knee errors
are weighed linearly, above it logarithmically. A single knee therefore
favours models trained near it. :func:`knee_psnr` evaluates the PSNR over a
grid of knees, and :func:`integrated_psnr` averages it uniformly in
``log10(knee)`` — one number that does not privilege any brightness scale.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from euclid_polish.config import Config

#: Knees (electrons) the curves are evaluated at: a 1-2-3-5 grid from noise
#: level (0.1 e⁻) to bright star cores (10⁴ e⁻). The integral spans the grid.
KNEE_GRID_E = (0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 50.0,
               100.0, 200.0, 300.0, 500.0, 1000.0, 2000.0, 3000.0, 5000.0, 10000.0)


def knee_psnr(pred_e: np.ndarray, truth_e: np.ndarray, *,
              knees: Sequence[float] = KNEE_GRID_E,
              peak_e: float = float(Config.PSNR_PEAK_E),
              truth_asinh: Sequence[np.ndarray] | None = None) -> np.ndarray:
    """PSNR (dB) of ``pred_e`` against ``truth_e`` at every knee.

    Arrays are ``(H, W)`` or ``(H, W, C)`` in electrons; the result is
    ``(K,)`` or ``(K, C)`` (one PSNR per band). The peak is the stretched
    PSNR peak at each knee, ``asinh(peak_e / knee)``, as in the standard
    metric. ``truth_asinh`` optionally supplies the stretched truth per knee
    so it is computed once when scoring several models on one field.
    """
    pred = np.asarray(pred_e, np.float32)
    truth = np.asarray(truth_e, np.float32)
    if pred.shape != truth.shape:
        raise ValueError(f"prediction {pred.shape} and truth {truth.shape} differ")
    spatial = (0, 1) if pred.ndim == 3 else None
    out = []
    for k, knee in enumerate(knees):
        q = np.float32(knee)
        t = truth_asinh[k] if truth_asinh is not None else np.arcsinh(truth / q)
        mse = np.mean((np.arcsinh(pred / q) - t) ** 2, axis=spatial, dtype=np.float64)
        peak = np.arcsinh(float(peak_e) / float(knee))
        out.append(10.0 * np.log10(peak * peak / np.maximum(mse, 1e-30)))
    return np.asarray(out, np.float64)


def stretched_truth(truth_e: np.ndarray, knees: Sequence[float] = KNEE_GRID_E
                    ) -> list[np.ndarray]:
    """``asinh(truth / knee)`` for every knee (reuse across models)."""
    truth = np.asarray(truth_e, np.float32)
    return [np.arcsinh(truth / np.float32(knee)) for knee in knees]


def integrated_psnr(curve: np.ndarray, knees: Sequence[float] = KNEE_GRID_E
                    ) -> np.ndarray:
    """Mean of a PSNR-vs-knee curve over ``log10(knee)`` (trapezoid rule).

    ``curve`` is ``(K,)`` or ``(K, C)``; the result drops the knee axis."""
    values = np.asarray(curve, np.float64)
    log_knee = np.log10(np.asarray(knees, np.float64))
    if values.shape[0] != len(log_knee):
        raise ValueError("curve and knee grid lengths differ")
    span = log_knee[-1] - log_knee[0]
    return np.trapezoid(values, log_knee, axis=0) / span
