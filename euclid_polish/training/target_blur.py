"""Gaussian smoothing for synthetic HR supervision and references."""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter

from euclid_polish.config import Config


def validate_target_fwhm_arcsec(value: float) -> float:
    """Validate and normalise a target Gaussian FWHM in arcseconds."""
    fwhm = float(value)
    if not math.isfinite(fwhm) or fwhm < 0.0:
        raise ValueError("target_psf_fwhm_arcsec must be finite and >= 0")
    return fwhm


def target_sigma_pixels(
    fwhm_arcsec: float = Config.TARGET_PSF_FWHM_ARCSEC,
    pixel_scale_arcsec: float = Config.DEFAULT_PIXEL_SCALE,
) -> float:
    """Convert a target FWHM in arcseconds to Gaussian sigma in pixels."""
    fwhm = validate_target_fwhm_arcsec(fwhm_arcsec)
    scale = float(pixel_scale_arcsec)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("pixel_scale_arcsec must be finite and > 0")
    return fwhm / scale / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def blur_target_array(
    data: np.ndarray,
    fwhm_arcsec: float = Config.TARGET_PSF_FWHM_ARCSEC,
    *,
    pixel_scale_arcsec: float = Config.DEFAULT_PIXEL_SCALE,
) -> np.ndarray:
    """Apply a per-channel, normalized Gaussian target PSF.

    ``data`` is ``(H, W, C)`` (or a 2-D single-channel plane). The blur is
    applied only over spatial axes with reflective boundaries; channels never
    mix. A zero FWHM returns a float32 copy without modifying the input.
    """
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim not in (2, 3):
        raise ValueError(f"target array must be 2-D or 3-D, got {arr.shape}")
    sigma = target_sigma_pixels(fwhm_arcsec, pixel_scale_arcsec)
    if sigma <= 0.0:
        return np.array(arr, dtype=np.float32, copy=True)
    axes_sigma = (sigma, sigma, 0.0) if arr.ndim == 3 else (sigma, sigma)
    return np.asarray(
        # ``mirror`` matches TensorFlow's REFLECT padding (the edge sample is
        # not duplicated), keeping the NumPy and graph paths identical.
        gaussian_filter(arr, sigma=axes_sigma, mode="mirror"),
        dtype=np.float32,
    )
