"""Pure-function PSF measurements.

These work on raw 2-D ``np.ndarray``s without any class wrapper, so
they can be used both by :class:`euclid_polish.psf.PSF` methods and
by debug / diagnostic scripts. Same code path either way → no risk
of the inline-debug version diverging from the production class.

Every function returns float-typed results in *pixel* units. Convert
to arcsec by multiplying by the kernel's ``pixel_scale``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def estimate_fwhm_pixels_1d(profile: np.ndarray) -> float:
    """1-D FWHM estimate on a single profile slice (integer-pixel
    counting — no interpolation between samples).

    Robust to flat plateaus; returns 0.0 if the profile never reaches
    half its peak (e.g. all-zero input).
    """
    profile = np.asarray(profile, dtype=np.float64)
    peak = profile.max()
    if peak <= 0:
        return 0.0
    above = profile > (peak / 2.0)
    if not above.any():
        return 0.0
    idx = np.where(above)[0]
    return float(idx[-1] - idx[0] + 1)


def _interp_fwhm_pixels_1d(profile: np.ndarray) -> float:
    """1-D FWHM with linear interpolation between the half-max
    crossings — sub-pixel precision. ``estimate_fwhm_pixels_1d``'s
    integer-pixel-counting cousin.

    Internal helper for :func:`fwhm_pixels_radial`.
    """
    profile = np.asarray(profile, dtype=np.float64)
    peak = profile.max()
    if peak <= 0:
        return 0.0
    half = peak / 2.0
    above = np.where(profile > half)[0]
    if len(above) < 2:
        return 0.0
    L, R = above[0], above[-1]
    # Linear-interp the half-power crossings on each side.
    if L > 0 and profile[L - 1] != profile[L]:
        Lx = (L - 1) + (profile[L - 1] - half) / (profile[L - 1] - profile[L])
    else:
        Lx = float(L)
    if R < len(profile) - 1 and profile[R] != profile[R + 1]:
        Rx = R + (profile[R] - half) / (profile[R] - profile[R + 1])
    else:
        Rx = float(R)
    return float(Rx - Lx)


def radial_profile(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Azimuthally-averaged radial profile around the brightest pixel.

    Returns ``(radii_pixels, mean_intensity)``: ``radii_pixels[k]``
    is the integer pixel-distance bin (so 0 = the peak pixel itself,
    1 = nearest-neighbour ring, …), and ``mean_intensity[k]`` is the
    average pixel value in that bin.

    Preferred way to measure a 2-D PSF's FWHM — orientation-free, so
    not biased by diffraction spikes that happen to align with the
    row :func:`estimate_fwhm_pixels_1d` samples.
    """
    data = np.asarray(data, dtype=np.float64)
    cy, cx = np.unravel_index(int(np.argmax(data)), data.shape)
    yy, xx = np.indices(data.shape)
    r_cont = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    # Bin by ROUNDED integer radius (not floored). For a Gaussian
    # profile the rounded-bin mean is much closer to the value at
    # integer ``r`` than the floored-bin mean — the latter mixes
    # all sub-integer radii into the bin, biasing high-curvature
    # regions (the peak) downward and corrupting FWHM estimates by
    # 0.5+ pixels.
    r = np.rint(r_cont).astype(int)
    r_max = int(r.max())
    sums = np.bincount(r.ravel(), weights=data.ravel(), minlength=r_max + 1)
    counts = np.bincount(r.ravel(), minlength=r_max + 1)
    prof = sums / np.maximum(counts, 1)
    radii = np.arange(r_max + 1, dtype=np.float64)
    return radii, prof


def fwhm_pixels_radial(data: np.ndarray) -> float:
    """FWHM in pixels from the azimuthally-averaged radial profile.

    More robust than a row-slice FWHM for any PSF with axis-aligned
    spike structure (HST 4-vane, Euclid 6-vane). Uses sub-pixel
    interpolation at the half-power crossing.
    """
    _, prof = radial_profile(data)
    if prof.size == 0 or prof[0] <= 0:
        return 0.0
    half = prof[0] / 2.0
    below = np.where(prof < half)[0]
    if below.size == 0:
        return 0.0
    r_half = int(below[0])
    if r_half > 0 and prof[r_half - 1] != prof[r_half]:
        rh = (r_half - 1) + (prof[r_half - 1] - half) / (
            prof[r_half - 1] - prof[r_half]
        )
    else:
        rh = float(r_half)
    return float(2.0 * rh)        # full-width = 2 × half-width


def fwhm_pixels_area_equivalent(data: np.ndarray) -> float:
    """FWHM in pixels from the half-max enclosed area.

    Counts pixels with value > peak/2, treats them as a circle of
    equivalent area, returns 2× the equivalent radius. Tolerates
    fragmented half-max regions (HST 4-vane spikes have a connected
    core + isolated spike pixels; this metric weights them by area).
    """
    data = np.asarray(data, dtype=np.float64)
    peak = data.max()
    if peak <= 0:
        return 0.0
    n_above = int((data > peak / 2.0).sum())
    if n_above == 0:
        return 0.0
    return float(2.0 * np.sqrt(n_above / np.pi))


def flux_centroid(data: np.ndarray) -> Tuple[float, float]:
    """Flux-weighted (sub-pixel) centroid ``(y, x)``.

    Uses only positive pixels so a sky-subtracted PSF with small
    negative residuals doesn't bias the centroid. Returns NaN on an
    all-zero / all-negative input.
    """
    data = np.asarray(data, dtype=np.float64)
    w = np.where(data > 0, data, 0.0)
    s = float(w.sum())
    if s <= 0:
        return float("nan"), float("nan")
    yy, xx = np.indices(data.shape, dtype=np.float64)
    return float((yy * w).sum() / s), float((xx * w).sum() / s)


def peak_offset_from_centre(data: np.ndarray) -> Tuple[float, float]:
    """``(argmax_y - centre_y, argmax_x - centre_x)`` — integer-pixel
    offset of the PSF's brightest pixel from the geometric centre of
    the stamp. Useful for "is this PSF properly centred?" checks.

    Equivalent in arcsec = multiply by ``pixel_scale``.
    """
    data = np.asarray(data)
    H, W = data.shape
    cy_geo = (H - 1) / 2.0
    cx_geo = (W - 1) / 2.0
    pky, pkx = np.unravel_index(int(np.argmax(data)), data.shape)
    return float(pky - cy_geo), float(pkx - cx_geo)
