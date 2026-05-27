"""Centralised PSF / ePSF operations.

This package owns every operation that turns raw PSF FITS bytes on
disk into a kernel ready to convolve a scene with:

  * :class:`PSF` (in :mod:`.core`) — the metadata-carrying object.
    Every operation (resample, crop, recentre, normalise, convolve,
    measure) is a method that returns a new ``PSF`` so they chain.
  * Loaders (in :mod:`.loaders`) — ``load_hst_f814w_psf`` and
    ``load_euclid_band_psf`` build ``PSF`` instances from the on-disk
    formats both instruments produce, with the right
    pixel-scale-fallback policy.
  * Measurements (in :mod:`.measurements`) — pure functions for FWHM
    (1-D row, 2-D area-equivalent, radial-profile), flux centroid,
    radial profile. Available as ``PSF`` methods AND standalone for
    arrays.

Naming convention: file uses ``psf``/``ePSF`` interchangeably. The
class is :class:`PSF` (covers both empirical ePSFs from photutils
and synthetic Gaussian PSFs).

Back-compat shim: ``euclid_polish.euclid.types.PSF`` re-exports
:class:`PSF` from this package so legacy imports keep working.
"""

from euclid_polish.psf.core import (
    PSF,
    DEFAULT_HR_PIXEL_SCALE,
)
from euclid_polish.psf.measurements import (
    estimate_fwhm_pixels_1d,
    fwhm_pixels_radial,
    fwhm_pixels_area_equivalent,
    flux_centroid,
    peak_offset_from_centre,
    radial_profile,
)
from euclid_polish.psf.loaders import (
    load_hst_f814w_psf,
    load_euclid_band_psf,
)

__all__ = [
    "PSF",
    "DEFAULT_HR_PIXEL_SCALE",
    "estimate_fwhm_pixels_1d",
    "fwhm_pixels_radial",
    "fwhm_pixels_area_equivalent",
    "flux_centroid",
    "peak_offset_from_centre",
    "radial_profile",
    "load_hst_f814w_psf",
    "load_euclid_band_psf",
]
