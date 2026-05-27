"""Instrument-specific PSF loaders.

These wrap the raw FITS read + the resample/crop/recentre pipeline
behind named functions, so consumer scripts don't have to remember
the per-instrument FITS-key idiosyncrasies (PIXSCALE vs PXSCALE,
HLSP COSMOS pixel-scale fallback, etc.).

Every loader returns a :class:`euclid_polish.psf.PSF` already on
the HR grid (``DEFAULT_HR_PIXEL_SCALE``), centre-cropped to the
requested side, sum=1 normalised, and optionally re-centred on the
peak. That's the contract every downstream operation in the project
needs.

The PSF class itself stays instrument-agnostic; the
HST-vs-Euclid-vs-anything-else logic lives here.
"""

from __future__ import annotations

import os
from typing import Optional

from euclid_polish.config import BandConfig, Config
from euclid_polish.psf.core import PSF, DEFAULT_HR_PIXEL_SCALE


def load_hst_f814w_psf(
    fits_path: str,
    *,
    target_side: int = 421,
    target_pixel_scale: float = DEFAULT_HR_PIXEL_SCALE,
    recentre: bool = False,
) -> PSF:
    """Load HST F814W ePSF onto the HR grid.

    Pipeline:

      1. ``PSF.from_fits(fits_path)`` — reads the file, picks
         ``PIXSCALE`` from the header (HST extractor convention),
         normalises to sum=1.
      2. ``.resampled_to(target_pixel_scale)`` — no-op if PIXSCALE
         already matches.
      3. ``.centre_cropped_to(target_side)`` — odd-side centre crop.
      4. Optional ``.recentred()`` — sub-pixel-shift the peak to the
         geometric centre. HST F814W ePSFs from ``fasrc_extract_hst_psf.py``
         already centre cleanly (peak on the geometric centre, no
         offset), so this is False by default. Set True if you have
         a non-standard PSF whose centroid drifts off-pixel.

    Returns
    -------
    A :class:`PSF` ready for ``fftconvolve``.
    """
    if not os.path.isfile(fits_path):
        raise FileNotFoundError(f"HST F814W PSF FITS not found: {fits_path}")
    psf = PSF.from_fits(fits_path)
    psf = psf.resampled_to(target_pixel_scale)
    psf = psf.centre_cropped_to(target_side)
    if recentre:
        psf = psf.recentred(on="peak", subpixel=True)
    return psf


def load_euclid_band_psf(
    band: BandConfig,
    *,
    psf_dir: str = Config.EUCLID_PSF_DIR,
    target_side: int = 421,
    target_pixel_scale: float = DEFAULT_HR_PIXEL_SCALE,
    recentre: bool = True,
) -> PSF:
    """Load a Euclid band's empirical PSF onto the HR grid.

    Pipeline mirrors :func:`load_hst_f814w_psf` but uses the band's
    ``psf_fits_filename`` (looked up in ``psf_dir``). Recentring
    defaults to **True** here because the Euclid VIS ePSF FITS the
    project ships has a known 1-pixel offset (peak at
    ``(510, 511)`` of the 1023² stamp instead of (511, 511)) —
    historically that offset propagated through A_θ as a
    half-LR-pixel learned shift. ``recentred()`` removes it on load.

    Pass ``recentre=False`` only if you're investigating that
    offset explicitly.
    """
    fits_path = os.path.join(psf_dir, band.psf_fits_filename)
    if not os.path.isfile(fits_path):
        raise FileNotFoundError(
            f"Euclid {band.name} PSF FITS not found: {fits_path}"
        )
    psf = PSF.from_fits(fits_path)
    psf = psf.resampled_to(target_pixel_scale)
    psf = psf.centre_cropped_to(target_side)
    if recentre:
        psf = psf.recentred(on="peak", subpixel=True)
    return psf
