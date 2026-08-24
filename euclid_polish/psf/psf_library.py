"""
Per-band PSF loading and management.

The PSF extraction pipeline (:mod:`euclid_polish.psf.psf_extractor`) is
band-agnostic — it takes a directory of star cutouts and produces an
empirical ePSF as a single normalised FITS file. To support the multi-band
forward model we just need a thin wrapper that, given a band name, returns
the appropriate PSF:

  1. If an empirical ePSF file exists at the configured per-band path,
     load it and (optionally) resample it onto the HR grid.
  2. Otherwise return a Gaussian fallback at the band's nominal FWHM.

Two CLI flows fit the same scheme — the user calls

    polish extract-psf --band Y_E --cutout-dir data/euclid_nisp_stars/Y

and the resulting FITS goes to ``data/euclid_psf/euclid_psf_Y.fits`` (the
path is computed from :attr:`BandConfig.psf_fits_filename`).
"""

from __future__ import annotations

import math
import os

import numpy as np

from euclid_polish.config import BandConfig, Config
from euclid_polish.psf import PSF
from euclid_polish.psf.psf_set import PSFSet

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def psf_path_for_band(band: BandConfig, psf_dir: str = Config.EUCLID_PSF_DIR) -> str:
    """Return the canonical file path for ``band``'s empirical ePSF FITS."""
    return os.path.join(psf_dir, band.psf_fits_filename)


# ---------------------------------------------------------------------------
# Per-band kernel sizing (single source of truth)
# ---------------------------------------------------------------------------

def psf_half_pixels_for_band(
    band: BandConfig, pixel_scale: float = Config.DEFAULT_PIXEL_SCALE,
) -> int:
    """Half-side of ``band``'s PSF kernel in pixels at ``pixel_scale``.

    Driven by :attr:`Config.PSF_HALF_SUPPORT_FWHM_FACTOR`:
        half = max(PSF_MIN_HALF_PIXELS,
                   ceil(factor × fwhm / pixel_scale))

    The full kernel side is ``2 × half + 1`` (always odd, so the centre
    pixel sits on the centroid).
    """
    if pixel_scale <= 0:
        raise ValueError(f"pixel_scale must be positive, got {pixel_scale}")
    factor = float(Config.PSF_HALF_SUPPORT_FWHM_FACTOR)
    half = int(math.ceil(factor * band.psf_fwhm_arcsec / pixel_scale))
    return max(int(Config.PSF_MIN_HALF_PIXELS), half)


def psf_side_pixels_for_band(
    band: BandConfig, pixel_scale: float = Config.DEFAULT_PIXEL_SCALE,
) -> int:
    """Full odd kernel side ``2 × half + 1`` for ``band`` at ``pixel_scale``."""
    return 2 * psf_half_pixels_for_band(band, pixel_scale) + 1


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def make_gaussian_psf(
    fwhm_arcsec: float, pixel_scale: float, *, size: int | None = None,
) -> PSF:
    """Build a normalised Gaussian :class:`PSF` stamp at ``pixel_scale``.

    ``size`` (full odd side in pixels) is auto-derived from
    :attr:`Config.PSF_HALF_SUPPORT_FWHM_FACTOR` when omitted, so the
    kernel side scales with the band FWHM rather than being clipped to a
    fixed 65×65. Pass ``size`` explicitly only for tests / fixed-size
    fixtures.
    """
    if size is None:
        factor = float(Config.PSF_HALF_SUPPORT_FWHM_FACTOR)
        half = max(int(Config.PSF_MIN_HALF_PIXELS),
                   int(math.ceil(factor * fwhm_arcsec / pixel_scale)))
        size = 2 * half + 1
    elif size % 2 == 0:
        size += 1
    sigma_pix = fwhm_arcsec / pixel_scale / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    half = size // 2
    yy, xx = np.indices((size, size), dtype=np.float64) - half
    g = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma_pix * sigma_pix))
    g /= g.sum()
    return PSF(
        data=g.astype(np.float32),
        pixel_scale=pixel_scale,
        fwhm_arcsec=fwhm_arcsec,
        oversampling=1,
    )


def load_band_psf(
    band: BandConfig,
    *,
    target_pixel_scale: float = Config.DEFAULT_PIXEL_SCALE,
    psf_dir: str = Config.EUCLID_PSF_DIR,
    require_empirical: bool = False,
) -> PSF:
    """Return the best available PSF for ``band`` at ``target_pixel_scale``.

    Resolution order:

      1. ``psf_dir / band.psf_fits_filename`` exists → load it, normalise,
         clean background, resample to ``target_pixel_scale`` if needed.
      2. Otherwise, build a Gaussian PSF at ``band.psf_fwhm_arcsec``.

    With ``require_empirical=True`` the second branch raises instead of
    falling back — useful for production runs where we don't want the
    network silently trained on Gaussian NISP PSFs.
    """
    path = psf_path_for_band(band, psf_dir)
    if os.path.isfile(path):
        # Use whatever size was produced at extraction time — the extractor
        # is the single source of truth for empirical-PSF dimensions.
        psf = PSF.from_fits(path)
        psf = psf.background_cleaned()
        return psf.resampled_to(target_pixel_scale)

    if require_empirical:
        raise FileNotFoundError(
            f"No empirical PSF for band {band.name} at {path}. "
            "Run `polish extract-psf --band ... --cutout-dir ...` or pass "
            "require_empirical=False to allow a Gaussian fallback."
        )
    # Gaussian fallback: size auto-derived from FWHM (the band's
    # ``psf_fwhm_arcsec`` × ``PSF_HALF_SUPPORT_FWHM_FACTOR``).
    target_side = psf_side_pixels_for_band(band, target_pixel_scale)
    return make_gaussian_psf(band.psf_fwhm_arcsec, target_pixel_scale,
                             size=target_side)


def load_all_band_psfs(
    *,
    target_pixel_scale: float = Config.DEFAULT_PIXEL_SCALE,
    psf_dir: str = Config.EUCLID_PSF_DIR,
    require_empirical: bool = False,
) -> dict[str, PSF]:
    """Load PSFs for every band in :attr:`Config.BANDS`.

    Returned dict is keyed by band name (``'VIS'`` etc.) and is ready to
    pass to :class:`euclid_polish.sky.observation_simulator.ObservationSimulator`.
    """
    return {
        band.name: load_band_psf(
            band,
            target_pixel_scale=target_pixel_scale,
            psf_dir=psf_dir,
            require_empirical=require_empirical,
        )
        for band in Config.BANDS
    }


# ---------------------------------------------------------------------------
# PSF *sets* (position-dependent ensembles)
# ---------------------------------------------------------------------------

def load_band_psf_set(
    band: BandConfig,
    *,
    target_pixel_scale: float = Config.DEFAULT_PIXEL_SCALE,
    psf_dir: str = Config.EUCLID_PSF_DIR,
    require_empirical: bool = False,
) -> PSFSet:
    """Return the position-dependent :class:`PSFSet` for ``band``.

    Resolution order mirrors :func:`load_band_psf`:

      1. ``psf_dir / band.psf_fits_filename`` exists → ``PSFSet.from_fits``
         (reads the K cluster PSFs from the multi-extension file; a legacy
         single-PSF FITS loads as a 1-element set), resampled to
         ``target_pixel_scale``.
      2. Otherwise a Gaussian fallback wrapped as a **1-element** set.

    A 1-element set behaves exactly like the single PSF (its ``sample`` is
    deterministic), so callers can always treat a band uniformly as a set.
    """
    path = psf_path_for_band(band, psf_dir)
    if os.path.isfile(path):
        # background_cleaned applies the radial cosine edge taper while
        # preserving faint wings; the Gaussian fallback is analytically clean.
        return (PSFSet.from_fits(path)
                .background_cleaned()
                .resampled_to(target_pixel_scale))

    if require_empirical:
        raise FileNotFoundError(
            f"No empirical PSF for band {band.name} at {path}. "
            "Run the extraction step or pass require_empirical=False."
        )
    target_side = psf_side_pixels_for_band(band, target_pixel_scale)
    gaussian = make_gaussian_psf(band.psf_fwhm_arcsec, target_pixel_scale,
                                 size=target_side)
    return PSFSet.from_psfs([gaussian])


def load_all_band_psf_sets(
    *,
    target_pixel_scale: float = Config.DEFAULT_PIXEL_SCALE,
    psf_dir: str = Config.EUCLID_PSF_DIR,
    require_empirical: bool = False,
) -> dict[str, PSFSet]:
    """Load a :class:`PSFSet` for every band in :attr:`Config.BANDS`.

    Ready to pass to :class:`euclid_polish.sky.observation_simulator.ObservationSimulator`
    as ``psf_sets_by_band`` so generation draws one random cluster PSF per scene
    (optionally roll-rotated; no blending).
    """
    return {
        band.name: load_band_psf_set(
            band,
            target_pixel_scale=target_pixel_scale,
            psf_dir=psf_dir,
            require_empirical=require_empirical,
        )
        for band in Config.BANDS
    }


# ---------------------------------------------------------------------------
# Inventory (for the CLI to report which bands have empirical PSFs)
# ---------------------------------------------------------------------------

def psf_inventory(psf_dir: str = Config.EUCLID_PSF_DIR) -> dict[str, str | None]:
    """Report which bands currently have an empirical PSF on disk.

    Returns ``{band_name: path or None}``. ``None`` means a Gaussian
    fallback would be used. Useful for CLI introspection.
    """
    out: dict[str, str | None] = {}
    for band in Config.BANDS:
        path = psf_path_for_band(band, psf_dir)
        out[band.name] = path if os.path.isfile(path) else None
    return out
