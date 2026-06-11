"""Photometric conversions onto the one scale the model trains in:
**electrons over the full stack**.

The stack zero-point ``band.sim_zeropoint_e`` is the AB magnitude of a
source contributing 1 e⁻ over the band's full integration
(``Config.sim_zeropoint_e``; VIS ≈ 33.88). Two inputs convert onto this
single scale:

  * **Catalog magnitudes** (Euclid MER, AB) → electrons, used to build the
    star-anchor delta targets.
  * **Archive ADU/s images** (each calibrated by its FITS ``MAGZERO``
    keyword) → electrons, used to feed real cutouts to the model.

Keeping both conversions in one place is what makes the star-anchor target
(from catalog mag) and the model's input (from ADU/s) photometrically
consistent — see ``scripts/verify_star_photometry.py``.
"""

from __future__ import annotations

import math

import numpy as np

from euclid_polish.config import BandConfig, Config


def uJy_to_ab_mag(flux_uJy: float) -> float:
    """AB magnitude of a flux quoted in microJansky:
    ``mag = Config.AB_ZP_UJY − 2.5·log10(flux_µJy)`` (3631 Jy = AB 0)."""
    return float(Config.AB_ZP_UJY - 2.5 * math.log10(float(flux_uJy)))


def uJy_to_electrons(flux_uJy: float, band: BandConfig) -> float:
    """Catalogue flux (µJy, AB) → electrons over ``band``'s stack — the scale
    the model and the star-anchor delta-targets use. Routes through the AB
    system: ``e⁻ = flux_µJy · 10**(0.4·(band.sim_zeropoint_e − AB_ZP_UJY))``.
    Equivalent to ``ab_mag_to_electrons(uJy_to_ab_mag(flux_µJy), band)``."""
    return float(flux_uJy) * float(
        10.0 ** (0.4 * (band.sim_zeropoint_e - Config.AB_ZP_UJY)))


def ab_mag_to_electrons(mag: float, band: BandConfig) -> float:
    """AB magnitude → flux in electrons over ``band``'s stack.

    ``flux_e = 10 ** (-0.4 · (mag − band.sim_zeropoint_e))``. A source at
    ``mag == sim_zeropoint_e`` yields exactly 1 e⁻; 2.5 mag brighter is 10×.
    """
    return float(10.0 ** (-0.4 * (float(mag) - band.sim_zeropoint_e)))


def adu_per_s_to_electrons_factor(magzero: float, band: BandConfig) -> float:
    """Multiplicative factor taking archive ADU/s pixels (calibrated to the
    image's ``MAGZERO``) to electrons over ``band``'s stack:
    ``10 ** ((band.sim_zeropoint_e − magzero) / 2.5)``.
    """
    return float(10.0 ** ((band.sim_zeropoint_e - float(magzero)) / 2.5))


def adu_per_s_to_electrons(arr: np.ndarray, magzero: float,
                           band: BandConfig) -> np.ndarray:
    """Convert an archive ADU/s image to electrons over ``band``'s stack."""
    factor = np.float32(adu_per_s_to_electrons_factor(magzero, band))
    return (np.asarray(arr, dtype=np.float32) * factor).astype(np.float32)


def pixel_solid_angle_sr(pixel_scale_arcsec: float) -> float:
    """Solid angle of a square pixel, in steradian.

    ``Ω = (s · π / 180 / 3600)²`` for a pixel ``s`` arcsec on a side.
    """
    s_rad = float(pixel_scale_arcsec) * math.pi / 180.0 / 3600.0
    return s_rad * s_rad


def mjy_per_sr_to_electrons_factor(band: BandConfig,
                                   pixel_scale_arcsec: float) -> float:
    """Multiplicative factor taking a **surface-brightness** image in MJy/sr
    (e.g. a SKIRT mock, ``BUNIT = MJy/sr``) to electrons-per-pixel over
    ``band``'s stack, given the angular scale the image is placed on.

    Surface brightness is *intensive*: the per-pixel flux is the brightness
    times the pixel solid angle, so the assigned ``pixel_scale_arcsec`` sets
    both the apparent size and the integrated flux of the source. Routing
    through the AB system (``1 MJy/sr · Ω`` → µJy → electrons)::

        e⁻ = I[MJy/sr] · 1e12 · Ω_pix · 10**(0.4·(band.sim_zeropoint_e − AB_ZP_UJY))

    where ``1e12`` is MJy→Jy (1e6) then Jy→µJy (1e6) and ``Ω_pix`` is
    :func:`pixel_solid_angle_sr`.
    """
    omega = pixel_solid_angle_sr(pixel_scale_arcsec)
    ujy_per_mjy_sr = 1.0e12 * omega          # µJy per (MJy/sr) at this pixel
    return float(ujy_per_mjy_sr * 10.0 ** (
        0.4 * (band.sim_zeropoint_e - Config.AB_ZP_UJY)))


def mjy_per_sr_to_electrons(arr: np.ndarray, band: BandConfig,
                            pixel_scale_arcsec: float) -> np.ndarray:
    """Convert a surface-brightness image (MJy/sr) to electrons over
    ``band``'s stack at the given angular pixel scale."""
    factor = np.float32(mjy_per_sr_to_electrons_factor(band, pixel_scale_arcsec))
    return (np.asarray(arr, dtype=np.float32) * factor).astype(np.float32)
