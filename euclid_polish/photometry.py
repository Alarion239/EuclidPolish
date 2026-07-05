"""THE canonical unit-conversion module — every photometric conversion in the
codebase MUST route through here (or through the ``BandConfig`` properties it
builds on). No other module may re-derive these formulas inline.

The one scale the model trains in is **electrons over the full stack**: the
total electron count a source deposits across a band's complete integration
(``BandConfig.t_total_s`` = exposure_time × n_exposures). Everything converts
onto or off that scale through exactly one anchor per band:

    ``band.sim_zeropoint_e`` — the AB magnitude of a source contributing
    exactly 1 e⁻ over the stack, defined ONCE in config as
    ``zeropoint_ab_e_per_s + 2.5·log10(t_total_s)``, where
    ``zeropoint_ab_e_per_s`` is the instrument's calibrated electron-rate
    zeropoint (VIS: 25.92 from the Q1 VIS PF ADU zeropoint 24.57 s⁻¹ and
    gain 3.48 e⁻/ADU, McCracken+ 2025; NISP: Schirmer+ 2022/2025).

Supported conversions (and their exact provenance):

  * **AB magnitude ↔ electrons** — the AB definition
    ``m = ZP − 2.5·log10(flux)`` applied at the stack zeropoint.
  * **µJy ↔ AB magnitude** — the AB system is *defined* by
    ``m = 8.90 − 2.5·log10(F[Jy])`` (Oke & Gunn 1983 normalisation,
    f_ν(AB 0) ≈ 3631 Jy); in µJy the constant is exactly
    ``8.90 + 2.5·log10(1e6) = 23.90`` (``Config.AB_ZP_UJY``). Euclid MER
    catalogue flux columns (``flux_vis_psf``, ``flux_*_aper``, …) are µJy.
  * **µJy → electrons** — composition of the two above (single expression so
    no intermediate rounding).
  * **Archive ADU/s → electrons** — Euclid archive mosaics are served in
    ADU/s calibrated to their FITS ``MAGZERO`` keyword (1 ADU/s ⇔
    m_AB = MAGZERO). Matching magnitudes on both scales gives the factor
    ``10^((sim_zeropoint_e − MAGZERO)/2.5)`` (VIS: ≈7.6e3 at the Q1
    mosaics' MAGZERO = 24.6); validated
    against MER catalogue photometry by ``scripts/verify_star_photometry.py``.
  * **MJy/sr (surface brightness) → electrons/pixel** — for SKIRT mocks
    (``BUNIT = MJy/sr``); intensive quantity × pixel solid angle → µJy →
    electrons.

Import discipline: this module depends on ``config`` and ``numpy`` ONLY — no
astroquery, no TF — so cluster scripts and generators can import it without
side effects. (It lived at ``catalog/photometry.py`` until 2026-07-05, where
importing it dragged in the astroquery-backed archive client via the package
``__init__`` — which is why stray inline copies of these formulas existed.)

The client-side viewer (``static/cutout_viewer.js``) necessarily mirrors the
display math in JS; it consumes the constants served by
``viewer_data.color_constants()`` (including the precomputed
``zeropoint_ab_e_total``) and must not hard-code any of them.
"""

from __future__ import annotations

import math

import numpy as np

from euclid_polish.config import BandConfig, Config


# --------------------------------------------------------------------------- #
# AB magnitude ↔ µJy (pure AB-system definition, band-independent)
# --------------------------------------------------------------------------- #

def uJy_to_ab_mag(flux_uJy: float) -> float:
    """AB magnitude of a flux quoted in microJansky.

    ``mag = Config.AB_ZP_UJY − 2.5·log10(flux_µJy)`` with ``AB_ZP_UJY = 23.90``
    exactly (the AB definition ``m = 8.90 − 2.5·log10(F[Jy])`` re-expressed in
    µJy). Inverse of :func:`ab_mag_to_uJy`.
    """
    return float(Config.AB_ZP_UJY - 2.5 * math.log10(float(flux_uJy)))


def ab_mag_to_uJy(mag: float) -> float:
    """Flux in microJansky of an AB magnitude — exact inverse of
    :func:`uJy_to_ab_mag`: ``F_µJy = 10^((AB_ZP_UJY − mag)/2.5)``."""
    return float(10.0 ** ((Config.AB_ZP_UJY - float(mag)) / 2.5))


# --------------------------------------------------------------------------- #
# AB magnitude ↔ electrons over the stack (anchored at band.sim_zeropoint_e)
# --------------------------------------------------------------------------- #

def ab_mag_to_electrons(mag, band: BandConfig):
    """AB magnitude → flux in electrons over ``band``'s stack.

    ``flux_e = 10^(−0.4·(mag − band.sim_zeropoint_e))``. A source at
    ``mag == sim_zeropoint_e`` yields exactly 1 e⁻; 2.5 mag brighter is 10×.
    Accepts scalars or arrays; inverse of :func:`electrons_to_ab_mag`.
    """
    out = 10.0 ** (-0.4 * (np.asarray(mag, dtype=np.float64)
                           - band.sim_zeropoint_e))
    return float(out) if np.ndim(mag) == 0 else out


def electrons_to_ab_mag(flux_e, band: BandConfig):
    """Flux in electrons over ``band``'s stack → AB magnitude.

    ``mag = band.sim_zeropoint_e − 2.5·log10(flux_e)`` — exact inverse of
    :func:`ab_mag_to_electrons`. Accepts scalars or arrays; non-positive
    fluxes give ``nan`` (no magnitude exists) rather than raising.
    """
    f = np.asarray(flux_e, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = band.sim_zeropoint_e - 2.5 * np.log10(np.where(f > 0, f, np.nan))
    return float(out) if np.ndim(flux_e) == 0 else out


# --------------------------------------------------------------------------- #
# µJy → electrons (composition, kept as one expression)
# --------------------------------------------------------------------------- #

def uJy_to_electrons(flux_uJy: float, band: BandConfig) -> float:
    """Catalogue flux (µJy, AB) → electrons over ``band``'s stack — the scale
    the model and the star-anchor delta-targets use. Routes through the AB
    system: ``e⁻ = flux_µJy · 10^(0.4·(band.sim_zeropoint_e − AB_ZP_UJY))``.
    Algebraically identical to ``ab_mag_to_electrons(uJy_to_ab_mag(f), band)``
    (pinned by a test) but avoids the intermediate log/exp round trip."""
    return float(flux_uJy) * float(
        10.0 ** (0.4 * (band.sim_zeropoint_e - Config.AB_ZP_UJY)))


# --------------------------------------------------------------------------- #
# Archive ADU/s (MAGZERO-calibrated) → electrons over the stack
# --------------------------------------------------------------------------- #

def adu_per_s_to_electrons_factor(magzero: float, band: BandConfig) -> float:
    """Multiplicative factor taking archive ADU/s pixels (calibrated so that
    1 ADU/s ⇔ m_AB = the image's ``MAGZERO``) to electrons over ``band``'s
    stack: ``10^((band.sim_zeropoint_e − magzero)/2.5)``.

    Derivation: a source of magnitude m gives ``10^((MAGZERO − m)/2.5)`` ADU/s
    and must give ``10^((sim_zeropoint_e − m)/2.5)`` electrons-over-stack;
    the ratio is the m-independent factor above (VIS: ≈7.6e3 at the Q1
    mosaics' MAGZERO = 24.6 under the 25.92 e⁻/s zeropoint).
    """
    return float(10.0 ** ((band.sim_zeropoint_e - float(magzero)) / 2.5))


def adu_per_s_to_electrons(arr: np.ndarray, magzero: float,
                           band: BandConfig) -> np.ndarray:
    """Convert an archive ADU/s image to electrons over ``band``'s stack."""
    factor = np.float32(adu_per_s_to_electrons_factor(magzero, band))
    return (np.asarray(arr, dtype=np.float32) * factor).astype(np.float32)


# --------------------------------------------------------------------------- #
# Surface brightness (MJy/sr) → electrons per pixel
# --------------------------------------------------------------------------- #

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
