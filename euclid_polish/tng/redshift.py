"""Redshift transformations for TNG atlas stamps.

The SKIRT atlas frames are *intrinsic* (z = 0) physical images — 100 pc/pixel
surface brightness in MJy/sr with no assumed distance. To place one at
redshift ``z`` every observable is derived from that single draw:

1. **Angular size** — a native pixel subtends ``100 pc / D_A(z)``, so the
   block-mean factor that lands the stamp on the 0.05″ HR grid is
   ``F(z) = θ_pix_HR[rad] · D_A(z) / 100 pc`` (:func:`rebin_factor_for_redshift`).
   D_A turns over at z ≈ 1.6, so F never exceeds ≈ 4.3 — distant galaxies
   stop shrinking, as on the real sky.
2. **Tolman dimming** — per-frequency surface brightness dims as
   ``I_ν,obs(ν_obs) = I_ν,em(ν_em) / (1+z)³`` (:func:`tolman_dimming_factor`).
3. **Spectral drift** — observed band b samples the rest-frame SED at
   ``λ_b / (1+z)``. A true K-correction is impossible with four bands, so
   :func:`band_drift_factors` combines (a) a deterministic part that
   log-log-interpolates the stamp's *own* 4-point SED at the blueshifted
   wavelengths, and (b) a stochastic tilt ``exp(ε · ln(λ_b/λ_H))`` with
   ``ε ~ N(0, σ(z))``, anchored at H (the band least exposed to the rest-UV
   extrapolation). The randomness broadens the colour distribution around the
   deterministic four-point estimate.

The same module derives lens velocity dispersion from a supplied stellar mass
through the Faber--Jackson relation. Atlas catalog ownership stays in
:mod:`euclid_polish.tng.catalog`.

The cosmological-distance helpers use the flat ΛCDM Collett-2015 parameters
and are shared with the strong-lens geometry sampler.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.integrate import trapezoid

from euclid_polish.config import Config
from euclid_polish.tng.types import TNG_NATIVE_PC_PER_PIXEL

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Pivot wavelengths (µm) of the four Euclid bands, in
#: ``Config.LR_INPUT_BAND_NAMES`` order (VIS, Y_E, J_E, H_E) — monotonically
#: increasing, which the drift interpolation relies on.
PIVOT_WAVELENGTH_UM: tuple[float, ...] = (0.715, 1.081, 1.367, 1.773)

_ARCSEC_TO_RAD = math.pi / (180.0 * 3600.0)
_PC_PER_MPC = 1.0e6


# ---------------------------------------------------------------------------
# Cosmological distances (flat ΛCDM, Collett-2015 cosmology)
# ---------------------------------------------------------------------------

def comoving_distance_mpc(
    z: float,
    *,
    H0: float = Config.LENS_COSMOLOGY_H0,
    Omega_m: float = Config.LENS_COSMOLOGY_OMEGA_M,
    Omega_L: float = Config.LENS_COSMOLOGY_OMEGA_L,
    n_int: int = 1024,
) -> float:
    """Line-of-sight comoving distance (Mpc), integrated numerically."""
    c_kms = 299_792.458
    DH = c_kms / H0       # Hubble distance, Mpc
    zs = np.linspace(0.0, z, n_int + 1)
    E = np.sqrt(Omega_m * (1.0 + zs) ** 3 + Omega_L)
    integrand = 1.0 / E
    return DH * float(trapezoid(integrand, zs))


def angular_diameter_distance(z1: float, z2: float | None = None) -> float:
    """Angular-diameter distance D_A (Mpc).

    If ``z2`` is ``None``: from observer (z=0) to z1. Otherwise from z1 to z2
    in a flat cosmology, via D_A(z1,z2) = (D_C(z2) - D_C(z1)) / (1+z2).
    """
    if z2 is None:
        return comoving_distance_mpc(z1) / (1.0 + z1)
    return (comoving_distance_mpc(z2) - comoving_distance_mpc(z1)) / (1.0 + z2)


def physical_pc_to_arcsec(length_pc: float, z: float) -> float:
    """Apparent angular size (arcsec) of a physical length at redshift ``z``."""
    da_pc = angular_diameter_distance(z) * _PC_PER_MPC
    if da_pc <= 0.0:
        return float("inf")
    return float(length_pc / da_pc / _ARCSEC_TO_RAD)


# ---------------------------------------------------------------------------
# Downsample factor from redshift
# ---------------------------------------------------------------------------

def rebin_factor_for_redshift(
    z: float,
    *,
    pixel_scale_arcsec: float = Config.DEFAULT_PIXEL_SCALE,
    native_pc_per_pixel: float = TNG_NATIVE_PC_PER_PIXEL,
) -> float:
    """Continuous block-mean factor placing a 100 pc/px stamp at redshift ``z``
    on the ``pixel_scale_arcsec`` grid.

    One output pixel must span ``F`` native pixels = ``F · 100 pc``, and that
    length must subtend ``pixel_scale_arcsec`` at D_A(z):
    ``F = θ_HR[rad] · D_A(z) / 100 pc``. Floored at 1, since the native grid
    cannot be upsampled — galaxies closer than z ≈ 0.10 render slightly
    smaller than physical. The caller stochastically rounds to an integer.
    """
    da_pc = angular_diameter_distance(z) * _PC_PER_MPC
    f = pixel_scale_arcsec * _ARCSEC_TO_RAD * da_pc / native_pc_per_pixel
    return max(1.0, float(f))


def compactness_factor(
    z: float,
    *,
    c0: float = Config.TNG_COMPACT_C0,
    beta: float = Config.TNG_COMPACT_BETA,
) -> float:
    """Size-evolution correction ``C(z) = c0·(1+z)^beta`` for the z = 0
    atlas morphologies: real galaxies at redshift ``z`` are more compact at
    fixed mass (van der Wel+ 2014). Applied as extra downsampling on top of
    the geometric F(z), flux-conserving (SB × C²); z, dimming and drift
    are untouched.
    """
    return float(c0 * (1.0 + z) ** beta)


# Photometric response to redshift: dimming + randomized spectral drift
# ---------------------------------------------------------------------------

def tolman_dimming_factor(z: float) -> float:
    """Cosmological surface-brightness dimming for per-frequency intensity.

    ``I_ν,obs(ν_obs) = I_ν,em((1+z)·ν_obs) / (1+z)³`` — this returns the
    (1+z)³ factor only; the band-shift part is :func:`band_drift_factors`'
    job. The SKIRT frames are MJy/sr (per-frequency), so the two factors
    together are the complete photometric response.
    """
    return float((1.0 + z) ** -3)


def band_drift_factors(
    sed_fnu: Sequence[float],
    z: float,
    rng: np.random.Generator | None = None,
    *,
    sigma0: float = Config.TNG_DRIFT_SIGMA0,
    sigma_slope: float = Config.TNG_DRIFT_SIGMA_SLOPE,
    parametric_k: float = Config.TNG_DRIFT_PARAMETRIC_K,
    include_dimming: bool = True,
    max_lnln_slope: float = 6.0,
    ratio_clip: tuple[float, float] = (1e-2, 1e2),
) -> tuple[np.ndarray, dict]:
    """Multiplicative per-band factors modelling redshift ``z``'s photometry.

    Three pieces per band (physics in the module docstring): the
    deterministic drift — the stamp's own SED log-log-interpolated at
    ``λ_b/(1+z)``, edge slope continued (clamped to ``±max_lnln_slope``)
    below the bluest point, the parametric tilt
    ``exp(k·ln(1+z)·ln(λ_b/λ_H))`` standing in for an unusable SED —
    times the stochastic tilt ``exp(ε·ln(λ_b/λ_H))`` with
    ``ε ~ N(0, σ0 + σ1·ln(1+z))``, times Tolman dimming ``(1+z)⁻³``
    (skippable for tests).

    Parameters
    ----------
    sed_fnu : 4 relative rest-frame f_ν values in ``LR_INPUT_BAND_NAMES``
        order (any common scale).
    z       : assigned redshift (the SKIRT frames are intrinsic, z = 0).
    rng     : drives the stochastic tilt; ``None`` → deterministic part only
        (so z = 0 with no rng returns exactly ones).

    Returns ``(factors[4] float64, meta)``; meta records the mode, the tilt
    ε, and the dimming applied.
    """
    lam = np.asarray(PIVOT_WAVELENGTH_UM, dtype=np.float64)
    ln_lam = np.log(lam)
    x = ln_lam - ln_lam[-1]                       # ln(λ_b / λ_H)  ≤ 0
    lnz1 = math.log1p(z)

    sed = np.asarray(sed_fnu, dtype=np.float64)
    usable = sed.shape == lam.shape and bool(
        np.all(np.isfinite(sed)) and np.all(sed > 0.0))
    if usable:
        ln_s = np.log(sed)
        ln_q = ln_lam - lnz1                      # blueshifted query points
        ln_interp = np.interp(ln_q, ln_lam, ln_s)
        # np.interp clamps below the bluest point; continue the edge slope
        # instead (slope clamped, since the rest-UV is least constrained).
        below = ln_q < ln_lam[0]
        if np.any(below):
            slope = (ln_s[1] - ln_s[0]) / (ln_lam[1] - ln_lam[0])
            slope = float(np.clip(slope, -max_lnln_slope, max_lnln_slope))
            ln_interp[below] = ln_s[0] + slope * (ln_q[below] - ln_lam[0])
        ratio = np.exp(ln_interp - ln_s)
        mode = "sed_interp"
    else:
        ratio = np.exp(parametric_k * lnz1 * x)
        mode = "parametric"
    ratio = np.clip(ratio, ratio_clip[0], ratio_clip[1])

    eps = 0.0
    if rng is not None:
        sigma = sigma0 + sigma_slope * lnz1
        eps = float(rng.normal(0.0, sigma))
    tilt = np.exp(eps * x)

    dimming = tolman_dimming_factor(z) if include_dimming else 1.0
    factors = dimming * ratio * tilt
    meta = {"drift_mode": mode, "drift_eps": float(eps),
            "dimming": float(dimming)}
    return factors, meta


# ---------------------------------------------------------------------------
# 1) Lens velocity dispersion from the TNG mass catalog
# ---------------------------------------------------------------------------

def sigma_v_from_stellar_mass(
    mstar_msun: float,
    rng: np.random.Generator | None = None,
    *,
    sigma_ref_kms: float = Config.LENS_FJ_SIGMA_REF_KMS,
    mstar_ref_msun: float = Config.LENS_FJ_MSTAR_REF_MSUN,
    slope: float = Config.LENS_FJ_SLOPE,
    scatter_dex: float = Config.LENS_FJ_SCATTER_DEX,
    clip_kms: tuple[float, float] = Config.LENS_SIGMA_V_CLIP_KMS,
) -> float:
    """Velocity dispersion (km/s) from stellar mass via Faber–Jackson:
    ``σ = σ_ref · (M*/M_ref)^slope`` with lognormal scatter, clipped to
    ``clip_kms``. NaN/non-positive mass → NaN (caller falls back to the
    uniform σ_v prior).
    """
    if not (isinstance(mstar_msun, (int, float)) and math.isfinite(mstar_msun)
            and mstar_msun > 0.0):
        return float("nan")
    sigma = sigma_ref_kms * (mstar_msun / mstar_ref_msun) ** slope
    if rng is not None and scatter_dex > 0.0:
        sigma *= 10.0 ** rng.normal(0.0, scatter_dex)
    return float(min(max(sigma, clip_kms[0]), clip_kms[1]))
