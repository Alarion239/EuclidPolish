"""Redshift realism for TNG SKIRT stamps: one z draw drives every observable.

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

The same module derives the lens **velocity dispersion from the TNG mass
catalog** (``tng_properties.csv``) via the Faber–Jackson relation, so a
TNG-lit lens galaxy bends light according to its own subhalo's mass.

The cosmological-distance helpers (flat ΛCDM, Collett-2015 parameters) live
here and are re-exported by
:mod:`euclid_polish.sky.generation.lens_population`.
"""

from __future__ import annotations

import csv
import math
import os
from collections.abc import Sequence

import numpy as np
from scipy.integrate import trapezoid

from euclid_polish.config import Config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: SKIRT atlas native pixel pitch (physical).
TNG_NATIVE_PC_PER_PIXEL = 100.0

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
# 0) Downsample factor from redshift
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


# ---------------------------------------------------------------------------
# Field redshift distribution
# ---------------------------------------------------------------------------

#: Inverse-CDF grids keyed by the sampler parameters; built once per set.
_ZCDF_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def _z_inverse_cdf_grid(form: str, z0: float, phi_scale: float,
                        z_min: float, z_max: float,
                        n: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    key = (str(form), float(z0), float(phi_scale), float(z_min), float(z_max))
    grid = _ZCDF_CACHE.get(key)
    if grid is None:
        zs = np.linspace(z_min, z_max, n)
        if form == "smail":
            pdf = zs ** 2 * np.exp(-((zs / z0) ** 1.5))
        elif form == "volume":
            # dV_c/dz ∝ D_C(z)²/E(z) (Hogg 1999, eq. 28), times the declining
            # comoving number density of massive galaxies.
            E = np.sqrt(Config.LENS_COSMOLOGY_OMEGA_M * (1.0 + zs) ** 3
                        + Config.LENS_COSMOLOGY_OMEGA_L)
            # D_C (Mpc) up to each grid point: the exact value at z_min plus
            # a cumulative trapezoid of the Hubble distance over the grid.
            DH = 299_792.458 / Config.LENS_COSMOLOGY_H0
            dc0 = comoving_distance_mpc(z_min)
            dz = zs[1] - zs[0]
            dc = dc0 + DH * np.concatenate(
                ([0.0], np.cumsum((1.0 / E[1:] + 1.0 / E[:-1]) * 0.5 * dz)))
            pdf = dc ** 2 / E * np.exp(-((zs / phi_scale) ** 2))
        else:
            raise ValueError(f"unknown n(z) form {form!r}")
        cdf = np.cumsum(pdf)
        cdf -= cdf[0]
        cdf /= cdf[-1]
        grid = (cdf, zs)
        _ZCDF_CACHE[key] = grid
    return grid


def sample_galaxy_redshift(
    rng: np.random.Generator,
    *,
    form: str = Config.TNG_Z_FORM,
    z0: float = Config.TNG_Z0,
    phi_scale: float = Config.TNG_Z_PHI_SCALE,
    z_min: float = Config.TNG_Z_MIN,
    z_max: float = Config.TNG_Z_MAX,
) -> float:
    """Draw one redshift for a TNG stamp, truncated to ``[z_min, z_max]``.

    ``form="volume"`` (default): ``dN/dz ∝ dV_c/dz · exp(-(z/phi_scale)²)``
    — our default prior for the atlas's massive galaxies, which are visible
    across a large survey volume: comoving volume element times a smooth
    approximation to the Muzzin+ 2013 decline of the log M*≳11 number
    density. It is not a fitted TNG light cone. Median z ≈ 1.15; about 5%
    of draws land below z = 0.4
    (where atlas giants
    appear arcsec-sized).

    ``form="smail"``: ``n(z) ∝ z² exp(-(z/z0)^1.5)`` — the full
    flux-limited population (Smail+ 1995; Euclid Red Book median 0.9 with
    z0 = 0.65). Over-draws low z for the massive-only atlas.

    Sampling is by inverse CDF on a cached grid; ``z_min = 0.10`` is where
    the 100 pc native pixel matches the 0.05″ HR grid (rebin factor 1).
    """
    cdf, zs = _z_inverse_cdf_grid(form, z0, phi_scale, z_min, z_max)
    return float(np.interp(rng.random(), cdf, zs))


#: VIS-magnitude anchor: atlas subhalo 167396 (log M* = 10.55) measures
#: m_VIS = 21.36 through the full pipeline at z = 0.5.
_MAG_ANCHOR_LOGM = 10.55
_MAG_ANCHOR_Z    = 0.5
_MAG_ANCHOR_VIS  = 21.36


def predicted_vis_mag(logm: float, z: float) -> float:
    """Coarse VIS magnitude of a ``logm`` galaxy at redshift ``z``:
    L ∝ M plus the luminosity-distance modulus relative to the measured
    anchor (no K-correction). Used to skip undetectably faint field draws
    before the expensive 4-band stamp load.
    """
    dl = (1.0 + z) * comoving_distance_mpc(z)
    dl0 = (1.0 + _MAG_ANCHOR_Z) * comoving_distance_mpc(_MAG_ANCHOR_Z)
    return (_MAG_ANCHOR_VIS + 2.5 * (_MAG_ANCHOR_LOGM - logm)
            + 5.0 * math.log10(dl / dl0))


#: Inverse-CDF grid for the Schechter mass draw, keyed by its parameters.
_MFCDF_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def sample_target_logmass(
    rng: np.random.Generator,
    *,
    logm_star: float = Config.TNG_MF_LOGM_STAR,
    alpha: float = Config.TNG_MF_ALPHA,
    logm_min: float = Config.TNG_MF_LOGM_MIN,
    logm_max: float = Config.TNG_MF_LOGM_MAX,
) -> float:
    """Draw one log10 stellar mass from the Schechter mass function,
    ``φ dlogM ∝ (M/M*)^(α+1) e^(-M/M*)`` (Baldry+ 2012 / Muzzin+ 2013),
    truncated to ``[logm_min, logm_max]``. The mass-rescaled TNG field
    population follows the observed mass distribution: giants only in the
    rare exponential tail.
    """
    key = (float(logm_star), float(alpha), float(logm_min), float(logm_max))
    grid = _MFCDF_CACHE.get(key)
    if grid is None:
        lm = np.linspace(logm_min, logm_max, 2048)
        x = 10.0 ** (lm - logm_star)
        pdf = x ** (alpha + 1.0) * np.exp(-x)
        cdf = np.cumsum(pdf)
        cdf -= cdf[0]
        cdf /= cdf[-1]
        grid = (cdf, lm)
        _MFCDF_CACHE[key] = grid
    cdf, lm = grid
    return float(np.interp(rng.random(), cdf, lm))


# ---------------------------------------------------------------------------
# 2) Photometric response to redshift: dimming + randomized spectral drift
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

def default_tng_properties_csv() -> str:
    """The property cache written by the TNG-infographic render
    (``euclid_polish.tng.properties.gather_properties``)."""
    return os.path.join(Config.DATA_DIR, "_tng_infographics",
                        "tng_properties.csv")


#: Per-path property cache: {csv_path: {subhalo_id: {field: value}}}.
_TNG_PROPS_CACHE: dict[str, dict[str, dict[str, float]]] = {}


def load_tng_properties(csv_path: str | None = None,
                        ) -> dict[str, dict[str, float]]:
    """Read ``tng_properties.csv`` → ``{subhalo_id: {sfr, mass_stars, m_halo,
    reff}}`` (floats, NaN where missing). Missing file → empty dict. Cached
    per path; the generator looks masses up per lens draw.
    """
    path = csv_path or default_tng_properties_csv()
    cached = _TNG_PROPS_CACHE.get(path)
    if cached is not None:
        return cached
    rows: dict[str, dict[str, float]] = {}
    if os.path.isfile(path):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                gid = str(row.get("id", "")).strip()
                if not gid:
                    continue
                props: dict[str, float] = {}
                for k, v in row.items():
                    if k == "id":
                        continue
                    try:
                        props[k] = float(v)
                    except (TypeError, ValueError):
                        props[k] = float("nan")
                rows[gid] = props
    _TNG_PROPS_CACHE[path] = rows
    return rows


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
