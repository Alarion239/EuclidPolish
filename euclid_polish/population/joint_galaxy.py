"""Joint analytical COSMOS--Euclid galaxy population calibration.

The latent population follows the classical bivariate construction: an
evolving Schechter luminosity function multiplied by a lognormal physical-size
distribution at fixed luminosity and redshift.  COSMOS constrains the latent
redshift dependence.  Euclid observes the same population after a photometric
response, a resolution-limited size response, and a surface-brightness-aware
selection function.

The cached COSMOS product does not contain rest-frame magnitudes.  Consequently
``M_eff = m_F814W - DM(z)`` is an observed-F814W absolute-like coordinate and
the fitted ``M_star`` evolution absorbs the mean K-correction.  This is kept
explicit in every artifact; it must not be interpreted as a rest-frame LF.
"""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.cosmology import Planck15
from scipy.optimize import least_squares, minimize, minimize_scalar
from scipy.special import gammaln, ndtr

from euclid_polish.photometry import ab_mag_to_uJy

COSMOS_FIT_MAG_MIN = 18.0
COSMOS_FIT_MAG_MAX = 27.25
COSMOS_FIT_Z_MIN = 0.05
COSMOS_FIT_Z_MAX = 5.5
COSMOS_AREA_ARCMIN2 = 0.54 * 3600.0

LF_Z_EDGES = np.linspace(COSMOS_FIT_Z_MIN, COSMOS_FIT_Z_MAX, 45)
LF_MAG_EDGES = np.arange(
    COSMOS_FIT_MAG_MIN, COSMOS_FIT_MAG_MAX + 1e-9, 0.25,
)
LATENT_Z_EDGES = np.linspace(COSMOS_FIT_Z_MIN, COSMOS_FIT_Z_MAX, 56)
LATENT_MAG_EDGES = np.arange(18.0, 30.0001, 0.20)
LATENT_LOG_RE_EDGES = np.arange(-2.40, 0.801, 0.08)
EUCLID_MAG_EDGES = np.arange(20.0, 28.0001, 0.25)
EUCLID_LOG_RE_EDGES = np.arange(np.log10(0.075), np.log10(1.5001), 0.10)
TNG_DRAW_VIS_MAG_EDGES = np.arange(18.0, 30.0001, 0.20)
BRIGHT_TRANSFER_MAG_MAX = 24.0


@dataclass(frozen=True)
class SchechterEvolutionFit:
    log_phi_star: float
    m_star_0: float
    alpha: float
    m_star_log1pz_slope: float
    log_phi_log1pz_slope: float
    alpha_log1pz_slope: float
    m_star_log1pz_quadratic: float
    log_phi_log1pz_quadratic: float
    cosmic_variance_fractional_scatter: float
    poisson_deviance: float
    negative_binomial_deviance: float
    dof: int
    standard_errors: tuple[float, ...]


@dataclass(frozen=True)
class SizeEvolutionFit:
    log10_r0_kpc: float
    magnitude_slope: float
    log1pz_slope: float
    magnitude_curvature: float
    magnitude_redshift_interaction: float
    scatter_dex: float
    scatter_magnitude_slope: float
    residual_rms_dex: float
    fitted_rows: int
    clipped_rows: int
    standard_errors: tuple[float, ...]


@dataclass(frozen=True)
class EuclidResponseFit:
    population_scale: float
    vis_minus_f814w_mag: float
    magnitude_slope: float
    scatter_mag: float
    measurement_flux_error_uJy: float
    size_scale: float
    size_floor_arcsec: float
    completeness_m50: float
    completeness_width_mag: float
    surface_brightness_penalty: float
    bright_transfer_magnitude_max: float
    bright_poisson_deviance: float
    bright_dof: int
    poisson_deviance: float
    dof: int
    standard_errors: tuple[float, ...]


def _finite_covariance_errors(result: Any, dof: int) -> tuple[float, ...]:
    """Approximate 1-sigma errors from a least-squares Jacobian."""
    try:
        jacobian = np.asarray(result.jac, dtype=np.float64)
        covariance = np.linalg.pinv(jacobian.T @ jacobian)
        scale = 2.0 * float(result.cost) / max(1, int(dof))
        errors = np.sqrt(np.maximum(np.diag(covariance) * scale, 0.0))
    except (ValueError, np.linalg.LinAlgError):
        errors = np.full(len(result.x), np.nan, dtype=np.float64)
    return tuple(float(value) for value in errors)


def signed_poisson_residual(
    observed: np.ndarray, predicted: np.ndarray,
) -> np.ndarray:
    """Signed square-root contribution to the Cash deviance."""
    count = np.asarray(observed, dtype=np.float64)
    mean = np.clip(np.asarray(predicted, dtype=np.float64), 1e-12, None)
    term = mean - count
    positive = count > 0.0
    term[positive] += count[positive] * np.log(
        count[positive] / mean[positive]
    )
    return np.sign(mean - count) * np.sqrt(np.maximum(2.0 * term, 0.0))


def negative_binomial_nll(
    observed: np.ndarray, predicted: np.ndarray, fractional_scatter: float,
) -> float:
    """Negative-binomial NLL with ``Var(N)=mu+(tau*mu)^2``."""
    count = np.asarray(observed, dtype=np.float64)
    mean = np.clip(np.asarray(predicted, dtype=np.float64), 1e-12, None)
    tau = max(float(fractional_scatter), 1e-8)
    shape = 1.0 / tau**2
    log_probability = (
        gammaln(count + shape) - gammaln(shape) - gammaln(count + 1.0)
        + shape * (math.log(shape) - np.log(shape + mean))
        + count * (np.log(mean) - np.log(shape + mean))
    )
    return -float(np.sum(log_probability))


def signed_negative_binomial_residual(
    observed: np.ndarray, predicted: np.ndarray, fractional_scatter: float,
) -> np.ndarray:
    """Signed square-root negative-binomial deviance contribution."""
    count = np.asarray(observed, dtype=np.float64)
    mean = np.clip(np.asarray(predicted, dtype=np.float64), 1e-12, None)
    tau = max(float(fractional_scatter), 1e-8)
    shape = 1.0 / tau**2
    with np.errstate(divide="ignore", invalid="ignore"):
        first = np.where(count > 0.0, count * np.log(count / mean), 0.0)
        second = (count + shape) * np.log(
            (count + shape) / (mean + shape)
        )
    deviance = 2.0 * np.maximum(first - second, 0.0)
    return np.sign(mean - count) * np.sqrt(deviance)


def _fit_fractional_overdispersion(
    observed: np.ndarray, predicted: np.ndarray,
) -> float:
    result = minimize_scalar(
        lambda log_tau: negative_binomial_nll(
            observed, predicted, math.exp(float(log_tau)),
        ),
        bounds=(math.log(1e-4), math.log(1.0)),
        method="bounded",
        options={"xatol": 1e-8},
    )
    if not result.success:
        raise RuntimeError("failed to fit count overdispersion")
    return float(math.exp(result.x))


def read_cosmos_population(path: str | Path) -> dict[str, np.ndarray]:
    """Read the broad COSMOS population and finite measured-size subset."""
    with np.load(path, allow_pickle=False) as data:
        required = (
            "mag_hst_f814w", "z_phot", "re_combined_arcsec",
            "logssfr_lephare",
        )
        missing = [name for name in required if name not in data.files]
        if missing:
            raise ValueError(f"COSMOS prior lacks required arrays: {missing}")
        magnitude = np.asarray(data["mag_hst_f814w"], dtype=np.float64)
        redshift = np.asarray(data["z_phot"], dtype=np.float64)
        radius = np.asarray(data["re_combined_arcsec"], dtype=np.float64)
        logssfr = np.asarray(data["logssfr_lephare"], dtype=np.float64)

    population = (
        np.isfinite(magnitude) & np.isfinite(redshift)
        & (redshift >= COSMOS_FIT_Z_MIN) & (redshift < COSMOS_FIT_Z_MAX)
        & (magnitude >= COSMOS_FIT_MAG_MIN) & (magnitude < 30.0)
    )
    if int(np.sum(population)) < 1000:
        raise ValueError("COSMOS prior has too few usable population rows")
    return {
        "magnitude": magnitude[population],
        "redshift": redshift[population],
        "radius_arcsec": radius[population],
        "logssfr": logssfr[population],
        "has_radius": (
            np.isfinite(radius[population]) & (radius[population] > 0.0)
        ),
    }


def read_euclid_population(
    path: str | Path, *, maximum_spurious_probability: float = 0.5,
) -> dict[str, np.ndarray | int | float | str]:
    """Read probability-weighted Euclid magnitudes and circularized size proxy."""
    magnitude: list[float] = []
    radius: list[float] = []
    weight: list[float] = []
    cone_index: list[int] = []
    magnitude_error: list[float] = []
    flux_error_uJy: list[float] = []
    rows = 0
    missing_probability = 0
    invalid_probability = 0
    missing_size = 0
    missing_magnitude_error = 0
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            raw_probability = str(row.get("point_like_prob", "")).strip()
            if not raw_probability:
                missing_probability += 1
                continue
            try:
                point_probability = float(raw_probability)
                spurious = float(row.get("spurious_prob") or 0.0)
                mag = float(row["mag_vis"])
                semimajor = float(row["semimajor_axis"])
                ellipticity = float(row["ellipticity"])
                cone = int(row.get("cone_index") or 0)
            except (KeyError, TypeError, ValueError):
                missing_size += 1
                continue
            try:
                aperture_flux = float(row["flux_vis_aper_uJy"])
                aperture_flux_error = float(row["fluxerr_vis_aper_uJy"])
            except (KeyError, TypeError, ValueError):
                missing_magnitude_error += 1
                continue
            if not np.isfinite(point_probability) or not 0.0 <= point_probability <= 1.0:
                invalid_probability += 1
                continue
            if not (
                np.isfinite(mag) and np.isfinite(spurious)
                and spurious <= maximum_spurious_probability
                and np.isfinite(semimajor) and semimajor > 0.0
                and np.isfinite(ellipticity) and 0.0 <= ellipticity < 1.0
            ):
                missing_size += 1
                continue
            if not (
                np.isfinite(aperture_flux) and aperture_flux > 0.0
                and np.isfinite(aperture_flux_error)
                and aperture_flux_error > 0.0
            ):
                missing_magnitude_error += 1
                continue
            circularized = 0.1 * semimajor * math.sqrt(1.0 - ellipticity)
            if not np.isfinite(circularized) or circularized <= 0.0:
                missing_size += 1
                continue
            magnitude.append(mag)
            radius.append(circularized)
            weight.append(1.0 - point_probability)
            cone_index.append(cone)
            magnitude_error.append(
                (2.5 / math.log(10.0)) * aperture_flux_error / aperture_flux
            )
            flux_error_uJy.append(aperture_flux_error)
    if not weight or float(np.sum(weight)) <= 0.0:
        raise ValueError(f"No usable probability-weighted Euclid galaxies in {path}")
    return {
        "magnitude": np.asarray(magnitude, dtype=np.float64),
        "radius_arcsec": np.asarray(radius, dtype=np.float64),
        "weight": np.asarray(weight, dtype=np.float64),
        "cone_index": np.asarray(cone_index, dtype=np.int64),
        "magnitude_error": np.asarray(magnitude_error, dtype=np.float64),
        "flux_error_uJy": np.asarray(flux_error_uJy, dtype=np.float64),
        "catalog_rows": rows,
        "missing_probability_rows": missing_probability,
        "invalid_probability_rows": invalid_probability,
        "missing_size_rows": missing_size,
        "missing_magnitude_error_rows": missing_magnitude_error,
        "classification_weighting": "galaxy_weight=1-POINT_LIKE_PROB",
        "size_estimator": (
            "0.1 arcsec/pixel * SEMIMAJOR_AXIS * sqrt(1-ELLIPTICITY); "
            "a MER detection proxy, not a fitted half-light radius"
        ),
    }


def _lf_expected_counts(
    parameters: np.ndarray,
    *,
    z_edges: np.ndarray = LF_Z_EDGES,
    magnitude_edges: np.ndarray = LF_MAG_EDGES,
    area_arcmin2: float = COSMOS_AREA_ARCMIN2,
) -> np.ndarray:
    (
        log_phi, m_star_0, alpha_0, q_evolution, p_evolution,
        alpha_evolution, q_quadratic, p_quadratic,
    ) = parameters
    z = 0.5 * (z_edges[:-1] + z_edges[1:])
    magnitude = 0.5 * (magnitude_edges[:-1] + magnitude_edges[1:])
    zz, mm = np.meshgrid(z, magnitude, indexing="ij")
    log1pz = np.log10(1.0 + zz)
    m_star = (
        m_star_0 + q_evolution * log1pz
        + q_quadratic * log1pz**2
    )
    alpha = alpha_0 + alpha_evolution * log1pz
    absolute_like = mm - Planck15.distmod(z).value[:, None]
    ratio = np.power(
        10.0, np.clip(0.4 * (m_star - absolute_like), -20.0, 20.0)
    )
    phi_star = np.exp(log_phi) * np.power(
        1.0 + zz, p_evolution + p_quadratic * log1pz,
    )
    density = (
        0.4 * np.log(10.0) * phi_star
        * np.power(ratio, alpha + 1.0) * np.exp(-ratio)
    )
    steradian_to_arcmin2 = ((180.0 / np.pi) * 60.0) ** 2
    volume = (
        Planck15.differential_comoving_volume(z).value
        / steradian_to_arcmin2
    )[:, None]
    return (
        area_arcmin2 * volume * np.diff(z_edges)[:, None]
        * np.diff(magnitude_edges)[None, :] * density
    )


def fit_schechter_evolution(
    magnitude: np.ndarray,
    redshift: np.ndarray,
    *,
    area_arcmin2: float = COSMOS_AREA_ARCMIN2,
) -> tuple[SchechterEvolutionFit, np.ndarray, np.ndarray]:
    """Fit an evolving Schechter intensity to the complete COSMOS window."""
    selected = (
        np.isfinite(magnitude) & np.isfinite(redshift)
        & (magnitude >= COSMOS_FIT_MAG_MIN)
        & (magnitude < COSMOS_FIT_MAG_MAX)
        & (redshift >= COSMOS_FIT_Z_MIN) & (redshift < COSMOS_FIT_Z_MAX)
    )
    observed, _, _ = np.histogram2d(
        redshift[selected], magnitude[selected],
        bins=(LF_Z_EDGES, LF_MAG_EDGES),
    )

    def poisson_residual(parameters: np.ndarray) -> np.ndarray:
        predicted = _lf_expected_counts(
            parameters, area_arcmin2=area_arcmin2,
        )
        return signed_poisson_residual(observed, predicted).ravel()

    initial = np.asarray([
        math.log(0.003), -22.0, -1.4, -0.7, -0.5, 0.0, 0.0, 0.0,
    ])
    lower = np.asarray([
        math.log(1e-8), -30.0, -2.4, -15.0, -10.0, -1.5, -15.0, -15.0,
    ])
    upper = np.asarray([
        math.log(1.0), -10.0, -0.30, 15.0, 10.0, 1.5, 15.0, 15.0,
    ])
    result = least_squares(
        poisson_residual, initial, bounds=(lower, upper),
        max_nfev=2000,
        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-11,
    )
    predicted = _lf_expected_counts(result.x, area_arcmin2=area_arcmin2)
    fractional_scatter = _fit_fractional_overdispersion(observed, predicted)
    deviance = float(np.sum(signed_poisson_residual(observed, predicted) ** 2))
    nb_deviance = float(np.sum(signed_negative_binomial_residual(
        observed, predicted, fractional_scatter,
    ) ** 2))
    dof = max(1, observed.size - len(result.x))
    fit = SchechterEvolutionFit(
        log_phi_star=float(result.x[0]),
        m_star_0=float(result.x[1]),
        alpha=float(result.x[2]),
        m_star_log1pz_slope=float(result.x[3]),
        log_phi_log1pz_slope=float(result.x[4]),
        alpha_log1pz_slope=float(result.x[5]),
        m_star_log1pz_quadratic=float(result.x[6]),
        log_phi_log1pz_quadratic=float(result.x[7]),
        cosmic_variance_fractional_scatter=fractional_scatter,
        poisson_deviance=deviance,
        negative_binomial_deviance=nb_deviance,
        dof=dof,
        standard_errors=_finite_covariance_errors(result, dof),
    )
    return fit, observed, predicted


def fit_size_evolution(
    magnitude: np.ndarray,
    redshift: np.ndarray,
    radius_arcsec: np.ndarray,
) -> SizeEvolutionFit:
    """Fit the lognormal physical-size relation with deterministic clipping."""
    valid = (
        np.isfinite(magnitude) & np.isfinite(redshift)
        & np.isfinite(radius_arcsec) & (radius_arcsec > 0.0)
        & (magnitude >= COSMOS_FIT_MAG_MIN)
        & (magnitude < COSMOS_FIT_MAG_MAX)
        & (redshift >= COSMOS_FIT_Z_MIN) & (redshift < COSMOS_FIT_Z_MAX)
    )
    magnitude = magnitude[valid]
    redshift = redshift[valid]
    radius_arcsec = radius_arcsec[valid]
    distance_modulus = Planck15.distmod(redshift).value
    absolute_like = magnitude - distance_modulus
    kpc_per_arcsec = Planck15.kpc_proper_per_arcmin(redshift).value / 60.0
    log_radius = np.log10(radius_arcsec * kpc_per_arcsec)
    magnitude_coordinate = absolute_like + 20.0
    redshift_coordinate = np.log10(1.0 + redshift)
    design = np.column_stack((
        np.ones(len(magnitude)), magnitude_coordinate,
        redshift_coordinate, magnitude_coordinate**2,
        magnitude_coordinate * redshift_coordinate,
    ))
    keep = np.ones(len(magnitude), dtype=bool)
    for _ in range(4):
        coefficients, *_ = np.linalg.lstsq(
            design[keep], log_radius[keep], rcond=None,
        )
        residual = log_radius - design @ coefficients
        center = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - center)))
        robust_sigma = max(1.4826 * mad, 0.03)
        keep = np.abs(residual - center) <= 4.5 * robust_sigma
    coefficients, *_ = np.linalg.lstsq(design[keep], log_radius[keep], rcond=None)
    residual = log_radius[keep] - design[keep] @ coefficients
    initial_scatter = max(float(np.sqrt(np.mean(residual * residual))), 0.03)
    x_fit = magnitude_coordinate[keep]
    design_fit = design[keep]
    radius_fit = log_radius[keep]

    def objective(parameters: np.ndarray) -> float:
        mean = design_fit @ parameters[:5]
        log_scatter = np.clip(
            parameters[5] + parameters[6] * x_fit, -4.0, 1.0,
        )
        scatter = np.exp(log_scatter)
        standardized = (radius_fit - mean) / scatter
        return float(np.sum(log_scatter + 0.5 * standardized**2))

    initial = np.asarray([
        *coefficients, math.log(initial_scatter), 0.0,
    ])
    result = minimize(
        objective, initial, method="L-BFGS-B",
        bounds=(
            (-3.0, 3.0), (-1.0, 1.0), (-6.0, 6.0),
            (-0.20, 0.20), (-1.0, 1.0),
            (math.log(0.02), math.log(1.0)), (-0.20, 0.20),
        ),
        options={"maxiter": 600, "ftol": 1e-12, "gtol": 1e-7},
    )
    if not result.success:
        raise RuntimeError(f"size-distribution fit failed: {result.message}")
    coefficients = np.asarray(result.x[:5], dtype=np.float64)
    scatter = float(math.exp(result.x[5]))
    residual = radius_fit - design_fit @ coefficients
    residual_rms = float(np.sqrt(np.mean(residual**2)))
    fitted_scatter = np.exp(np.clip(
        result.x[5] + result.x[6] * x_fit, -4.0, 1.0,
    ))
    weighted_design = design_fit / fitted_scatter[:, None]
    mean_covariance = np.linalg.pinv(weighted_design.T @ weighted_design)
    scatter_design = np.column_stack((np.ones(len(x_fit)), x_fit))
    scatter_covariance = np.linalg.pinv(
        2.0 * scatter_design.T @ scatter_design,
    )
    errors = np.concatenate((
        np.sqrt(np.maximum(np.diag(mean_covariance), 0.0)),
        np.asarray([
            scatter * math.sqrt(max(scatter_covariance[0, 0], 0.0)),
            math.sqrt(max(scatter_covariance[1, 1], 0.0)),
        ]),
    ))
    return SizeEvolutionFit(
        log10_r0_kpc=float(coefficients[0]),
        magnitude_slope=float(coefficients[1]),
        log1pz_slope=float(coefficients[2]),
        magnitude_curvature=float(coefficients[3]),
        magnitude_redshift_interaction=float(coefficients[4]),
        scatter_dex=scatter,
        scatter_magnitude_slope=float(result.x[6]),
        residual_rms_dex=residual_rms,
        fitted_rows=int(np.sum(keep)),
        clipped_rows=int(len(keep) - np.sum(keep)),
        standard_errors=tuple(float(value) for value in errors),
    )


def _schechter_density(
    fit: SchechterEvolutionFit, z: np.ndarray, magnitude: np.ndarray,
) -> np.ndarray:
    zz, mm = np.meshgrid(z, magnitude, indexing="ij")
    log1pz = np.log10(1.0 + zz)
    m_star = (
        fit.m_star_0 + fit.m_star_log1pz_slope * log1pz
        + fit.m_star_log1pz_quadratic * log1pz**2
    )
    alpha = fit.alpha + fit.alpha_log1pz_slope * log1pz
    absolute_like = mm - Planck15.distmod(z).value[:, None]
    ratio = np.power(
        10.0, np.clip(0.4 * (m_star - absolute_like), -20.0, 20.0)
    )
    phi_star = np.exp(fit.log_phi_star) * np.power(
        1.0 + zz,
        fit.log_phi_log1pz_slope
        + fit.log_phi_log1pz_quadratic * log1pz,
    )
    return (
        0.4 * np.log(10.0) * phi_star
        * np.power(ratio, alpha + 1.0) * np.exp(-ratio)
    )


def latent_population_cube(
    lf_fit: SchechterEvolutionFit,
    size_fit: SizeEvolutionFit,
    *,
    z_edges: np.ndarray = LATENT_Z_EDGES,
    magnitude_edges: np.ndarray = LATENT_MAG_EDGES,
    log_radius_edges: np.ndarray = LATENT_LOG_RE_EDGES,
) -> dict[str, np.ndarray]:
    """Evaluate the intrinsic density on a deterministic quadrature grid."""
    z = 0.5 * (z_edges[:-1] + z_edges[1:])
    magnitude = 0.5 * (magnitude_edges[:-1] + magnitude_edges[1:])
    log_radius = 0.5 * (log_radius_edges[:-1] + log_radius_edges[1:])
    lf_density = _schechter_density(lf_fit, z, magnitude)
    steradian_to_arcmin2 = ((180.0 / np.pi) * 60.0) ** 2
    volume = (
        Planck15.differential_comoving_volume(z).value
        / steradian_to_arcmin2
    )[:, None]
    zm_density = (
        lf_density * volume * np.diff(z_edges)[:, None]
        * np.diff(magnitude_edges)[None, :]
    )
    absolute_like = (
        magnitude[None, :] - Planck15.distmod(z).value[:, None]
    )
    magnitude_coordinate = absolute_like + 20.0
    redshift_coordinate = np.log10(1.0 + z)[:, None]
    mean_log_kpc = (
        size_fit.log10_r0_kpc
        + size_fit.magnitude_slope * magnitude_coordinate
        + size_fit.log1pz_slope * redshift_coordinate
        + size_fit.magnitude_curvature * magnitude_coordinate**2
        + size_fit.magnitude_redshift_interaction
        * magnitude_coordinate * redshift_coordinate
    )
    scatter = size_fit.scatter_dex * np.exp(
        np.clip(
            size_fit.scatter_magnitude_slope * magnitude_coordinate,
            -3.0, 3.0,
        )
    )
    kpc_per_arcsec = Planck15.kpc_proper_per_arcmin(z).value / 60.0
    mean_log_arcsec = mean_log_kpc - np.log10(kpc_per_arcsec)[:, None]
    upper = (
        log_radius_edges[None, None, 1:] - mean_log_arcsec[:, :, None]
    ) / scatter[:, :, None]
    lower = (
        log_radius_edges[None, None, :-1] - mean_log_arcsec[:, :, None]
    ) / scatter[:, :, None]
    radius_probability = ndtr(upper) - ndtr(lower)
    cube = zm_density[:, :, None] * radius_probability
    return {
        "density": cube,
        "z": z,
        "magnitude": magnitude,
        "log_radius": log_radius,
        "z_edges": z_edges,
        "magnitude_edges": magnitude_edges,
        "log_radius_edges": log_radius_edges,
    }


def _unpack_response(parameters: np.ndarray) -> tuple[float, ...]:
    return (
        float(np.exp(parameters[0])),
        float(parameters[1]),
        float(np.exp(parameters[2])),
        float(np.exp(parameters[3])),
        float(np.exp(parameters[4])),
        float(np.exp(parameters[5])),
        float(parameters[6]),
        float(np.exp(parameters[7])),
        float(np.exp(parameters[8])),
    )


def euclid_flux_error_model(
    euclid: dict[str, np.ndarray | int | float | str],
) -> float:
    """Return the robust galaxy-weighted VIS aperture-flux error in microJy.

    MER aperture-flux errors are approximately homoscedastic in flux space.
    Their weighted median is therefore used directly in the likelihood instead
    of applying a faint-source delta-method conversion to magnitude error.
    """
    error = np.asarray(euclid["flux_error_uJy"], dtype=np.float64)
    weight = np.asarray(euclid["weight"], dtype=np.float64)
    valid = (
        np.isfinite(error) & (error > 0.0)
        & np.isfinite(weight) & (weight > 0.0)
    )
    if int(np.sum(valid)) < 100:
        raise ValueError("Euclid catalogue has insufficient VIS flux-error rows")
    values = error[valid]
    weights = weight[valid]
    order = np.argsort(values)
    values = values[order]
    cumulative = np.cumsum(weights[order])
    result = float(np.interp(0.5 * cumulative[-1], cumulative, values))
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("Euclid VIS flux-error estimate is invalid")
    return result


def _flux_space_magnitude_probability(
    mean_magnitude: np.ndarray,
    intrinsic_scatter_mag: float,
    magnitude_edges: np.ndarray,
    measurement_flux_error_uJy: float,
) -> np.ndarray:
    """Probability of observed magnitude bins after Gaussian flux noise."""
    if not np.isfinite(measurement_flux_error_uJy) or measurement_flux_error_uJy <= 0:
        raise ValueError("Euclid VIS flux error must be positive")
    nodes, weights = np.polynomial.hermite.hermgauss(9)
    intrinsic_magnitude = (
        np.asarray(mean_magnitude, dtype=np.float64)[:, None]
        + math.sqrt(2.0) * intrinsic_scatter_mag * nodes[None, :]
    )
    true_flux = np.asarray(ab_mag_to_uJy(intrinsic_magnitude))
    upper_flux = np.asarray(ab_mag_to_uJy(magnitude_edges[:-1]))
    lower_flux = np.asarray(ab_mag_to_uJy(magnitude_edges[1:]))
    probability = ndtr(
        (upper_flux[None, None, :] - true_flux[:, :, None])
        / measurement_flux_error_uJy
    ) - ndtr(
        (lower_flux[None, None, :] - true_flux[:, :, None])
        / measurement_flux_error_uJy
    )
    return np.sum(
        probability * (weights / math.sqrt(math.pi))[None, :, None], axis=1,
    )


def predict_euclid_histogram(
    latent_magnitude_radius_density: np.ndarray,
    latent_magnitude: np.ndarray,
    latent_log_radius: np.ndarray,
    parameters: np.ndarray,
    *,
    magnitude_edges: np.ndarray = EUCLID_MAG_EDGES,
    log_radius_edges: np.ndarray = EUCLID_LOG_RE_EDGES,
    measurement_flux_error_uJy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pass the latent population through the fitted Euclid response."""
    (
        population_scale, offset, slope, intrinsic_scatter, size_scale, size_floor,
        m50, width, surface_brightness_penalty,
    ) = _unpack_response(parameters)
    latent_m, latent_log_r = np.meshgrid(
        latent_magnitude, latent_log_radius, indexing="ij",
    )
    weights = np.asarray(latent_magnitude_radius_density, dtype=np.float64).ravel()
    source_magnitude = latent_m.ravel()
    source_radius = np.power(10.0, latent_log_r.ravel())
    mean_vis = 24.0 + slope * (source_magnitude - 24.0) + offset
    observed_radius = np.sqrt(
        np.square(size_scale * source_radius) + size_floor**2
    )
    observed_log_radius = np.log10(observed_radius)
    radius_index = np.searchsorted(
        log_radius_edges, observed_log_radius, side="right",
    ) - 1
    magnitude_centers = 0.5 * (magnitude_edges[:-1] + magnitude_edges[1:])
    magnitude_probability = _flux_space_magnitude_probability(
        mean_vis, intrinsic_scatter, magnitude_edges,
        measurement_flux_error_uJy,
    )
    mean_surface_brightness = (
        magnitude_centers[None, :]
        + 2.5 * np.log10(2.0 * np.pi * observed_radius[:, None] ** 2)
    )
    logit = (
        (m50 - magnitude_centers[None, :]) / width
        - surface_brightness_penalty * (mean_surface_brightness - 24.0)
    )
    completeness = 1.0 / (1.0 + np.exp(-np.clip(logit, -60.0, 60.0)))
    contribution = (
        population_scale * weights[:, None]
        * magnitude_probability * completeness
    )
    predicted = np.zeros(
        (len(magnitude_edges) - 1, len(log_radius_edges) - 1),
        dtype=np.float64,
    )
    for index in range(predicted.shape[1]):
        selected = radius_index == index
        if np.any(selected):
            predicted[:, index] = np.sum(contribution[selected], axis=0)
    return predicted, completeness


def tng_draw_population_cube(
    latent_cube: dict[str, np.ndarray],
    response_fit: EuclidResponseFit,
    *,
    vis_magnitude_edges: np.ndarray = TNG_DRAW_VIS_MAG_EDGES,
) -> dict[str, np.ndarray]:
    """Project the latent population into pre-observation TNG draw space.

    Only the fitted F814W-to-true-VIS transfer and Euclid field normalization
    enter.  MER size broadening, catalogue completeness, radius censoring, and
    the empirical Euclid measurement-error curve are deliberately excluded;
    those effects belong after TNG rendering.
    """
    latent_magnitude = np.asarray(latent_cube["magnitude"], dtype=np.float64)
    mean_vis = (
        24.0
        + response_fit.magnitude_slope * (latent_magnitude - 24.0)
        + response_fit.vis_minus_f814w_mag
    )
    intrinsic_scatter = float(response_fit.scatter_mag)
    upper = (
        vis_magnitude_edges[None, 1:] - mean_vis[:, None]
    ) / intrinsic_scatter
    lower = (
        vis_magnitude_edges[None, :-1] - mean_vis[:, None]
    ) / intrinsic_scatter
    magnitude_probability = ndtr(upper) - ndtr(lower)
    density = response_fit.population_scale * np.einsum(
        "zmr,mv->zvr",
        np.asarray(latent_cube["density"], dtype=np.float64),
        magnitude_probability,
        optimize=True,
    )
    return {
        "density": density,
        "z": np.asarray(latent_cube["z"], dtype=np.float64),
        "z_edges": np.asarray(latent_cube["z_edges"], dtype=np.float64),
        "vis_magnitude": 0.5 * (
            vis_magnitude_edges[:-1] + vis_magnitude_edges[1:]
        ),
        "vis_magnitude_edges": np.asarray(
            vis_magnitude_edges, dtype=np.float64,
        ),
        "log_radius": np.asarray(
            latent_cube["log_radius"], dtype=np.float64,
        ),
        "log_radius_edges": np.asarray(
            latent_cube["log_radius_edges"], dtype=np.float64,
        ),
    }


def fit_euclid_response(
    latent_cube: dict[str, np.ndarray],
    euclid: dict[str, np.ndarray | int | float | str],
    *,
    area_arcmin2: float,
    unresolved_policy: str = "keep",
    unresolved_radius_arcsec: float = 0.10,
    log_radius_edges: np.ndarray = EUCLID_LOG_RE_EDGES,
    measurement_flux_error_uJy: float | None = None,
    bright_transfer_magnitude_max: float = BRIGHT_TRANSFER_MAG_MAX,
) -> tuple[EuclidResponseFit, np.ndarray, np.ndarray]:
    """Fit a frozen bright transfer, then Euclid size and completeness.

    The normalization and affine F814W-to-VIS relation are determined only
    from the high-S/N magnitude counts brighter than
    ``bright_transfer_magnitude_max``.  Those four parameters are frozen while
    the full magnitude-size plane determines the size proxy and faint
    completeness response.
    """
    if area_arcmin2 <= 0.0:
        raise ValueError("Euclid area must be positive")
    if unresolved_policy not in {"keep", "drop", "censor"}:
        raise ValueError(f"unsupported unresolved policy: {unresolved_policy}")
    radius_edges = np.asarray(log_radius_edges, dtype=np.float64)
    threshold = math.log10(unresolved_radius_arcsec)
    if unresolved_policy != "keep" and not np.any(
        np.isclose(radius_edges, threshold, atol=1e-10, rtol=0.0)
    ):
        raise ValueError("unresolved radius must be an explicit histogram edge")
    unresolved_bins = radius_edges[1:] <= threshold + 1e-10
    resolved_bins = radius_edges[:-1] >= threshold - 1e-10
    magnitude = np.asarray(euclid["magnitude"], dtype=np.float64)
    radius = np.asarray(euclid["radius_arcsec"], dtype=np.float64)
    weight = np.asarray(euclid["weight"], dtype=np.float64)
    observed, _, _ = np.histogram2d(
        magnitude, np.log10(radius),
        bins=(EUCLID_MAG_EDGES, radius_edges),
        weights=weight,
    )
    latent_magnitude_radius = np.sum(
        np.asarray(latent_cube["density"], dtype=np.float64), axis=0,
    )
    flux_error = (
        euclid_flux_error_model(euclid)
        if measurement_flux_error_uJy is None
        else float(measurement_flux_error_uJy)
    )
    if not np.isfinite(flux_error) or flux_error <= 0.0:
        raise ValueError("explicit Euclid VIS flux error is invalid")

    magnitude_centers = 0.5 * (EUCLID_MAG_EDGES[:-1] + EUCLID_MAG_EDGES[1:])
    bright_bins = magnitude_centers < bright_transfer_magnitude_max
    if int(np.sum(bright_bins)) < 8:
        raise ValueError("bright transfer window contains too few magnitude bins")
    observed_magnitude, _ = np.histogram(
        magnitude, bins=EUCLID_MAG_EDGES, weights=weight,
    )
    latent_magnitude_density = np.sum(latent_magnitude_radius, axis=1)

    def bright_prediction(parameters: np.ndarray) -> np.ndarray:
        scale = float(np.exp(parameters[0]))
        offset = float(parameters[1])
        slope = float(np.exp(parameters[2]))
        scatter = float(np.exp(parameters[3]))
        mean_vis = (
            24.0 + slope * (
                np.asarray(latent_cube["magnitude"], dtype=np.float64) - 24.0
            ) + offset
        )
        probability = _flux_space_magnitude_probability(
            mean_vis, scatter, EUCLID_MAG_EDGES, flux_error,
        )
        return scale * latent_magnitude_density @ probability

    def bright_data_residual(parameters: np.ndarray) -> np.ndarray:
        predicted = bright_prediction(parameters) * area_arcmin2
        return signed_poisson_residual(
            observed_magnitude[bright_bins], predicted[bright_bins],
        )

    def bright_residual(parameters: np.ndarray) -> np.ndarray:
        priors = np.asarray([
            parameters[0] / 0.35,
            parameters[1] / 0.70,
            parameters[2] / 0.25,
            (parameters[3] - math.log(0.25)) / 0.8,
        ])
        return np.concatenate((bright_data_residual(parameters), priors))

    bright_initial = np.asarray([0.0, 0.0, 0.0, math.log(0.25)])
    bright_lower = np.asarray([
        math.log(0.2), -1.5, math.log(0.5), math.log(0.02),
    ])
    bright_upper = np.asarray([
        math.log(5.0), 1.5, math.log(1.5), math.log(1.20),
    ])
    bright_result = least_squares(
        bright_residual, bright_initial,
        bounds=(bright_lower, bright_upper), max_nfev=800,
        xtol=2e-9, ftol=2e-9, gtol=2e-9,
    )
    frozen_transfer = np.asarray(bright_result.x, dtype=np.float64)

    def response_parameters(parameters: np.ndarray) -> np.ndarray:
        return np.concatenate((frozen_transfer, parameters))

    def data_residual(response_only_parameters: np.ndarray) -> np.ndarray:
        parameters = response_parameters(response_only_parameters)
        predicted_density, _ = predict_euclid_histogram(
            latent_magnitude_radius,
            np.asarray(latent_cube["magnitude"]),
            np.asarray(latent_cube["log_radius"]),
            parameters,
            log_radius_edges=radius_edges,
            measurement_flux_error_uJy=flux_error,
        )
        predicted = predicted_density * area_arcmin2
        if unresolved_policy == "keep":
            return signed_poisson_residual(observed, predicted).ravel()
        resolved = signed_poisson_residual(
            observed[:, resolved_bins], predicted[:, resolved_bins],
        ).ravel()
        if unresolved_policy == "drop":
            return resolved
        unresolved = signed_poisson_residual(
            np.sum(observed[:, unresolved_bins], axis=1),
            np.sum(predicted[:, unresolved_bins], axis=1),
        ).ravel()
        return np.concatenate((resolved, unresolved))

    def residual(response_only_parameters: np.ndarray) -> np.ndarray:
        catalogue_residual = data_residual(response_only_parameters)
        priors = np.asarray([
            response_only_parameters[0] / 0.60,
            (response_only_parameters[1] - math.log(0.09)) / 0.55,
            (response_only_parameters[2] - 25.2) / 1.4,
            (response_only_parameters[3] - math.log(0.40)) / 0.8,
            (response_only_parameters[4] - math.log(0.20)) / 1.2,
        ])
        return np.concatenate((catalogue_residual, priors))

    initial = np.asarray([
        0.0, math.log(0.09), 25.2, math.log(0.40), math.log(0.20),
    ])
    lower = np.asarray([
        math.log(0.2), math.log(0.02), 22.0, math.log(0.04), math.log(0.005),
    ])
    upper = np.asarray([
        math.log(5.0), math.log(0.35), 28.5, math.log(2.0), math.log(5.0),
    ])
    result = least_squares(
        residual, initial, bounds=(lower, upper), max_nfev=1200,
        xtol=2e-9, ftol=2e-9, gtol=2e-9,
    )
    all_parameters = response_parameters(result.x)
    predicted_density, _ = predict_euclid_histogram(
        latent_magnitude_radius,
        np.asarray(latent_cube["magnitude"]),
        np.asarray(latent_cube["log_radius"]),
        all_parameters,
        log_radius_edges=radius_edges,
        measurement_flux_error_uJy=flux_error,
    )
    final_data_residual = data_residual(result.x)
    deviance = float(np.sum(final_data_residual**2))
    dof = max(1, final_data_residual.size - len(all_parameters))
    values = _unpack_response(all_parameters)
    bright_data = bright_data_residual(bright_result.x)
    bright_deviance = float(np.sum(bright_data**2))
    bright_dof = max(1, bright_data.size - len(bright_result.x))
    bright_errors = np.asarray(
        _finite_covariance_errors(bright_result, bright_dof), dtype=np.float64,
    )
    response_errors = np.asarray(
        _finite_covariance_errors(result, dof), dtype=np.float64,
    )
    transformed_errors = np.concatenate((bright_errors, response_errors))
    for index in (0, 2, 3, 4, 5, 7, 8):
        transformed_errors[index] *= values[index]
    fit = EuclidResponseFit(
        population_scale=values[0],
        vis_minus_f814w_mag=values[1],
        magnitude_slope=values[2],
        scatter_mag=values[3],
        measurement_flux_error_uJy=flux_error,
        size_scale=values[4],
        size_floor_arcsec=values[5],
        completeness_m50=values[6],
        completeness_width_mag=values[7],
        surface_brightness_penalty=values[8],
        bright_transfer_magnitude_max=float(bright_transfer_magnitude_max),
        bright_poisson_deviance=bright_deviance,
        bright_dof=bright_dof,
        poisson_deviance=deviance,
        dof=dof,
        standard_errors=tuple(float(value) for value in transformed_errors),
    )
    return fit, observed, predicted_density


def fit_payload(fit: Any) -> dict[str, Any]:
    """Convert a fit dataclass to a JSON-safe mapping."""
    return asdict(fit)
