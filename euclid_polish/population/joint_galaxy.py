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
from typing import Any, Protocol, cast

import numpy as np
from astropy.cosmology import Planck15
from scipy.ndimage import gaussian_filter
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

class _FLRWCosmology(Protocol):
    """Subset of the FLRW API used by this calibration."""

    def distmod(self, z: Any) -> Any: ...

    def differential_comoving_volume(self, z: Any) -> Any: ...

    def kpc_proper_per_arcmin(self, z: Any) -> Any: ...


_COSMOLOGY = cast(_FLRWCosmology, Planck15)


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
        mass_key = next(
            (name for name in ("logmass_lephare", "logmass") if name in data.files),
            None,
        )
        mass = (
            np.asarray(data[mass_key], dtype=np.float64)
            if mass_key is not None else np.full(magnitude.shape, np.nan)
        )

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
        "logmass": mass[population],
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


def read_phz_population(
    catalog_path: str | Path, pdf_path: str | Path,
) -> dict[str, np.ndarray | int]:
    """Read PHZ scalars and align the compact redshift PDFs by object ID."""
    with np.load(pdf_path, allow_pickle=False) as data:
        pdf_ids = np.asarray(data["object_id"]).astype(str)
        pdf_probability = np.asarray(data["probability"], dtype=np.float64)
        pdf_z_edges = np.asarray(data["z_edges"], dtype=np.float64)
    if pdf_probability.shape != (pdf_ids.size, pdf_z_edges.size - 1):
        raise ValueError("PHZ PDF cache arrays are not aligned")
    if (
        pdf_probability.size
        and (
            not np.isfinite(pdf_probability).all()
            or np.any(pdf_probability < 0.0)
        )
    ):
        raise ValueError("PHZ PDF cache contains invalid probabilities")
    if pdf_ids.size != np.unique(pdf_ids).size:
        raise ValueError("PHZ PDF cache contains duplicate object IDs")
    if pdf_probability.size and not np.allclose(
        np.sum(pdf_probability, axis=1), 1.0, rtol=1e-5, atol=1e-6,
    ):
        raise ValueError("PHZ PDF cache rows are not normalized")
    pdf_by_id = {value: index for index, value in enumerate(pdf_ids)}
    fields = {
        "magnitude": "mag_vis",
        "galaxy_probability": "phz_gal_prob",
        "qso_probability": "phz_qso_prob",
        "physical_redshift": "phz_pp_median_redshift",
        "logmass": "phz_pp_median_stellarmass",
        "logmass_p16": "phz_pp_stellarmass_p16",
        "logmass_p84": "phz_pp_stellarmass_p84",
        "logsfr": "phz_pp_median_sfr",
        "logsfr_p16": "phz_pp_sfr_p16",
        "logsfr_p84": "phz_pp_sfr_p84",
        "sfhage": "phz_pp_median_sfhage",
        "sfhage_p16": "phz_pp_sfhage_p16",
        "sfhage_p84": "phz_pp_sfhage_p84",
        "rest_u": "phz_pp_median_mu",
        "rest_v": "phz_pp_median_mv",
        "rest_j": "phz_pp_median_mj",
        "rest_vis": "phz_pp_median_mvis",
        "physical_flags": "phz_phys_flags",
        "quality_flag": "phz_phys_quality_flag",
    }
    values: dict[str, list[float]] = {key: [] for key in fields}
    object_ids: list[str] = []
    cone_index: list[int] = []
    pdf_index: list[int] = []
    with Path(catalog_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            object_id = str(row.get("object_id") or "")
            if not object_id:
                continue
            object_ids.append(object_id)
            try:
                cone_index.append(int(row.get("cone_index") or 0))
            except (TypeError, ValueError):
                cone_index.append(0)
            pdf_index.append(pdf_by_id.get(object_id, -1))
            for output, column in fields.items():
                try:
                    value = float(row.get(column) or "nan")
                except (TypeError, ValueError):
                    value = float("nan")
                values[output].append(value)
    aligned_pdf = np.zeros(
        (len(object_ids), pdf_probability.shape[1]), dtype=np.float64,
    )
    has_pdf = np.asarray(pdf_index, dtype=np.int64) >= 0
    if np.any(has_pdf):
        aligned_pdf[has_pdf] = pdf_probability[np.asarray(pdf_index)[has_pdf]]
    return {
        "object_id": np.asarray(object_ids).astype(str),
        "cone_index": np.asarray(cone_index, dtype=np.int64),
        "has_pdf": has_pdf,
        "pdf_probability": aligned_pdf,
        "pdf_z_edges": pdf_z_edges,
        **{
            key: np.asarray(value, dtype=np.float64)
            for key, value in values.items()
        },
    }


def _phz_observed_plane(
    phz: dict[str, np.ndarray | int], magnitude_edges: np.ndarray,
    *, cone_mask: np.ndarray | None = None,
) -> np.ndarray:
    probability = np.asarray(phz["pdf_probability"], dtype=np.float64)
    magnitude = np.asarray(phz["magnitude"], dtype=np.float64)
    weight = np.asarray(phz["galaxy_probability"], dtype=np.float64)
    valid = (
        np.asarray(phz["has_pdf"], dtype=bool)
        & np.isfinite(magnitude) & (magnitude < 24.5)
        & np.isfinite(weight) & (weight >= 0.0) & (weight <= 1.0)
    )
    if cone_mask is not None:
        valid &= np.asarray(cone_mask, dtype=bool)
    result = np.zeros(
        (probability.shape[1], magnitude_edges.size - 1), dtype=np.float64,
    )
    magnitude_bin = np.searchsorted(
        magnitude_edges, magnitude, side="right",
    ) - 1
    for index in np.flatnonzero(valid):
        column = int(magnitude_bin[index])
        if 0 <= column < result.shape[1]:
            result[:, column] += weight[index] * probability[index]
    return result


def _rebin_probability_mass(
    probability: np.ndarray,
    source_edges: np.ndarray,
    target_edges: np.ndarray,
) -> np.ndarray:
    """Conservatively move binned PDF mass onto another grid.

    The compact PHZ cache uses the luminosity-function redshift grid, while
    the generation cube uses the finer latent grid.  Treat the probability
    density as uniform inside each cached bin and distribute its mass by bin
    overlap.  Both grids must cover the same redshift interval so no posterior
    mass is silently discarded or invented.
    """
    values = np.asarray(probability, dtype=np.float64)
    source = np.asarray(source_edges, dtype=np.float64)
    target = np.asarray(target_edges, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != source.size - 1:
        raise ValueError("PHZ probability matrix does not match its redshift grid")
    if (
        source.size < 2
        or target.size < 2
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
        or np.any(np.diff(source) <= 0.0)
        or np.any(np.diff(target) <= 0.0)
    ):
        raise ValueError("PHZ and analytical redshift grids must be finite and increasing")
    if not np.allclose(
        [source[0], source[-1]],
        [target[0], target[-1]],
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError("PHZ and analytical redshift grids cover different ranges")

    overlap = np.maximum(
        0.0,
        np.minimum(source[1:, None], target[None, 1:])
        - np.maximum(source[:-1, None], target[None, :-1]),
    )
    transfer = overlap / np.diff(source)[:, None]
    rebinned = values @ transfer
    if not np.allclose(
        np.sum(rebinned, axis=1),
        np.sum(values, axis=1),
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("PHZ PDF rebinning did not conserve probability mass")
    return rebinned


def _scaled_plane_prediction(
    baseline: np.ndarray, observed: np.ndarray,
    correction: np.ndarray | None = None,
) -> np.ndarray:
    prediction = np.asarray(baseline, dtype=np.float64).copy()
    if correction is not None:
        prediction *= np.asarray(correction, dtype=np.float64)
    totals = np.sum(prediction, axis=0)
    observed_totals = np.sum(observed, axis=0)
    scale = np.divide(
        observed_totals, totals, out=np.zeros_like(totals), where=totals > 0.0,
    )
    return prediction * scale[None, :]


def _poisson_deviance(observed: np.ndarray, predicted: np.ndarray) -> float:
    predicted = np.maximum(np.asarray(predicted, dtype=np.float64), 1e-12)
    observed = np.asarray(observed, dtype=np.float64)
    logarithmic = np.zeros_like(observed)
    positive = observed > 0.0
    logarithmic[positive] = observed[positive] * np.log(
        observed[positive] / predicted[positive]
    )
    return float(2.0 * np.sum(predicted - observed + logarithmic))


def _fit_plane_correction(
    baseline: np.ndarray, observed: np.ndarray, magnitude_edges: np.ndarray,
) -> np.ndarray:
    predicted = _scaled_plane_prediction(baseline, observed)
    ratio = (observed + 0.5) / (predicted + 0.5)
    correction = np.exp(gaussian_filter(np.log(ratio), sigma=(1.0, 1.0)))
    baseline_total = np.sum(baseline, axis=0)
    corrected_total = np.sum(baseline * correction, axis=0)
    correction *= np.divide(
        baseline_total, corrected_total,
        out=np.ones_like(baseline_total), where=corrected_total > 0.0,
    )[None, :]
    centers = 0.5 * (magnitude_edges[:-1] + magnitude_edges[1:])
    taper = np.clip((24.5 - centers) / 0.5, 0.0, 1.0)
    correction = np.exp(
        np.log(np.maximum(correction, 1e-12)) * taper[None, :]
    )
    corrected_total = np.sum(baseline * correction, axis=0)
    correction *= np.divide(
        baseline_total, corrected_total,
        out=np.ones_like(baseline_total), where=corrected_total > 0.0,
    )[None, :]
    return correction


def fit_phz_redshift_correction(
    draw: dict[str, np.ndarray], phz: dict[str, np.ndarray | int],
) -> dict[str, Any]:
    """Fit and cross-validate a density-preserving PHZ redshift correction."""
    z_edges = np.asarray(draw["z_edges"], dtype=np.float64)
    magnitude_edges = np.asarray(draw["vis_magnitude_edges"], dtype=np.float64)
    pdf_edges = np.asarray(phz["pdf_z_edges"], dtype=np.float64)
    grids_match = z_edges.shape == pdf_edges.shape and np.allclose(
        z_edges, pdf_edges, rtol=0.0, atol=1e-10,
    )
    phz_for_fit = phz
    if not grids_match:
        phz_for_fit = {
            **phz,
            "pdf_probability": _rebin_probability_mass(
                np.asarray(phz["pdf_probability"], dtype=np.float64),
                pdf_edges,
                z_edges,
            ),
            "pdf_z_edges": z_edges,
        }
    baseline = np.sum(np.asarray(draw["density"], dtype=np.float64), axis=2)
    observed = _phz_observed_plane(phz_for_fit, magnitude_edges)
    correction = _fit_plane_correction(baseline, observed, magnitude_edges)
    corrected = _scaled_plane_prediction(baseline, observed, correction)
    baseline_prediction = _scaled_plane_prediction(baseline, observed)
    supported = observed >= 1.0
    fractional = np.abs(corrected - observed) / np.maximum(observed, 1.0)
    median_fractional = (
        float(np.median(fractional[supported])) if np.any(supported) else float("inf")
    )
    cone_index = np.asarray(phz_for_fit["cone_index"], dtype=np.int64)
    cones = np.unique(cone_index)
    folds: list[dict[str, Any]] = []
    for fold_cones in np.array_split(cones, 4) if cones.size >= 4 else []:
        test = np.isin(cone_index, fold_cones)
        train_observed = _phz_observed_plane(
            phz_for_fit, magnitude_edges, cone_mask=~test,
        )
        test_observed = _phz_observed_plane(
            phz_for_fit, magnitude_edges, cone_mask=test,
        )
        fold_correction = _fit_plane_correction(
            baseline, train_observed, magnitude_edges,
        )
        baseline_test = _scaled_plane_prediction(baseline, test_observed)
        corrected_test = _scaled_plane_prediction(
            baseline, test_observed, fold_correction,
        )
        baseline_deviance = _poisson_deviance(test_observed, baseline_test)
        corrected_deviance = _poisson_deviance(test_observed, corrected_test)
        folds.append({
            "test_cones": [int(value) for value in fold_cones],
            "baseline_deviance": baseline_deviance,
            "corrected_deviance": corrected_deviance,
            "improvement_fraction": (
                (baseline_deviance - corrected_deviance) / baseline_deviance
                if baseline_deviance > 0.0 else 0.0
            ),
        })
    mean_improvement = (
        float(np.mean([fold["improvement_fraction"] for fold in folds]))
        if folds else 0.0
    )
    corrected_density = baseline * correction
    density_change = float(
        abs(np.sum(corrected_density) - np.sum(baseline))
        / max(np.sum(baseline), 1e-12)
    )
    return {
        "version": 1,
        "z_edges": z_edges.tolist(),
        "input_pdf_z_edges": pdf_edges.tolist(),
        "pdf_rebinned": not grids_match,
        "vis_magnitude_edges": magnitude_edges.tolist(),
        "factor": correction.tolist(),
        "observed_weighted_counts": observed.tolist(),
        "baseline_weighted_counts": baseline_prediction.tolist(),
        "corrected_weighted_counts": corrected.tolist(),
        "baseline_deviance": _poisson_deviance(observed, baseline_prediction),
        "corrected_deviance": _poisson_deviance(observed, corrected),
        "median_absolute_fractional_residual": median_fractional,
        "density_change_fraction": density_change,
        "cross_validation": {
            "folds": folds,
            "mean_improvement_fraction": mean_improvement,
        },
    }


def _effective_weight(weight: np.ndarray) -> float:
    total = float(np.sum(weight))
    squared = float(np.sum(weight * weight))
    return total * total / squared if squared > 0.0 else 0.0


def _weighted_location_scale(
    values: np.ndarray, weights: np.ndarray,
) -> tuple[float, float]:
    total = float(np.sum(weights))
    if total <= 0.0:
        return float("nan"), float("nan")
    mean = float(np.sum(weights * values) / total)
    variance = float(np.sum(weights * (values - mean) ** 2) / total)
    return mean, max(math.sqrt(max(variance, 0.0)), 0.05)


def fit_physical_conditionals(
    cosmos: dict[str, np.ndarray], phz: dict[str, np.ndarray | int],
    response: EuclidResponseFit,
    *, minimum_effective_weight: float = 64.0,
) -> dict[str, Any]:
    """Fit compact mass/activity conditionals, blending PHZ into COSMOS."""
    z_edges = np.asarray([0.05, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 4.0, 5.5])
    magnitude_edges = np.asarray([18.0, 20.0, 22.0, 23.0, 24.0, 24.5, 26.0, 28.0, 30.0])
    phz_magnitude = np.asarray(phz["magnitude"], dtype=np.float64)
    phz_redshift = np.asarray(phz["physical_redshift"], dtype=np.float64)
    phz_mass = np.asarray(phz["logmass"], dtype=np.float64)
    phz_ssfr = np.asarray(phz["logsfr"], dtype=np.float64) - phz_mass
    phz_mass_p16 = np.asarray(phz["logmass_p16"], dtype=np.float64)
    phz_mass_p84 = np.asarray(phz["logmass_p84"], dtype=np.float64)
    phz_sfr_p16 = np.asarray(phz["logsfr_p16"], dtype=np.float64)
    phz_sfr_p84 = np.asarray(phz["logsfr_p84"], dtype=np.float64)
    phz_age = np.asarray(phz["sfhage"], dtype=np.float64)
    phz_age_p16 = np.asarray(phz["sfhage_p16"], dtype=np.float64)
    phz_age_p84 = np.asarray(phz["sfhage_p84"], dtype=np.float64)
    phz_weight = np.asarray(phz["galaxy_probability"], dtype=np.float64)
    phz_flags = np.asarray(phz["physical_flags"], dtype=np.float64)
    phz_quality = np.asarray(phz["quality_flag"], dtype=np.float64)
    phz_blend = np.clip((24.5 - phz_magnitude) / 0.5, 0.0, 1.0)
    phz_valid = (
        np.isfinite(phz_magnitude) & np.isfinite(phz_redshift)
        & np.isfinite(phz_mass) & np.isfinite(phz_ssfr)
        & np.isfinite(phz_mass_p16) & np.isfinite(phz_mass_p84)
        & np.isfinite(phz_sfr_p16) & np.isfinite(phz_sfr_p84)
        & np.isfinite(phz_age) & np.isfinite(phz_age_p16)
        & np.isfinite(phz_age_p84)
        & (phz_mass_p16 <= phz_mass) & (phz_mass <= phz_mass_p84)
        & (phz_sfr_p16 <= np.asarray(phz["logsfr"], dtype=np.float64))
        & (np.asarray(phz["logsfr"], dtype=np.float64) <= phz_sfr_p84)
        & (phz_age_p16 <= phz_age) & (phz_age <= phz_age_p84)
        & np.isfinite(phz_weight) & (phz_weight >= 0.0) & (phz_weight <= 1.0)
        & (phz_flags == 0.0) & (phz_quality == 0.0)
        & (phz_ssfr < -8.2) & (phz_magnitude < 24.5)
    )
    pivot = 24.0
    cosmos_magnitude = (
        pivot
        + response.magnitude_slope
        * (np.asarray(cosmos["magnitude"], dtype=np.float64) - pivot)
        + response.vis_minus_f814w_mag
    )
    cosmos_redshift = np.asarray(cosmos["redshift"], dtype=np.float64)
    cosmos_mass = np.asarray(cosmos["logmass"], dtype=np.float64)
    cosmos_ssfr = np.asarray(cosmos["logssfr"], dtype=np.float64)
    cosmos_blend = np.clip((cosmos_magnitude - 24.0) / 0.5, 0.0, 1.0)
    cosmos_valid = (
        np.isfinite(cosmos_magnitude) & np.isfinite(cosmos_redshift)
        & np.isfinite(cosmos_mass) & np.isfinite(cosmos_ssfr)
        & (cosmos_ssfr < -8.2) & (cosmos_blend > 0.0)
    )
    magnitude = np.concatenate([
        phz_magnitude[phz_valid], cosmos_magnitude[cosmos_valid],
    ])
    redshift = np.concatenate([
        phz_redshift[phz_valid], cosmos_redshift[cosmos_valid],
    ])
    mass = np.concatenate([phz_mass[phz_valid], cosmos_mass[cosmos_valid]])
    ssfr = np.concatenate([phz_ssfr[phz_valid], cosmos_ssfr[cosmos_valid]])
    weight = np.concatenate([
        phz_weight[phz_valid] * phz_blend[phz_valid],
        cosmos_blend[cosmos_valid],
    ])
    activity = np.where(ssfr < -11.0, "quenched", "star_forming")
    z_bin = np.searchsorted(z_edges, redshift, side="right") - 1
    magnitude_bin = np.searchsorted(magnitude_edges, magnitude, side="right") - 1
    shape = (z_edges.size - 1, magnitude_edges.size - 1)
    quenched_fraction = np.zeros(shape, dtype=np.float64)
    pooled_radius = np.zeros(shape, dtype=np.int64)
    class_payload: dict[str, dict[str, Any]] = {}
    class_stats: dict[str, dict[str, np.ndarray]] = {}
    all_cells_valid = True
    for label in ("quenched", "star_forming"):
        global_selected = activity == label
        global_effective = _effective_weight(weight[global_selected])
        if global_effective < minimum_effective_weight:
            all_cells_valid = False
        global_mass_mean, global_mass_sigma = _weighted_location_scale(
            mass[global_selected], weight[global_selected],
        )
        class_payload[label] = {
            "global_mass_mean": global_mass_mean,
            "global_mass_sigma": global_mass_sigma,
            "global_effective_weight": global_effective,
        }
        class_stats[label] = {
            key: np.full(shape, np.nan, dtype=np.float64)
            for key in (
                "mass_mean", "mass_sigma", "ssfr_mean", "ssfr_sigma",
                "effective_weight",
            )
        }
    for zi in range(shape[0]):
        for mi in range(shape[1]):
            selected_by_class: dict[str, np.ndarray] = {}
            selected_all = np.zeros(magnitude.shape, dtype=bool)
            selected_radius = max(shape)
            for radius in range(max(shape) + 1):
                local = (
                    (np.abs(z_bin - zi) <= radius)
                    & (np.abs(magnitude_bin - mi) <= radius)
                    & (z_bin >= 0) & (magnitude_bin >= 0)
                )
                candidate = {
                    label: local & (activity == label)
                    for label in ("quenched", "star_forming")
                }
                if all(
                    _effective_weight(weight[mask]) >= minimum_effective_weight
                    for mask in candidate.values()
                ):
                    selected_by_class = candidate
                    selected_all = local
                    selected_radius = radius
                    break
            if not selected_by_class:
                all_cells_valid = False
                selected_by_class = {
                    label: activity == label
                    for label in ("quenched", "star_forming")
                }
                selected_all = np.ones(magnitude.shape, dtype=bool)
            pooled_radius[zi, mi] = selected_radius
            total_weight = float(np.sum(weight[selected_all]))
            quenched_fraction[zi, mi] = (
                float(np.sum(weight[selected_by_class["quenched"]]))
                / total_weight if total_weight > 0.0 else 0.5
            )
            for label, selected in selected_by_class.items():
                mass_mean, mass_sigma = _weighted_location_scale(
                    mass[selected], weight[selected],
                )
                ssfr_mean, ssfr_sigma = _weighted_location_scale(
                    ssfr[selected], weight[selected],
                )
                class_stats[label]["mass_mean"][zi, mi] = mass_mean
                class_stats[label]["mass_sigma"][zi, mi] = mass_sigma
                class_stats[label]["ssfr_mean"][zi, mi] = ssfr_mean
                class_stats[label]["ssfr_sigma"][zi, mi] = ssfr_sigma
                class_stats[label]["effective_weight"][zi, mi] = (
                    _effective_weight(weight[selected])
                )
    for label in class_payload:
        class_payload[label].update({
            key: value.tolist() for key, value in class_stats[label].items()
        })
    return {
        "version": 1,
        "z_edges": z_edges.tolist(),
        "vis_magnitude_edges": magnitude_edges.tolist(),
        "activity_threshold_logssfr": -11.0,
        "pathological_upper_logssfr": -8.2,
        "minimum_effective_weight": minimum_effective_weight,
        "phz_rows": int(np.sum(phz_valid)),
        "cosmos_rows": int(np.sum(cosmos_valid)),
        "quenched_fraction": quenched_fraction.tolist(),
        "pooled_radius": pooled_radius.tolist(),
        "classes": class_payload,
        "all_cells_valid": all_cells_valid,
    }


def validate_physical_conditionals(
    draw: dict[str, np.ndarray], correction: dict[str, Any],
    conditionals: dict[str, Any], *, samples: int = 20_000, seed: int = 731,
) -> dict[str, Any]:
    """Fixed-seed Monte Carlo check of the compact physical sampler."""
    density = np.asarray(draw["density"], dtype=np.float64)
    factor = np.asarray(correction["factor"], dtype=np.float64)
    density = density * factor[:, :, None]
    marginal = np.sum(density, axis=2)
    fine_z = 0.5 * (
        np.asarray(draw["z_edges"][:-1]) + np.asarray(draw["z_edges"][1:])
    )
    fine_magnitude = 0.5 * (
        np.asarray(draw["vis_magnitude_edges"][:-1])
        + np.asarray(draw["vis_magnitude_edges"][1:])
    )
    z_edges = np.asarray(conditionals["z_edges"], dtype=np.float64)
    magnitude_edges = np.asarray(
        conditionals["vis_magnitude_edges"], dtype=np.float64,
    )
    z_lookup = np.clip(
        np.searchsorted(z_edges, fine_z, side="right") - 1,
        0, z_edges.size - 2,
    )
    magnitude_lookup = np.clip(
        np.searchsorted(magnitude_edges, fine_magnitude, side="right") - 1,
        0, magnitude_edges.size - 2,
    )
    quenched = np.asarray(conditionals["quenched_fraction"], dtype=np.float64)
    class_payload = conditionals["classes"]
    expected_quenched = 0.0
    expected_mass = 0.0
    total = float(np.sum(marginal))
    expected_redshift = np.sum(marginal, axis=1) / total
    mass_edges = np.linspace(6.0, 13.0, 29)
    expected_mass_probability = np.zeros(mass_edges.size - 1, dtype=np.float64)
    for zi in range(marginal.shape[0]):
        for mi in range(marginal.shape[1]):
            cell_weight = marginal[zi, mi] / total
            q = quenched[z_lookup[zi], magnitude_lookup[mi]]
            expected_quenched += cell_weight * q
            expected_mass += cell_weight * (
                q * class_payload["quenched"]["mass_mean"][z_lookup[zi]][magnitude_lookup[mi]]
                + (1.0 - q)
                * class_payload["star_forming"]["mass_mean"][z_lookup[zi]][magnitude_lookup[mi]]
            )
            for label, class_probability in (
                ("quenched", q), ("star_forming", 1.0 - q),
            ):
                payload = class_payload[label]
                mean = float(payload["mass_mean"][z_lookup[zi]][magnitude_lookup[mi]])
                sigma = max(
                    float(payload["mass_sigma"][z_lookup[zi]][magnitude_lookup[mi]]),
                    1e-6,
                )
                expected_mass_probability += (
                    cell_weight * class_probability
                    * np.diff(ndtr((mass_edges - mean) / sigma))
                )
    rng = np.random.default_rng(seed)
    flat_probability = marginal.ravel() / total
    flat = rng.choice(flat_probability.size, size=samples, p=flat_probability)
    z_index, magnitude_index = np.unravel_index(flat, marginal.shape)
    coarse_z = z_lookup[z_index]
    coarse_magnitude = magnitude_lookup[magnitude_index]
    q_probability = quenched[coarse_z, coarse_magnitude]
    is_quenched = rng.random(samples) < q_probability
    mass_draw = np.empty(samples, dtype=np.float64)
    for label, class_mask in (
        ("quenched", is_quenched), ("star_forming", ~is_quenched),
    ):
        payload = class_payload[label]
        mean = np.asarray(payload["mass_mean"])[coarse_z, coarse_magnitude]
        sigma = np.asarray(payload["mass_sigma"])[coarse_z, coarse_magnitude]
        mass_draw[class_mask] = rng.normal(mean[class_mask], sigma[class_mask])
    sampled_redshift = np.bincount(
        z_index, minlength=marginal.shape[0],
    ).astype(np.float64) / samples
    sampled_mass_probability = np.histogram(mass_draw, bins=mass_edges)[0] / samples
    quenched_error = abs(float(np.mean(is_quenched)) - expected_quenched)
    mass_error = abs(float(np.mean(mass_draw)) - expected_mass) / max(
        abs(expected_mass), 1.0,
    )
    redshift_marginal_error = float(np.max(np.abs(
        sampled_redshift - expected_redshift,
    )))
    mass_marginal_error = float(np.max(np.abs(
        sampled_mass_probability - expected_mass_probability,
    )))
    maximum_error = max(
        quenched_error, mass_error, redshift_marginal_error,
        mass_marginal_error,
    )
    return {
        "samples": int(samples),
        "seed": int(seed),
        "expected_quenched_fraction": expected_quenched,
        "sampled_quenched_fraction": float(np.mean(is_quenched)),
        "expected_mean_logmass": expected_mass,
        "sampled_mean_logmass": float(np.mean(mass_draw)),
        "quenched_fraction_error": quenched_error,
        "mean_logmass_fractional_error": mass_error,
        "redshift_marginal_maximum_error": redshift_marginal_error,
        "mass_marginal_maximum_error": mass_marginal_error,
        "redshift_expected_probability": expected_redshift.tolist(),
        "redshift_sampled_probability": sampled_redshift.tolist(),
        "mass_edges": mass_edges.tolist(),
        "mass_expected_probability": expected_mass_probability.tolist(),
        "mass_sampled_probability": sampled_mass_probability.tolist(),
        "maximum_error": maximum_error,
        "valid": maximum_error <= 0.02,
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
    absolute_like = mm - _COSMOLOGY.distmod(z).value[:, None]
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
        _COSMOLOGY.differential_comoving_volume(z).value
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
    distance_modulus = _COSMOLOGY.distmod(redshift).value
    absolute_like = magnitude - distance_modulus
    kpc_per_arcsec = _COSMOLOGY.kpc_proper_per_arcmin(redshift).value / 60.0
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
    absolute_like = mm - _COSMOLOGY.distmod(z).value[:, None]
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
        _COSMOLOGY.differential_comoving_volume(z).value
        / steradian_to_arcmin2
    )[:, None]
    zm_density = (
        lf_density * volume * np.diff(z_edges)[:, None]
        * np.diff(magnitude_edges)[None, :]
    )
    absolute_like = (
        magnitude[None, :] - _COSMOLOGY.distmod(z).value[:, None]
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
    kpc_per_arcsec = _COSMOLOGY.kpc_proper_per_arcmin(z).value / 60.0
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
