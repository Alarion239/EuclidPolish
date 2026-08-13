"""Minimal Euclid-only joint brightness and half-light-radius prior."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.special import ndtr

from euclid_polish.population.magnitude_law import StraightMagnitudeLaw

JOINT_EUCLID_GALAXY_VERSION = 6
RADIUS_MODEL_VERSION = 1
RADIUS_PIVOT_MAG = 23.0
LOG_RADIUS_MIN = float(np.log10(0.03))
LOG_RADIUS_MAX = float(np.log10(10.0))
GALAXY_GENERATION_DENSITY_CAP_ARCMIN2 = 100.0


def generation_magnitude_law(
    fitted_law: StraightMagnitudeLaw,
) -> StraightMagnitudeLaw:
    """Return the bright-preserving, faint-truncated generation law."""
    return fitted_law.truncated_to_density(
        GALAXY_GENERATION_DENSITY_CAP_ARCMIN2
    )


@dataclass(frozen=True)
class ConditionalRadiusLaw:
    """Straight mean relation with constant Gaussian scatter in log radius."""

    version: int
    pivot_mag: float
    intercept_log10_arcsec: float
    slope_log10_arcsec_per_mag: float
    scatter_dex: float
    log_radius_min: float
    log_radius_max: float
    fitted_rows: int
    clipped_rows: int
    weighted_rows: float
    residual_rms_dex: float
    r_squared: float
    covariance: tuple[tuple[float, float], tuple[float, float]]
    selection: str

    @classmethod
    def from_payload(cls, payload: dict) -> ConditionalRadiusLaw:
        values = dict(payload)
        values["covariance"] = tuple(
            tuple(float(item) for item in row)
            for row in values["covariance"]
        )
        law = cls(**values)
        if law.version != RADIUS_MODEL_VERSION:
            raise ValueError("Euclid radius law has an unsupported version")
        if not (
            np.isfinite(law.intercept_log10_arcsec)
            and np.isfinite(law.slope_log10_arcsec_per_mag)
            and np.isfinite(law.scatter_dex)
            and law.scatter_dex > 0.0
            and law.log_radius_min < law.log_radius_max
            and law.fitted_rows >= 100
        ):
            raise ValueError("Euclid radius law is invalid")
        return law

    def to_payload(self) -> dict:
        return asdict(self)

    def mean(self, magnitude: np.ndarray | float) -> np.ndarray:
        values = np.asarray(magnitude, dtype=np.float64)
        return (
            self.intercept_log10_arcsec
            + self.slope_log10_arcsec_per_mag
            * (values - self.pivot_mag)
        )


def fit_conditional_radius_law(
    magnitude: np.ndarray,
    radius_arcsec: np.ndarray,
    weight: np.ndarray,
) -> ConditionalRadiusLaw:
    """Fit log10(R_e/arcsec) = alpha + beta(m-23) with robust clipping."""
    magnitude = np.asarray(magnitude, dtype=np.float64)
    radius = np.asarray(radius_arcsec, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    valid = (
        np.isfinite(magnitude) & np.isfinite(radius) & np.isfinite(weight)
        & (radius > 0.0) & (weight > 0.0)
    )
    magnitude, radius, weight = (
        magnitude[valid], radius[valid], weight[valid]
    )
    if magnitude.size < 100 or float(np.sum(weight)) <= 0.0:
        raise ValueError("At least 100 clean PHZ/MER Sérsic radii are required")
    x = magnitude - RADIUS_PIVOT_MAG
    y = np.log10(radius)
    design = np.column_stack((np.ones_like(x), x))
    keep = np.ones(x.size, dtype=bool)
    for _ in range(4):
        root_weight = np.sqrt(weight[keep])
        coefficients, *_ = np.linalg.lstsq(
            design[keep] * root_weight[:, None],
            y[keep] * root_weight,
            rcond=None,
        )
        residual = y - design @ coefficients
        center = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - center)))
        scale = max(1.4826 * mad, 0.03)
        keep = np.abs(residual - center) <= 4.5 * scale
    if int(np.sum(keep)) < 100:
        raise ValueError("Radius clipping left fewer than 100 objects")
    yk, wk, dk = y[keep], weight[keep], design[keep]
    root_weight = np.sqrt(wk)
    coefficients, *_ = np.linalg.lstsq(
        dk * root_weight[:, None], yk * root_weight, rcond=None,
    )
    residual = yk - dk @ coefficients
    weighted_rows = float(np.sum(wk))
    variance = float(np.sum(wk * residual**2) / weighted_rows)
    scatter = max(float(np.sqrt(variance)), 0.03)
    centered = yk - float(np.sum(wk * yk) / weighted_rows)
    total = float(np.sum(wk * centered**2))
    r_squared = 1.0 - float(np.sum(wk * residual**2)) / total if total > 0 else 0.0
    covariance = scatter**2 * np.linalg.pinv(
        (dk * root_weight[:, None]).T @ (dk * root_weight[:, None])
    )
    return ConditionalRadiusLaw(
        version=RADIUS_MODEL_VERSION,
        pivot_mag=RADIUS_PIVOT_MAG,
        intercept_log10_arcsec=float(coefficients[0]),
        slope_log10_arcsec_per_mag=float(coefficients[1]),
        scatter_dex=scatter,
        log_radius_min=LOG_RADIUS_MIN,
        log_radius_max=LOG_RADIUS_MAX,
        fitted_rows=int(np.sum(keep)),
        clipped_rows=int(keep.size - np.sum(keep)),
        weighted_rows=weighted_rows,
        residual_rms_dex=float(np.sqrt(np.mean(residual**2))),
        r_squared=float(r_squared),
        covariance=tuple(tuple(float(value) for value in row) for row in covariance),
        selection=(
            "PHZ_GAL_PROB >= 0.5; positive VIS 2FWHM flux; positive MER "
            "morphology VIS Sérsic R_e; SERSIC_VISNIR_FLAGS = 0; weighted "
            "by PHZ_GAL_PROB"
        ),
    )


def fit_conditional_radius_law_from_aggregate_moments(
    magnitude: np.ndarray,
    selected_rows: np.ndarray,
    expected_weight: np.ndarray,
    weighted_radius_sum_arcsec: np.ndarray,
    weighted_radius2_sum_arcsec2: np.ndarray,
) -> ConditionalRadiusLaw:
    """Fit the simple radius law from bracket-level sufficient statistics.

    Within each magnitude bracket the positive radius distribution is treated
    as log-normal.  Its first two linear-radius moments determine the mean and
    scatter in log radius; no object rows are needed.
    """
    magnitude = np.asarray(magnitude, dtype=np.float64)
    selected = np.asarray(selected_rows, dtype=np.float64)
    weight = np.asarray(expected_weight, dtype=np.float64)
    radius_sum = np.asarray(weighted_radius_sum_arcsec, dtype=np.float64)
    radius2_sum = np.asarray(weighted_radius2_sum_arcsec2, dtype=np.float64)
    if not (
        magnitude.shape == selected.shape == weight.shape
        == radius_sum.shape == radius2_sum.shape
    ):
        raise ValueError("Aggregate radius-moment arrays must have one shape")
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_radius = radius_sum / weight
        mean_radius2 = radius2_sum / weight
        variance_ln = np.log(mean_radius2 / np.square(mean_radius))
        mean_log10 = (
            np.log(mean_radius) - 0.5 * variance_ln
        ) / np.log(10.0)
        scatter_dex = np.sqrt(variance_ln) / np.log(10.0)
    valid = (
        np.isfinite(magnitude) & np.isfinite(selected) & (selected > 0.0)
        & np.isfinite(weight) & (weight > 0.0)
        & np.isfinite(mean_log10)
        & np.isfinite(scatter_dex) & (scatter_dex > 0.0)
    )
    magnitude = magnitude[valid]
    selected = selected[valid]
    weight = weight[valid]
    mean_log10 = mean_log10[valid]
    scatter_dex = np.maximum(scatter_dex[valid], 0.03)
    total_selected = int(round(float(np.sum(selected))))
    total_weight = float(np.sum(weight))
    if magnitude.size < 8 or total_selected < 100 or total_weight <= 0.0:
        raise ValueError(
            "At least eight populated magnitude brackets and 100 clean "
            "PHZ/MER Sersic radii are required"
        )

    x = magnitude - RADIUS_PIVOT_MAG
    design = np.column_stack((np.ones_like(x), x))
    # The uncertainty of a bracket mean scales as sigma / sqrt(N).  PHZ
    # probability weight supplies the effective N for this intentionally
    # compact model.
    fit_weight = weight / np.square(scatter_dex)
    root_weight = np.sqrt(fit_weight)
    coefficients, *_ = np.linalg.lstsq(
        design * root_weight[:, None], mean_log10 * root_weight, rcond=None,
    )
    residual = mean_log10 - design @ coefficients
    pooled_variance = float(np.sum(
        weight * (np.square(scatter_dex) + np.square(residual))
    ) / total_weight)
    scatter = max(float(np.sqrt(pooled_variance)), 0.03)
    centered = mean_log10 - float(np.average(mean_log10, weights=weight))
    total = float(np.sum(weight * np.square(centered)))
    residual_total = float(np.sum(weight * np.square(residual)))
    r_squared = 1.0 - residual_total / total if total > 0.0 else 0.0
    covariance = np.linalg.pinv(
        (design * root_weight[:, None]).T
        @ (design * root_weight[:, None])
    )
    return ConditionalRadiusLaw(
        version=RADIUS_MODEL_VERSION,
        pivot_mag=RADIUS_PIVOT_MAG,
        intercept_log10_arcsec=float(coefficients[0]),
        slope_log10_arcsec_per_mag=float(coefficients[1]),
        scatter_dex=scatter,
        log_radius_min=LOG_RADIUS_MIN,
        log_radius_max=LOG_RADIUS_MAX,
        fitted_rows=total_selected,
        clipped_rows=0,
        weighted_rows=total_weight,
        residual_rms_dex=float(np.sqrt(np.mean(np.square(residual)))),
        r_squared=float(r_squared),
        covariance=tuple(
            tuple(float(value) for value in row) for row in covariance
        ),
        selection=(
            "aggregate Q1 magnitude brackets; PHZ_GAL_PROB >= 0.5; "
            "positive VIS 2FWHM flux; positive MER morphology VIS Sersic "
            "R_e; SERSIC_VISNIR_FLAGS = 0; weighted by PHZ_GAL_PROB"
        ),
    )


def joint_density_grid(
    magnitude_law: StraightMagnitudeLaw,
    radius_law: ConditionalRadiusLaw,
    *,
    magnitude_edges: np.ndarray | None = None,
    log_radius_edges: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return cell-integrated density for the simple Euclid joint law."""
    mag_edges = np.asarray(
        magnitude_edges
        if magnitude_edges is not None
        else np.linspace(magnitude_law.mag_bright, magnitude_law.mag_faint, 151),
        dtype=np.float64,
    )
    radius_edges = np.asarray(
        log_radius_edges
        if log_radius_edges is not None
        else np.linspace(radius_law.log_radius_min, radius_law.log_radius_max, 101),
        dtype=np.float64,
    )
    magnitude = 0.5 * (mag_edges[:-1] + mag_edges[1:])
    log_radius = 0.5 * (radius_edges[:-1] + radius_edges[1:])
    magnitude_density = magnitude_law.density(magnitude)
    mean = radius_law.mean(magnitude)
    upper = (
        radius_edges[None, 1:] - mean[:, None]
    ) / radius_law.scatter_dex
    lower = (
        radius_edges[None, :-1] - mean[:, None]
    ) / radius_law.scatter_dex
    probability = ndtr(upper) - ndtr(lower)
    probability /= np.sum(probability, axis=1, keepdims=True)
    density = magnitude_density[:, None] * np.diff(mag_edges)[:, None] * probability
    return {
        "density": density,
        "magnitude": magnitude,
        "magnitude_edges": mag_edges,
        "log_radius": log_radius,
        "log_radius_edges": radius_edges,
    }
