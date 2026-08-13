"""Minimal Euclid-only joint brightness and half-light-radius prior."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr

from euclid_polish.population.magnitude_law import (
    FaintCappedMagnitudeLaw,
    StraightMagnitudeLaw,
)

JOINT_EUCLID_GALAXY_VERSION = 9
SUPPORTED_JOINT_EUCLID_GALAXY_VERSIONS = frozenset({7, 8, 9})
LINEAR_RADIUS_MODEL_VERSION = 1
CONSTANT_TAIL_RADIUS_MODEL_VERSION = 2
RADIUS_MODEL_VERSION = 3
RADIUS_PIVOT_MAG = 23.0
LOG_RADIUS_MIN = float(np.log10(0.03))
LOG_RADIUS_MAX = float(np.log10(10.0))
GALAXY_FAINT_DENSITY_CAP_ARCMIN2_MAG = 100.0
RADIUS_FIT_MIN_SELECTED_PER_MAG_BIN = 20
RADIUS_FIT_EFFECTIVE_WEIGHT_CAP = 1000.0
RADIUS_FIT_FAINT_MAGNITUDE = 25.5
RADIUS_TAIL_TAPER_END_MAGNITUDE = 27.0


def generation_magnitude_law(
    fitted_law: StraightMagnitudeLaw,
) -> FaintCappedMagnitudeLaw:
    """Return the fitted bright line with a flat faint-count tail."""
    return FaintCappedMagnitudeLaw(
        straight_law=fitted_law,
        density_cap_arcmin2_mag=GALAXY_FAINT_DENSITY_CAP_ARCMIN2_MAG,
    )


@dataclass(frozen=True)
class ConditionalRadiusLaw:
    """Bounded conditional law for Euclid Sersic radius in log space."""

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
    bright_intercept_log10_arcsec: float | None = None
    break_magnitude: float | None = None
    tail_fraction: float = 0.0
    tail_distribution: str = "none"
    fit_min_selected_per_magnitude_bin: int = 0
    fit_effective_weight_cap: float = 0.0
    fit_faint_magnitude: float | None = None
    tail_taper_start_magnitude: float | None = None
    tail_taper_end_magnitude: float | None = None

    @classmethod
    def from_payload(cls, payload: dict) -> ConditionalRadiusLaw:
        values = dict(payload)
        values.setdefault("bright_intercept_log10_arcsec", None)
        values.setdefault("break_magnitude", None)
        values.setdefault("tail_fraction", 0.0)
        values.setdefault("tail_distribution", "none")
        values.setdefault("fit_min_selected_per_magnitude_bin", 0)
        values.setdefault("fit_effective_weight_cap", 0.0)
        values.setdefault("fit_faint_magnitude", None)
        values.setdefault("tail_taper_start_magnitude", None)
        values.setdefault("tail_taper_end_magnitude", None)
        values["covariance"] = tuple(
            tuple(float(item) for item in row)
            for row in values["covariance"]
        )
        law = cls(**values)
        if law.version not in {
            LINEAR_RADIUS_MODEL_VERSION,
            CONSTANT_TAIL_RADIUS_MODEL_VERSION,
            RADIUS_MODEL_VERSION,
        }:
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
        if law.version in {
            CONSTANT_TAIL_RADIUS_MODEL_VERSION, RADIUS_MODEL_VERSION,
        } and not (
            law.bright_intercept_log10_arcsec is not None
            and np.isfinite(law.bright_intercept_log10_arcsec)
            and law.break_magnitude is not None
            and np.isfinite(law.break_magnitude)
            and 0.0 <= law.tail_fraction < 1.0
            and law.tail_distribution == "uniform_log_radius"
            and law.fit_min_selected_per_magnitude_bin >= 1
            and law.fit_effective_weight_cap > 0.0
        ):
            raise ValueError("Euclid broken radius law is invalid")
        if law.version == RADIUS_MODEL_VERSION and not (
            law.fit_faint_magnitude is not None
            and np.isfinite(law.fit_faint_magnitude)
            and law.tail_taper_start_magnitude is not None
            and np.isfinite(law.tail_taper_start_magnitude)
            and law.tail_taper_end_magnitude is not None
            and np.isfinite(law.tail_taper_end_magnitude)
            and law.tail_taper_start_magnitude < law.tail_taper_end_magnitude
            and np.isclose(
                law.fit_faint_magnitude,
                law.tail_taper_start_magnitude,
            )
        ):
            raise ValueError("Euclid radius-tail taper is invalid")
        return law

    def to_payload(self) -> dict:
        return asdict(self)

    def core_mean(self, magnitude: np.ndarray | float) -> np.ndarray:
        """Return the Gaussian-core location in log10 arcsec."""
        values = np.asarray(magnitude, dtype=np.float64)
        faint = (
            self.intercept_log10_arcsec
            + self.slope_log10_arcsec_per_mag
            * (values - self.pivot_mag)
        )
        if self.version == LINEAR_RADIUS_MODEL_VERSION:
            return faint
        return np.where(
            values < float(self.break_magnitude),
            float(self.bright_intercept_log10_arcsec),
            faint,
        )

    def mean(self, magnitude: np.ndarray | float) -> np.ndarray:
        """Return the conditional mean log10 radius."""
        core = self.core_mean(magnitude)
        tail_fraction = self.tail_fraction_at(magnitude)
        if not np.any(tail_fraction > 0.0):
            return core
        uniform_mean = 0.5 * (self.log_radius_min + self.log_radius_max)
        return (1.0 - tail_fraction) * core + tail_fraction * uniform_mean

    def tail_fraction_at(self, magnitude: np.ndarray | float) -> np.ndarray:
        """Return the broad-component fraction at each magnitude."""
        values = np.asarray(magnitude, dtype=np.float64)
        if self.tail_fraction <= 0.0:
            return np.zeros_like(values)
        if self.version < RADIUS_MODEL_VERSION:
            return np.full_like(values, self.tail_fraction)
        start = float(self.tail_taper_start_magnitude)
        end = float(self.tail_taper_end_magnitude)
        taper = np.clip((end - values) / (end - start), 0.0, 1.0)
        return self.tail_fraction * taper

    def bin_probability(
        self, magnitude: np.ndarray, log_radius_edges: np.ndarray,
    ) -> np.ndarray:
        """Return bounded conditional probability in each log-radius bin."""
        values = np.atleast_1d(np.asarray(magnitude, dtype=np.float64))
        edges = np.asarray(log_radius_edges, dtype=np.float64)
        mean = self.core_mean(values)
        upper = (edges[None, 1:] - mean[:, None]) / self.scatter_dex
        lower = (edges[None, :-1] - mean[:, None]) / self.scatter_dex
        core = ndtr(upper) - ndtr(lower)
        core /= np.sum(core, axis=1, keepdims=True)
        tail_fraction = self.tail_fraction_at(values)
        if not np.any(tail_fraction > 0.0):
            return core
        tail = np.diff(edges) / (edges[-1] - edges[0])
        return (
            (1.0 - tail_fraction[:, None]) * core
            + tail_fraction[:, None] * tail[None, :]
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
        version=LINEAR_RADIUS_MODEL_VERSION,
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
        version=LINEAR_RADIUS_MODEL_VERSION,
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


def fit_conditional_radius_law_from_binned_counts(
    magnitude_edges: np.ndarray,
    log_radius_edges: np.ndarray,
    selected_counts: np.ndarray,
    expected_counts: np.ndarray,
    *,
    fit_bright: float,
    fit_faint: float,
) -> ConditionalRadiusLaw:
    """Fit the conditional lognormal law to bounded joint Q1 counts.

    The likelihood uses the probability integrated over each log-radius bin
    and normalizes it over the unchanged 0.03--10 arcsec draw domain.  This
    avoids inferring a global lognormal scatter from outlier-sensitive linear
    radius moments.
    """
    mag_edges = np.asarray(magnitude_edges, dtype=np.float64)
    radius_edges = np.asarray(log_radius_edges, dtype=np.float64)
    selected = np.asarray(selected_counts, dtype=np.float64)
    counts = np.asarray(expected_counts, dtype=np.float64)
    expected_shape = (mag_edges.size - 1, radius_edges.size - 1)
    if (
        mag_edges.ndim != 1
        or radius_edges.ndim != 1
        or mag_edges.size < 3
        or radius_edges.size < 3
        or selected.shape != expected_shape
        or counts.shape != expected_shape
        or not np.all(np.isfinite(mag_edges))
        or not np.all(np.isfinite(radius_edges))
        or not np.all(np.diff(mag_edges) > 0.0)
        or not np.all(np.diff(radius_edges) > 0.0)
        or not np.all(np.isfinite(selected) & (selected >= 0.0))
        or not np.all(np.isfinite(counts) & (counts >= 0.0))
    ):
        raise ValueError("Binned radius-fit inputs are malformed")
    magnitude = 0.5 * (mag_edges[:-1] + mag_edges[1:])
    fit_rows = (
        (magnitude >= float(fit_bright))
        & (magnitude <= float(fit_faint))
        & (np.sum(counts, axis=1) > 0.0)
    )
    selected_fit = selected[fit_rows]
    counts_fit = counts[fit_rows]
    magnitude_fit = magnitude[fit_rows]
    total_selected = int(round(float(np.sum(selected_fit))))
    total_weight = float(np.sum(counts_fit))
    if magnitude_fit.size < 8 or total_selected < 100 or total_weight <= 0.0:
        raise ValueError(
            "At least eight populated magnitude bins and 100 bounded "
            "PHZ/MER Sersic radii are required"
        )

    radius_center = 0.5 * (radius_edges[:-1] + radius_edges[1:])
    row_weight = np.sum(counts_fit, axis=1)
    row_mean = np.sum(counts_fit * radius_center[None, :], axis=1) / row_weight
    x = magnitude_fit - RADIUS_PIVOT_MAG
    design = np.column_stack((np.ones_like(x), x))
    root_weight = np.sqrt(row_weight)
    initial_coefficients, *_ = np.linalg.lstsq(
        design * root_weight[:, None], row_mean * root_weight, rcond=None,
    )
    initial_residual = (
        radius_center[None, :] - design @ initial_coefficients[:, None]
    )
    initial_scatter = float(np.sqrt(
        np.sum(counts_fit * np.square(initial_residual)) / total_weight
    ))
    initial = np.asarray([
        float(initial_coefficients[0]),
        float(initial_coefficients[1]),
        math.log(min(max(initial_scatter, 0.05), 1.0)),
    ])

    def objective(parameters: np.ndarray) -> float:
        intercept, slope, log_scatter = parameters
        scatter = math.exp(float(log_scatter))
        mean = intercept + slope * x
        upper = (radius_edges[None, 1:] - mean[:, None]) / scatter
        lower = (radius_edges[None, :-1] - mean[:, None]) / scatter
        probability = ndtr(upper) - ndtr(lower)
        normalization = np.sum(probability, axis=1, keepdims=True)
        probability = probability / np.maximum(normalization, 1e-300)
        return -float(
            np.sum(counts_fit * np.log(np.maximum(probability, 1e-300)))
            / total_weight
        )

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=(
            (float(radius_edges[0] - 1.0), float(radius_edges[-1] + 1.0)),
            (-1.0, 1.0),
            (math.log(0.03), math.log(1.5)),
        ),
        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 2000},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ValueError(f"Binned conditional-radius fit failed: {result.message}")
    intercept, slope, log_scatter = (float(value) for value in result.x)
    scatter = math.exp(log_scatter)
    modeled_row_mean = intercept + slope * x
    residual = row_mean - modeled_row_mean
    centered = row_mean - float(np.average(row_mean, weights=row_weight))
    total = float(np.sum(row_weight * np.square(centered)))
    residual_total = float(np.sum(row_weight * np.square(residual)))
    r_squared = 1.0 - residual_total / total if total > 0.0 else 0.0
    covariance = scatter**2 * np.linalg.pinv(
        (design * root_weight[:, None]).T
        @ (design * root_weight[:, None])
    )
    return ConditionalRadiusLaw(
        version=LINEAR_RADIUS_MODEL_VERSION,
        pivot_mag=RADIUS_PIVOT_MAG,
        intercept_log10_arcsec=intercept,
        slope_log10_arcsec_per_mag=slope,
        scatter_dex=scatter,
        log_radius_min=float(radius_edges[0]),
        log_radius_max=float(radius_edges[-1]),
        fitted_rows=total_selected,
        clipped_rows=0,
        weighted_rows=total_weight,
        residual_rms_dex=float(np.sqrt(
            np.sum(row_weight * np.square(residual)) / total_weight
        )),
        r_squared=float(r_squared),
        covariance=tuple(
            tuple(float(value) for value in row) for row in covariance
        ),
        selection=(
            "bounded aggregate Q1 magnitude x log-radius bins; "
            f"{float(fit_bright):g} <= VIS 2FWHM <= "
            f"{float(fit_faint):g}; PHZ_GAL_PROB >= 0.5; 0.03 <= MER "
            "morphology VIS Sersic R_e < 10 arcsec; "
            "SERSIC_VISNIR_FLAGS = 0; weighted by PHZ_GAL_PROB"
        ),
    )


def fit_broken_conditional_radius_law_from_binned_counts(
    magnitude_edges: np.ndarray,
    log_radius_edges: np.ndarray,
    selected_counts: np.ndarray,
    expected_counts: np.ndarray,
    *,
    minimum_selected_per_bin: int = RADIUS_FIT_MIN_SELECTED_PER_MAG_BIN,
    effective_weight_cap: float = RADIUS_FIT_EFFECTIVE_WEIGHT_CAP,
    fit_faint_magnitude: float = RADIUS_FIT_FAINT_MAGNITUDE,
    tail_taper_end_magnitude: float = RADIUS_TAIL_TAPER_END_MAGNITUDE,
) -> ConditionalRadiusLaw:
    """Fit a broken Gaussian core plus a broad log-radius tail to Q1 bins.

    Every sufficiently populated magnitude bracket contributes at most
    ``effective_weight_cap`` to the conditional likelihood.  This prevents the
    millions of faint detections from erasing the measured bright plateau and
    sharp transition.  A uniform component in log radius represents the broad
    Q1 Sersic tail through the count turnover, then tapers to zero so the
    incompleteness-dominated faint upturn is not extrapolated.
    """
    mag_edges = np.asarray(magnitude_edges, dtype=np.float64)
    radius_edges = np.asarray(log_radius_edges, dtype=np.float64)
    selected = np.asarray(selected_counts, dtype=np.float64)
    counts = np.asarray(expected_counts, dtype=np.float64)
    expected_shape = (mag_edges.size - 1, radius_edges.size - 1)
    if (
        mag_edges.ndim != 1
        or radius_edges.ndim != 1
        or mag_edges.size < 3
        or radius_edges.size < 3
        or selected.shape != expected_shape
        or counts.shape != expected_shape
        or not np.all(np.isfinite(mag_edges))
        or not np.all(np.isfinite(radius_edges))
        or not np.all(np.diff(mag_edges) > 0.0)
        or not np.all(np.diff(radius_edges) > 0.0)
        or not np.all(np.isfinite(selected) & (selected >= 0.0))
        or not np.all(np.isfinite(counts) & (counts >= 0.0))
        or int(minimum_selected_per_bin) < 1
        or not np.isfinite(effective_weight_cap)
        or effective_weight_cap <= 0.0
        or not np.isfinite(fit_faint_magnitude)
        or not np.isfinite(tail_taper_end_magnitude)
        or tail_taper_end_magnitude <= fit_faint_magnitude
    ):
        raise ValueError("Broken binned radius-fit inputs are malformed")
    magnitude = 0.5 * (mag_edges[:-1] + mag_edges[1:])
    selected_by_magnitude = np.sum(selected, axis=1)
    expected_by_magnitude = np.sum(counts, axis=1)
    fit_rows = (
        (selected_by_magnitude >= int(minimum_selected_per_bin))
        & (expected_by_magnitude > 0.0)
        & (magnitude <= float(fit_faint_magnitude))
    )
    selected_fit = selected[fit_rows]
    counts_fit = counts[fit_rows]
    magnitude_fit = magnitude[fit_rows]
    row_weight = np.sum(counts_fit, axis=1)
    total_selected = int(round(float(np.sum(selected_fit))))
    total_weight = float(np.sum(counts_fit))
    if magnitude_fit.size < 20 or total_selected < 100 or total_weight <= 0.0:
        raise ValueError(
            "At least 20 populated magnitude bins and 100 bounded PHZ/MER "
            "Sersic radii are required"
        )
    effective_row_weight = np.minimum(row_weight, float(effective_weight_cap))
    effective_counts = counts_fit * (
        effective_row_weight / row_weight
    )[:, None]
    effective_total = float(np.sum(effective_counts))
    radius_center = 0.5 * (radius_edges[:-1] + radius_edges[1:])
    row_mean = np.sum(counts_fit * radius_center[None, :], axis=1) / row_weight
    uniform_probability = np.diff(radius_edges) / (
        radius_edges[-1] - radius_edges[0]
    )

    transition_candidates = (
        (magnitude_fit[:-1] >= 16.5)
        & (magnitude_fit[1:] <= 19.5)
    )
    difference = np.diff(row_mean)
    difference[~transition_candidates] = -np.inf
    if np.any(np.isfinite(difference)):
        transition_index = int(np.nanargmax(difference))
        initial_break = float(
            0.5 * (magnitude_fit[transition_index] + magnitude_fit[transition_index + 1])
        )
    else:
        initial_break = 18.0
    initial_break = float(np.clip(initial_break, 17.4, 19.2))
    bright_rows = magnitude_fit < initial_break - 0.2
    peak_rows = (
        (magnitude_fit >= initial_break + 0.5)
        & (magnitude_fit <= initial_break + 1.5)
    )
    initial_bright = float(np.median(row_mean[bright_rows]))
    initial_peak = float(np.median(row_mean[peak_rows]))
    faint_rows = magnitude_fit >= initial_break + 1.0
    faint_x = magnitude_fit[faint_rows] - initial_break
    faint_design = np.column_stack((np.ones_like(faint_x), faint_x))
    faint_root_weight = np.sqrt(effective_row_weight[faint_rows])
    faint_coefficients, *_ = np.linalg.lstsq(
        faint_design * faint_root_weight[:, None],
        row_mean[faint_rows] * faint_root_weight,
        rcond=None,
    )
    initial = np.asarray([
        initial_bright,
        initial_peak,
        initial_break,
        float(np.clip(faint_coefficients[1], -0.4, 0.05)),
        0.2,
        0.12,
    ])
    bounds = (
        (float(radius_edges[0]), 0.3),
        (-0.5, 0.7),
        (17.4, 19.2),
        (-0.4, 0.05),
        (0.03, 1.0),
        (0.0, 0.5),
    )

    def probabilities(parameters: np.ndarray) -> np.ndarray:
        bright, peak, break_magnitude, slope, scatter, tail_fraction = parameters
        core_mean = np.where(
            magnitude_fit < break_magnitude,
            bright,
            peak + slope * (magnitude_fit - break_magnitude),
        )
        upper = (radius_edges[None, 1:] - core_mean[:, None]) / scatter
        lower = (radius_edges[None, :-1] - core_mean[:, None]) / scatter
        core = ndtr(upper) - ndtr(lower)
        core /= np.maximum(np.sum(core, axis=1, keepdims=True), 1e-300)
        return (
            (1.0 - tail_fraction) * core
            + tail_fraction * uniform_probability[None, :]
        )

    def objective(parameters: np.ndarray) -> float:
        probability = probabilities(parameters)
        return -float(
            np.sum(effective_counts * np.log(np.maximum(probability, 1e-300)))
            / effective_total
        )

    starts = []
    for break_start in np.unique(np.clip(
        initial_break + np.linspace(-0.3, 0.4, 8), 17.4, 19.2,
    )):
        start = initial.copy()
        start[2] = break_start
        starts.append(minimize(
            objective,
            start,
            method="Nelder-Mead",
            bounds=bounds,
            options={"xatol": 1e-10, "fatol": 1e-11, "maxiter": 20_000},
        ))
    successful = [candidate for candidate in starts if candidate.success]
    result = min(successful, key=lambda candidate: candidate.fun) if successful else starts[0]
    if result.success:
        refined = minimize(
            objective,
            result.x,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-14, "gtol": 1e-9, "maxiter": 5000},
        )
        if refined.success and refined.fun <= result.fun:
            result = refined
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ValueError(f"Broken conditional-radius fit failed: {result.message}")
    bright, peak, break_magnitude, slope, scatter, tail_fraction = (
        float(value) for value in result.x
    )
    intercept = peak + slope * (RADIUS_PIVOT_MAG - break_magnitude)
    uniform_mean = 0.5 * (radius_edges[0] + radius_edges[-1])
    core_model = np.where(
        magnitude_fit < break_magnitude,
        bright,
        intercept + slope * (magnitude_fit - RADIUS_PIVOT_MAG),
    )
    modeled_row_mean = (
        (1.0 - tail_fraction) * core_model + tail_fraction * uniform_mean
    )
    residual = row_mean - modeled_row_mean
    centered = row_mean - float(np.average(
        row_mean, weights=effective_row_weight,
    ))
    total = float(np.sum(effective_row_weight * np.square(centered)))
    residual_total = float(np.sum(
        effective_row_weight * np.square(residual),
    ))
    r_squared = 1.0 - residual_total / total if total > 0.0 else 0.0
    faint_design = np.column_stack((
        np.ones(int(np.sum(faint_rows))),
        magnitude_fit[faint_rows] - RADIUS_PIVOT_MAG,
    ))
    faint_root_weight = np.sqrt(effective_row_weight[faint_rows])
    covariance = scatter**2 * np.linalg.pinv(
        (faint_design * faint_root_weight[:, None]).T
        @ (faint_design * faint_root_weight[:, None])
    )
    return ConditionalRadiusLaw(
        version=RADIUS_MODEL_VERSION,
        pivot_mag=RADIUS_PIVOT_MAG,
        intercept_log10_arcsec=intercept,
        slope_log10_arcsec_per_mag=slope,
        scatter_dex=scatter,
        log_radius_min=float(radius_edges[0]),
        log_radius_max=float(radius_edges[-1]),
        fitted_rows=total_selected,
        clipped_rows=0,
        weighted_rows=total_weight,
        residual_rms_dex=float(np.sqrt(
            residual_total / float(np.sum(effective_row_weight))
        )),
        r_squared=float(r_squared),
        covariance=tuple(
            tuple(float(value) for value in row) for row in covariance
        ),
        selection=(
            "bounded aggregate Q1 magnitude x log-radius bins; at least "
            f"{int(minimum_selected_per_bin)} selected radii per magnitude "
            f"bin; each magnitude bin capped at {effective_weight_cap:g} "
            f"effective PHZ weight; fit stops at VIS {fit_faint_magnitude:g} "
            f"and broad tail tapers to zero by VIS {tail_taper_end_magnitude:g}; "
            "PHZ_GAL_PROB >= 0.5; 0.03 <= MER "
            "morphology VIS Sersic R_e < 10 arcsec; "
            "SERSIC_VISNIR_FLAGS = 0"
        ),
        bright_intercept_log10_arcsec=bright,
        break_magnitude=break_magnitude,
        tail_fraction=tail_fraction,
        tail_distribution="uniform_log_radius",
        fit_min_selected_per_magnitude_bin=int(minimum_selected_per_bin),
        fit_effective_weight_cap=float(effective_weight_cap),
        fit_faint_magnitude=float(fit_faint_magnitude),
        tail_taper_start_magnitude=float(fit_faint_magnitude),
        tail_taper_end_magnitude=float(tail_taper_end_magnitude),
    )


def joint_density_grid(
    magnitude_law: StraightMagnitudeLaw | FaintCappedMagnitudeLaw,
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
    probability = radius_law.bin_probability(magnitude, radius_edges)
    density = magnitude_density[:, None] * np.diff(mag_edges)[:, None] * probability
    return {
        "density": density,
        "magnitude": magnitude,
        "magnitude_edges": mag_edges,
        "log_radius": log_radius,
        "log_radius_edges": radius_edges,
    }
