"""Minimal Euclid-only joint brightness and half-light-radius prior."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import cast

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr

from euclid_polish.population.magnitude_law import (
    ContinuousBrightBridgeFaintCappedMagnitudeLaw,
    StraightMagnitudeLaw,
)

JOINT_EUCLID_GALAXY_VERSION = 13
JOINT_EUCLID_GALAXY_KIND = (
    "euclid_vis2fwhm_circularized_sersic_re_joint"
)
RADIUS_MODEL_VERSION = 5
APERTURE_FWHM_MODEL_VERSION = 1
RADIUS_PIVOT_MAG = 23.0
RADIUS_FIT_MIN_SELECTED_PER_MAG_BIN = 20
RADIUS_FIT_EFFECTIVE_WEIGHT_CAP = 1000.0
RADIUS_FIT_FAINT_MAGNITUDE = 25.5
BRIGHT_BRIDGE_JOIN_MAGNITUDES = (16.4, 19.0, 20.9)

type _Covariance2x2 = tuple[tuple[float, float], tuple[float, float]]


def fit_continuous_generation_magnitude_law(
    fitted_law: StraightMagnitudeLaw,
    bright_bins: list[dict],
    *,
    footprint_area_arcmin2: float,
    density_cap_arcmin2_mag: float,
) -> tuple[
    ContinuousBrightBridgeFaintCappedMagnitudeLaw,
    dict[str, float | int | list[float]],
]:
    """Fit a three-slope continuous bridge into the trusted main Q1 law.

    The very bright 0.1-mag bins contain only a handful of objects, so they
    are not retained as independent generation parameters.  Three log-linear
    slopes cover fixed, interpretable intervals ending at VIS 16.4, 19.0, and
    20.9, then join the well-measured main line continuously.  Bin-integrated
    Poisson deviance includes empty bins through their finite modeled-count
    penalty and therefore needs no logarithmic pseudo-count.  The caller
    supplies the faint cap from the observed Q1 differential-count peak so
    this law cannot silently extrapolate to a fixed project-wide ceiling.
    """
    if not bright_bins:
        raise ValueError("Q1 bright-count bins are required")
    try:
        lower = np.asarray([
            float(item["mag_lo"]) for item in bright_bins
        ], dtype=np.float64)
        upper = np.asarray([
            float(item["mag_hi"]) for item in bright_bins
        ], dtype=np.float64)
        expected = np.asarray([
            float(item["expected_galaxies"]) for item in bright_bins
        ], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Q1 bright-count bins are malformed") from exc
    area = float(footprint_area_arcmin2)
    density_cap = float(density_cap_arcmin2_mag)
    joins = BRIGHT_BRIDGE_JOIN_MAGNITUDES
    if (
        lower.size < 30
        or lower.shape != upper.shape
        or lower.shape != expected.shape
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or not np.all(np.isfinite(expected) & (expected >= 0.0))
        or not np.all(upper > lower)
        or not np.allclose(lower[1:], upper[:-1], atol=1e-10, rtol=0.0)
        or not np.isclose(lower[0], fitted_law.mag_bright)
        or upper[-1] < joins[-1] - 1e-10
        or not np.isfinite(area)
        or area <= 0.0
        or not np.isfinite(density_cap)
        or density_cap <= 0.0
    ):
        raise ValueError("Q1 bright-count bins are malformed")

    fit_bins = lower < joins[-1] - 1e-10
    lower = lower[fit_bins]
    upper = upper[fit_bins]
    expected = expected[fit_bins]
    if lower.size < 30 or upper[-1] < joins[-1] - 1e-10:
        raise ValueError("Q1 bright-count bins do not cover the bridge")

    def interval_density(
        bright_slopes: tuple[float, float, float],
    ) -> np.ndarray:
        law = ContinuousBrightBridgeFaintCappedMagnitudeLaw(
            straight_law=fitted_law,
            bright_slopes=bright_slopes,
            bright_join_magnitudes=joins,
            density_cap_arcmin2_mag=density_cap,
        )
        result = np.empty_like(lower)
        for index, (lo, hi) in enumerate(zip(lower, upper, strict=True)):
            mass = 0.0
            for bright, faint, slope, intercept in law._line_components():
                overlap_bright = max(lo, bright)
                overlap_faint = min(hi, faint)
                if overlap_faint <= overlap_bright:
                    continue
                mass += law._line_integral(
                    slope,
                    intercept,
                    overlap_bright,
                    overlap_faint,
                )
            result[index] = mass
        return result

    def objective(parameters: np.ndarray) -> float:
        slopes = (
            float(parameters[0]),
            float(parameters[1]),
            float(parameters[2]),
        )
        modeled = area * interval_density(slopes)
        return float(np.sum(
            modeled - expected * np.log(np.maximum(modeled, 1e-300))
        ))

    bounds = (
        (0.05, 3.0),
        (0.01, 1.0),
        (0.05, 1.0),
    )
    starts = []
    for first_slope in (1.0, 1.5, 2.0):
        for second_slope in (0.2, 0.35):
            for third_slope in (0.4, 0.6):
                starts.append(minimize(
                    objective,
                    np.asarray([
                        first_slope, second_slope, third_slope,
                    ]),
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={
                        "ftol": 1e-14, "gtol": 1e-9, "maxiter": 3000,
                    },
                ))
    successful = [candidate for candidate in starts if candidate.success]
    result = min(
        successful or starts,
        key=lambda candidate: float(candidate.fun),
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ValueError(f"Continuous bright-count fit failed: {result.message}")
    bright_slopes = (
        float(result.x[0]),
        float(result.x[1]),
        float(result.x[2]),
    )
    law = ContinuousBrightBridgeFaintCappedMagnitudeLaw(
        straight_law=fitted_law,
        bright_slopes=bright_slopes,
        bright_join_magnitudes=joins,
        density_cap_arcmin2_mag=density_cap,
    )
    modeled = area * interval_density(bright_slopes)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.where(
            expected > 0.0,
            np.log(expected / np.maximum(modeled, 1e-300)),
            0.0,
        )
    deviance = float(2.0 * np.sum(
        modeled - expected + expected * log_ratio
    ))
    return law, {
        "bright_fit_bin_count": int(lower.size),
        "bright_fit_zero_bin_count": int(np.sum(expected <= 0.0)),
        "bright_fit_expected_galaxies": float(np.sum(expected)),
        "bright_fit_poisson_deviance": deviance,
        "bright_fit_deviance_per_bin": deviance / float(lower.size),
        "bright_fit_parameter_count": 3,
        "bright_bridge_join_magnitudes": list(joins),
        "bright_bridge_slopes": list(law.bright_slopes),
    }


@dataclass(frozen=True)
class ConditionalRadiusLaw:
    """Current bounded conditional law for circularized Euclid Sersic radius."""

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
    fit_min_selected_per_magnitude_bin: int
    fit_effective_weight_cap: float
    fit_faint_magnitude: float

    @classmethod
    def from_payload(cls, payload: dict) -> ConditionalRadiusLaw:
        """Load the current radius law, ignoring surplus keys in v11 artifacts."""
        try:
            covariance = tuple(
                tuple(float(item) for item in row)
                for row in payload["covariance"]
            )
            law = cls(
                version=int(payload["version"]),
                pivot_mag=float(payload["pivot_mag"]),
                intercept_log10_arcsec=float(
                    payload["intercept_log10_arcsec"]
                ),
                slope_log10_arcsec_per_mag=float(
                    payload["slope_log10_arcsec_per_mag"]
                ),
                scatter_dex=float(payload["scatter_dex"]),
                log_radius_min=float(payload["log_radius_min"]),
                log_radius_max=float(payload["log_radius_max"]),
                fitted_rows=int(payload["fitted_rows"]),
                clipped_rows=int(payload["clipped_rows"]),
                weighted_rows=float(payload["weighted_rows"]),
                residual_rms_dex=float(payload["residual_rms_dex"]),
                r_squared=float(payload["r_squared"]),
                covariance=cast(_Covariance2x2, covariance),
                selection=str(payload["selection"]),
                fit_min_selected_per_magnitude_bin=int(
                    payload["fit_min_selected_per_magnitude_bin"]
                ),
                fit_effective_weight_cap=float(
                    payload["fit_effective_weight_cap"]
                ),
                fit_faint_magnitude=float(payload["fit_faint_magnitude"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Euclid radius law payload is malformed") from exc
        covariance = np.asarray(law.covariance, dtype=np.float64)
        if not (
            law.version == RADIUS_MODEL_VERSION
            and np.isfinite(law.pivot_mag)
            and np.isfinite(law.intercept_log10_arcsec)
            and np.isfinite(law.slope_log10_arcsec_per_mag)
            and np.isfinite(law.scatter_dex)
            and law.scatter_dex > 0.0
            and np.isfinite(law.log_radius_min)
            and np.isfinite(law.log_radius_max)
            and law.log_radius_min < law.log_radius_max
            and law.fitted_rows >= 100
            and law.clipped_rows >= 0
            and np.isfinite(law.weighted_rows)
            and law.weighted_rows > 0.0
            and np.isfinite(law.residual_rms_dex)
            and law.residual_rms_dex >= 0.0
            and np.isfinite(law.r_squared)
            and covariance.shape == (2, 2)
            and np.all(np.isfinite(covariance))
            and bool(law.selection.strip())
            and law.fit_min_selected_per_magnitude_bin >= 1
            and law.fit_effective_weight_cap > 0.0
            and np.isfinite(law.fit_faint_magnitude)
        ):
            raise ValueError("Euclid radius law is invalid")
        return law

    def to_payload(self) -> dict:
        return asdict(self)

    def mean(self, magnitude: np.ndarray | float) -> np.ndarray:
        """Return the conditional mean log10 circularized radius."""
        values = np.asarray(magnitude, dtype=np.float64)
        return (
            self.intercept_log10_arcsec
            + self.slope_log10_arcsec_per_mag
            * (values - self.pivot_mag)
        )

    def bin_probability(
        self, magnitude: np.ndarray, log_radius_edges: np.ndarray,
    ) -> np.ndarray:
        """Return bounded conditional probability in each log-radius bin."""
        values = np.atleast_1d(np.asarray(magnitude, dtype=np.float64))
        edges = np.asarray(log_radius_edges, dtype=np.float64)
        mean = self.mean(values)
        upper = (edges[None, 1:] - mean[:, None]) / self.scatter_dex
        lower = (edges[None, :-1] - mean[:, None]) / self.scatter_dex
        probability = ndtr(upper) - ndtr(lower)
        return probability / np.sum(probability, axis=1, keepdims=True)


@dataclass(frozen=True)
class ConditionalApertureFWHMDistribution:
    """Empirical MER photometric FWHM distribution conditioned on brightness."""

    version: int
    magnitude_edges: tuple[float, ...]
    fwhm_edges_arcsec: tuple[float, ...]
    probability: tuple[tuple[float, ...], ...]
    source_magnitude_bin: tuple[int, ...]
    selection: str
    out_of_support_policy: str

    @classmethod
    def from_payload(
        cls, payload: dict,
    ) -> ConditionalApertureFWHMDistribution:
        try:
            distribution = cls(
                version=int(payload["version"]),
                magnitude_edges=tuple(
                    float(value) for value in payload["magnitude_edges"]
                ),
                fwhm_edges_arcsec=tuple(
                    float(value) for value in payload["fwhm_edges_arcsec"]
                ),
                probability=tuple(
                    tuple(float(value) for value in row)
                    for row in payload["probability"]
                ),
                source_magnitude_bin=tuple(
                    int(value) for value in payload["source_magnitude_bin"]
                ),
                selection=str(payload["selection"]),
                out_of_support_policy=str(payload["out_of_support_policy"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "MER aperture-FWHM distribution payload is malformed"
            ) from exc
        magnitude_edges = np.asarray(
            distribution.magnitude_edges, dtype=np.float64,
        )
        fwhm_edges = np.asarray(
            distribution.fwhm_edges_arcsec, dtype=np.float64,
        )
        probability = np.asarray(
            distribution.probability, dtype=np.float64,
        )
        source = np.asarray(
            distribution.source_magnitude_bin, dtype=np.int64,
        )
        expected_shape = (magnitude_edges.size - 1, fwhm_edges.size - 1)
        if not (
            distribution.version == APERTURE_FWHM_MODEL_VERSION
            and magnitude_edges.size >= 2
            and fwhm_edges.size >= 2
            and probability.shape == expected_shape
            and source.shape == (expected_shape[0],)
            and np.all(np.isfinite(magnitude_edges))
            and np.all(np.diff(magnitude_edges) > 0.0)
            and np.all(np.isfinite(fwhm_edges) & (fwhm_edges > 0.0))
            and np.all(np.diff(fwhm_edges) > 0.0)
            and np.all(np.isfinite(probability) & (probability >= 0.0))
            and np.allclose(
                np.sum(probability, axis=1), 1.0, rtol=0.0, atol=1e-10,
            )
            and np.all((source >= 0) & (source < expected_shape[0]))
            and bool(distribution.selection.strip())
            and distribution.out_of_support_policy
            == "nearest_observed_magnitude_bin"
        ):
            raise ValueError("MER aperture-FWHM distribution is invalid")
        return distribution

    def to_payload(self) -> dict:
        return asdict(self)

    @property
    def minimum_arcsec(self) -> float:
        return float(self.fwhm_edges_arcsec[0])

    @property
    def maximum_arcsec(self) -> float:
        return float(self.fwhm_edges_arcsec[-1])

    def _magnitude_bin(self, magnitude: float) -> int:
        value = float(magnitude)
        if not math.isfinite(value):
            raise ValueError("VIS 2FWHM magnitude must be finite")
        return int(np.clip(
            np.searchsorted(self.magnitude_edges, value, side="right") - 1,
            0,
            len(self.magnitude_edges) - 2,
        ))

    def sample(
        self, magnitude: float, rng: np.random.Generator,
    ) -> float:
        """Draw the MER FWHM paired with a sampled VIS-2FWHM magnitude."""
        magnitude_bin = self._magnitude_bin(magnitude)
        probability = np.asarray(
            self.probability[magnitude_bin], dtype=np.float64,
        )
        fwhm_bin = int(rng.choice(probability.size, p=probability))
        return float(rng.uniform(
            self.fwhm_edges_arcsec[fwhm_bin],
            self.fwhm_edges_arcsec[fwhm_bin + 1],
        ))

    def mean(self, magnitude: np.ndarray | float) -> np.ndarray:
        values = np.atleast_1d(np.asarray(magnitude, dtype=np.float64))
        if not np.all(np.isfinite(values)):
            raise ValueError("VIS 2FWHM magnitudes must be finite")
        indices = np.clip(
            np.searchsorted(self.magnitude_edges, values, side="right") - 1,
            0,
            len(self.magnitude_edges) - 2,
        )
        centres = 0.5 * (
            np.asarray(self.fwhm_edges_arcsec[:-1])
            + np.asarray(self.fwhm_edges_arcsec[1:])
        )
        probability = np.asarray(self.probability, dtype=np.float64)
        return probability[indices] @ centres


def fit_conditional_aperture_fwhm_distribution(
    magnitude_edges: np.ndarray,
    fwhm_edges_arcsec: np.ndarray,
    expected_counts: np.ndarray,
    *,
    selection: str,
) -> ConditionalApertureFWHMDistribution:
    """Normalize grouped Q1 magnitude-FWHM counts without breaking pairing."""
    mag_edges = np.asarray(magnitude_edges, dtype=np.float64)
    fwhm_edges = np.asarray(fwhm_edges_arcsec, dtype=np.float64)
    counts = np.asarray(expected_counts, dtype=np.float64)
    expected_shape = (mag_edges.size - 1, fwhm_edges.size - 1)
    if not (
        mag_edges.size >= 2
        and fwhm_edges.size >= 2
        and counts.shape == expected_shape
        and np.all(np.isfinite(mag_edges))
        and np.all(np.diff(mag_edges) > 0.0)
        and np.all(np.isfinite(fwhm_edges) & (fwhm_edges > 0.0))
        and np.all(np.diff(fwhm_edges) > 0.0)
        and np.all(np.isfinite(counts) & (counts >= 0.0))
        and bool(str(selection).strip())
    ):
        raise ValueError("MER aperture-FWHM fit inputs are malformed")
    weight = np.sum(counts, axis=1)
    populated = np.flatnonzero(weight > 0.0)
    if not populated.size or float(np.sum(weight)) <= 0.0:
        raise ValueError("MER aperture-FWHM histogram is empty")
    source = np.arange(weight.size, dtype=np.int64)
    for index in np.flatnonzero(weight <= 0.0):
        source[index] = int(populated[np.argmin(np.abs(populated - index))])
    probability = counts[source] / weight[source, None]
    return ConditionalApertureFWHMDistribution(
        version=APERTURE_FWHM_MODEL_VERSION,
        magnitude_edges=tuple(float(value) for value in mag_edges),
        fwhm_edges_arcsec=tuple(float(value) for value in fwhm_edges),
        probability=tuple(
            tuple(float(value) for value in row) for row in probability
        ),
        source_magnitude_bin=tuple(int(value) for value in source),
        selection=str(selection),
        out_of_support_policy="nearest_observed_magnitude_bin",
    )



def fit_linear_conditional_radius_law_from_binned_counts(
    magnitude_edges: np.ndarray,
    log_radius_edges: np.ndarray,
    selected_counts: np.ndarray,
    expected_counts: np.ndarray,
    *,
    minimum_selected_per_bin: int = RADIUS_FIT_MIN_SELECTED_PER_MAG_BIN,
    effective_weight_cap: float = RADIUS_FIT_EFFECTIVE_WEIGHT_CAP,
    fit_faint_magnitude: float = RADIUS_FIT_FAINT_MAGNITUDE,
) -> ConditionalRadiusLaw:
    """Fit one bounded Gaussian log-radius relation with no generated tail.

    The science-clean Q1 conditional means are almost exactly linear.  Each
    sufficiently populated magnitude bracket contributes at most
    ``effective_weight_cap`` to the likelihood, so the millions of faint
    measurements do not erase the sparse bright relation.  The distribution
    is integrated over the unchanged 0.03--10 arcsec support and has no
    separate broad component.
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
    ):
        raise ValueError("Linear binned radius-fit inputs are malformed")

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

    effective_row_weight = np.minimum(
        row_weight, float(effective_weight_cap),
    )
    effective_counts = counts_fit * (
        effective_row_weight / row_weight
    )[:, None]
    effective_total = float(np.sum(effective_counts))
    radius_center = 0.5 * (radius_edges[:-1] + radius_edges[1:])
    row_mean = np.sum(
        counts_fit * radius_center[None, :], axis=1,
    ) / row_weight
    x = magnitude_fit - RADIUS_PIVOT_MAG
    design = np.column_stack((np.ones_like(x), x))
    root_weight = np.sqrt(effective_row_weight)
    initial_coefficients, *_ = np.linalg.lstsq(
        design * root_weight[:, None],
        row_mean * root_weight,
        rcond=None,
    )
    initial_mean = design @ initial_coefficients
    initial_scatter = float(np.sqrt(
        np.sum(
            effective_counts
            * np.square(radius_center[None, :] - initial_mean[:, None])
        )
        / effective_total
    ))
    initial = np.asarray([
        float(initial_coefficients[0]),
        float(initial_coefficients[1]),
        math.log(min(max(initial_scatter, 0.05), 1.0)),
    ])

    def probabilities(parameters: np.ndarray) -> np.ndarray:
        intercept, slope, log_scatter = parameters
        scatter = math.exp(float(log_scatter))
        mean = intercept + slope * x
        upper = (radius_edges[None, 1:] - mean[:, None]) / scatter
        lower = (radius_edges[None, :-1] - mean[:, None]) / scatter
        probability = ndtr(upper) - ndtr(lower)
        return probability / np.maximum(
            np.sum(probability, axis=1, keepdims=True), 1e-300,
        )

    def objective(parameters: np.ndarray) -> float:
        probability = probabilities(parameters)
        return -float(
            np.sum(
                effective_counts
                * np.log(np.maximum(probability, 1e-300))
            )
            / effective_total
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
        options={"ftol": 1e-14, "gtol": 1e-9, "maxiter": 5000},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ValueError(f"Linear conditional-radius fit failed: {result.message}")
    intercept, slope, log_scatter = (float(value) for value in result.x)
    scatter = math.exp(log_scatter)
    modeled_row_mean = intercept + slope * x
    residual = row_mean - modeled_row_mean
    centered = row_mean - float(np.average(
        row_mean, weights=effective_row_weight,
    ))
    total = float(np.sum(effective_row_weight * np.square(centered)))
    residual_total = float(np.sum(
        effective_row_weight * np.square(residual),
    ))
    r_squared = 1.0 - residual_total / total if total > 0.0 else 0.0
    covariance = scatter**2 * np.linalg.pinv(
        (design * root_weight[:, None]).T
        @ (design * root_weight[:, None])
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
        covariance=(
            (float(covariance[0, 0]), float(covariance[0, 1])),
            (float(covariance[1, 0]), float(covariance[1, 1])),
        ),
        selection=(
            "bounded aggregate Q1 magnitude x log-radius bins; at least "
            f"{int(minimum_selected_per_bin)} selected radii per magnitude "
            f"bin; each magnitude bin capped at {effective_weight_cap:g} "
            f"effective PHZ weight; fit stops at VIS {fit_faint_magnitude:g}; "
            "straight truncated-Gaussian circularized Sersic radius with no "
            "generated broad tail; PHZ_GAL_PROB >= 0.5; literature-style "
            "MER quality and morphology-fit cuts; 0.03 <= circularized VIS "
            "Sersic R_e < 10 arcsec"
        ),
        fit_min_selected_per_magnitude_bin=int(minimum_selected_per_bin),
        fit_effective_weight_cap=float(effective_weight_cap),
        fit_faint_magnitude=float(fit_faint_magnitude),
    )

def joint_density_grid(
    magnitude_law: ContinuousBrightBridgeFaintCappedMagnitudeLaw,
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
