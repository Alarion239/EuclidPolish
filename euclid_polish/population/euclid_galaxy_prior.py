"""Minimal Euclid-only joint brightness and half-light-radius prior."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.special import ndtr

from euclid_polish.population.magnitude_law import StraightMagnitudeLaw

JOINT_EUCLID_GALAXY_VERSION = 4
RADIUS_MODEL_VERSION = 1
RADIUS_PIVOT_MAG = 23.0
LOG_RADIUS_MIN = float(np.log10(0.03))
LOG_RADIUS_MAX = float(np.log10(10.0))


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
