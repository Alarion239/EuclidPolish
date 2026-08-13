"""Straight differential number-count laws in apparent magnitude.

The analytical brightness priors used by generation are straight lines in
logarithmic surface density,

    log10(dN / dA / dm) = slope * m + intercept.

This module owns fitting, validation, integration, and inverse-CDF sampling so
galaxies and stars cannot silently implement different conventions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StraightMagnitudeLaw:
    """A finite-domain straight log-density law."""

    slope: float
    intercept: float
    mag_bright: float
    mag_faint: float
    fit_bright: float
    fit_faint: float
    covariance: tuple[tuple[float, float], tuple[float, float]]
    r_squared: float
    rms_log10_density: float
    source: str

    def __post_init__(self) -> None:
        values = (
            self.slope, self.intercept, self.mag_bright, self.mag_faint,
            self.fit_bright, self.fit_faint, self.r_squared,
            self.rms_log10_density,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("straight magnitude law contains non-finite values")
        if not self.mag_bright < self.mag_faint:
            raise ValueError("straight magnitude-law limits must be ordered")
        if not self.fit_bright < self.fit_faint:
            raise ValueError("straight magnitude-law fit limits must be ordered")
        covariance = np.asarray(self.covariance, dtype=np.float64)
        if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
            raise ValueError("straight magnitude-law covariance is invalid")
        if not self.source:
            raise ValueError("straight magnitude law requires source provenance")

    def log10_density(self, magnitude: np.ndarray | float) -> np.ndarray:
        values = np.asarray(magnitude, dtype=np.float64)
        return self.slope * values + self.intercept

    def density(self, magnitude: np.ndarray | float) -> np.ndarray:
        return np.power(10.0, self.log10_density(magnitude))

    def integrated_density(self) -> float:
        """Surface density integrated over the configured magnitude domain."""
        beta = self.slope * math.log(10.0)
        if abs(beta) < 1e-12:
            return float(
                10.0 ** self.intercept * (self.mag_faint - self.mag_bright)
            )
        anchor = math.exp(beta * self.mag_bright)
        return float(
            10.0 ** self.intercept
            * anchor
            * math.expm1(beta * (self.mag_faint - self.mag_bright))
            / beta
        )

    def truncated_to_density(self, density_arcmin2: float) -> StraightMagnitudeLaw:
        """Keep the bright end and shorten the faint limit to a density cap.

        The fitted slope and normalization remain untouched.  If the full
        configured interval already integrates below ``density_arcmin2``, the
        law is returned unchanged.
        """
        target = float(density_arcmin2)
        if not math.isfinite(target) or target <= 0.0:
            raise ValueError("straight magnitude-law density cap must be positive")
        if self.integrated_density() <= target:
            return self

        beta = self.slope * math.log(10.0)
        normalization = 10.0 ** self.intercept
        if abs(beta) < 1e-12:
            faint = self.mag_bright + target / normalization
        else:
            bright_term = math.exp(beta * self.mag_bright)
            faint_term = bright_term + target * beta / normalization
            if faint_term <= 0.0 or not math.isfinite(faint_term):
                raise ValueError("density cap cannot be represented on this magnitude law")
            faint = math.log(faint_term) / beta
        if not self.mag_bright < faint < self.mag_faint:
            raise ValueError("density cap produced an invalid faint magnitude limit")
        return replace(self, mag_faint=float(faint))

    def sample(self, rng: np.random.Generator) -> float:
        """Draw one magnitude from the exact finite-domain inverse CDF."""
        span = self.mag_faint - self.mag_bright
        beta = self.slope * math.log(10.0)
        u = float(rng.random())
        offset = (
            u * span
            if abs(beta) < 1e-12
            else math.log1p(u * math.expm1(beta * span)) / beta
        )
        return float(self.mag_bright + offset)

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": "straight_log10_differential_counts",
            "equation": "log10(dN_dA_dm) = slope * magnitude + intercept",
            "slope": float(self.slope),
            "intercept": float(self.intercept),
            "mag_bright": float(self.mag_bright),
            "mag_faint": float(self.mag_faint),
            "fit_bright": float(self.fit_bright),
            "fit_faint": float(self.fit_faint),
            "covariance": [list(row) for row in self.covariance],
            "r_squared": float(self.r_squared),
            "rms_log10_density": float(self.rms_log10_density),
            "surface_density_arcmin2": self.integrated_density(),
            "source": self.source,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> StraightMagnitudeLaw:
        if payload.get("kind") != "straight_log10_differential_counts":
            raise ValueError("magnitude distribution is not a straight count law")
        try:
            covariance_array = np.asarray(payload["covariance"], dtype=np.float64)
            covariance = tuple(
                tuple(float(value) for value in row)
                for row in covariance_array
            )
            law = cls(
                slope=float(payload["slope"]),
                intercept=float(payload["intercept"]),
                mag_bright=float(payload["mag_bright"]),
                mag_faint=float(payload["mag_faint"]),
                fit_bright=float(payload["fit_bright"]),
                fit_faint=float(payload["fit_faint"]),
                covariance=covariance,  # type: ignore[arg-type]
                r_squared=float(payload["r_squared"]),
                rms_log10_density=float(payload["rms_log10_density"]),
                source=str(payload["source"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("straight magnitude-law payload is malformed") from exc
        saved_density = float(payload.get("surface_density_arcmin2", float("nan")))
        if not math.isfinite(saved_density) or not math.isclose(
            saved_density, law.integrated_density(), rel_tol=2e-8, abs_tol=1e-10,
        ):
            raise ValueError("straight magnitude-law density is inconsistent")
        return law


@dataclass(frozen=True)
class StraightRegionFit:
    """Selected consecutive straight region plus its fitted law diagnostics."""

    start: int
    stop: int
    slope: float
    intercept: float
    covariance: np.ndarray
    r_squared: float
    rms: float


def _weighted_line(
    x: np.ndarray, y: np.ndarray, sigma: np.ndarray,
) -> tuple[float, float, np.ndarray, float, float]:
    design = np.column_stack([x, np.ones(x.size, dtype=np.float64)])
    weight = 1.0 / np.maximum(np.asarray(sigma, dtype=np.float64), 1e-8) ** 2
    normal = design.T @ (weight[:, None] * design)
    covariance = np.linalg.inv(normal)
    coefficients = covariance @ (design.T @ (weight * y))
    fitted = design @ coefficients
    residual = y - fitted
    mean = float(np.mean(y))
    total = float(np.sum((y - mean) ** 2))
    error = float(np.sum(residual ** 2))
    r_squared = 1.0 - error / total if total > 1e-20 else 1.0
    dof = max(1, x.size - 2)
    covariance = covariance * (float(np.sum(weight * residual ** 2)) / dof)
    return (
        float(coefficients[0]), float(coefficients[1]), covariance,
        float(r_squared), float(np.sqrt(np.mean(residual ** 2))),
    )


def fit_straight_region(
    magnitude: np.ndarray,
    density: np.ndarray,
    density_sigma: np.ndarray,
    *,
    minimum_span_mag: float,
    minimum_r_squared: float,
) -> StraightRegionFit:
    """Select the widest consecutive positive-count straight region."""
    x = np.asarray(magnitude, dtype=np.float64)
    d = np.asarray(density, dtype=np.float64)
    ds = np.asarray(density_sigma, dtype=np.float64)
    if x.ndim != 1 or d.shape != x.shape or ds.shape != x.shape:
        raise ValueError("straight-region arrays must be aligned vectors")
    valid = np.isfinite(x) & np.isfinite(d) & np.isfinite(ds) & (d > 0.0) & (ds > 0.0)
    indices = np.flatnonzero(valid)
    if indices.size < 3:
        raise ValueError("too few positive bins for a straight magnitude fit")
    typical_step = float(np.median(np.diff(x[indices])))
    best: tuple[tuple[float, float, int], StraightRegionFit] | None = None
    for left in range(indices.size - 2):
        for right in range(left + 3, indices.size + 1):
            selected = indices[left:right]
            if np.any(np.diff(selected) != 1):
                break
            xx = x[selected]
            span = float(xx[-1] - xx[0])
            if span + 0.51 * typical_step < float(minimum_span_mag):
                continue
            yy = np.log10(d[selected])
            sigma_log = ds[selected] / (d[selected] * math.log(10.0))
            slope, intercept, covariance, r_squared, rms = _weighted_line(
                xx, yy, sigma_log,
            )
            if slope <= 0.0 or r_squared < float(minimum_r_squared):
                continue
            fit = StraightRegionFit(
                start=int(selected[0]), stop=int(selected[-1] + 1),
                slope=slope, intercept=intercept, covariance=covariance,
                r_squared=r_squared, rms=rms,
            )
            rank = (span, -rms, selected.size)
            if best is None or rank > best[0]:
                best = (rank, fit)
    if best is None:
        raise ValueError(
            "no consecutive magnitude window passes the straight-line quality gate"
        )
    return best[1]


def fit_shared_slope(
    series: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[float, list[float], np.ndarray, float, float]:
    """Fit one slope and one independent intercept per supplied survey."""
    if len(series) < 2:
        raise ValueError("shared-slope fit requires at least two surveys")
    rows: list[np.ndarray] = []
    values: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for survey_index, (x, density, sigma) in enumerate(series):
        xx = np.asarray(x, dtype=np.float64)
        dd = np.asarray(density, dtype=np.float64)
        ss = np.asarray(sigma, dtype=np.float64)
        if xx.ndim != 1 or dd.shape != xx.shape or ss.shape != xx.shape:
            raise ValueError("shared-slope survey arrays must be aligned")
        if xx.size < 3 or np.any(dd <= 0.0) or np.any(ss <= 0.0):
            raise ValueError("shared-slope surveys require positive fitted bins")
        design = np.zeros((xx.size, len(series) + 1), dtype=np.float64)
        design[:, 0] = xx
        design[:, survey_index + 1] = 1.0
        rows.append(design)
        values.append(np.log10(dd))
        sigma_log = ss / (dd * math.log(10.0))
        weights.append(1.0 / np.maximum(sigma_log, 1e-8) ** 2)
    design = np.vstack(rows)
    y = np.concatenate(values)
    weight = np.concatenate(weights)
    normal = design.T @ (weight[:, None] * design)
    covariance = np.linalg.inv(normal)
    coefficients = covariance @ (design.T @ (weight * y))
    fitted = design @ coefficients
    residual = y - fitted
    survey_means = np.concatenate([
        np.full(values[index].shape, np.mean(values[index]))
        for index in range(len(series))
    ])
    total = float(np.sum((y - survey_means) ** 2))
    error = float(np.sum(residual ** 2))
    dof = max(1, y.size - coefficients.size)
    covariance *= float(np.sum(weight * residual ** 2)) / dof
    return (
        float(coefficients[0]),
        [float(value) for value in coefficients[1:]],
        covariance,
        1.0 - error / total if total > 1e-20 else 1.0,
        float(np.sqrt(np.mean(residual ** 2))),
    )
