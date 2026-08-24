"""Straight differential number-count laws in apparent magnitude.

The analytical brightness priors used by generation are straight lines in
logarithmic surface density,

    log10(dN / dA / dm) = slope * m + intercept.

This module owns fitting, validation, integration, and inverse-CDF sampling so
galaxies and stars cannot silently implement different conventions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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
class ContinuousBrightBridgeFaintCappedMagnitudeLaw:
    """Three continuous bright count lines, a main line, and a faint cap.

    ``bright_slopes`` describes three consecutive log10-density lines ending
    at ``bright_join_magnitudes``.  The final bright line joins the nested,
    well-constrained main count law continuously.  Every intercept is derived
    recursively from those joins, so the bridge adds only three fitted
    parameters and cannot acquire per-bin degrees of freedom.
    """

    straight_law: StraightMagnitudeLaw
    bright_slopes: tuple[float, float, float]
    bright_join_magnitudes: tuple[float, float, float]
    density_cap_arcmin2_mag: float

    def __post_init__(self) -> None:
        slopes = np.asarray(self.bright_slopes, dtype=np.float64)
        joins = np.asarray(self.bright_join_magnitudes, dtype=np.float64)
        cap = float(self.density_cap_arcmin2_mag)
        if (
            slopes.shape != (3,)
            or not np.all(np.isfinite(slopes) & (slopes > 0.0))
        ):
            raise ValueError(
                "continuous bright-bridge magnitude-law slopes are invalid"
            )
        if (
            joins.shape != (3,)
            or not np.all(np.isfinite(joins))
            or not np.all(np.diff(joins) > 0.0)
        ):
            raise ValueError(
                "continuous bright-bridge magnitude-law joins are invalid"
            )
        if not math.isfinite(cap) or cap <= 0.0:
            raise ValueError("faint magnitude-law density cap must be positive")
        if self.straight_law.slope <= 0.0:
            raise ValueError(
                "continuous bright-bridge law requires a positive main slope"
            )
        object.__setattr__(
            self, "bright_slopes", tuple(float(value) for value in slopes),
        )
        object.__setattr__(
            self,
            "bright_join_magnitudes",
            tuple(float(value) for value in joins),
        )
        object.__setattr__(self, "density_cap_arcmin2_mag", cap)
        if not (
            self.mag_bright < self.bright_join_magnitudes[0]
            and self.bright_join_magnitudes[-1] < self.break_magnitude
        ):
            raise ValueError(
                "continuous bright-bridge joins must lie before the faint cap"
            )
        if not self.break_magnitude < self.mag_faint:
            raise ValueError(
                "continuous bright-bridge faint cap must lie inside the domain"
            )

    @property
    def mag_bright(self) -> float:
        return self.straight_law.mag_bright

    @property
    def mag_faint(self) -> float:
        return self.straight_law.mag_faint

    @property
    def slope(self) -> float:
        return self.straight_law.slope

    @property
    def intercept(self) -> float:
        return self.straight_law.intercept

    @property
    def source(self) -> str:
        return self.straight_law.source

    @property
    def bright_intercepts(self) -> tuple[float, float, float]:
        """Return all three bridge intercepts implied by continuity."""
        slope1, slope2, slope3 = self.bright_slopes
        join1, join2, join3 = self.bright_join_magnitudes
        intercept3 = (self.slope - slope3) * join3 + self.intercept
        intercept2 = (slope3 - slope2) * join2 + intercept3
        intercept1 = (slope2 - slope1) * join1 + intercept2
        return float(intercept1), float(intercept2), float(intercept3)

    @property
    def break_magnitude(self) -> float:
        """Magnitude where the main line reaches the flat faint cap."""
        return float(
            (math.log10(self.density_cap_arcmin2_mag) - self.intercept)
            / self.slope
        )

    def log10_density(self, magnitude: np.ndarray | float) -> np.ndarray:
        values = np.asarray(magnitude, dtype=np.float64)
        result = np.minimum(
            self.straight_law.log10_density(values),
            math.log10(self.density_cap_arcmin2_mag),
        )
        for slope, intercept, join in reversed(tuple(zip(
            self.bright_slopes,
            self.bright_intercepts,
            self.bright_join_magnitudes,
            strict=True,
        ))):
            result = np.where(values < join, slope * values + intercept, result)
        return result

    def density(self, magnitude: np.ndarray | float) -> np.ndarray:
        return np.power(10.0, self.log10_density(magnitude))

    @staticmethod
    def _line_integral(
        slope: float, intercept: float, bright: float, faint: float,
    ) -> float:
        beta = slope * math.log(10.0)
        normalization = 10.0 ** intercept
        if abs(beta) < 1e-12:
            return float(normalization * (faint - bright))
        return float(
            normalization
            * math.exp(beta * bright)
            * math.expm1(beta * (faint - bright))
            / beta
        )

    def _line_components(
        self,
    ) -> tuple[tuple[float, float, float, float], ...]:
        edges = (
            self.mag_bright,
            *self.bright_join_magnitudes,
            self.break_magnitude,
        )
        slopes = (*self.bright_slopes, self.slope)
        intercepts = (*self.bright_intercepts, self.intercept)
        return tuple(
            (float(edges[index]), float(edges[index + 1]), slope, intercept)
            for index, (slope, intercept) in enumerate(zip(
                slopes, intercepts, strict=True,
            ))
        )

    def _component_masses(self) -> tuple[float, ...]:
        line_masses = tuple(
            self._line_integral(slope, intercept, bright, faint)
            for bright, faint, slope, intercept in self._line_components()
        )
        faint = self.density_cap_arcmin2_mag * (
            self.mag_faint - self.break_magnitude
        )
        return (*line_masses, faint)

    def integrated_density(self) -> float:
        """Return the exact mass of every bridge, main, and faint component."""
        return float(sum(self._component_masses()))

    @staticmethod
    def _invert_line_mass(
        target: float, slope: float, intercept: float, bright: float,
    ) -> float:
        beta = slope * math.log(10.0)
        normalization = 10.0 ** intercept
        if abs(beta) < 1e-12:
            return float(bright + target / normalization)
        bright_term = math.exp(beta * bright)
        return float(
            math.log(bright_term + target * beta / normalization) / beta
        )

    def sample(self, rng: np.random.Generator) -> float:
        """Draw from the exact five-component finite-domain inverse CDF."""
        masses = self._component_masses()
        target = float(rng.random()) * sum(masses)
        for mass, (bright, _faint, slope, intercept) in zip(
            masses[:-1], self._line_components(), strict=True,
        ):
            if target < mass:
                return self._invert_line_mass(
                    target, slope, intercept, bright,
                )
            target -= mass
        return float(
            self.break_magnitude + target / self.density_cap_arcmin2_mag
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": (
                "continuous_three_slope_bright_bridge_main_flat_faint_counts"
            ),
            "equation": (
                "three continuous bright log10-linear count segments; fitted "
                "main line; constant faint density cap"
            ),
            "straight_law": self.straight_law.to_payload(),
            "bright_slopes": list(self.bright_slopes),
            "bright_intercepts": list(self.bright_intercepts),
            "bright_join_magnitudes": list(self.bright_join_magnitudes),
            "density_cap_arcmin2_mag": self.density_cap_arcmin2_mag,
            "break_magnitude": self.break_magnitude,
            "surface_density_arcmin2": self.integrated_density(),
            "source": self.source,
        }

    @classmethod
    def from_payload(
        cls, payload: dict[str, Any],
    ) -> ContinuousBrightBridgeFaintCappedMagnitudeLaw:
        if payload.get("kind") != (
            "continuous_three_slope_bright_bridge_main_flat_faint_counts"
        ):
            raise ValueError(
                "magnitude distribution is not a continuous bright-bridge law"
            )
        try:
            law = cls(
                straight_law=StraightMagnitudeLaw.from_payload(
                    payload["straight_law"]
                ),
                bright_slopes=tuple(
                    float(value) for value in payload["bright_slopes"]
                ),
                bright_join_magnitudes=tuple(
                    float(value)
                    for value in payload["bright_join_magnitudes"]
                ),
                density_cap_arcmin2_mag=float(
                    payload["density_cap_arcmin2_mag"]
                ),
            )
            saved_bright_intercepts = np.asarray(
                payload["bright_intercepts"], dtype=np.float64,
            )
            saved_break = float(payload["break_magnitude"])
            saved_density = float(payload["surface_density_arcmin2"])
            saved_source = str(payload["source"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "continuous bright-bridge magnitude-law payload is malformed"
            ) from exc
        checks = (
            saved_bright_intercepts.shape == (3,)
            and np.all(np.isfinite(saved_bright_intercepts))
            and np.allclose(
                saved_bright_intercepts,
                law.bright_intercepts,
                rtol=2e-8,
                atol=1e-10,
            )
            and math.isfinite(saved_break)
            and math.isclose(
                saved_break,
                law.break_magnitude,
                rel_tol=2e-8,
                abs_tol=1e-10,
            )
            and math.isfinite(saved_density)
            and math.isclose(
                saved_density,
                law.integrated_density(),
                rel_tol=2e-8,
                abs_tol=1e-10,
            )
            and saved_source == law.source
        )
        if not checks:
            raise ValueError(
                "continuous bright-bridge magnitude-law payload is inconsistent"
            )
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
