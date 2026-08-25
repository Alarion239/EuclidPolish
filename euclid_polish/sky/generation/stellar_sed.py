"""Empirical Gaia-to-Euclid stellar colours for synthetic point sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from euclid_polish.population.magnitude_law import StraightMagnitudeLaw


@dataclass(frozen=True)
class StellarSED:
    """One sampled stellar SED, normalised to a supplied VIS magnitude."""

    temperature_k: float
    extinction_av: float
    magnitudes: dict[str, float]


@dataclass(frozen=True)
class EmpiricalStellarPrior:
    """Activated Gaia latent CDF plus matched Euclid colour mapping."""

    bp_rp_quantiles: np.ndarray
    temperature_quantiles_k: np.ndarray
    band_coefficients: np.ndarray
    residual_covariance: np.ndarray
    magnitude_law: StraightMagnitudeLaw | None = None
    color_model: dict[str, np.ndarray] | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> EmpiricalStellarPrior:
        gaia = payload.get("gaia") or {}
        mapping = payload.get("euclid_mapping") or {}
        colors = np.asarray(gaia.get("bp_rp_quantiles") or [], dtype=np.float64)
        temperatures = np.asarray(
            gaia.get("temperature_quantiles_k") or [], dtype=np.float64,
        )
        coefficients_by_name = mapping.get("g_to_band_offset_coefficients") or {}
        coefficients = np.asarray([
            coefficients_by_name.get(key, [])
            for key in ("mag_vis", "mag_y_e", "mag_j_e", "mag_h_e")
        ], dtype=np.float64)
        covariance = np.asarray(mapping.get("residual_covariance"), dtype=np.float64)
        if colors.size < 2 or temperatures.size < 2:
            raise ValueError("stellar prior requires Gaia colour and temperature CDFs")
        if coefficients.shape != (4, 3) or covariance.shape != (4, 4):
            raise ValueError("stellar prior has invalid Euclid mapping dimensions")
        distribution = payload.get("population", {}).get("magnitude_distribution", {})
        try:
            magnitude_law = StraightMagnitudeLaw.from_payload(distribution)
        except ValueError as exc:
            raise ValueError(
                "stellar prior requires a straight magnitude-count law"
            ) from exc
        color_model_payload = payload.get("color_model") or {}
        color_model: dict[str, np.ndarray] | None = None
        if color_model_payload.get("kind") == "gaia_euclid_latent_locus_v1":
            try:
                parsed = {
                    key: np.asarray(color_model_payload[key], dtype=np.float64)
                    for key in (
                        "bp_rp_edges", "bp_rp_nodes", "temperature_nodes_k",
                        "locus_colors", "intrinsic_color_covariance",
                        "magnitude_edges", "magnitude_node_weights",
                    )
                }
                if (
                    parsed["bp_rp_edges"].ndim != 1
                    or parsed["bp_rp_nodes"].ndim != 1
                    or parsed["temperature_nodes_k"].shape
                    != parsed["bp_rp_nodes"].shape
                    or parsed["locus_colors"].shape
                    != (parsed["bp_rp_nodes"].size, 3)
                    or parsed["intrinsic_color_covariance"].shape != (3, 3)
                    or parsed["magnitude_edges"].ndim != 1
                    or parsed["magnitude_node_weights"].shape
                    != (parsed["magnitude_edges"].size - 1,
                        parsed["bp_rp_nodes"].size)
                    or parsed["bp_rp_edges"].size
                    != parsed["bp_rp_nodes"].size + 1
                ):
                    raise ValueError("invalid latent stellar locus dimensions")
                if (
                    not all(np.all(np.isfinite(value)) for value in parsed.values())
                    or np.any(np.diff(parsed["bp_rp_edges"]) <= 0)
                    or np.any(np.diff(parsed["magnitude_edges"]) <= 0)
                    or np.any(parsed["magnitude_node_weights"] < 0)
                    or np.any(~np.isfinite(parsed["magnitude_node_weights"]))
                ):
                    raise ValueError("latent stellar locus contains non-finite values")
                covariance = _positive_semidefinite_covariance(
                    parsed["intrinsic_color_covariance"], floor=1e-4,
                )
                parsed["intrinsic_color_covariance"] = covariance
                parsed["magnitude_node_weights"] = _normalise_rows(
                    parsed["magnitude_node_weights"],
                )
                color_model = parsed
            except (KeyError, TypeError, ValueError):
                color_model = None
        if color_model is None:
            raise ValueError("stellar prior requires a valid latent colour model")
        return cls(
            colors, temperatures, coefficients, covariance,
            magnitude_law, color_model,
        )

    def sample_magnitude(self, rng: np.random.Generator) -> float:
        """Sample VIS magnitude from the activated straight count law."""
        if self.magnitude_law is None:
            raise ValueError("stellar prior has no straight magnitude-count law")
        return self.magnitude_law.sample(rng)

    def sample(self, rng: np.random.Generator, mag_vis: float) -> StellarSED:
        if self.color_model is None:
            raise ValueError("stellar prior has no latent colour model")
        return self._sample_latent_locus(rng, mag_vis)

    def _sample_latent_locus(
        self, rng: np.random.Generator, mag_vis: float,
    ) -> StellarSED:
        model = self.color_model
        assert model is not None
        edges = model["magnitude_edges"]
        bin_index = int(np.searchsorted(edges, mag_vis, side="right") - 1)
        bin_index = max(0, min(bin_index, edges.size - 2))
        node_weights = model["magnitude_node_weights"][bin_index]
        node_index = int(rng.choice(node_weights.size, p=node_weights))
        bp_edges = model["bp_rp_edges"]
        bp_rp = float(rng.uniform(bp_edges[node_index], bp_edges[node_index + 1]))
        bp_nodes = model["bp_rp_nodes"]
        temperatures = model["temperature_nodes_k"]
        temperature = float(np.interp(bp_rp, bp_nodes, temperatures))
        locus = np.asarray([
            np.interp(bp_rp, bp_nodes, model["locus_colors"][:, index])
            for index in range(3)
        ])
        covariance = model["intrinsic_color_covariance"]
        residual = np.zeros(3, dtype=np.float64)
        for _ in range(12):
            residual = rng.multivariate_normal(np.zeros(3), covariance)
            mahalanobis = float(residual @ np.linalg.solve(covariance, residual))
            if mahalanobis <= 16.0:
                break
        colors = locus + residual
        magnitudes = {
            "VIS": float(mag_vis),
            "Y_E": float(mag_vis - colors[0]),
            "J_E": float(mag_vis - colors[0] - colors[1]),
            "H_E": float(mag_vis - colors[0] - colors[1] - colors[2]),
        }
        return StellarSED(
            temperature_k=temperature,
            extinction_av=0.0,
            magnitudes=magnitudes,
        )



def _normalise_rows(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    totals = values.sum(axis=1, keepdims=True)
    return np.divide(
        values, totals, out=np.full_like(values, 1.0 / values.shape[1]),
        where=totals > 0,
    )


def _positive_semidefinite_covariance(
    values: np.ndarray, *, floor: float,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return (eigenvectors * np.maximum(eigenvalues, float(floor))) @ eigenvectors.T



def sample_stellar_sed(
    rng: np.random.Generator, mag_vis: float,
    prior: EmpiricalStellarPrior | None = None,
) -> StellarSED:
    """Draw one star from the required empirical calibration."""
    if prior is None:
        raise ValueError(
            "an active empirical stellar prior is required for star generation"
        )
    return prior.sample(rng, mag_vis)
