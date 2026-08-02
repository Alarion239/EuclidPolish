"""Temperature-driven stellar colours for synthetic point sources.

The simulator knows a star's VIS magnitude but needs a diverse, correlated
four-band SED.  We draw a temperature from a cool-dwarf-dominated mixture,
integrate a Planck ``f_nu`` spectrum over lightweight top-hat approximations
to the Euclid VIS/Y/J/H passbands, and add modest extinction plus per-band
scatter for line blanketing/metallicity not represented by a blackbody.

This is intentionally an approximation, not a detailed stellar-photosphere
library. It removes the far more damaging fixed-colour shortcut while
remaining cheap enough for fresh on-the-fly star draws during every training
visit. No terrestrial atmospheric term is present: Euclid is space based.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from euclid_polish.config import Config

_SECOND_RADIATION_CONSTANT_UM_K = 14387.77
_BAND_NAMES = Config.LR_INPUT_BAND_NAMES
_TEMPERATURE_MIN_K = min(c[3] for c in Config.STAR_TEMPERATURE_COMPONENTS)
_TEMPERATURE_MAX_K = max(c[4] for c in Config.STAR_TEMPERATURE_COMPONENTS)


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
    magnitude_edges: np.ndarray | None = None
    magnitude_cdf: np.ndarray | None = None
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
        edges = np.asarray(distribution.get("edges") or [], dtype=np.float64)
        cdf = np.asarray(distribution.get("cdf") or [], dtype=np.float64)
        if edges.size < 2 or cdf.size != edges.size:
            raise ValueError("stellar prior requires an empirical magnitude CDF")
        else:
            if (not np.all(np.isfinite(edges)) or not np.all(np.isfinite(cdf))
                    or np.any(np.diff(edges) <= 0)
                    or cdf[0] < -1e-9 or cdf[-1] <= 0.0
                    or np.any(np.diff(cdf) < -1e-9)):
                raise ValueError("stellar prior has an invalid magnitude CDF")
            else:
                cdf = np.clip(cdf / cdf[-1], 0.0, 1.0)
                cdf[0] = 0.0
                cdf[-1] = 1.0
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
        return cls(colors, temperatures, coefficients, covariance, edges, cdf,
                   color_model)

    def sample_magnitude(
        self, rng: np.random.Generator, *, slope: float,
        m_bright: float, m_faint: float,
    ) -> float:
        """Sample VIS magnitude from the fitted empirical CDF."""
        if self.magnitude_edges is None or self.magnitude_cdf is None:
            raise ValueError("stellar prior has no empirical magnitude CDF")
        u = float(rng.random())
        index = int(np.searchsorted(self.magnitude_cdf, u, side="right") - 1)
        index = max(0, min(index, self.magnitude_edges.size - 2))
        lo = float(self.magnitude_cdf[index])
        hi = float(self.magnitude_cdf[index + 1])
        fraction = 0.0 if hi <= lo else (u - lo) / (hi - lo)
        return float(
            self.magnitude_edges[index]
            + fraction * (self.magnitude_edges[index + 1]
                          - self.magnitude_edges[index])
        )

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


def _sample_exponential_magnitude(
    rng: np.random.Generator, *, slope: float,
    m_bright: float, m_faint: float,
) -> float:
    """Sample ``dN/dm ∝ 10^(slope·m)`` for legacy priors."""
    span = float(m_faint) - float(m_bright)
    if span <= 0.0:
        return float(m_bright)
    beta = float(slope) * np.log(10.0)
    u = float(rng.random())
    t = (u * span if abs(beta) < 1e-9
         else np.log1p(u * np.expm1(beta * span)) / beta)
    return float(m_bright + t)


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


def _planck_fnu(wavelength_um: np.ndarray, temperature_k: float) -> np.ndarray:
    """Planck ``f_nu`` at wavelength in microns, arbitrary normalisation."""
    wavelength = np.asarray(wavelength_um, dtype=np.float64)
    x = _SECOND_RADIATION_CONSTANT_UM_K / (wavelength * float(temperature_k))
    return np.power(1.0 / wavelength, 3) / np.expm1(x)


def _band_mean_fnu(temperature_k: float, band_name: str) -> float:
    """Photon-counting AB-like mean ``f_nu`` over one top-hat passband."""
    lo, hi = Config.STAR_BANDPASS_UM[band_name]
    wavelength = np.geomspace(float(lo), float(hi), 64)
    fnu = _planck_fnu(wavelength, temperature_k)
    # A flat f_nu AB source stays flat under the d(lambda)/lambda weighting
    # appropriate to a photon-counting response.
    return float(np.trapezoid(fnu, x=np.log(wavelength)) / np.log(hi / lo))


def blackbody_band_offsets_mag(temperature_k: float) -> dict[str, float]:
    """Return integrated ``m_band - m_VIS`` for a blackbody temperature."""
    temperature = float(temperature_k)
    if not _TEMPERATURE_MIN_K <= temperature <= _TEMPERATURE_MAX_K:
        raise ValueError(
            f"temperature_k must be in [{_TEMPERATURE_MIN_K:g}, "
            f"{_TEMPERATURE_MAX_K:g}]"
        )
    flux = np.asarray([
        _band_mean_fnu(temperature, name) for name in _BAND_NAMES
    ])
    offsets = -2.5 * np.log10(flux / flux[0])
    return {name: float(offsets[k]) for k, name in enumerate(_BAND_NAMES)}


# Integrate once, then interpolate per star. This keeps on-the-fly sampling at
# a few scalar RNG/interpolation operations rather than four numerical
# integrations per visit.
_TEMPERATURE_GRID_K = np.geomspace(
    _TEMPERATURE_MIN_K, _TEMPERATURE_MAX_K, 512,
)
_OFFSET_GRID_MAG = np.asarray([
    [blackbody_band_offsets_mag(t)[name] for name in _BAND_NAMES]
    for t in _TEMPERATURE_GRID_K
])


def _draw_temperature(rng: np.random.Generator) -> float:
    components = Config.STAR_TEMPERATURE_COMPONENTS
    weights = np.asarray([component[0] for component in components], dtype=float)
    weights /= weights.sum()
    index = int(rng.choice(len(components), p=weights))
    _weight, median_k, sigma_log, lower_k, upper_k = components[index]
    value = float(np.exp(rng.normal(np.log(median_k), sigma_log)))
    return float(np.clip(value, lower_k, upper_k))


def _temperature_offsets(temperature_k: float) -> np.ndarray:
    log_grid = np.log(_TEMPERATURE_GRID_K)
    log_t = np.log(float(temperature_k))
    return np.asarray([
        np.interp(log_t, log_grid, _OFFSET_GRID_MAG[:, k])
        for k in range(len(_BAND_NAMES))
    ])


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
