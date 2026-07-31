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
        return cls(colors, temperatures, coefficients, covariance)

    def sample(self, rng: np.random.Generator, mag_vis: float) -> StellarSED:
        quantile = float(rng.random())
        grid = np.linspace(0.0, 1.0, self.bp_rp_quantiles.size)
        bp_rp = float(np.interp(quantile, grid, self.bp_rp_quantiles))
        temperature_grid = np.linspace(0.0, 1.0, self.temperature_quantiles_k.size)
        temperature = float(np.interp(
            1.0 - quantile, temperature_grid, self.temperature_quantiles_k,
        ))
        vis_coeff = self.band_coefficients[0]
        denominator = 1.0 + vis_coeff[2]
        if abs(denominator) < 0.05:
            denominator = 0.05 if denominator >= 0 else -0.05
        g_mag = (
            float(mag_vis) - vis_coeff[0] - vis_coeff[1] * bp_rp
            + 20.0 * vis_coeff[2]
        ) / denominator
        features = np.asarray([1.0, bp_rp, g_mag - 20.0])
        predicted = g_mag + self.band_coefficients @ features
        residual = rng.multivariate_normal(
            np.zeros(4), self.residual_covariance, check_valid="ignore",
        )
        predicted += residual - residual[0]
        predicted += float(mag_vis) - predicted[0]
        return StellarSED(
            temperature_k=temperature,
            extinction_av=0.0,
            magnitudes={
                name: float(predicted[index])
                for index, name in enumerate(_BAND_NAMES)
            },
        )


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
    """Draw temperature, extinction, and correlated Euclid magnitudes."""
    if prior is not None:
        return prior.sample(rng, mag_vis)
    temperature = _draw_temperature(rng)
    extinction_av = float(np.clip(
        rng.exponential(Config.STAR_EXTINCTION_AV_SCALE_MAG),
        0.0,
        Config.STAR_EXTINCTION_AV_MAX_MAG,
    ))
    offsets = _temperature_offsets(temperature)

    # Lightweight optical/NIR extinction law. Only differential extinction
    # matters because the caller fixes the observed VIS magnitude.
    wavelength = np.asarray([
        Config.Color.PIVOT_WAVELENGTH_UM[name] for name in _BAND_NAMES
    ])
    extinction = extinction_av * np.power(wavelength / 0.55, -1.7)
    offsets += extinction - extinction[0]
    offsets += np.asarray([
        Config.STAR_SED_BAND_CORRECTION_MAG[name] for name in _BAND_NAMES
    ])
    offsets[1:] += rng.normal(
        0.0, Config.STAR_SED_BAND_SCATTER_MAG, len(_BAND_NAMES) - 1,
    )
    offsets = np.clip(
        offsets,
        Config.STAR_COLOR_OFFSET_MIN_MAG,
        Config.STAR_COLOR_OFFSET_MAX_MAG,
    )
    offsets[0] = 0.0
    magnitudes = {
        name: float(mag_vis + offsets[k])
        for k, name in enumerate(_BAND_NAMES)
    }
    return StellarSED(
        temperature_k=temperature,
        extinction_av=extinction_av,
        magnitudes=magnitudes,
    )
