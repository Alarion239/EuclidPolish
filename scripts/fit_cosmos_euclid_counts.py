#!/usr/bin/env python3
"""Fit one smooth analytical galaxy population to COSMOS and Euclid.

The latent distribution is an evolving Schechter intensity in redshift and
F814W absolute-like magnitude, multiplied by a lognormal physical-size
distribution.  COSMOS constrains the latent redshift and size relations;
Euclid constrains the projection through a photometric response, a
resolution-limited size response, and surface-brightness-dependent
completeness.  No TNG catalogue or image is read by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "euclid_mpl_cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from euclid_polish.config import Config
from euclid_polish.population.joint_galaxy import (
    COSMOS_AREA_ARCMIN2,
    COSMOS_FIT_MAG_MAX,
    COSMOS_FIT_MAG_MIN,
    COSMOS_FIT_Z_MAX,
    COSMOS_FIT_Z_MIN,
    EUCLID_LOG_RE_EDGES,
    EUCLID_MAG_EDGES,
    LF_MAG_EDGES,
    LF_Z_EDGES,
    fit_euclid_response,
    fit_payload,
    fit_physical_conditionals,
    fit_phz_redshift_correction,
    fit_schechter_evolution,
    fit_size_evolution,
    latent_population_cube,
    predict_euclid_histogram,
    read_cosmos_population,
    read_euclid_population,
    read_phz_population,
    signed_poisson_residual,
    tng_draw_population_cube,
    validate_physical_conditionals,
)
from euclid_polish.web.helpers.q1_galaxy_counts import (
    read_q1_galaxy_aperture_counts,
    read_q1_galaxy_aperture_fit,
)

DEFAULT_COSMOS = Config.COSMOS_POPULATION_PRIOR_PATH
DEFAULT_EUCLID = "data/population_comparison/euclid_population.csv"
DEFAULT_EUCLID_META = "data/population_comparison/euclid_population_meta.json"
DEFAULT_EUCLID_PHZ_PDF = "data/population_comparison/euclid_population_phz_pdf.npz"
DEFAULT_OUTPUT_DIR = "data/population_comparison/cosmos2025"
OUTPUT_JSON = "joint_population_fit.json"
OUTPUT_OVERVIEW = "joint_population_fit.png"
OUTPUT_PLANES = "joint_population_joint_planes.png"
OUTPUT_PARAMETERS = "joint_population_parameters.png"
OUTPUT_CROSS_VALIDATION = "joint_population_cross_validation.png"
OUTPUT_CORE_MARGINALS = "joint_population_core_marginals.png"

EUCLID_UNRESOLVED_RADIUS_ARCSEC = 0.10
EUCLID_CENSORED_LOG_RE_EDGES = np.unique(np.append(
    EUCLID_LOG_RE_EDGES, math.log10(EUCLID_UNRESOLVED_RADIUS_ARCSEC),
))


def _read_area(path: Path) -> tuple[float, dict[str, Any]]:
    payload = json.loads(path.read_text())
    area = float(payload["area_arcmin2"])
    if area <= 0.0:
        raise ValueError(f"No positive area_arcmin2 in {path}")
    return area, payload


def _verify_phz_sidecar(path: Path, meta: dict[str, Any]) -> None:
    expected = str(meta.get("phz_pdf_sha256") or "")
    if len(expected) != 64:
        raise ValueError("Euclid cache metadata has no PHZ PDF checksum")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(
            "PHZ PDF checksum mismatch; the existing fit and active calibration "
            "were preserved"
        )


def _fingerprint(paths: list[Path], identity: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(identity, sort_keys=True).encode("utf-8"))
    for path in paths:
        digest.update(str(path).encode("utf-8"))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _histogram_density(
    values: np.ndarray,
    edges: np.ndarray,
    *,
    area: float | None = None,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges, weights=weights)
    divisor = np.diff(edges)
    if area is not None:
        divisor = divisor * area
    return counts.astype(np.float64) / divisor


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return float("nan")
    order = np.argsort(values[valid])
    value = values[valid][order]
    weight = weights[valid][order]
    cdf = np.cumsum(weight) - 0.5 * weight
    cdf /= np.sum(weight)
    return float(np.interp(q, cdf, value))


def _median_radius_by_magnitude(
    magnitude: np.ndarray,
    radius: np.ndarray,
    weights: np.ndarray,
    edges: np.ndarray,
) -> list[float | None]:
    result: list[float | None] = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        selected = (magnitude >= lower) & (magnitude < upper)
        value = _weighted_quantile(radius[selected], weights[selected], 0.5)
        result.append(float(value) if np.isfinite(value) else None)
    return result


def _parameter_rows(lf_fit, size_fit, response_fit) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(group: str, names: list[tuple[str, str, str]], values, errors):
        for index, (key, label, unit) in enumerate(names):
            rows.append({
                "group": group,
                "key": key,
                "label": label,
                "value": float(values[index]),
                "standard_error": (
                    float(errors[index]) if np.isfinite(errors[index]) else None
                ),
                "unit": unit,
            })

    add("intrinsic luminosity function", [
        ("log_phi_star", "ln phi* at z=0", "ln(Mpc^-3 mag^-1)"),
        ("m_star_0", "effective M* at z=0", "mag"),
        ("alpha", "faint-end slope alpha", ""),
        ("m_star_log1pz_slope", "M* evolution Q", "mag / log10(1+z)"),
        ("log_phi_log1pz_slope", "density evolution P", "power of (1+z)"),
        ("alpha_log1pz_slope", "alpha evolution", "per log10(1+z)"),
        ("m_star_log1pz_quadratic", "M* evolution Q2", "mag / log10(1+z)^2"),
        ("log_phi_log1pz_quadratic", "density evolution P2", "power / log10(1+z)"),
    ], [
        lf_fit.log_phi_star, lf_fit.m_star_0, lf_fit.alpha,
        lf_fit.m_star_log1pz_slope, lf_fit.log_phi_log1pz_slope,
        lf_fit.alpha_log1pz_slope, lf_fit.m_star_log1pz_quadratic,
        lf_fit.log_phi_log1pz_quadratic,
    ], lf_fit.standard_errors)
    rows.append({
        "group": "count covariance",
        "key": "cosmic_variance_fractional_scatter",
        "label": "extra-Poisson fractional scatter tau",
        "value": float(lf_fit.cosmic_variance_fractional_scatter),
        "standard_error": None,
        "unit": "fraction per COSMOS z-m cell",
    })
    add("intrinsic size relation", [
        ("log10_r0_kpc", "log10 R0", "kpc"),
        ("size_magnitude_slope", "size-luminosity slope", "dex / mag"),
        ("size_log1pz_slope", "size evolution", "dex / log10(1+z)"),
        ("size_magnitude_curvature", "size-luminosity curvature", "dex / mag^2"),
        (
            "size_magnitude_redshift_interaction",
            "size luminosity-redshift interaction",
            "dex / mag / log10(1+z)",
        ),
        ("size_scatter_dex", "intrinsic log-size scatter", "dex"),
        ("size_scatter_magnitude_slope", "log-size scatter magnitude slope", "per mag"),
    ], [
        size_fit.log10_r0_kpc, size_fit.magnitude_slope,
        size_fit.log1pz_slope, size_fit.magnitude_curvature,
        size_fit.magnitude_redshift_interaction, size_fit.scatter_dex,
        size_fit.scatter_magnitude_slope,
    ], size_fit.standard_errors)
    add("Euclid observation response", [
        ("population_scale", "Euclid field normalization", "ratio"),
        ("vis_minus_f814w_mag", "VIS-F814W offset", "mag"),
        ("magnitude_slope", "VIS magnitude slope", ""),
        ("scatter_mag", "intrinsic F814W-to-VIS scatter", "mag"),
        ("size_scale", "MER size-proxy scale", "ratio"),
        ("size_floor_arcsec", "MER size-proxy floor", "arcsec"),
        ("completeness_m50", "point-source m50", "AB mag"),
        ("completeness_width_mag", "completeness width", "mag"),
        ("surface_brightness_penalty", "surface-brightness penalty", "logit / mag arcsec^-2"),
    ], [
        response_fit.population_scale, response_fit.vis_minus_f814w_mag,
        response_fit.magnitude_slope, response_fit.scatter_mag,
        response_fit.size_scale, response_fit.size_floor_arcsec,
        response_fit.completeness_m50, response_fit.completeness_width_mag,
        response_fit.surface_brightness_penalty,
    ], response_fit.standard_errors)
    rows.append({
        "group": "Euclid observation response",
        "key": "measurement_flux_error_uJy",
        "label": "catalogue VIS aperture-flux error",
        "value": float(response_fit.measurement_flux_error_uJy),
        "standard_error": None,
        "unit": "microJy; empirical weighted median",
    })
    return rows


def _diagnostics(
    cosmos: dict[str, np.ndarray],
    euclid: dict[str, Any],
    euclid_area: float,
    lf_observed: np.ndarray,
    lf_predicted: np.ndarray,
    cube: dict[str, np.ndarray],
    euclid_observed: np.ndarray,
    euclid_predicted_density: np.ndarray,
    response_fit,
    *,
    euclid_log_radius_edges: np.ndarray,
    unresolved_radius_arcsec: float,
) -> dict[str, Any]:
    latent = np.asarray(cube["density"])
    latent_m = np.asarray(cube["magnitude"])
    latent_z = np.asarray(cube["z"])
    latent_log_r = np.asarray(cube["log_radius"])
    latent_mr = np.sum(latent, axis=0)
    tng_draw = tng_draw_population_cube(cube, response_fit)
    tng_draw_density = np.asarray(tng_draw["density"])
    tng_magnitude = np.asarray(tng_draw["vis_magnitude"])
    comparison_magnitude = (tng_magnitude >= 20.0) & (tng_magnitude < 28.0)

    def tng_marginals(density: np.ndarray) -> dict[str, Any]:
        return {
            "redshift": {
                "x": np.asarray(tng_draw["z"]).tolist(),
                "density": (
                    np.sum(density, axis=(1, 2))
                    / np.diff(np.asarray(tng_draw["z_edges"]))
                ).tolist(),
                "unit": "objects / arcmin2 / dz",
            },
            "magnitude": {
                "x": tng_magnitude.tolist(),
                "density": (
                    np.sum(density, axis=(0, 2))
                    / np.diff(np.asarray(tng_draw["vis_magnitude_edges"]))
                ).tolist(),
                "unit": "objects / arcmin2 / mag",
                "label": "true pre-noise Euclid VIS AB magnitude",
            },
            "angular_radius": {
                "x": np.asarray(tng_draw["log_radius"]).tolist(),
                "density": (
                    np.sum(density, axis=(0, 1))
                    / np.diff(np.asarray(tng_draw["log_radius_edges"]))
                ).tolist(),
                "unit": "objects / arcmin2 / dex",
                "label": "true circularized Re / arcsec",
            },
            "surface_density_arcmin2": float(np.sum(density)),
        }

    tng_full = tng_marginals(tng_draw_density)
    tng_comparison_density = np.zeros_like(tng_draw_density)
    tng_comparison_density[:, comparison_magnitude, :] = (
        tng_draw_density[:, comparison_magnitude, :]
    )
    tng_comparison = tng_marginals(tng_comparison_density)
    tng_comparison["magnitude"]["density"] = [
        value if selected else None
        for value, selected in zip(
            tng_comparison["magnitude"]["density"],
            comparison_magnitude,
            strict=True,
        )
    ]

    cosmos_mag_observed = np.sum(lf_observed, axis=0) / (
        COSMOS_AREA_ARCMIN2 * np.diff(LF_MAG_EDGES)
    )
    cosmos_mag_model = np.sum(lf_predicted, axis=0) / (
        COSMOS_AREA_ARCMIN2 * np.diff(LF_MAG_EDGES)
    )
    euclid_mag_observed = np.sum(euclid_observed, axis=1) / (
        euclid_area * np.diff(EUCLID_MAG_EDGES)
    )
    euclid_mag_model = np.sum(euclid_predicted_density, axis=1) / np.diff(
        EUCLID_MAG_EDGES
    )

    cosmos_z_observed = np.sum(lf_observed, axis=1) / (
        COSMOS_AREA_ARCMIN2 * np.diff(LF_Z_EDGES)
    )
    cosmos_z_model = np.sum(lf_predicted, axis=1) / (
        COSMOS_AREA_ARCMIN2 * np.diff(LF_Z_EDGES)
    )

    log_r_edges = np.arange(-2.4, 0.81, 0.12)
    has_radius = np.asarray(cosmos["has_radius"], dtype=bool)
    cosmos_radius = np.asarray(cosmos["radius_arcsec"])[has_radius]
    cosmos_radius_density = _histogram_density(
        np.log10(cosmos_radius), log_r_edges, area=COSMOS_AREA_ARCMIN2,
    )
    cosmos_model_magnitude = (
        (latent_m >= COSMOS_FIT_MAG_MIN)
        & (latent_m < COSMOS_FIT_MAG_MAX)
    )
    latent_radius_density = np.sum(
        latent[:, cosmos_model_magnitude, :], axis=(0, 1),
    ) / np.diff(np.asarray(cube["log_radius_edges"]))
    euclid_radius_observed = np.sum(euclid_observed, axis=0) / (
        euclid_area * np.diff(euclid_log_radius_edges)
    )
    euclid_radius_model = np.sum(euclid_predicted_density, axis=0) / np.diff(
        euclid_log_radius_edges
    )
    threshold_log_radius = math.log10(unresolved_radius_arcsec)
    unresolved_radius_bins = (
        euclid_log_radius_edges[1:] <= threshold_log_radius + 1e-10
    )
    resolved_radius_bins = ~unresolved_radius_bins
    unresolved_log_width = threshold_log_radius - euclid_log_radius_edges[0]

    median_edges = np.arange(20.0, 27.5001, 0.5)
    cosmos_magnitude = np.asarray(cosmos["magnitude"])[has_radius]
    cosmos_median = _median_radius_by_magnitude(
        cosmos_magnitude, cosmos_radius, np.ones(len(cosmos_radius)), median_edges,
    )
    resolved_euclid_rows = (
        np.asarray(euclid["radius_arcsec"]) >= unresolved_radius_arcsec
    )
    euclid_median = _median_radius_by_magnitude(
        np.asarray(euclid["magnitude"])[resolved_euclid_rows],
        np.asarray(euclid["radius_arcsec"])[resolved_euclid_rows],
        np.asarray(euclid["weight"])[resolved_euclid_rows], median_edges,
    )
    latent_m_grid, latent_r_grid = np.meshgrid(
        latent_m, np.power(10.0, latent_log_r), indexing="ij",
    )
    latent_model_median = _median_radius_by_magnitude(
        latent_m_grid.ravel(), latent_r_grid.ravel(), latent_mr.ravel(),
        median_edges,
    )
    euclid_m_centers = 0.5 * (EUCLID_MAG_EDGES[:-1] + EUCLID_MAG_EDGES[1:])
    euclid_r_centers = np.power(
        10.0,
        0.5 * (euclid_log_radius_edges[:-1] + euclid_log_radius_edges[1:]),
    )
    em, er = np.meshgrid(euclid_m_centers, euclid_r_centers, indexing="ij")
    euclid_model_median = _median_radius_by_magnitude(
        em.ravel(), er.ravel(), euclid_predicted_density.ravel(), median_edges,
    )

    mu_edges = np.arange(16.0, 32.0001, 0.4)
    cosmos_mu = (
        cosmos_magnitude + 2.5 * np.log10(2.0 * np.pi * cosmos_radius**2)
    )
    cosmos_mu_shape = _histogram_density(cosmos_mu, mu_edges)
    cosmos_mu_shape /= max(np.sum(cosmos_mu_shape * np.diff(mu_edges)), 1e-12)
    latent_mu = (
        latent_m_grid + 2.5 * np.log10(2.0 * np.pi * latent_r_grid**2)
    )
    latent_mu_shape = _histogram_density(
        latent_mu.ravel(), mu_edges, weights=latent_mr.ravel(),
    )
    latent_mu_shape /= max(np.sum(latent_mu_shape * np.diff(mu_edges)), 1e-12)
    euclid_mu = (
        np.asarray(euclid["magnitude"])[resolved_euclid_rows]
        + 2.5 * np.log10(
            2.0 * np.pi
            * np.asarray(euclid["radius_arcsec"])[resolved_euclid_rows] ** 2
        )
    )
    euclid_mu_observed = _histogram_density(
        euclid_mu, mu_edges, area=euclid_area,
        weights=np.asarray(euclid["weight"])[resolved_euclid_rows],
    )
    euclid_model_mu = (
        em + 2.5 * np.log10(2.0 * np.pi * er**2)
    )
    euclid_mu_model = _histogram_density(
        euclid_model_mu.ravel(), mu_edges,
        weights=euclid_predicted_density.ravel(),
    )

    completeness_magnitude = np.linspace(20.0, 28.0, 161)
    completeness: dict[str, list[float]] = {}
    for radius in (0.10, 0.20, 0.50):
        mu = completeness_magnitude + 2.5 * math.log10(
            2.0 * math.pi * radius**2
        )
        logit = (
            (response_fit.completeness_m50 - completeness_magnitude)
            / response_fit.completeness_width_mag
            - response_fit.surface_brightness_penalty * (mu - 24.0)
        )
        completeness[f"{radius:.2f}"] = (
            1.0 / (1.0 + np.exp(-np.clip(logit, -60.0, 60.0)))
        ).tolist()

    def series(x, observed, model, unit, label):
        return {
            "x": np.asarray(x).tolist(),
            "observed": np.asarray(observed).tolist(),
            "model": np.asarray(model).tolist(),
            "unit": unit,
            "label": label,
        }

    euclid_radius_series = series(
        0.5 * (euclid_log_radius_edges[:-1] + euclid_log_radius_edges[1:]),
        euclid_radius_observed, euclid_radius_model,
        "objects / arcmin2 / dex", "log10 MER size proxy / arcsec",
    )
    for key in ("observed", "model"):
        euclid_radius_series[key] = [
            value if resolved else None
            for value, resolved in zip(
                euclid_radius_series[key], resolved_radius_bins, strict=True,
            )
        ]
    euclid_radius_series["censored"] = {
        "upper_radius_arcsec": unresolved_radius_arcsec,
        "observed_density": (
            float(np.sum(euclid_observed[:, unresolved_radius_bins]))
            / euclid_area / unresolved_log_width
        ),
        "model_density": (
            float(np.sum(euclid_predicted_density[:, unresolved_radius_bins]))
            / unresolved_log_width
        ),
        "interpretation": (
            "aggregate probability mass with MER radius below the resolution "
            "threshold; exact proxy radii do not enter the likelihood"
        ),
    }

    return {
        "magnitude_counts": {
            "cosmos": series(
                0.5 * (LF_MAG_EDGES[:-1] + LF_MAG_EDGES[1:]),
                cosmos_mag_observed, cosmos_mag_model,
                "objects / arcmin2 / mag", "HST F814W AB magnitude",
            ),
            "euclid": series(
                euclid_m_centers, euclid_mag_observed, euclid_mag_model,
                "objects / arcmin2 / mag", "Euclid VIS AB magnitude",
            ),
        },
        "redshift": series(
            0.5 * (LF_Z_EDGES[:-1] + LF_Z_EDGES[1:]),
            cosmos_z_observed, cosmos_z_model,
            "objects / arcmin2 / dz", "photometric redshift",
        ),
        "tng_draw": {
            "full": tng_full,
            "comparison_window": {
                **tng_comparison,
                "vis_magnitude_min": 20.0,
                "vis_magnitude_max": 28.0,
            },
            "definition": (
                "latent Schechter x lognormal population, scaled to the "
                "Euclid field normalization and transformed from F814W to "
                "true VIS using intrinsic scatter only; no MER broadening, "
                "measurement error, completeness, or radius censoring"
            ),
        },
        "angular_radius": {
            "cosmos": series(
                0.5 * (log_r_edges[:-1] + log_r_edges[1:]),
                cosmos_radius_density,
                np.interp(
                    0.5 * (log_r_edges[:-1] + log_r_edges[1:]),
                    latent_log_r, latent_radius_density,
                ),
                "objects / arcmin2 / dex", "log10 circularized Re / arcsec",
            ),
            "euclid": euclid_radius_series,
        },
        "median_radius_by_magnitude": {
            "x": (0.5 * (median_edges[:-1] + median_edges[1:])).tolist(),
            "cosmos_observed": cosmos_median,
            "cosmos_model": latent_model_median,
            "euclid_observed": euclid_median,
            "euclid_model": euclid_model_median,
            "unit": "arcsec",
        },
        "surface_brightness": {
            "x": (0.5 * (mu_edges[:-1] + mu_edges[1:])).tolist(),
            "cosmos_observed": cosmos_mu_shape.tolist(),
            "cosmos_model": latent_mu_shape.tolist(),
            "euclid_observed": euclid_mu_observed.tolist(),
            "euclid_model": euclid_mu_model.tolist(),
            "unit": "mag / arcsec2",
        },
        "completeness": {
            "magnitude": completeness_magnitude.tolist(),
            "by_radius_arcsec": completeness,
        },
        "joint_planes": {
            "cosmos_observed": (
                lf_observed / COSMOS_AREA_ARCMIN2
            ).tolist(),
            "cosmos_model": (
                lf_predicted / COSMOS_AREA_ARCMIN2
            ).tolist(),
            "cosmos_z_edges": LF_Z_EDGES.tolist(),
            "cosmos_magnitude_edges": LF_MAG_EDGES.tolist(),
            "euclid_observed": (euclid_observed / euclid_area).tolist(),
            "euclid_model": euclid_predicted_density.tolist(),
            "euclid_magnitude_edges": EUCLID_MAG_EDGES.tolist(),
            "euclid_log_radius_edges": euclid_log_radius_edges.tolist(),
            "euclid_unresolved_radius_arcsec": unresolved_radius_arcsec,
        },
        "predicted_euclid_redshift": {
            "x": latent_z.tolist(),
            "note": (
                "Model projection only: the cached Euclid catalogue has no "
                "PHZ redshift column."
            ),
        },
    }


def _q1_brightness_plot() -> dict[str, Any]:
    counts = read_q1_galaxy_aperture_counts()["apertures"]["f2"]
    curve = read_q1_galaxy_aperture_fit()["apertures"]["f2"]
    return {
        "observed_x": np.asarray([
            0.5 * (float(item["mag_lo"]) + float(item["mag_hi"]))
            for item in counts["bins"]
        ]),
        "observed_density": np.asarray([
            float(item["density_arcmin2_mag"]) for item in counts["bins"]
        ]),
        "law_x": np.asarray(curve["x"], dtype=np.float64),
        "law_density": np.asarray(curve["density"], dtype=np.float64),
        "fit_interval": (
            float(curve["law"]["fit_bright"]),
            float(curve["law"]["fit_faint"]),
        ),
        "extrapolated_interval": tuple(
            float(value) for value in curve["extrapolated_faint_interval"]
        ),
    }


def _plot_q1_brightness(ax, brightness: dict[str, Any]) -> None:
    ax.scatter(
        brightness["observed_x"], brightness["observed_density"], s=20,
        facecolors="none", edgecolors="#1267d6", linewidths=1.2,
        label="Q1 MER + PHZ 2FWHM counts",
    )
    ax.plot(
        brightness["law_x"], brightness["law_density"], color="#7a3db8",
        linewidth=2.7, label="Q1-normalized straight law",
    )
    ax.axvspan(*brightness["fit_interval"], color="#1267d6", alpha=0.07)
    ax.axvspan(
        *brightness["extrapolated_interval"], color="#7a3db8", alpha=0.08,
    )
    ax.set(
        yscale="log", xlim=(14, 29),
        xlabel="VIS 2FWHM aperture magnitude [AB]",
        ylabel="objects / arcmin² / mag",
        title="Independent goal brightness",
    )
    ax.legend(frameon=False, fontsize=8)


def _plot_overview(
    path: Path, diagnostics: dict[str, Any], brightness: dict[str, Any],
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 13.2), constrained_layout=True)
    ax = axes[0, 0]
    _plot_q1_brightness(ax, brightness)

    redshift = diagnostics["redshift"]
    ax = axes[0, 1]
    ax.plot(redshift["x"], redshift["observed"], "o", ms=3.5,
            color="#242424", label="COSMOS observed")
    ax.plot(redshift["x"], redshift["model"], color="#008c68", lw=2.2,
            label="shared model")
    ax.set(yscale="log", xlabel="photometric redshift", ylabel=redshift["unit"],
           title="Redshift distribution (COSMOS constraint)")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    for survey, color in (("cosmos", "#008c68"), ("euclid", "#cf3d2e")):
        item = diagnostics["angular_radius"][survey]
        observed = np.asarray([
            np.nan if value is None else value for value in item["observed"]
        ])
        model = np.asarray([
            np.nan if value is None else value for value in item["model"]
        ])
        ax.plot(item["x"], observed, "o", ms=3.5, color=color,
                alpha=0.7, label=f"{survey.title()} observed")
        ax.plot(item["x"], model, color=color, lw=2.2,
                label=f"{survey.title()} model")
        if survey == "euclid" and "censored" in item:
            censored = item["censored"]
            x = math.log10(censored["upper_radius_arcsec"])
            ax.scatter(
                [x], [censored["observed_density"]], marker="<", s=65,
                color="#7a3db8", label="Euclid radius-censored mass",
            )
            ax.scatter(
                [x], [censored["model_density"]], marker="x", s=48,
                color=color, label="model censored mass",
            )
    ax.set(yscale="log", xlabel="log10 angular radius / arcsec",
           ylabel="density (survey-specific units)", title="Angular-size distributions")
    ax.legend(frameon=False, fontsize=8)

    median = diagnostics["median_radius_by_magnitude"]
    ax = axes[1, 1]
    for key, color, style, label in (
        ("cosmos_observed", "#242424", "o", "COSMOS observed"),
        ("cosmos_model", "#008c68", "-", "COSMOS model"),
        ("euclid_observed", "#1267d6", "o", "Euclid observed proxy"),
        ("euclid_model", "#cf3d2e", "-", "Euclid response model"),
    ):
        y = np.asarray([np.nan if value is None else value for value in median[key]])
        if style == "o":
            ax.plot(median["x"], y, style, ms=4, color=color, label=label)
        else:
            ax.plot(median["x"], y, style, lw=2.2, color=color, label=label)
    ax.set(yscale="log", xlabel="survey AB magnitude", ylabel="median radius / arcsec",
           title="Magnitude-conditioned angular size")
    ax.legend(frameon=False, fontsize=8)

    surface = diagnostics["surface_brightness"]
    ax = axes[2, 0]
    for key, color, style, label in (
        ("cosmos_observed", "#242424", "o", "COSMOS observed shape"),
        ("cosmos_model", "#008c68", "-", "COSMOS latent model"),
        ("euclid_observed", "#1267d6", "o", "Euclid observed"),
        ("euclid_model", "#cf3d2e", "-", "Euclid response model"),
    ):
        ax.plot(surface["x"], surface[key], style, color=color,
                ms=3.2 if style == "o" else None,
                lw=2.1 if style == "-" else None, label=label)
    ax.set(yscale="log", xlabel="mean surface brightness / mag arcsec⁻²",
           ylabel="density (survey-specific units)", title="Derived surface brightness")
    ax.legend(frameon=False, fontsize=8)

    complete = diagnostics["completeness"]
    ax = axes[2, 1]
    for index, (radius, values) in enumerate(complete["by_radius_arcsec"].items()):
        ax.plot(complete["magnitude"], values, lw=2.2,
                color=("#1267d6", "#008c68", "#cf3d2e")[index],
                label=f"observed radius {radius} arcsec")
    ax.axhline(0.5, color="#777777", ls="--", lw=1)
    ax.set(xlabel="Euclid VIS magnitude", ylabel="detection probability",
           ylim=(-0.03, 1.03), title="Fitted Euclid completeness surface")
    ax.legend(frameon=False, fontsize=8)
    for axis in axes.ravel():
        axis.grid(alpha=0.18)
    fig.suptitle("Independent Q1 2FWHM brightness; staged Schechter × lognormal geometry",
                 fontsize=15)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_core_marginals(
    path: Path, diagnostics: dict[str, Any], brightness: dict[str, Any],
) -> None:
    """Plot the independent brightness law and staged geometry marginals."""
    redshift = diagnostics["redshift"]
    radius = diagnostics["angular_radius"]
    tng_draw = diagnostics["tng_draw"]
    tng_full = tng_draw["full"]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), constrained_layout=True)

    axes[0].scatter(
        redshift["x"], redshift["observed"], s=22, facecolors="none",
        edgecolors="#008c68", linewidths=1.4, label="COSMOS observed",
    )
    axes[0].plot(
        redshift["x"], redshift["model"], color="#008c68",
        linewidth=2.2, label="shared intrinsic fit",
    )
    axes[0].plot(
        tng_full["redshift"]["x"], tng_full["redshift"]["density"],
        color="#7a3db8", linewidth=3.0,
        label="brightness-marginalized staged geometry",
    )
    axes[0].text(
        0.98, 0.96, "No redshift column in cached Euclid MER",
        transform=axes[0].transAxes, ha="right", va="top", fontsize=9,
        color="0.35",
    )
    axes[0].set(
        xlabel="photometric redshift",
        ylabel="objects / arcmin² / dz",
        title="Redshift density",
    )
    axes[0].set_yscale("log")

    _plot_q1_brightness(axes[1], brightness)

    for survey, color in (("cosmos", "#008c68"), ("euclid", "#1267d6")):
        item = radius[survey]
        label = "COSMOS measured Re" if survey == "cosmos" else "Euclid MER proxy"
        angular_radius = np.power(10.0, np.asarray(item["x"]))
        observed = np.asarray([
            np.nan if value is None else value for value in item["observed"]
        ])
        model = np.asarray([
            np.nan if value is None else value for value in item["model"]
        ])
        axes[2].scatter(
            angular_radius, observed, s=22, facecolors="none",
            edgecolors=color, linewidths=1.4, label=f"{label} observed",
        )
        axes[2].plot(
            angular_radius, model, color=color, linewidth=2.2,
            label=f"fit through {survey.upper()} response",
        )
        if survey == "euclid" and "censored" in item:
            censored = item["censored"]
            x = censored["upper_radius_arcsec"]
            axes[2].scatter(
                [x], [censored["observed_density"]], marker="<", s=75,
                color="#7a3db8", label=f"Euclid censored below {x:.2f}″",
            )
            axes[2].scatter(
                [x], [censored["model_density"]], marker="x", s=55,
                color="#cf3d2e", label="model censored mass",
            )
            axes[2].axvline(x, color="0.45", linestyle="--", linewidth=1.1)
    axes[2].set(
        xlabel="angular radius / arcsec",
        ylabel="objects / arcmin² / dex",
        title="Angular-radius density",
    )
    axes[2].set_yscale("log")
    axes[2].set_xscale("log")
    axes[2].plot(
        np.power(10.0, np.asarray(tng_full["angular_radius"]["x"])),
        tng_full["angular_radius"]["density"],
        color="#7a3db8", linewidth=3.0,
        label="brightness-marginalized staged geometry",
    )

    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8.5)
    fig.suptitle(
        "Independent Q1 brightness and staged geometry before Euclid observation",
        fontsize=15,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_planes(path: Path, diagnostics: dict[str, Any]) -> None:
    planes = diagnostics["joint_planes"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    entries = (
        ("cosmos_observed", "COSMOS observed", "cosmos_magnitude_edges", "cosmos_z_edges",
         "HST F814W magnitude", "redshift"),
        ("cosmos_model", "Shared model in COSMOS", "cosmos_magnitude_edges", "cosmos_z_edges",
         "HST F814W magnitude", "redshift"),
        ("euclid_observed", "Euclid observed", "euclid_log_radius_edges", "euclid_magnitude_edges",
         "log10 size proxy / arcsec", "VIS magnitude"),
        (
            "euclid_model", "Shared model through Euclid response",
            "euclid_log_radius_edges", "euclid_magnitude_edges",
         "log10 size proxy / arcsec", "VIS magnitude"),
    )
    positive = [
        value for key, *_rest in entries
        for value in np.asarray(planes[key]).ravel()
        if np.isfinite(value) and value > 0.0
    ]
    norm = LogNorm(vmin=max(np.percentile(positive, 2), 1e-5),
                   vmax=np.percentile(positive, 99.7))
    for ax, (key, title, xkey, ykey, xlabel, ylabel) in zip(
        axes.ravel(), entries, strict=True,
    ):
        values = np.asarray(planes[key])
        xedges = np.asarray(planes[xkey])
        yedges = np.asarray(planes[ykey])
        image = ax.pcolormesh(xedges, yedges, values, shading="auto", norm=norm,
                              cmap="magma")
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    fig.colorbar(image, ax=axes.ravel().tolist(), label="objects / arcmin² / cell")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_parameters(path: Path, rows: list[dict[str, Any]], quality: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 11.0), constrained_layout=True)
    ax.axis("off")
    lines = [
        "FITTED JOINT POPULATION PARAMETERS",
        "",
    ]
    group = None
    for row in rows:
        if row["group"] != group:
            group = row["group"]
            lines.extend((group.upper(), "─" * len(group)))
        error = row["standard_error"]
        uncertainty = f" ± {error:.4g}" if error is not None else ""
        unit = f"  {row['unit']}" if row["unit"] else ""
        lines.append(f"{row['label']:<34} {row['value']:>11.5g}{uncertainty}{unit}")
        if row["key"] in {"log_phi_star", "m_star_0", "alpha", "size_scatter_dex"}:
            lines.append("")
    lines.extend((
        "",
        "FIT QUALITY",
        "───────────",
        f"COSMOS reduced Poisson deviance: {quality['cosmos_reduced_poisson_deviance']:.3f}",
        "COSMOS reduced overdispersed deviance: "
        f"{quality['cosmos_reduced_negative_binomial_deviance']:.3f}",
        "Euclid resolved-plus-censored reduced Cash deviance: "
        f"{quality['euclid_reduced_poisson_deviance']:.3f}",
    ))
    for item in quality["warnings"]:
        lines.extend(textwrap.wrap(
            f"warning: {item}", width=105,
            subsequent_indent="         ",
        ))
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left",
            family="monospace", fontsize=10.2, linespacing=1.28)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _cross_validate_euclid_response(
    cube: dict[str, np.ndarray], euclid: dict[str, Any], area_arcmin2: float,
    *,
    unresolved_radius_arcsec: float,
    log_radius_edges: np.ndarray,
    folds: int = 4,
) -> dict[str, Any]:
    cone_index = np.asarray(euclid["cone_index"], dtype=np.int64)
    cones = np.unique(cone_index)
    if cones.size < folds:
        raise ValueError("Euclid cross-validation needs at least four cones")
    results: list[dict[str, Any]] = []
    array_keys = (
        "magnitude", "radius_arcsec", "magnitude_error", "flux_error_uJy", "weight",
        "cone_index",
    )
    for fold in range(folds):
        test_cones = cones[fold::folds]
        is_test = np.isin(cone_index, test_cones)
        train = {
            key: np.asarray(euclid[key])[~is_test]
            for key in array_keys
        }
        test_area = area_arcmin2 * len(test_cones) / len(cones)
        train_area = area_arcmin2 - test_area
        fitted, _observed_train, _predicted_train = fit_euclid_response(
            cube, train, area_arcmin2=train_area,
            unresolved_policy="censor",
            unresolved_radius_arcsec=unresolved_radius_arcsec,
            log_radius_edges=log_radius_edges,
        )
        observed, _, _ = np.histogram2d(
            np.asarray(euclid["magnitude"])[is_test],
            np.log10(np.asarray(euclid["radius_arcsec"])[is_test]),
            bins=(EUCLID_MAG_EDGES, log_radius_edges),
            weights=np.asarray(euclid["weight"])[is_test],
        )
        predicted_density, _ = predict_euclid_histogram(
            np.sum(np.asarray(cube["density"]), axis=0),
            np.asarray(cube["magnitude"]),
            np.asarray(cube["log_radius"]),
            np.asarray([
                math.log(fitted.population_scale),
                fitted.vis_minus_f814w_mag,
                math.log(fitted.magnitude_slope),
                math.log(fitted.scatter_mag),
                math.log(fitted.size_scale),
                math.log(fitted.size_floor_arcsec),
                fitted.completeness_m50,
                math.log(fitted.completeness_width_mag),
                math.log(fitted.surface_brightness_penalty),
            ]),
            log_radius_edges=log_radius_edges,
            measurement_flux_error_uJy=fitted.measurement_flux_error_uJy,
        )
        predicted = predicted_density * test_area
        threshold = math.log10(unresolved_radius_arcsec)
        unresolved_bins = log_radius_edges[1:] <= threshold + 1e-10
        resolved_bins = ~unresolved_bins
        resolved_residual = signed_poisson_residual(
            observed[:, resolved_bins], predicted[:, resolved_bins],
        )
        censored_residual = signed_poisson_residual(
            np.sum(observed[:, unresolved_bins], axis=1),
            np.sum(predicted[:, unresolved_bins], axis=1),
        )
        deviance = float(
            np.sum(resolved_residual**2) + np.sum(censored_residual**2)
        )
        comparison_cells = resolved_residual.size + censored_residual.size
        results.append({
            "fold": fold + 1,
            "test_cones": [int(value) for value in test_cones],
            "test_area_arcmin2": test_area,
            "reduced_poisson_deviance": deviance / comparison_cells,
            "observed_weighted_density_arcmin2": (
                float(np.sum(observed)) / test_area
            ),
            "predicted_density_arcmin2": (
                float(np.sum(predicted_density))
            ),
        })
    return {
        "folds": results,
        "mean_reduced_poisson_deviance": float(np.mean([
            item["reduced_poisson_deviance"] for item in results
        ])),
        "interpretation": (
            "Four cone-group folds; the Euclid response is refitted on the "
            "other cones and evaluated on unseen cones with exact radius bins "
            f"at or above {unresolved_radius_arcsec:.2f} arcsec and one "
            "magnitude-conditioned left-censored bin below it. The cached "
            "cones share one query radius, so total area is apportioned equally."
        ),
    }


def _plot_cross_validation(path: Path, payload: dict[str, Any]) -> None:
    folds = payload["folds"]
    x = np.arange(1, len(folds) + 1)
    deviance = np.asarray([
        item["reduced_poisson_deviance"] for item in folds
    ])
    density_ratio = np.asarray([
        item["predicted_density_arcmin2"]
        / item["observed_weighted_density_arcmin2"]
        for item in folds
    ])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    axes[0].bar(x, deviance, color="#1267d6")
    axes[0].axhline(1.0, color="0.4", linestyle="--", linewidth=1.2)
    axes[0].set(xlabel="held-out cone group", ylabel="reduced Poisson deviance",
                title="Unseen Euclid magnitude-size plane")
    axes[1].bar(x, density_ratio, color="#008c68")
    axes[1].axhline(1.0, color="0.4", linestyle="--", linewidth=1.2)
    axes[1].set(xlabel="held-out cone group", ylabel="predicted / observed density",
                title="Held-out surface-density normalization")
    for axis in axes:
        axis.set_xticks(x)
        axis.grid(alpha=0.2, axis="y")
    fig.suptitle("Four-fold Euclid cone validation", fontsize=15)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    cosmos_path = Path(args.cosmos)
    euclid_path = Path(args.euclid)
    euclid_meta_path = Path(args.euclid_meta)
    euclid_phz_pdf_path = Path(args.euclid_phz_pdf)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    euclid_area, euclid_meta = _read_area(euclid_meta_path)
    _verify_phz_sidecar(euclid_phz_pdf_path, euclid_meta)

    cosmos = read_cosmos_population(cosmos_path)
    euclid = read_euclid_population(
        euclid_path,
        maximum_spurious_probability=args.maximum_spurious_probability,
    )
    phz = read_phz_population(euclid_path, euclid_phz_pdf_path)
    lf_fit, lf_observed, lf_predicted = fit_schechter_evolution(
        np.asarray(cosmos["magnitude"]), np.asarray(cosmos["redshift"]),
    )
    size_fit = fit_size_evolution(
        np.asarray(cosmos["magnitude"]), np.asarray(cosmos["redshift"]),
        np.asarray(cosmos["radius_arcsec"]),
    )
    cube = latent_population_cube(lf_fit, size_fit)
    response_fit, euclid_observed, euclid_predicted_density = fit_euclid_response(
        cube, euclid, area_arcmin2=euclid_area,
        unresolved_policy="censor",
        unresolved_radius_arcsec=EUCLID_UNRESOLVED_RADIUS_ARCSEC,
        log_radius_edges=EUCLID_CENSORED_LOG_RE_EDGES,
    )
    generation_draw = tng_draw_population_cube(cube, response_fit)
    phz_redshift = fit_phz_redshift_correction(generation_draw, phz)
    physical_conditionals = fit_physical_conditionals(
        cosmos, phz, response_fit,
    )
    physical_monte_carlo = validate_physical_conditionals(
        generation_draw, phz_redshift, physical_conditionals,
    )
    cross_validation = _cross_validate_euclid_response(
        cube, euclid, euclid_area,
        unresolved_radius_arcsec=EUCLID_UNRESOLVED_RADIUS_ARCSEC,
        log_radius_edges=EUCLID_CENSORED_LOG_RE_EDGES,
    )
    diagnostics = _diagnostics(
        cosmos, euclid, euclid_area, lf_observed, lf_predicted, cube,
        euclid_observed, euclid_predicted_density, response_fit,
        euclid_log_radius_edges=EUCLID_CENSORED_LOG_RE_EDGES,
        unresolved_radius_arcsec=EUCLID_UNRESOLVED_RADIUS_ARCSEC,
    )

    cosmos_poisson_reduced = lf_fit.poisson_deviance / max(1, lf_fit.dof)
    cosmos_overdispersed_reduced = (
        lf_fit.negative_binomial_deviance / max(1, lf_fit.dof)
    )
    euclid_reduced = response_fit.poisson_deviance / max(1, response_fit.dof)
    bright_reduced = (
        response_fit.bright_poisson_deviance
        / max(1, response_fit.bright_dof)
    )
    warnings: list[str] = []
    if cosmos_overdispersed_reduced > 5.0:
        warnings.append(
            "a single smooth Schechter component does not capture all COSMOS "
            "redshift-magnitude structure"
        )
    if lf_fit.cosmic_variance_fractional_scatter > 0.20:
        warnings.append(
            "extra-Poisson COSMOS scatter is too large to interpret as pure "
            "cosmic variance; it also absorbs selection and model mismatch"
        )
    if euclid_reduced > 5.0:
        warnings.append(
            "the Euclid magnitude-size plane retains structure beyond the "
            "single lognormal size and selection response"
        )
    if bright_reduced > 5.0:
        warnings.append(
            "the frozen bright F814W-to-VIS transfer matches integrated counts "
            "but not every high-S/N magnitude bin"
        )
    if cross_validation["mean_reduced_poisson_deviance"] > 5.0:
        warnings.append(
            "held-out Euclid cones confirm that the remaining magnitude-size "
            "structure is predictive mismatch, not only training overfit"
        )
    warnings.append(
        "M_eff is F814W-DM(z); mean K-correction is absorbed by M* evolution"
    )
    warnings.append(
        "Euclid SEMIMAJOR_AXIS is a detection-size proxy, not a half-light radius"
    )
    warnings.append(
        "Euclid MER radii below 0.10 arcsec are left-censored; their exact "
        "proxy values do not enter the response likelihood"
    )
    quality = {
        "valid": bool(
            cosmos_overdispersed_reduced <= 5.0 and euclid_reduced <= 5.0
        ),
        "cosmos_poisson_deviance": lf_fit.poisson_deviance,
        "cosmos_dof": lf_fit.dof,
        "cosmos_reduced_poisson_deviance": cosmos_poisson_reduced,
        "cosmos_negative_binomial_deviance": (
            lf_fit.negative_binomial_deviance
        ),
        "cosmos_reduced_negative_binomial_deviance": (
            cosmos_overdispersed_reduced
        ),
        "euclid_poisson_deviance": response_fit.poisson_deviance,
        "euclid_dof": response_fit.dof,
        "euclid_reduced_poisson_deviance": euclid_reduced,
        "euclid_bright_transfer_poisson_deviance": (
            response_fit.bright_poisson_deviance
        ),
        "euclid_bright_transfer_dof": response_fit.bright_dof,
        "euclid_bright_transfer_reduced_poisson_deviance": bright_reduced,
        "euclid_cross_validated_reduced_poisson_deviance": (
            cross_validation["mean_reduced_poisson_deviance"]
        ),
        "warnings": warnings,
    }
    phz_coverage = dict(euclid_meta.get("phz_coverage") or {})
    phz_cache_quality = dict(euclid_meta.get("phz_quality") or {})
    phz_gates = {
        "classification_coverage": bool(
            float(phz_coverage.get("classification_fraction") or 0.0) >= 0.90
        ),
        "redshift_pdf_coverage": bool(
            float(phz_coverage.get("valid_pdf_fraction") or 0.0) >= 0.80
        ),
        "archive_pdf_provenance": bool(
            euclid_meta.get("phz_pdf_activation_eligible", True)
        ),
        "pdf_normalization": bool(
            phz_cache_quality.get("all_retained_pdfs_normalized")
        ),
        "cross_validated_improvement": bool(
            float(
                (phz_redshift.get("cross_validation") or {}).get(
                    "mean_improvement_fraction"
                ) or 0.0
            ) >= 0.10
        ),
        "redshift_cell_residual": bool(
            float(phz_redshift["median_absolute_fractional_residual"]) <= 0.20
        ),
        "physical_effective_weight": bool(
            physical_conditionals.get("all_cells_valid")
        ),
        "physical_posterior_inputs": bool(
            float(phz_coverage.get("valid_physical_fraction") or 0.0) > 0.0
        ),
        "monte_carlo_reproduction": bool(physical_monte_carlo.get("valid")),
        "density_preservation": bool(
            float(phz_redshift["density_change_fraction"]) <= 0.01
        ),
        "existing_joint_fit": bool(quality["valid"]),
    }
    phz_valid = all(phz_gates.values())
    quality["phz_valid"] = phz_valid
    quality["phz_quality_gates"] = phz_gates
    quality["valid"] = bool(quality["valid"] and phz_valid)
    if not phz_valid:
        failed = ", ".join(key for key, passed in phz_gates.items() if not passed)
        warnings.append(f"PHZ activation gates failed: {failed}")
    parameters = _parameter_rows(lf_fit, size_fit, response_fit)
    overview_path = output_dir / OUTPUT_OVERVIEW
    planes_path = output_dir / OUTPUT_PLANES
    parameters_path = output_dir / OUTPUT_PARAMETERS
    cross_validation_path = output_dir / OUTPUT_CROSS_VALIDATION
    core_marginals_path = output_dir / OUTPUT_CORE_MARGINALS
    if not args.no_plot:
        brightness_plot = _q1_brightness_plot()
        _plot_overview(overview_path, diagnostics, brightness_plot)
        _plot_planes(planes_path, diagnostics)
        _plot_parameters(parameters_path, parameters, quality)
        _plot_cross_validation(cross_validation_path, cross_validation)
        _plot_core_marginals(
            core_marginals_path, diagnostics, brightness_plot,
        )

    algorithm = {
        "name": "flexibly evolving Schechter x lognormal size joint survey fit",
        "version": 6,
        "cosmology": "Planck15",
        "cosmos_fit_window": {
            "magnitude": [COSMOS_FIT_MAG_MIN, COSMOS_FIT_MAG_MAX],
            "redshift": [COSMOS_FIT_Z_MIN, COSMOS_FIT_Z_MAX],
        },
        "euclid_response": (
            "affine F814W-to-true-VIS transfer with fitted intrinsic Gaussian "
            "scatter fitted and frozen using VIS<24 counts; Gaussian VIS "
            "measurement noise in flux space using the robust catalogue aperture-"
            "flux error; quadrature size "
            "scale plus resolution floor; logistic completeness in VIS "
            "magnitude and derived mean surface brightness; exact MER radii "
            "below 0.10 arcsec are replaced by a left-censoring event"
        ),
        "tng_draw_target": (
            "latent population times Euclid field normalization, transformed "
            "to true VIS with intrinsic scatter only; no measurement error, "
            "MER size response, completeness, or censoring"
        ),
        "tng_draw_window": [18.0, 30.0],
        "count_covariance": (
            "negative-binomial fractional scatter fitted after the Poisson "
            "mean; Var(N)=mu+(tau*mu)^2"
        ),
        "validation": (
            "four folds across twelve cached Euclid cones using the same "
            "left-censored radius likelihood, plus a four-fold PHZ "
            "redshift-magnitude correction and fixed-seed physical sampler check"
        ),
    }
    fingerprint = _fingerprint(
        [cosmos_path, euclid_path, euclid_meta_path, euclid_phz_pdf_path], algorithm,
    )
    payload: dict[str, Any] = {
        "version": 3,
        "kind": "joint_intrinsic_galaxy_population",
        "fingerprint": fingerprint,
        "method": algorithm,
        "interpretation": (
            "One latent smooth galaxy distribution is observed through two "
            "survey responses. COSMOS constrains redshift, luminosity and "
            "physical-size evolution; Euclid constrains the projected VIS "
            "magnitude-size distribution and its selection. No TNG data enter."
        ),
        "inputs": {
            "cosmos_population_npz": str(cosmos_path),
            "cosmos_area_arcmin2": COSMOS_AREA_ARCMIN2,
            "cosmos_population_rows": int(len(cosmos["magnitude"])),
            "cosmos_measured_size_rows": int(np.sum(cosmos["has_radius"])),
            "euclid_catalog_csv": str(euclid_path),
            "euclid_meta_json": str(euclid_meta_path),
            "euclid_phz_pdf_npz": str(euclid_phz_pdf_path),
            "euclid_area_arcmin2": euclid_area,
            "euclid_cone_count": int(euclid_meta.get("cone_count", 1)),
            "euclid_catalog_rows": int(euclid["catalog_rows"]),
            "euclid_expected_galaxies_with_sizes": float(
                np.sum(np.asarray(euclid["weight"]))
            ),
            "euclid_unresolved_radius_policy": "left_censor",
            "euclid_unresolved_radius_arcsec": (
                EUCLID_UNRESOLVED_RADIUS_ARCSEC
            ),
            "euclid_unresolved_weighted_galaxies": float(np.sum(
                np.asarray(euclid["weight"])[
                    np.asarray(euclid["radius_arcsec"])
                    < EUCLID_UNRESOLVED_RADIUS_ARCSEC
                ]
            )),
            "classification_weighting": euclid["classification_weighting"],
            "size_estimator": euclid["size_estimator"],
            "missing_probability_rows": int(euclid["missing_probability_rows"]),
            "invalid_probability_rows": int(euclid["invalid_probability_rows"]),
            "missing_size_rows": int(euclid["missing_size_rows"]),
            "missing_magnitude_error_rows": int(
                euclid["missing_magnitude_error_rows"]
            ),
        },
        "model": {
            "luminosity_function": fit_payload(lf_fit),
            "size_relation": fit_payload(size_fit),
            "euclid_response": fit_payload(response_fit),
        },
        "phz_redshift_correction": phz_redshift,
        "physical_conditionals": physical_conditionals,
        "phz_inputs": {
            "coverage": phz_coverage,
            "pdf_rows": int(np.sum(np.asarray(phz["has_pdf"], dtype=bool))),
            "pdf_source": euclid_meta.get(
                "phz_pdf_source", "archive_full_pdf",
            ),
            "pdf_activation_eligible": bool(
                euclid_meta.get("phz_pdf_activation_eligible", True)
            ),
            "qso_probability_treatment": "diagnostic_only_not_renormalized",
            "full_pdf_compaction": (
                "601-point PDF rebinned to LF_Z_EDGES"
                if euclid_meta.get("phz_pdf_source") != "summary_reconstruction"
                else "diagnostic Gaussian mixture reconstructed from cached PHZ modes"
            ),
        },
        "phz_quality_gates": phz_gates,
        "phz_monte_carlo": physical_monte_carlo,
        "parameters": parameters,
        "fit_quality": quality,
        "diagnostics": diagnostics,
        "cross_validation": cross_validation,
        "outputs": {
            "overview_png": str(overview_path),
            "joint_planes_png": str(planes_path),
            "parameters_png": str(parameters_path),
            "cross_validation_png": str(cross_validation_path),
            "core_marginals_png": str(core_marginals_path),
        },
        "generation_status": (
            "pre-observation TNG draw target is defined and plotted; this "
            "artifact does not yet select or render TNG galaxies"
        ),
    }
    output_path = output_dir / OUTPUT_JSON
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, output_path)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosmos", default=DEFAULT_COSMOS)
    parser.add_argument("--euclid", default=DEFAULT_EUCLID)
    parser.add_argument("--euclid-meta", default=DEFAULT_EUCLID_META)
    parser.add_argument("--euclid-phz-pdf", default=DEFAULT_EUCLID_PHZ_PDF)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--maximum-spurious-probability", type=float, default=0.5)
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main() -> None:
    payload = run(build_parser().parse_args())
    quality = payload["fit_quality"]
    print(f"Wrote {Path(payload['outputs']['overview_png']).parent / OUTPUT_JSON}")
    print(
        "COSMOS reduced deviance "
        f"{quality['cosmos_reduced_poisson_deviance']:.3f} Poisson / "
        f"{quality['cosmos_reduced_negative_binomial_deviance']:.3f} "
        "overdispersed; "
        "Euclid reduced deviance "
        f"{quality['euclid_reduced_poisson_deviance']:.3f}; "
        "held-out Euclid "
        f"{quality['euclid_cross_validated_reduced_poisson_deviance']:.3f}"
    )
    for warning in quality["warnings"]:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
