"""Versioned calibration artifacts for galaxy density and stellar priors."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

from euclid_polish.config import Config
from euclid_polish.population.euclid_galaxy_prior import (
    APERTURE_FWHM_MODEL_VERSION,
    BRIGHT_BRIDGE_JOIN_MAGNITUDES,
    JOINT_EUCLID_GALAXY_KIND,
    JOINT_EUCLID_GALAXY_VERSION,
    RADIUS_MODEL_VERSION,
    ConditionalApertureFWHMDistribution,
    ConditionalRadiusLaw,
    fit_conditional_aperture_fwhm_distribution,
    fit_continuous_generation_magnitude_law,
    fit_linear_conditional_radius_law_from_binned_counts,
    joint_density_grid,
)
from euclid_polish.population.magnitude_law import (
    ContinuousBrightBridgeFaintCappedMagnitudeLaw,
    StraightMagnitudeLaw,
)
from euclid_polish.sky.generation.cosmos_tng_prior import (
    brightness_transfer_payload,
)


def calibration_dir() -> Path:
    return Path(Config.DATA_DIR) / "population_comparison" / "calibrations"


def active_transfer_path() -> Path:
    return calibration_dir() / "photometric_transfer_active.json"


def density_calibration_path() -> Path:
    return calibration_dir() / "tng_density_calibration.json"


def active_density_path() -> Path:
    return calibration_dir() / "tng_density_active.json"


def star_candidate_path() -> Path:
    return calibration_dir() / "star_population_candidate.json"


def active_star_path() -> Path:
    return calibration_dir() / "star_population_active.json"


def active_joint_galaxy_path() -> Path:
    return calibration_dir() / "joint_galaxy_population_active.json"


def joint_galaxy_candidate_path() -> Path:
    return calibration_dir() / "euclid_galaxy_population_candidate.json"


def _read(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def joint_galaxy_candidate() -> dict[str, Any] | None:
    """Return the persisted Euclid-only brightness-radius candidate."""
    source = _read(joint_galaxy_candidate_path())
    if not source:
        return None
    try:
        fitted_magnitude_law = StraightMagnitudeLaw.from_payload(
            source["fitted_magnitude_law"]
        )
        magnitude_law = (
            ContinuousBrightBridgeFaintCappedMagnitudeLaw.from_payload(
                source["magnitude_law"]
            )
        )
        radius_law = ConditionalRadiusLaw.from_payload(source["radius_law"])
        aperture_fwhm = ConditionalApertureFWHMDistribution.from_payload(
            source["aperture_fwhm_distribution"]
        )
        density = float(source["generation"]["surface_density_arcmin2"])
        density_cap = float(
            source["generation"]["differential_density_cap_arcmin2_mag"]
        )
        break_magnitude = float(source["generation"]["break_magnitude"])
        magnitude_plot = source["magnitude_plot"]
        radius_plot = source["plots"]["radius"]
        relation_plot = source["plots"]["conditional_radius"]
        magnitude_x = np.asarray(magnitude_plot["law"]["x"], dtype=np.float64)
        magnitude_density = np.asarray(
            magnitude_plot["law"]["density"], dtype=np.float64,
        )
        generation_x = np.asarray(
            magnitude_plot["generation_law"]["x"], dtype=np.float64,
        )
        generation_density = np.asarray(
            magnitude_plot["generation_law"]["density"], dtype=np.float64,
        )
        radius_x = np.asarray(radius_plot["x"], dtype=np.float64)
        radius_density = np.asarray(radius_plot["density"], dtype=np.float64)
        q1_weighted_radius_density = np.asarray(
            radius_plot["q1_weighted_density"], dtype=np.float64,
        )
        relation_x = np.asarray(relation_plot["magnitude"], dtype=np.float64)
        relation_mean = np.asarray(
            relation_plot["model_mean_log10_arcsec"], dtype=np.float64,
        )
        fwhm_plot = source["plots"]["conditional_aperture_fwhm"]
        fwhm_magnitude = np.asarray(
            fwhm_plot["magnitude"], dtype=np.float64,
        )
        fwhm_mean = np.asarray(
            fwhm_plot["model_mean_arcsec"], dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError):
        return None
    if (
        source.get("version") != JOINT_EUCLID_GALAXY_VERSION
        or source.get("kind") != JOINT_EUCLID_GALAXY_KIND
        or radius_law.version != RADIUS_MODEL_VERSION
        or aperture_fwhm.version != APERTURE_FWHM_MODEL_VERSION
        or len(str(source.get("fingerprint") or "")) != 64
        or not source.get("valid")
        or not np.isclose(density, magnitude_law.integrated_density())
        or not np.isclose(
            density_cap,
            magnitude_law.density_cap_arcmin2_mag,
        )
        or not np.isclose(break_magnitude, magnitude_law.break_magnitude)
        or magnitude_law.straight_law != fitted_magnitude_law
        or not np.allclose(
            magnitude_law.bright_join_magnitudes,
            BRIGHT_BRIDGE_JOIN_MAGNITUDES,
            rtol=0.0,
            atol=1e-12,
        )
        or magnitude_x.size < 2
        or magnitude_x.shape != magnitude_density.shape
        or generation_x.size < 3
        or generation_x.shape != generation_density.shape
        or radius_x.size < 2
        or radius_x.shape != radius_density.shape
        or radius_x.shape != q1_weighted_radius_density.shape
        or relation_x.size < 2
        or relation_x.shape != relation_mean.shape
        or fwhm_magnitude.size < 2
        or fwhm_magnitude.shape != fwhm_mean.shape
        or not np.all(np.isfinite(magnitude_x))
        or not np.all(np.isfinite(magnitude_density) & (magnitude_density > 0.0))
        or not np.all(np.isfinite(generation_x))
        or not np.all(
            np.isfinite(generation_density) & (generation_density >= 0.0)
        )
        or not np.any(generation_density > 0.0)
        or not np.allclose(generation_density, magnitude_law.density(generation_x))
        or not np.all(np.isfinite(radius_x))
        or not np.all(np.isfinite(radius_density) & (radius_density >= 0.0))
        or not np.all(
            np.isfinite(q1_weighted_radius_density)
            & (q1_weighted_radius_density >= 0.0)
        )
        or not np.all(np.isfinite(relation_x))
        or not np.all(np.isfinite(relation_mean))
        or not np.all(np.isfinite(fwhm_magnitude))
        or not np.all(np.isfinite(fwhm_mean) & (fwhm_mean > 0.0))
    ):
        return None
    return source


def fit_euclid_joint_galaxy_candidate() -> dict[str, Any]:
    """Fit the minimal aggregate Euclid VIS-2FWHM x Sersic-R_e model."""
    from euclid_polish.web.helpers.q1_galaxy_counts import (
        q1_galaxy_counts_path,
        q1_galaxy_fit_path,
        read_q1_galaxy_aperture_counts,
        read_q1_galaxy_aperture_fit,
    )
    from euclid_polish.web.helpers.q1_galaxy_radius_statistics import (
        q1_galaxy_radius_statistics_path,
        read_q1_galaxy_radius_statistics,
    )

    fit_payload = read_q1_galaxy_aperture_fit()
    count_payload = read_q1_galaxy_aperture_counts()
    radius_payload = read_q1_galaxy_radius_statistics()
    try:
        magnitude_curve = fit_payload["apertures"]["f2"]
        fitted_magnitude_law = StraightMagnitudeLaw.from_payload(
            magnitude_curve["law"]
        )
        count_bins = count_payload["apertures"]["f2"]["bins"]
        magnitude_bins = radius_payload["magnitude_bins"]
        joint_bins = radius_payload["joint_bins"]
        magnitude_fwhm_bins = radius_payload["magnitude_fwhm_bins"]
        magnitude_edges = np.asarray(
            radius_payload["magnitude_edges"], dtype=np.float64,
        )
        radius_bins = radius_payload["radius_bins"]
        radius_edges = np.asarray(
            radius_payload["radius_edges_arcsec"], dtype=np.float64,
        )
        fwhm_bins = radius_payload["fwhm_bins"]
        fwhm_edges = np.asarray(
            radius_payload["fwhm_edges_arcsec"], dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Query the complete Q1 VIS 2FWHM and Sersic-radius brackets first"
        ) from exc
    if not count_payload.get("complete") or not radius_payload.get("complete"):
        raise ValueError(
            "Complete all Q1 VIS 2FWHM and Sersic-radius brackets before fitting"
        )

    try:
        observed_magnitude_x = np.asarray([
            0.5 * (float(item["mag_lo"]) + float(item["mag_hi"]))
            for item in count_bins
        ], dtype=np.float64)
        observed_magnitude_density = np.asarray([
            float(item["density_arcmin2_mag"])
            for item in count_bins
        ], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Q1 VIS 2FWHM count bins are malformed") from exc
    if (
        observed_magnitude_x.size == 0
        or observed_magnitude_x.shape != observed_magnitude_density.shape
        or not np.all(np.isfinite(observed_magnitude_x))
        or not np.all(
            np.isfinite(observed_magnitude_density)
            & (observed_magnitude_density >= 0.0)
        )
        or not np.any(observed_magnitude_density > 0.0)
    ):
        raise ValueError("Q1 VIS 2FWHM count bins are malformed")
    observed_peak_index = int(np.argmax(observed_magnitude_density))
    observed_peak_magnitude = float(observed_magnitude_x[observed_peak_index])
    observed_peak_density = float(
        observed_magnitude_density[observed_peak_index]
    )

    selected_grid = np.zeros(
        (magnitude_edges.size - 1, radius_edges.size - 1), dtype=np.float64,
    )
    weight_grid = np.zeros_like(selected_grid)
    for item in joint_bins:
        mag_index = int(item["magnitude_bin"])
        radius_index = int(item["radius_bin"])
        selected_grid[mag_index, radius_index] = float(item["selected_radii"])
        weight_grid[mag_index, radius_index] = float(item["expected_radii"])
    radius_law = fit_linear_conditional_radius_law_from_binned_counts(
        magnitude_edges,
        np.log10(radius_edges),
        selected_grid,
        weight_grid,
    )
    fwhm_weight_grid = np.zeros(
        (magnitude_edges.size - 1, fwhm_edges.size - 1), dtype=np.float64,
    )
    for item in magnitude_fwhm_bins:
        mag_index = int(item["magnitude_bin"])
        fwhm_index = int(item["fwhm_bin"])
        fwhm_weight_grid[mag_index, fwhm_index] = float(
            item["expected_fwhm"]
        )
    aperture_fwhm = fit_conditional_aperture_fwhm_distribution(
        magnitude_edges,
        fwhm_edges,
        fwhm_weight_grid,
        selection=(
            f"{radius_payload['selection']}; "
            f"{radius_payload['fwhm_selection']}"
        ),
    )
    log_radius_edges = np.log10(radius_edges)
    magnitude_law, bright_fit_diagnostics = (
        fit_continuous_generation_magnitude_law(
            fitted_magnitude_law,
            list(count_bins),
            footprint_area_arcmin2=float(
                count_payload["footprint_area_arcmin2"]
            ),
            density_cap_arcmin2_mag=observed_peak_density,
        )
    )
    generation_x = np.unique(np.concatenate((
        np.linspace(
            magnitude_law.mag_bright,
            magnitude_law.mag_faint,
            301,
            dtype=np.float64,
        ),
        np.asarray(
            [
                *magnitude_law.bright_join_magnitudes,
                magnitude_law.break_magnitude,
            ],
            dtype=np.float64,
        ),
    )))
    generation_density = magnitude_law.density(generation_x)
    grid = joint_density_grid(
        magnitude_law, radius_law, log_radius_edges=log_radius_edges,
    )
    diagnostic_magnitude_edges = np.unique(np.concatenate((
        np.linspace(
            magnitude_law.mag_bright,
            magnitude_law.mag_faint,
            6001,
            dtype=np.float64,
        ),
        np.asarray(
            [
                *magnitude_law.bright_join_magnitudes,
                magnitude_law.break_magnitude,
                float(radius_law.fit_faint_magnitude),
            ],
            dtype=np.float64,
        ),
    )))
    diagnostic_magnitude = 0.5 * (
        diagnostic_magnitude_edges[:-1]
        + diagnostic_magnitude_edges[1:]
    )
    diagnostic_magnitude_mass = (
        magnitude_law.density(diagnostic_magnitude)
        * np.diff(diagnostic_magnitude_edges)
    )

    def density_above_radius(radius_arcsec: float) -> float:
        threshold = float(np.log10(radius_arcsec))
        probability = radius_law.bin_probability(
            diagnostic_magnitude,
            np.asarray(
                [
                    radius_law.log_radius_min,
                    threshold,
                    radius_law.log_radius_max,
                ],
                dtype=np.float64,
            ),
        )[:, 1]
        return float(np.sum(diagnostic_magnitude_mass * probability))

    radius_density = (
        np.sum(grid["density"], axis=0)
        / np.diff(grid["log_radius_edges"])
    )
    observed_radius_density = np.asarray([
        float(item["density_arcmin2_dex"]) for item in radius_bins
    ], dtype=np.float64)
    q1_probability = radius_law.bin_probability(
        0.5 * (magnitude_edges[:-1] + magnitude_edges[1:]),
        log_radius_edges,
    )
    q1_expected_by_magnitude = np.sum(weight_grid, axis=1)
    q1_weighted_radius_density = (
        np.sum(q1_expected_by_magnitude[:, None] * q1_probability, axis=0)
        / float(radius_payload["footprint_area_arcmin2"])
        / np.diff(log_radius_edges)
    )
    relation_x = 0.5 * (magnitude_edges[:-1] + magnitude_edges[1:])
    relation_observed: list[float | None] = []
    log_radius_centers = 0.5 * (
        log_radius_edges[:-1] + log_radius_edges[1:]
    )
    for row in weight_grid:
        expected = float(np.sum(row))
        if expected <= 0.0:
            relation_observed.append(None)
            continue
        relation_observed.append(float(
            np.sum(row * log_radius_centers) / expected
        ))
    relation_model = radius_law.mean(relation_x)
    fwhm_centers = 0.5 * (fwhm_edges[:-1] + fwhm_edges[1:])
    fwhm_observed_mean: list[float | None] = []
    for row in fwhm_weight_grid:
        expected = float(np.sum(row))
        if expected <= 0.0:
            fwhm_observed_mean.append(None)
            continue
        fwhm_observed_mean.append(float(
            np.sum(row * fwhm_centers) / expected
        ))
    fwhm_model_mean = aperture_fwhm.mean(relation_x)
    fit_row_mask = np.sum(selected_grid, axis=1) >= (
        radius_law.fit_min_selected_per_magnitude_bin
    )
    fit_row_mask &= relation_x <= float(radius_law.fit_faint_magnitude)
    conditional_fit_interval = [
        float(relation_x[fit_row_mask][0]),
        float(relation_x[fit_row_mask][-1]),
    ]
    observed_radius_probability = (
        observed_radius_density * np.diff(log_radius_edges)
    )
    observed_radius_probability /= np.sum(observed_radius_probability)
    modeled_radius_probability = (
        q1_weighted_radius_density * np.diff(log_radius_edges)
    )
    modeled_radius_probability /= np.sum(modeled_radius_probability)
    fit_expected = weight_grid[fit_row_mask]
    fit_expected_by_magnitude = np.sum(fit_expected, axis=1)
    fit_effective_by_magnitude = np.minimum(
        fit_expected_by_magnitude,
        radius_law.fit_effective_weight_cap,
    )
    fit_effective_counts = fit_expected * (
        fit_effective_by_magnitude / fit_expected_by_magnitude
    )[:, None]
    conditional_cross_entropy = float(-np.sum(
        fit_effective_counts
        * np.log(np.maximum(q1_probability[fit_row_mask], 1e-300))
    ) / np.sum(fit_effective_counts))
    core = {
        "version": JOINT_EUCLID_GALAXY_VERSION,
        "kind": JOINT_EUCLID_GALAXY_KIND,
        "valid": True,
        "validated": True,
        "fitted_magnitude_law": fitted_magnitude_law.to_payload(),
        "magnitude_law": magnitude_law.to_payload(),
        "radius_law": radius_law.to_payload(),
        "aperture_fwhm_distribution": aperture_fwhm.to_payload(),
        "magnitude_plot": {
            "label": "Q1 MER + PHZ VIS 2FWHM",
            "law": {
                "x": list(magnitude_curve["x"]),
                "density": list(magnitude_curve["density"]),
            },
            "generation_law": {
                "x": generation_x.tolist(),
                "density": generation_density.tolist(),
            },
            "observed": {
                "x": observed_magnitude_x.tolist(),
                "density": observed_magnitude_density.tolist(),
            },
            "observed_support": {
                "turnover_magnitude": observed_peak_magnitude,
                "peak_differential_density_arcmin2_mag": (
                    observed_peak_density
                ),
                "density_cap_policy": (
                    "hold the generation law at the maximum observed Q1 "
                    "VIS 2FWHM differential density"
                ),
            },
            "fit_interval": [
                fitted_magnitude_law.fit_bright,
                fitted_magnitude_law.fit_faint,
            ],
            "fitted_law_interval": [
                fitted_magnitude_law.mag_bright,
                fitted_magnitude_law.mag_faint,
            ],
            "generation_interval": [
                magnitude_law.mag_bright, magnitude_law.mag_faint,
            ],
            "continuous_bright_interval": [
                magnitude_law.mag_bright,
                magnitude_law.bright_join_magnitudes[-1],
            ],
            "bright_join_magnitudes": list(
                magnitude_law.bright_join_magnitudes
            ),
            "bright_slopes": list(magnitude_law.bright_slopes),
            "bright_fit_diagnostics": bright_fit_diagnostics,
            "break_magnitude": magnitude_law.break_magnitude,
            "differential_density_cap_arcmin2_mag": (
                magnitude_law.density_cap_arcmin2_mag
            ),
            "extrapolated_interval": [
                float(count_payload["faint"]),
                fitted_magnitude_law.mag_faint,
            ],
        },
        "plots": {
            "radius": {
                "x": grid["log_radius"].tolist(),
                "density": radius_density.tolist(),
                "q1_weighted_density": q1_weighted_radius_density.tolist(),
                "observed_density": observed_radius_density.tolist(),
                "unit": "objects / arcmin2 / dex",
                "model_semantics": (
                    "nominal continuous-space circularized Euclid VIS "
                    "Sersic R_e = R_e,major sqrt(q); TNG output pixels are "
                    "not remeasured during generation"
                ),
            },
            "conditional_radius": {
                "magnitude": relation_x.tolist(),
                "observed_mean_log10_arcsec": relation_observed,
                "model_mean_log10_arcsec": relation_model.tolist(),
                "model_core_low_log10_arcsec": (
                    relation_model - radius_law.scatter_dex
                ).tolist(),
                "model_core_high_log10_arcsec": (
                    relation_model + radius_law.scatter_dex
                ).tolist(),
                "fit_interval": conditional_fit_interval,
                "model_kind": "straight_truncated_gaussian_no_tail",
            },
            "conditional_aperture_fwhm": {
                "magnitude": relation_x.tolist(),
                "observed_mean_arcsec": fwhm_observed_mean,
                "model_mean_arcsec": fwhm_model_mean.tolist(),
                "model_kind": "empirical_mer_fwhm_given_vis_2fwhm_magnitude",
                "out_of_support_policy": (
                    aperture_fwhm.out_of_support_policy
                ),
            },
            "fit_diagnostics": {
                "conditional_cross_entropy": conditional_cross_entropy,
                "q1_marginal_total_variation": float(
                    0.5 * np.sum(np.abs(
                        modeled_radius_probability
                        - observed_radius_probability
                    ))
                ),
                "generation_density_re_ge_1_arcsec": density_above_radius(1.0),
                "generation_density_re_ge_2_arcsec": density_above_radius(2.0),
                "generation_density_re_ge_5_arcsec": density_above_radius(5.0),
                "generation_density_re_ge_8_arcsec": density_above_radius(8.0),
                "generation_fraction_fainter_than_radius_fit": float(
                    np.sum(
                        diagnostic_magnitude_mass[
                            diagnostic_magnitude
                            > float(radius_law.fit_faint_magnitude)
                        ]
                    )
                    / np.sum(diagnostic_magnitude_mass)
                ),
                "q1_fraction_fainter_than_radius_fit": float(
                    np.sum(
                        q1_expected_by_magnitude[
                            relation_x > radius_law.fit_faint_magnitude
                        ]
                    )
                    / np.sum(q1_expected_by_magnitude)
                ),
                **bright_fit_diagnostics,
            },
        },
        "generation": {
            "surface_density_arcmin2": magnitude_law.integrated_density(),
            "differential_density_cap_arcmin2_mag": (
                magnitude_law.density_cap_arcmin2_mag
            ),
            "differential_density_cap_source": (
                "maximum_observed_q1_vis_2fwhm_differential_density"
            ),
            "density_cap_observed_magnitude": observed_peak_magnitude,
            "break_magnitude": magnitude_law.break_magnitude,
            "fitted_surface_density_arcmin2": (
                fitted_magnitude_law.integrated_density()
            ),
            "vis_magnitude_min": magnitude_law.mag_bright,
            "vis_magnitude_max": magnitude_law.mag_faint,
            "fitted_vis_magnitude_max": fitted_magnitude_law.mag_faint,
            "faint_end_policy": (
                "continuous_three_slope_bright_bridge_then_fitted_main_"
                "then_flat_at_observed_q1_peak"
            ),
            "faint_radius_policy": (
                "straight_truncated_gaussian_at_all_magnitudes_no_tail"
            ),
            "radius_semantics": "circularized_sersic_half_light_radius",
            "radius_min_arcsec": 10.0 ** radius_law.log_radius_min,
            "radius_max_arcsec": 10.0 ** radius_law.log_radius_max,
            "sampling_order": "radius_marginal_then_brightness_given_radius",
            "aperture_fwhm_sampling": (
                "MER_FWHM_given_sampled_VIS_2FWHM_magnitude"
            ),
            "aperture_fwhm_min_arcsec": aperture_fwhm.minimum_arcsec,
            "aperture_fwhm_max_arcsec": aperture_fwhm.maximum_arcsec,
            "morphology_assignment": "balanced_random_tng_atlas",
            "position_process": "homogeneous_poisson",
        },
        "provenance": {
            "brightness": (
                "Q1 MER + PHZ VIS 2FWHM continuous three-slope bright "
                "bridge with fixed joins, fitted main count line, and a "
                "faint tail held at the maximum observed Q1 differential "
                "density"
            ),
            "radius": (
                "Q1 MER morphology circularized VIS Sersic radius "
                "R_e,major sqrt(q), bounded joint magnitude x log-radius "
                "bins joined to PHZ"
            ),
            "radius_model": (
                "one straight magnitude-dependent truncated Gaussian in "
                "log10 circularized Sersic radius over 0.03--10 arcsec; "
                "no bright break and no generated broad tail"
            ),
            "aperture_fwhm": (
                "Q1 MER catalogue FWHM used by A-PHOT, sampled from the "
                "aggregate magnitude x FWHM histogram conditional on the "
                "same VIS 2FWHM brightness assigned to the synthetic galaxy"
            ),
            "aperture_fwhm_model": (
                "empirical 0.025-arcsec bins with nearest observed "
                "magnitude-bin continuation outside populated support"
            ),
            "radius_selection": str(radius_payload["selection"]),
            "radius_acquisition": str(radius_payload["acquisition"]),
            "cosmos_used": False,
            "object_catalog_used": False,
            "random_cones_used": False,
            "q1_counts_sha256": hashlib.sha256(
                q1_galaxy_counts_path().read_bytes()
            ).hexdigest(),
            "q1_brightness_fit_sha256": hashlib.sha256(
                q1_galaxy_fit_path().read_bytes()
            ).hexdigest(),
            "q1_radius_statistics_sha256": hashlib.sha256(
                q1_galaxy_radius_statistics_path().read_bytes()
            ).hexdigest(),
            "radius_magnitude_bins": len(magnitude_bins),
            "radius_histogram_bins": len(radius_bins),
            "joint_populated_bins": len(joint_bins),
            "fwhm_histogram_bins": len(fwhm_bins),
            "magnitude_fwhm_populated_bins": len(magnitude_fwhm_bins),
        },
    }
    fingerprint = hashlib.sha256(json.dumps(
        core, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    payload = {**core, "fingerprint": fingerprint, "active": False}
    _write(joint_galaxy_candidate_path(), payload)
    return payload


def joint_galaxy_state() -> dict[str, Any]:
    candidate = joint_galaxy_candidate()
    active = _read(active_joint_galaxy_path())
    return {
        "candidate": candidate,
        "active": active,
        "is_active": bool(
            candidate and active and candidate.get("valid")
            and candidate.get("fingerprint") == active.get("fingerprint")
        ),
    }


def activate_joint_galaxy_candidate() -> dict[str, Any]:
    """Atomically activate the Euclid-only joint draw model."""
    candidate = joint_galaxy_candidate()
    if not candidate or not candidate.get("valid"):
        raise ValueError("No structurally valid joint galaxy fit is available")
    payload = {**candidate, "active": True}
    _write(active_joint_galaxy_path(), payload)
    from euclid_polish.web import job_config

    job_config.update({
        "galaxy_density_arcmin2": float(
            payload["generation"]["surface_density_arcmin2"]
        )
    })
    return payload


def _catalog_weighted_fingerprint() -> str | None:
    """Fingerprint all catalog inputs used by probability-weighted fits."""
    from euclid_polish.web.helpers.population_comparison import (
        euclid_catalog_meta_path,
        euclid_catalog_path,
    )
    meta = _read(euclid_catalog_meta_path())
    if not meta or not euclid_catalog_path().is_file():
        return None
    digest = hashlib.sha256()
    try:
        with euclid_catalog_path().open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode())
                digest.update(b"\n")
    except OSError:
        return None
    identity = {
        "catalog_version": meta.get("catalog_version"),
        "area_arcmin2": meta.get("area_arcmin2"),
        "radius_arcmin": meta.get("radius_arcmin"),
        "cones": meta.get("cones"),
        "rows": meta.get("rows"),
        "rows_digest": digest.hexdigest(),
    }
    return hashlib.sha256(json.dumps(
        identity, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def photometric_candidate() -> dict[str, Any] | None:
    return brightness_transfer_payload(Config.JOINT_GALAXY_POPULATION_FIT_PATH)


def active_transfer() -> dict[str, Any] | None:
    return _read(active_transfer_path())


def activate_photometric_transfer(
    *, allow_quality_warnings: bool = False,
) -> dict[str, Any]:
    candidate = photometric_candidate()
    if candidate is None:
        raise ValueError("No fixed-normalization photometric fit is available")
    quality = candidate.get("fit_quality") or {}
    if not quality.get("valid", False) and not allow_quality_warnings:
        warnings = "; ".join(quality.get("warnings") or [])
        raise ValueError(
            "Fixed-normalization fit failed its quality gate"
            + (f": {warnings}" if warnings else "")
        )
    payload = {
        **candidate,
        "active": True,
        "validated": bool(quality.get("valid", False)),
        "activated_with_quality_warnings": bool(
            not quality.get("valid", False)
        ),
    }
    _write(active_transfer_path(), payload)
    return payload


def transfer_state() -> dict[str, Any]:
    candidate = photometric_candidate()
    active = active_transfer()
    return {
        "candidate": candidate,
        "active": active,
        "is_active": bool(
            candidate and active
            and candidate.get("version") == 3
            and active.get("version") == 3
            and candidate.get("valid")
            and candidate.get("fingerprint") == active.get("fingerprint")
        ),
    }


def source_transfer_fingerprints(rows: list[dict[str, Any]]) -> list[str]:
    fingerprints: set[str] = set()
    for row in rows:
        source = str(row.get("brightness_transfer") or "")
        if not source:
            continue
        parts = source.split(":", 2)
        if len(parts) >= 2 and len(parts[1]) == 64:
            fingerprints.add(parts[1])
        else:
            fingerprints.add("legacy:" + hashlib.sha256(
                source.encode("utf-8")
            ).hexdigest())
    return sorted(fingerprints)


def current_transfer_compatibility(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    source = source_transfer_fingerprints(rows)
    state = transfer_state()
    active = state.get("active") or {}
    expected = active.get("fingerprint")
    compatible = bool(expected and source == [expected])
    return {
        "compatible": compatible,
        "source_fingerprints": source,
        "active_fingerprint": expected,
        "reason": (
            "matched active fixed-normalization transfer"
            if compatible else
            "not actionable—brightness transfer changed"
        ),
    }


def _isotonic(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted PAVA, returned at the original sorted locations."""
    blocks: list[list[float]] = []
    for index, (value, weight) in enumerate(zip(values, weights, strict=True)):
        blocks.append([float(index), float(index), float(value), float(weight)])
        while len(blocks) >= 2 and blocks[-2][2] > blocks[-1][2]:
            right = blocks.pop()
            left = blocks.pop()
            total = left[3] + right[3]
            mean = (left[2] * left[3] + right[2] * right[3]) / total
            blocks.append([left[0], right[1], mean, total])
    result = np.empty(len(values), dtype=np.float64)
    for start, end, mean, _weight in blocks:
        result[int(start):int(end) + 1] = mean
    return result


def _invert_response(
    densities: np.ndarray, response: np.ndarray, target: float,
) -> float | None:
    if target < response[0] or target > response[-1]:
        return None
    if float(response[-1] - response[0]) <= 1e-6:
        return None
    return float(np.interp(target, response, densities))


def fit_density_response(
    densities: list[float],
    field_detections: list[list[float]],
    euclid_field_detections: list[float],
    *,
    transfer_fingerprint: str,
    active_transfer_fingerprint: str | None,
    field_area_arcmin2: float,
    euclid_cone_detection_densities: list[float] | None = None,
    bootstraps: int = 1000,
    seed: int = 71031,
) -> dict[str, Any]:
    """Fit and invert a paired monotone density-response sweep."""
    x = np.asarray(densities, dtype=np.float64)
    matrix = np.asarray(field_detections, dtype=np.float64)
    real = np.asarray(euclid_field_detections, dtype=np.float64)
    cone_densities = np.asarray(
        euclid_cone_detection_densities or [], dtype=np.float64,
    )
    cone_densities = cone_densities[np.isfinite(cone_densities)]
    if matrix.ndim != 2 or matrix.shape[0] != len(x):
        raise ValueError("density sweep must contain one field row per density")
    if matrix.shape[1] < 2 or real.size < 2 or field_area_arcmin2 <= 0:
        raise ValueError("density sweep needs at least two synthetic and real fields")
    order = np.argsort(x)
    x = x[order]
    matrix = matrix[order]
    means = np.mean(matrix, axis=1) / field_area_arcmin2
    weights = np.full(len(x), matrix.shape[1], dtype=np.float64)
    response = _isotonic(means, weights)
    target = float(np.mean(real) / field_area_arcmin2)
    estimate = _invert_response(x, response, target)
    fingerprint_match = bool(
        active_transfer_fingerprint
        and transfer_fingerprint == active_transfer_fingerprint
    )
    warnings: list[str] = []
    if not fingerprint_match:
        warnings.append("sweep brightness transfer is not the active transfer")
    if estimate is None:
        warnings.append("Euclid target is not bracketed by the sweep response")
    if float(response[-1] - response[0]) < 1.0:
        warnings.append("synthetic detection response is effectively flat")

    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(int(bootstraps)):
        synthetic_index = rng.integers(0, matrix.shape[1], matrix.shape[1])
        real_index = rng.integers(0, real.size, real.size)
        boot_means = np.mean(matrix[:, synthetic_index], axis=1) / field_area_arcmin2
        boot_response = _isotonic(boot_means, weights)
        boot_target = float(np.mean(real[real_index]) / field_area_arcmin2)
        if cone_densities.size >= 2 and float(np.mean(cone_densities)) > 0:
            cone_index = rng.integers(
                0, cone_densities.size, cone_densities.size,
            )
            boot_target *= float(
                np.mean(cone_densities[cone_index])
                / np.mean(cone_densities)
            )
        value = _invert_response(x, boot_response, boot_target)
        if value is not None and math.isfinite(value):
            samples.append(value)
    valid = not warnings and len(samples) >= max(20, bootstraps // 4)
    interval = None
    if samples:
        interval = {
            "median": float(np.median(samples)),
            "p16": float(np.percentile(samples, 16)),
            "p84": float(np.percentile(samples, 84)),
        }
    return {
        "version": 1,
        "method": "paired nested-thinning fields plus weighted isotonic response",
        "valid": valid,
        "warnings": warnings,
        "transfer_fingerprint": transfer_fingerprint,
        "active_transfer_fingerprint": active_transfer_fingerprint,
        "response_points": [
            {
                "density_arcmin2": float(density),
                "detected_density_arcmin2": float(detected),
                "isotonic_density_arcmin2": float(fitted),
            }
            for density, detected, fitted in zip(x, means, response, strict=True)
        ],
        "euclid_detected_density_arcmin2": target,
        "recommended_density_arcmin2": estimate,
        "interval_arcmin2": interval,
        "synthetic_fields_per_point": int(matrix.shape[1]),
        "euclid_fields": int(real.size),
        "euclid_cones": int(cone_densities.size),
    }


def _forward_detection_probabilities(
    f814w: np.ndarray,
    *,
    offset: float,
    slope: float,
    scatter: float,
    m50: float,
    width: float,
    magnitude_edges: np.ndarray,
    grid_step: float = 0.005,
) -> tuple[np.ndarray, float]:
    """Deterministically project an empirical F814W sample into VIS bins."""
    source = np.asarray(f814w, dtype=np.float64)
    if source.size == 0 or not np.isfinite(source).all():
        raise ValueError("COSMOS F814W prior is empty or non-finite")
    means = 24.0 + slope * (source - 24.0) + offset
    margin = max(1.0, 8.0 * scatter)
    grid_start = min(
        12.0, math.floor((float(means.min()) - margin) / grid_step) * grid_step,
    )
    grid_stop = max(
        40.0, math.ceil((float(means.max()) + margin) / grid_step) * grid_step,
    )
    grid = np.arange(grid_start, grid_stop + grid_step / 2.0, grid_step)
    grid_edges = np.concatenate((
        grid - grid_step / 2.0,
        np.asarray([grid[-1] + grid_step / 2.0]),
    ))
    density = np.histogram(means, bins=grid_edges)[0].astype(np.float64)
    density /= source.size * grid_step
    if scatter > 0.0:
        density = gaussian_filter1d(
            density,
            sigma=scatter / grid_step,
            mode="constant",
            cval=0.0,
        )
    argument = np.clip((grid - m50) / width, -60.0, 60.0)
    detected_mass = density / (1.0 + np.exp(argument)) * grid_step
    probabilities = np.asarray([
        detected_mass[(grid >= lower) & (grid < upper)].sum()
        for lower, upper in zip(
            magnitude_edges[:-1], magnitude_edges[1:], strict=True,
        )
    ], dtype=np.float64)
    return probabilities, float(probabilities.sum())


def fit_local_catalog_density(
    *,
    bootstraps: int = 2_000,
    seed: int = 71034,
) -> dict[str, Any]:
    """Infer the raw draw budget from local catalogs, without rendering fields.

    This evaluates the generator's empirical COSMOS distribution on a fixed
    magnitude grid, followed by the fitted F814W-to-VIS transfer and Euclid
    completeness curve. Dividing the probability-weighted extended-source
    density by that retained fraction gives the raw TNG draw budget.
    Cone bootstraps carry the dominant field-to-field uncertainty.
    """
    from euclid_polish.sky.generation.cosmos_tng_prior import (
        MORPHOLOGY_ACTIVITY_THRESHOLD_LOGSSFR,
        MORPHOLOGY_BALANCE_POWER,
        MORPHOLOGY_MIN_EFFECTIVE_DONORS,
        CosmosTngPrior,
        conditional_mass_quantiles,
        cross_validated_mass_bandwidth,
    )
    from euclid_polish.tng.radius_manifest import (
        load_parameter_summary,
    )
    from euclid_polish.web.helpers.population_comparison import (
        euclid_catalog_meta_path,
        euclid_catalog_path,
    )

    transfer = photometric_candidate()
    if transfer is None:
        raise ValueError("Fit the fixed-normalization brightness transfer first")
    coefficients = transfer.get("coefficients") or {}
    observation = transfer.get("observation_model") or {}
    try:
        offset = float(coefficients["offset_mag"])
        slope = float(coefficients["magnitude_slope"])
        scatter = float(coefficients["scatter_mag"])
        m50 = float(observation["completeness_m50"])
        width = float(observation["completeness_width_mag"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Brightness-transfer artifact is incomplete") from exc
    if not (slope > 0 and scatter >= 0 and width > 0):
        raise ValueError("Brightness-transfer coefficients are outside physical bounds")
    if bootstraps < 100:
        raise ValueError("Local calibration needs at least 100 bootstraps")

    prior = CosmosTngPrior(
        Config.COSMOS_POPULATION_PRIOR_PATH,
        photometric_fit_path=Config.JOINT_GALAXY_POPULATION_FIT_PATH,
    )
    if len(prior) < 1_000:
        raise ValueError("COSMOS/TNG prior has too few generator-ready rows")
    atlas_summary = load_parameter_summary(Config.TNG_ATLAS_PARAMETERS_PATH)
    summary_meta = atlas_summary["meta"]
    radius_fingerprint = str(summary_meta.get("manifest_fingerprint") or "")
    if not radius_fingerprint:
        raise ValueError("atlas parameter summary lacks a radius fingerprint")
    mass_by_id: dict[str, float] = {}
    sfr_by_id: dict[str, float] = {}
    for row in atlas_summary["rows"]:
        gid = str(row["subhalo_id"])
        mass = float(row["mass_stars_msun"])
        sfr = float(row["sfr_msun_yr"])
        previous = mass_by_id.setdefault(gid, mass)
        if not np.isclose(previous, mass, rtol=1e-12, atol=0.0):
            raise ValueError(f"TNG{gid} has inconsistent masses across orientations")
        previous_sfr = sfr_by_id.setdefault(gid, sfr)
        if not np.isclose(previous_sfr, sfr, rtol=1e-12, atol=0.0):
            raise ValueError(f"TNG{gid} has inconsistent SFR across orientations")
    atlas_ids = sorted(mass_by_id, key=int)
    atlas_mass = np.asarray([mass_by_id[gid] for gid in atlas_ids])
    atlas_sfr = np.asarray([sfr_by_id[gid] for gid in atlas_ids])
    if (
        not np.isfinite(atlas_mass).all() or np.any(atlas_mass <= 0.0)
        or not np.isfinite(atlas_sfr).all() or np.any(atlas_sfr < 0.0)
    ):
        raise ValueError("TNG atlas summary has invalid mass or SFR values")
    atlas_logmass = np.log10(atlas_mass)
    with np.errstate(divide="ignore", invalid="ignore"):
        atlas_logssfr = np.where(
            atlas_sfr > 0.0, np.log10(atlas_sfr) - atlas_logmass, -np.inf,
        )
    atlas_activity_class = np.where(
        atlas_logssfr < MORPHOLOGY_ACTIVITY_THRESHOLD_LOGSSFR,
        "quenched", "star_forming",
    )
    atlas_mass_quantile = conditional_mass_quantiles(
        atlas_logmass, atlas_activity_class,
    )
    transport_classes: dict[str, dict[str, Any]] = {}
    atlas_proxy_logmass = np.full(atlas_logmass.shape, np.nan, dtype=np.float64)
    for label in ("quenched", "star_forming"):
        atlas_indices = np.flatnonzero(atlas_activity_class == label)
        cosmos_indices = np.flatnonzero(prior.activity_class == label)
        if atlas_indices.size < 2 or cosmos_indices.size < 2:
            raise ValueError(
                f"quantile transport lacks a usable {label} population"
            )
        bandwidth = float(cross_validated_mass_bandwidth(
            atlas_mass_quantile[atlas_indices]
        ))
        cosmos_masses = prior.mass[cosmos_indices].astype(np.float64)
        atlas_proxy_logmass[atlas_indices] = np.quantile(
            cosmos_masses, atlas_mass_quantile[atlas_indices],
        )
        transport_classes[label] = {
            "tng_donors": int(atlas_indices.size),
            "cosmos_rows": int(cosmos_indices.size),
            "kernel_bandwidth_quantile": bandwidth,
            "native_tng_logmass_range": [
                float(np.min(atlas_logmass[atlas_indices])),
                float(np.max(atlas_logmass[atlas_indices])),
            ],
            "transported_proxy_logmass_range": [
                float(np.min(atlas_proxy_logmass[atlas_indices])),
                float(np.max(atlas_proxy_logmass[atlas_indices])),
            ],
        }
    eligible_indices = np.arange(len(prior), dtype=np.int64)
    excluded_mass_rows = 0
    meta = _read(euclid_catalog_meta_path())
    if not meta:
        raise ValueError("Query and cache several Euclid cones first")
    cone_count = int(meta.get("cone_count") or 0)
    area = float(meta.get("area_arcmin2") or 0.0)
    if cone_count < 3 or area <= 0:
        raise ValueError("Local density calibration needs at least three Euclid cones")

    cone_counts = np.zeros(cone_count, dtype=np.float64)
    magnitude_edges = np.arange(20.0, 28.0001, 0.5, dtype=np.float64)
    magnitude_counts = np.zeros(magnitude_edges.size - 1, dtype=np.float64)
    total_count = 0.0
    missing_probability = 0
    invalid_probability = 0
    catalog_digest = hashlib.sha256()
    with euclid_catalog_path().open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            catalog_digest.update(json.dumps({
                "object_id": row.get("object_id"),
                "cone_index": row.get("cone_index"),
                "mag_vis": row.get("mag_vis"),
                "spurious_prob": row.get("spurious_prob"),
                "point_like_prob": row.get("point_like_prob"),
            }, sort_keys=True, separators=(",", ":")).encode("utf-8"))
            catalog_digest.update(b"\n")
            try:
                spurious = float(row.get("spurious_prob") or 0.0)
                magnitude = float(row["mag_vis"])
                cone_index = int(row.get("cone_index") or -1)
            except (KeyError, TypeError, ValueError):
                continue
            try:
                point_probability = float(row["point_like_prob"])
            except (KeyError, TypeError, ValueError):
                missing_probability += 1
                continue
            if not math.isfinite(point_probability) or not 0.0 <= point_probability <= 1.0:
                invalid_probability += 1
                continue
            if not (
                math.isfinite(spurious) and spurious <= 0.5
                and math.isfinite(magnitude) and 20.0 <= magnitude < 28.0
            ):
                continue
            extended_weight = 1.0 - point_probability
            total_count += extended_weight
            bin_index = int(np.searchsorted(
                magnitude_edges, magnitude, side="right",
            ) - 1)
            if 0 <= bin_index < magnitude_counts.size:
                magnitude_counts[bin_index] += extended_weight
            if 0 <= cone_index < cone_count:
                cone_counts[cone_index] += extended_weight
    if total_count <= 0 or np.any(cone_counts <= 0):
        raise ValueError(
            "Euclid cone catalog lacks usable per-cone weighted extended sources"
        )

    bin_probabilities, retained_fraction = _forward_detection_probabilities(
        prior.f814w[eligible_indices],
        offset=offset,
        slope=slope,
        scatter=scatter,
        m50=m50,
        width=width,
        magnitude_edges=magnitude_edges,
    )
    prior_magnitude_counts = np.histogram(
        prior.f814w[eligible_indices], bins=magnitude_edges,
    )[0]
    if not 0.001 < retained_fraction < 0.999:
        raise ValueError("Fitted observation model has a degenerate retained fraction")

    cone_area = area / cone_count
    cone_densities = cone_counts / cone_area
    euclid_density = float(total_count / area)
    recommendation = euclid_density / retained_fraction
    predicted_bin_density = recommendation * bin_probabilities
    predicted_bin_counts = predicted_bin_density * area
    positive = magnitude_counts > 0.0
    deviance_terms = predicted_bin_counts.copy()
    deviance_terms[positive] = (
        magnitude_counts[positive] * np.log(
            magnitude_counts[positive]
            / np.maximum(predicted_bin_counts[positive], 1e-300)
        ) - (magnitude_counts[positive] - predicted_bin_counts[positive])
    )
    poisson_deviance = float(2.0 * np.sum(deviance_terms))
    magnitude_dof = max(1, int(magnitude_counts.size - 1))
    reduced_poisson_deviance = poisson_deviance / magnitude_dof

    # The forward probability is deterministic. Bootstrap only the measured
    # cone-to-cone variation, which is the dominant calibration uncertainty.
    rng = np.random.default_rng(seed)
    cone_indices = rng.integers(0, cone_count, size=(bootstraps, cone_count))
    target_samples = np.mean(cone_densities[cone_indices], axis=1)
    density_samples = target_samples / retained_fraction
    interval = {
        "median": float(np.median(density_samples)),
        "p16": float(np.percentile(density_samples, 16)),
        "p84": float(np.percentile(density_samples, 84)),
    }

    prior_digest = hashlib.sha256()
    for values in (prior.catalog_id, prior.f814w, prior.z, prior.mass, prior.re):
        prior_digest.update(np.ascontiguousarray(values).tobytes())
    prior_fingerprint = prior_digest.hexdigest()
    identity = {
        "version": 6,
        "method": "local_catalog_deterministic_forward_model_probability_weighted",
        "transfer_fingerprint": transfer["fingerprint"],
        "prior_f814w_fingerprint": prior_fingerprint,
        "tng_radius_manifest_fingerprint": radius_fingerprint,
        "euclid_cones": meta.get("cones"),
        "catalog_version": meta.get("catalog_version"),
        "catalog_area_arcmin2": area,
        "catalog_radius_arcmin": meta.get("radius_arcmin"),
        "catalog_weighted_fingerprint": _catalog_weighted_fingerprint(),
        "classification_weighting": "galaxy_weight=1-POINT_LIKE_PROB",
        "morphology_model": {
            "method": "activity_conditioned_empirical_mass_quantile_transport",
            "atlas_ids": atlas_ids,
            "atlas_logmass": atlas_logmass.tolist(),
            "atlas_mass_quantile": atlas_mass_quantile.tolist(),
            "atlas_proxy_logmass": atlas_proxy_logmass.tolist(),
            "atlas_activity_class": atlas_activity_class.tolist(),
            "atlas_parameter_summary_fingerprint": summary_meta.get(
                "summary_fingerprint"
            ),
            "native_tng_logmass_range": [
                float(np.min(atlas_logmass)), float(np.max(atlas_logmass)),
            ],
            "cosmos_target_logmass_range": [
                float(np.min(prior.mass)), float(np.max(prior.mass)),
            ],
            "activity_threshold_logssfr_yr": (
                MORPHOLOGY_ACTIVITY_THRESHOLD_LOGSSFR
            ),
            "minimum_effective_donors": MORPHOLOGY_MIN_EFFECTIVE_DONORS,
            "worker_balance_power": MORPHOLOGY_BALANCE_POWER,
            "classes": transport_classes,
            "eligible_cosmos_rows": int(len(eligible_indices)),
            "excluded_cosmos_rows": excluded_mass_rows,
            "changes_flux_or_size": False,
        },
        "selection": {
            "mag_min": 20.0, "mag_max": 28.0, "spurious_max": 0.5,
        },
        "forward_integration_grid_step_mag": 0.005,
        "seed": int(seed),
    }
    calibration_fingerprint = hashlib.sha256(json.dumps(
        identity, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
    warnings = list((transfer.get("fit_quality") or {}).get("warnings") or [])
    if reduced_poisson_deviance > 5.0:
        warnings.append(
            "quantile-transport COSMOS draw pool has high Poisson deviance"
        )
    warnings.append(
        "catalog-level calibration does not model rendering, crowding, or deblending"
    )
    quality_warnings = [
        warning for warning in warnings
        if "does not model rendering" not in warning
    ]
    result = {
        "version": 5,
        "method": (
            "empirical COSMOS/TNG generator distribution deterministically "
            "passed through the fitted Euclid brightness and completeness model"
        ),
        "valid": not quality_warnings,
        "validated": not quality_warnings,
        "warnings": warnings + [
            f"excluded {missing_probability:,} rows without point-like probability",
            f"excluded {invalid_probability:,} rows with invalid point-like probability",
        ],
        "transfer_fingerprint": transfer["fingerprint"],
        "tng_radius_manifest_fingerprint": radius_fingerprint,
        "active_transfer_fingerprint": (active_transfer() or {}).get("fingerprint"),
        "calibration_fingerprint": calibration_fingerprint,
        "catalog_weighted_fingerprint": _catalog_weighted_fingerprint(),
        "catalog_version": meta.get("catalog_version"),
        "catalog_area_arcmin2": area,
        "recommended_density_arcmin2": float(recommendation),
        "interval_arcmin2": interval,
        "euclid_detected_density_arcmin2": euclid_density,
        "retained_detection_fraction": retained_fraction,
        "magnitude_fit_quality": {
            "poisson_deviance": poisson_deviance,
            "dof": magnitude_dof,
            "reduced_poisson_deviance": reduced_poisson_deviance,
            "valid": reduced_poisson_deviance <= 5.0,
            "bins": [
                {
                    "mag_lo": float(lower),
                    "mag_hi": float(upper),
                    "euclid_detected_density_arcmin2": float(count / area),
                    "predicted_detected_density_arcmin2": float(predicted),
                }
                for lower, upper, count, predicted in zip(
                    magnitude_edges[:-1], magnitude_edges[1:],
                    magnitude_counts, predicted_bin_density, strict=True,
                )
            ],
        },
        "response_points": [
            {"density_arcmin2": 0.0, "detected_density_arcmin2": 0.0},
            {
                "density_arcmin2": float(recommendation),
                "detected_density_arcmin2": euclid_density,
            },
        ],
        "forward_integration_grid_step_mag": 0.005,
        "bootstrap_samples": int(bootstraps),
        "seed": int(seed),
        "cosmos_generator_rows": int(len(eligible_indices)),
        "cosmos_generator_rows_before_mass_support": int(len(prior)),
        "cosmos_f814w_support": {
            "minimum_mag": float(np.min(prior.f814w[eligible_indices])),
            "maximum_mag": float(np.max(prior.f814w[eligible_indices])),
            "bins": [
                {
                    "mag_lo": float(lower),
                    "mag_hi": float(upper),
                    "rows": int(count),
                }
                for lower, upper, count in zip(
                    magnitude_edges[:-1], magnitude_edges[1:],
                    prior_magnitude_counts, strict=True,
                )
            ],
        },
        "morphology_model": identity["morphology_model"],
        "cosmos_f814w_fingerprint": prior_fingerprint,
        "euclid_expected_extended_sources": float(total_count),
        "classification_weighting": {
            "star_weight": "POINT_LIKE_PROB",
            "galaxy_weight": "1 - POINT_LIKE_PROB",
            "missing_probability_rows": int(missing_probability),
            "invalid_probability_rows": int(invalid_probability),
        },
        "euclid_cones": cone_count,
        "euclid_cone_densities_arcmin2": cone_densities.tolist(),
        "selection": identity["selection"],
    }
    _write(density_calibration_path(), result)
    return result


def density_state() -> dict[str, Any]:
    candidate = _read(density_calibration_path())
    transfer = photometric_candidate() or {}
    if candidate and candidate.get("transfer_fingerprint") != transfer.get(
        "fingerprint"
    ):
        candidate = dict(candidate)
        candidate["valid"] = False
        candidate["warnings"] = list(candidate.get("warnings") or []) + [
            "brightness-transfer candidate changed after the sweep"
        ]
    active = _read(active_density_path())
    from euclid_polish.tng.radius_manifest import validate_manifest
    radius_status = validate_manifest(Config.TNG_SKIRT_DIR)
    current_radius_fingerprint = (
        radius_status.get("manifest_fingerprint") if radius_status.get("valid")
        else None
    )
    if candidate and candidate.get("tng_radius_manifest_fingerprint") != current_radius_fingerprint:
        candidate = dict(candidate)
        candidate["valid"] = False
        candidate["warnings"] = list(candidate.get("warnings") or []) + [
            "TNG radius manifest changed or is not submit-ready"
        ]
    current_catalog_fingerprint = _catalog_weighted_fingerprint()
    if candidate and candidate.get("catalog_weighted_fingerprint") != current_catalog_fingerprint:
        candidate = dict(candidate)
        candidate["valid"] = False
        candidate["warnings"] = list(candidate.get("warnings") or []) + [
            "Euclid weighted catalog changed after the density fit"
        ]
    return {
        "candidate": candidate,
        "active": active,
        "is_active": bool(
            candidate and active
            and candidate.get("valid")
            and candidate.get("calibration_fingerprint")
            == active.get("calibration_fingerprint")
        ),
    }


def activate_density_candidate() -> dict[str, Any]:
    """Activate a valid, transfer-matched local fit and update job config."""
    candidate = _read(density_calibration_path())
    if not candidate or not candidate.get("valid"):
        raise ValueError("No valid local density calibration is available")
    transfer = active_transfer() or {}
    if candidate.get("transfer_fingerprint") != transfer.get("fingerprint"):
        raise ValueError(
            "Density calibration used a different brightness transfer"
        )
    recommendation = candidate.get("recommended_density_arcmin2")
    if recommendation is None or not math.isfinite(float(recommendation)):
        raise ValueError("Density calibration has no finite recommendation")
    from euclid_polish.web import job_config

    payload = {
        **candidate,
        "active": True,
        "activated_density_arcmin2": float(recommendation),
    }
    job_config.update({"galaxy_density_arcmin2": float(recommendation)})
    _write(active_density_path(), payload)
    return payload


def galaxy_recommendation_state() -> dict[str, Any]:
    """Return every fitted generator parameter as one reviewable proposal."""
    transfer = photometric_candidate()
    density = density_state().get("candidate")
    coefficients = (transfer or {}).get("coefficients") or {}
    quality = (transfer or {}).get("fit_quality") or {}
    warnings = list(quality.get("warnings") or [])
    if density:
        warnings.extend(density.get("warnings") or [])
    recommendation_available = bool(
        transfer
        and density
        and density.get("valid")
        and density.get("recommended_density_arcmin2") is not None
        and density.get("transfer_fingerprint") == transfer.get("fingerprint")
    )
    return {
        "recommendation_available": recommendation_available,
        "validated": bool(recommendation_available and quality.get("valid")),
        "warnings": list(dict.fromkeys(str(item) for item in warnings)),
        "transfer_fingerprint": (transfer or {}).get("fingerprint"),
        "density_calibration_fingerprint": (
            (density or {}).get("calibration_fingerprint")
        ),
        "generator_parameters": {
            "galaxy_density_arcmin2": (
                (density or {}).get("recommended_density_arcmin2")
            ),
            "cosmos_vis_offset_mag": coefficients.get("offset_mag"),
            "cosmos_vis_magnitude_slope": coefficients.get("magnitude_slope"),
            "cosmos_vis_scatter_mag": coefficients.get("scatter_mag"),
        },
        "observation_model_diagnostics": (
            (transfer or {}).get("observation_model") or {}
        ),
        "density_interval_arcmin2": (density or {}).get("interval_arcmin2"),
    }


def activate_galaxy_recommendation() -> dict[str, Any]:
    """Freeze and apply the complete fitted generator parameter proposal."""
    state = galaxy_recommendation_state()
    if not state["recommendation_available"]:
        raise ValueError(
            "Run the local joint galaxy calibration before activating parameters"
        )
    transfer = activate_photometric_transfer(allow_quality_warnings=True)
    density = activate_density_candidate()
    return {
        **state,
        "active": True,
        "brightness_transfer": transfer,
        "density_calibration": density,
    }


def star_state() -> dict[str, Any]:
    candidate = _read(star_candidate_path())
    active = _read(active_star_path())
    candidate_current = _current_star_artifact(candidate)
    active_current = _current_star_artifact(active)
    if candidate and not candidate_current:
        candidate = {
            **candidate,
            "valid": False,
            "warnings": list(candidate.get("warnings") or []) + [
                "refit required: stellar counts must come from Q1 "
                "PHZ_STAR_PROB and colours from the fixed-Q1 sample"
            ],
        }
    is_active = False
    if (candidate is not None and active is not None
            and candidate_current and active_current):
        is_active = bool(
            candidate.get("valid")
            and candidate.get("fingerprint") == active.get("fingerprint")
        )
    return {
        "candidate": candidate,
        "active": active if active_current else None,
        "is_active": is_active,
    }


def _current_star_artifact(payload: dict[str, Any] | None) -> bool:
    return bool(
        payload
        and payload.get("version") == 6
        and (payload.get("fingerprint_inputs") or {}).get("fit_version")
        == "q1-phz-gaia-shared-straight-counts-latent-locus-v5"
    )


def active_star() -> dict[str, Any] | None:
    payload = _read(active_star_path())
    return payload if _current_star_artifact(payload) else None


def activate_star_candidate() -> dict[str, Any]:
    candidate = _read(star_candidate_path())
    if (candidate is None or not _current_star_artifact(candidate)
            or not candidate.get("valid")):
        raise ValueError("No valid fitted stellar population is available")
    payload = {**candidate, "active": True}
    _write(active_star_path(), payload)
    return payload


def write_star_candidate(payload: dict[str, Any]) -> None:
    _write(star_candidate_path(), payload)
