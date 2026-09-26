"""Versioned calibration artifacts for galaxy density and stellar priors."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from euclid_polish.config import Config
from euclid_polish.population.conditional_color_sfr import (
    COLOR_SFR_MODEL_VERSION,
    ConditionalColorSFRSampler,
)
from euclid_polish.population.conditional_color_sfr_fit import (
    fit_conditional_color_sfr_payload,
)
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
from euclid_polish.web import job_config
from euclid_polish.web.helpers import (
    population_comparison,
    q1_galaxy_counts,
    q1_galaxy_radius_statistics,
)


def calibration_dir() -> Path:
    return Path(Config.DATA_DIR) / "population_comparison" / "calibrations"


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


# The v15 candidate carries the exported colour+SFR forest (~20 MB), so the
# dashboard cannot afford to re-read and re-validate it on every poll. The
# cache key is the file identity; ``_write`` replaces atomically, so a refit
# always changes it.
_JOINT_CANDIDATE_CACHE: dict[str, Any] = {"key": None, "value": None}


def _file_identity(path: Path) -> tuple[int, int] | None:
    try:
        status = path.stat()
    except OSError:
        return None
    return (status.st_mtime_ns, status.st_size)


def joint_galaxy_candidate() -> dict[str, Any] | None:
    """Return the persisted, structurally validated joint-galaxy candidate."""
    identity = _file_identity(joint_galaxy_candidate_path())
    if identity is not None and _JOINT_CANDIDATE_CACHE["key"] == identity:
        return _JOINT_CANDIDATE_CACHE["value"]
    value = _validated_joint_galaxy_candidate()
    _JOINT_CANDIDATE_CACHE["key"] = identity
    _JOINT_CANDIDATE_CACHE["value"] = value
    return value


def _validated_joint_galaxy_candidate() -> dict[str, Any] | None:
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
        color_sampler = ConditionalColorSFRSampler(source["color_sfr_model"])
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
        colors_plot = source["plots"]["conditional_colors"]
        color_magnitude = np.asarray(
            colors_plot["magnitude"], dtype=np.float64,
        )
        color_mean_vis_minus_y = np.asarray(
            colors_plot["model_mean_vis_minus_y"], dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError):
        return None
    if (
        source.get("version") != JOINT_EUCLID_GALAXY_VERSION
        or source.get("kind") != JOINT_EUCLID_GALAXY_KIND
        or radius_law.version != RADIUS_MODEL_VERSION
        or aperture_fwhm.version != APERTURE_FWHM_MODEL_VERSION
        or int(source["color_sfr_model"].get("version") or 0)
        != COLOR_SFR_MODEL_VERSION
        or color_sampler.row_count < 2
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
        or color_magnitude.size < 2
        or color_magnitude.shape != color_mean_vis_minus_y.shape
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
        or not np.all(np.isfinite(color_magnitude))
        or not np.all(np.isfinite(color_mean_vis_minus_y))
    ):
        return None
    return source


def fit_euclid_joint_galaxy_candidate() -> dict[str, Any]:
    """Fit VIS-2FWHM x Sersic-R_e plus the empirical colour+SFR forest."""
    # Looked up on the modules at call time (tests patch these helpers).
    POPULATION_CATALOG_VERSION = population_comparison.CATALOG_VERSION  # noqa: N806
    euclid_catalog_meta_path = population_comparison.euclid_catalog_meta_path
    euclid_catalog_path = population_comparison.euclid_catalog_path
    q1_galaxy_counts_path = q1_galaxy_counts.q1_galaxy_counts_path
    q1_galaxy_fit_path = q1_galaxy_counts.q1_galaxy_fit_path
    read_q1_galaxy_aperture_counts = q1_galaxy_counts.read_q1_galaxy_aperture_counts
    read_q1_galaxy_aperture_fit = q1_galaxy_counts.read_q1_galaxy_aperture_fit
    q1_galaxy_radius_statistics_path = (
        q1_galaxy_radius_statistics.q1_galaxy_radius_statistics_path)
    read_q1_galaxy_radius_statistics = (
        q1_galaxy_radius_statistics.read_q1_galaxy_radius_statistics)

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
    color_sfr_model, color_diagnostics = fit_conditional_color_sfr_payload(
        euclid_catalog_path(),
        euclid_catalog_meta_path(),
        radius_law=radius_law,
        expected_catalog_version=POPULATION_CATALOG_VERSION,
    )
    color_meta = _read(euclid_catalog_meta_path()) or {}
    color_trend = color_diagnostics["mean_colors_by_magnitude"]
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
        "color_sfr_model": color_sfr_model,
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
            "conditional_colors": {
                "magnitude": list(color_trend["magnitude"]),
                "model_mean_vis_minus_y": list(color_trend["vis_minus_y"]),
                "model_mean_y_j": list(color_trend["y_minus_j"]),
                "model_mean_j_h": list(color_trend["j_minus_h"]),
                "model_kind": (
                    "conditional_color_sfr_forest_leaf_resampling"
                ),
                "deconvolution": color_sfr_model["deconvolution"],
                "magnitude_edges": color_diagnostics["magnitude_edges"],
                "observed_ratio_variance_by_magnitude": color_diagnostics[
                    "observed_ratio_variance_by_magnitude"
                ],
                "noise_ratio_variance_by_magnitude": color_diagnostics[
                    "noise_ratio_variance_by_magnitude"
                ],
                "sfr_valid_weight_fraction_by_magnitude": color_diagnostics[
                    "sfr_valid_weight_fraction_by_magnitude"
                ],
                "leaf_count_by_tree": color_diagnostics["leaf_count_by_tree"],
                "median_leaf_rows_by_tree": color_diagnostics[
                    "median_leaf_rows_by_tree"
                ],
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
                "generation_fraction_fainter_than_vis_25": float(
                    np.sum(
                        diagnostic_magnitude_mass[
                            diagnostic_magnitude >= 25.0
                        ]
                    )
                    / np.sum(diagnostic_magnitude_mass)
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
            "sampling_order": (
                "radius_marginal_then_brightness_given_radius_then_"
                "colors_sfr_given_brightness_and_radius"
            ),
            "color_sampling": (
                "empirical_forest_row_resampling_given_vis_2fwhm_and_re"
            ),
            "color_deconvolution": (
                "analytic_neighborhood_single_gaussian_extreme_deconvolution"
            ),
            "color_rendering": (
                "preserve_sampled_observed_Re_then_exact_VIS_2FWHM_"
                "normalization_then_per_band_nisp_over_vis_ratio_scalars"
            ),
            "redshift_sampling": (
                "none_redshift_enters_implicitly_through_empirical_colors"
            ),
            "sfr_morphology_matching": (
                "global_log_sfr_rank_kernel_ess64_balanced"
            ),
            "lens_color_policy": (
                "achromatic_tolman_dimming_plus_class_conditioned_"
                "empirical_colors"
            ),
            "aperture_fwhm_sampling": (
                "MER_FWHM_given_sampled_VIS_2FWHM_magnitude"
            ),
            "aperture_fwhm_min_arcsec": aperture_fwhm.minimum_arcsec,
            "aperture_fwhm_max_arcsec": aperture_fwhm.maximum_arcsec,
            "morphology_assignment": "sfr_rank_matched_tng_atlas",
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
            "colors": (
                "empirical NISP/VIS 2FWHM flux ratios plus PHZ SFR "
                "resampled from real Q1 rows through a conditional forest; "
                "measurement noise removed by analytic neighbourhood "
                "deconvolution; no per-galaxy redshift is assigned"
            ),
            "color_selection": color_sfr_model["selection"],
            "color_weight_policy": color_sfr_model["weight_policy"],
            "color_catalog_version": color_sfr_model["catalog_version"],
            "color_catalog_sha256": color_sfr_model["catalog_sha256"],
            "color_calibration_fingerprint": color_sfr_model[
                "calibration_fingerprint"
            ],
            "color_catalog_rows": color_diagnostics["catalog_rows"],
            "color_selected_rows": color_diagnostics["selected_rows"],
            "color_selected_galaxy_weight": color_diagnostics[
                "selected_galaxy_weight"
            ],
            "color_sfr_valid_weight_fraction": color_diagnostics[
                "sfr_valid_weight_fraction"
            ],
            "color_resolved_radius_weight_fraction": color_diagnostics[
                "resolved_radius_weight_fraction"
            ],
            "color_negative_ratio_row_fraction": color_diagnostics[
                "negative_ratio_row_fraction"
            ],
            "color_quenched_weight_fraction": color_diagnostics[
                "quenched_weight_fraction"
            ],
            "color_sklearn_version": color_sfr_model["sklearn_version"],
            "radius_selection": str(radius_payload["selection"]),
            "radius_acquisition": str(radius_payload["acquisition"]),
            "cosmos_used": False,
            "object_catalog_used": True,
            "random_cones_used": bool(color_meta.get("cones")),
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


def joint_galaxy_payload_summary(
    payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """The candidate/active payload without the packed forest arrays.

    Everything the dashboard renders (laws, plots, provenance, colour-model
    metadata) survives; only the encoded row table and trees are dropped —
    they are megabytes of base64 the browser never uses.
    """
    if not payload:
        return payload
    slim = dict(payload)
    model = payload.get("color_sfr_model")
    if isinstance(model, dict):
        slim["color_sfr_model"] = {
            key: value for key, value in model.items()
            if key not in ("rows", "trees")
        }
    return slim


def joint_galaxy_state_summary() -> dict[str, Any]:
    """``joint_galaxy_state`` with browser-sized payloads."""
    state = joint_galaxy_state()
    return {
        **state,
        "candidate": joint_galaxy_payload_summary(state.get("candidate")),
        "active": joint_galaxy_payload_summary(state.get("active")),
    }


def activate_joint_galaxy_candidate() -> dict[str, Any]:
    """Atomically activate the Euclid-only joint draw model."""
    candidate = joint_galaxy_candidate()
    if not candidate or not candidate.get("valid"):
        raise ValueError("No structurally valid joint galaxy fit is available")
    payload = {**candidate, "active": True}
    _write(active_joint_galaxy_path(), payload)
    job_config.update({
        "galaxy_density_arcmin2": float(
            payload["generation"]["surface_density_arcmin2"]
        )
    })
    return payload


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
