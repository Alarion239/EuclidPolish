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
from euclid_polish.population.magnitude_law import StraightMagnitudeLaw
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
    """Return a compact generation artifact derived from the joint fit."""
    source_path = Path(Config.JOINT_GALAXY_POPULATION_FIT_PATH)
    source = _read(source_path)
    if not source or source.get("version") not in {2, 3}:
        return None
    if source.get("kind") != "joint_intrinsic_galaxy_population":
        return None
    fingerprint = str(source.get("fingerprint") or "")
    model = source.get("model") or {}
    diagnostics = source.get("diagnostics") or {}
    tng_full = ((diagnostics.get("tng_draw") or {}).get("full") or {})
    try:
        geometry_density = float(tng_full["surface_density_arcmin2"])
        luminosity = dict(model["luminosity_function"])
        size = dict(model["size_relation"])
        response = dict(model["euclid_response"])
        vis_edges = ((source.get("method") or {}).get("tng_draw_window") or [
            18.0, 30.0,
        ])
        vis_min, vis_max = (float(vis_edges[0]), float(vis_edges[1]))
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    if (
        len(fingerprint) != 64
        or not math.isfinite(geometry_density) or geometry_density <= 0.0
        or not (math.isfinite(vis_min) and math.isfinite(vis_max)
                and vis_min < vis_max)
    ):
        return None
    quality = dict(source.get("fit_quality") or {})
    enhanced = source.get("version") == 3
    physical_conditionals = source.get("physical_conditionals")
    phz_correction = source.get("phz_redshift_correction")
    phz_gates = dict(source.get("phz_quality_gates") or {})
    if enhanced and (
        not isinstance(physical_conditionals, dict)
        or not isinstance(phz_correction, dict)
        or not phz_gates
    ):
        return None
    try:
        from euclid_polish.web.helpers.q1_galaxy_counts import (
            read_q1_galaxy_aperture_counts,
            read_q1_galaxy_aperture_fit,
        )
        aperture_fit = read_q1_galaxy_aperture_fit()
        aperture_curve = aperture_fit["apertures"]["f2"]
        magnitude_law = StraightMagnitudeLaw.from_payload(aperture_curve["law"])
    except (KeyError, TypeError, ValueError):
        return None
    plot_grid = np.asarray(aperture_curve.get("x", []), dtype=np.float64)
    plot_density = np.asarray(
        aperture_curve.get("density", []), dtype=np.float64,
    )
    if (
        plot_grid.size < 2 or plot_density.shape != plot_grid.shape
        or not np.all(np.isfinite(plot_grid))
        or not np.all(np.isfinite(plot_density))
    ):
        plot_grid = np.linspace(
            magnitude_law.mag_bright, magnitude_law.mag_faint, 301,
        )
        plot_density = magnitude_law.density(plot_grid)
    magnitude_plot: dict[str, Any] = {
        "law": {
            "x": plot_grid.tolist(),
            "density": plot_density.tolist(),
        },
        "fit_interval": [
            magnitude_law.fit_bright, magnitude_law.fit_faint,
        ],
        "sampling_interval": [
            magnitude_law.mag_bright, magnitude_law.mag_faint,
        ],
        "extrapolated_interval": list(
            aperture_curve.get("extrapolated_faint_interval") or [28.0, 29.0]
        ),
        "label": "Q1 MER + PHZ VIS 2FWHM straight law",
    }
    try:
        aperture_counts = read_q1_galaxy_aperture_counts()["apertures"]["f2"]
        magnitude_plot["observed"] = {
            "x": [
                0.5 * (float(item["mag_lo"]) + float(item["mag_hi"]))
                for item in aperture_counts["bins"]
            ],
            "density": [
                float(item["density_arcmin2_mag"])
                for item in aperture_counts["bins"]
            ],
        }
    except (KeyError, TypeError, ValueError):
        # The law remains sufficient for generation and for a fail-closed
        # calibration plot; raw points appear whenever the matching cache is
        # present.
        pass
    combined_fingerprint = hashlib.sha256(json.dumps({
        "geometry": fingerprint,
        "magnitude_law": magnitude_law.to_payload(),
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "version": 3,
        "kind": "joint_analytical_tng_draw",
        "valid": True,
        "validated": bool(quality.get("valid")),
        "warnings": list(quality.get("warnings") or []),
        "fingerprint": combined_fingerprint,
        "geometry_model_fingerprint": fingerprint,
        "source_artifact": str(source_path),
        "model": {
            "luminosity_function": luminosity,
            "size_relation": size,
            "euclid_response": response,
        },
        **({
            "phz_redshift_correction": phz_correction,
            "physical_conditionals": physical_conditionals,
            "phz_quality_gates": phz_gates,
        } if enhanced else {}),
        "magnitude_law": magnitude_law.to_payload(),
        "magnitude_plot": magnitude_plot,
        "generation": {
            "surface_density_arcmin2": magnitude_law.integrated_density(),
            "geometry_reference_density_arcmin2": geometry_density,
            "vis_magnitude_min": magnitude_law.mag_bright,
            "vis_magnitude_max": magnitude_law.mag_faint,
            "morphology_assignment": (
                "phz_mass_activity_quantile_transport"
                if enhanced else "balanced_random_tng_atlas"
            ),
            "position_process": "homogeneous_poisson",
        },
        "fit_quality": {
            key: quality.get(key)
            for key in (
                "valid", "cosmos_reduced_negative_binomial_deviance",
                "euclid_bright_transfer_reduced_poisson_deviance",
                "euclid_reduced_poisson_deviance",
                "euclid_cross_validated_reduced_poisson_deviance",
            )
        },
    }


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
    """Atomically activate the fitted joint draw model for future jobs."""
    candidate = joint_galaxy_candidate()
    if not candidate or not candidate.get("valid"):
        raise ValueError("No structurally valid joint galaxy fit is available")
    if candidate.get("physical_conditionals") and not candidate.get("validated"):
        raise ValueError(
            "The PHZ-enhanced joint galaxy fit has not passed activation gates"
        )
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
    return brightness_transfer_payload(Config.COSMOS_EUCLID_FIT_PATH)


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
    from euclid_polish.sky.generation.tng_radius_manifest import (
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
        Config.COSMOS_TNG_PRIOR_PATH,
        photometric_fit_path=Config.COSMOS_EUCLID_FIT_PATH,
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
    from euclid_polish.sky.generation.tng_radius_manifest import validate_manifest
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
                "refit required: stellar counts must come from Q1 PHZ_STAR_PROB"
            ],
        }
    return {
        "candidate": candidate,
        "active": active if active_current else None,
        "is_active": bool(
            candidate_current and active_current
            and candidate.get("valid")
            and candidate.get("fingerprint") == active.get("fingerprint")
        ),
    }


def _current_star_artifact(payload: dict[str, Any] | None) -> bool:
    return bool(
        payload
        and payload.get("version") == 4
        and (payload.get("fingerprint_inputs") or {}).get("fit_version")
        == "q1-phz-gaia-shared-straight-counts-latent-locus-v3"
    )


def active_star() -> dict[str, Any] | None:
    payload = _read(active_star_path())
    return payload if _current_star_artifact(payload) else None


def activate_star_candidate() -> dict[str, Any]:
    candidate = _read(star_candidate_path())
    if not _current_star_artifact(candidate) or not candidate.get("valid"):
        raise ValueError("No valid fitted stellar population is available")
    payload = {**candidate, "active": True}
    _write(active_star_path(), payload)
    return payload


def write_star_candidate(payload: dict[str, Any]) -> None:
    _write(star_candidate_path(), payload)
