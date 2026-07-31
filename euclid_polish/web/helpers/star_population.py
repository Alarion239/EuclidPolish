"""Same-footprint Gaia queries and a compact synthetic-star population fit."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from euclid_polish.config import Config
from euclid_polish.sky.generation.stellar_sed import EmpiricalStellarPrior
from euclid_polish.web.helpers.population_calibration import write_star_candidate
from euclid_polish.web.helpers.population_comparison import (
    euclid_catalog_meta_path,
    euclid_catalog_path,
)

_GAIA_COUNT_LIMIT_MAG = 20.5
_STAR_POPULATION_VERSION = 2


def gaia_catalog_path() -> Path:
    return Path(Config.DATA_DIR) / "population_comparison" / "gaia_population.csv"


def gaia_catalog_meta_path() -> Path:
    return gaia_catalog_path().with_suffix(".meta.json")


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _angular_separation_arcsec(
    ra1: float, dec1: float, ra2: float, dec2: float,
) -> float:
    d1, d2 = math.radians(dec1), math.radians(dec2)
    dra = math.radians(ra1 - ra2)
    cosine = math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(dra)
    return 3600.0 * math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def query_gaia_same_cones(
    *, progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Query Gaia DR3 for the exact cached Euclid cone footprints."""
    try:
        cone_meta = json.loads(euclid_catalog_meta_path().read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Query Euclid population cones first") from exc
    cones = cone_meta.get("cones") or []
    radius_arcmin = float(cone_meta.get("radius_arcmin") or 0.0)
    if not cones or radius_arcmin <= 0:
        raise ValueError("Cached Euclid catalog has no multi-cone provenance")

    from astroquery.gaia import Gaia

    rows: list[dict[str, Any]] = []
    for index, cone in enumerate(cones):
        if progress:
            progress(index, len(cones), f"Gaia cone {index + 1}/{len(cones)}")
        ra, dec = float(cone["ra"]), float(cone["dec"])
        query = f"""
        SELECT source_id, ra, dec, phot_g_mean_mag, phot_bp_mean_mag,
               phot_rp_mean_mag, bp_rp, teff_gspphot, ag_gspphot
        FROM gaiadr3.gaia_source
        WHERE CONTAINS(
          POINT('ICRS', ra, dec),
          CIRCLE('ICRS', {ra}, {dec}, {radius_arcmin / 60.0})
        ) = 1
          AND phot_g_mean_mag IS NOT NULL
        """
        result = Gaia.launch_job_async(query).get_results()
        cone_rows: list[dict[str, Any]] = []
        for raw in result:
            source_id = str(raw["source_id"]).strip()
            source_ra = float(raw["ra"])
            source_dec = float(raw["dec"])
            cone_rows.append({
                "source_id": source_id,
                "cone_index": index,
                "ra": source_ra,
                "dec": source_dec,
                "g_mag": _finite(raw["phot_g_mean_mag"]),
                "bp_mag": _finite(raw["phot_bp_mean_mag"]),
                "rp_mag": _finite(raw["phot_rp_mean_mag"]),
                "bp_rp": _finite(raw["bp_rp"]),
                "temperature_k": _finite(raw["teff_gspphot"]),
                "extinction_g_mag": _finite(raw["ag_gspphot"]),
                "central_selected_star": 0,
            })
        if cone_rows:
            closest = min(
                range(len(cone_rows)),
                key=lambda item: _angular_separation_arcsec(
                    ra, dec, cone_rows[item]["ra"], cone_rows[item]["dec"],
                ),
            )
            # A saved-star cone deliberately contains its selected centre.
            # Mark exactly the nearest Gaia source and exclude it from density.
            cone_rows[closest]["central_selected_star"] = 1
        rows.extend(cone_rows)

    output = gaia_catalog_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    columns = [
        "source_id", "cone_index", "ra", "dec", "g_mag", "bp_mag",
        "rp_mag", "bp_rp", "temperature_k", "extinction_g_mag",
        "central_selected_star",
    ]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output)
    meta = {
        "version": 1,
        "gaia_table": "gaiadr3.gaia_source",
        "cone_count": len(cones),
        "cones": cones,
        "radius_arcmin": radius_arcmin,
        "area_arcmin2": len(cones) * math.pi * radius_arcmin ** 2,
        "rows": len(rows),
        "central_sources_excluded_from_density": len(cones),
        "euclid_cone_selection_seed": cone_meta.get("selection_seed"),
    }
    gaia_catalog_meta_path().write_text(json.dumps(meta, indent=2, sort_keys=True))
    if progress:
        progress(len(cones), len(cones), "Gaia cones cached")
    return meta


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _robust_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = np.ones(len(y), dtype=np.float64)
    coefficients = np.zeros(x.shape[1], dtype=np.float64)
    for _ in range(12):
        root = np.sqrt(weights)
        coefficients = np.linalg.lstsq(x * root[:, None], y * root, rcond=None)[0]
        residual = y - x @ coefficients
        scale = max(1.4826 * np.median(np.abs(residual - np.median(residual))), 0.02)
        ratio = np.abs(residual) / (1.5 * scale)
        weights = np.ones_like(ratio)
        outliers = ratio > 1
        weights[outliers] = 1.0 / ratio[outliers]
    return coefficients, y - x @ coefficients


def _quantiles(values: np.ndarray, count: int = 33) -> list[float]:
    if values.size == 0:
        return []
    return [float(item) for item in np.quantile(values, np.linspace(0, 1, count))]


def _histogram(
    observed: np.ndarray, fitted: np.ndarray, *, bins: np.ndarray,
    observed_scale: float = 1.0, fitted_scale: float = 1.0,
) -> dict[str, Any]:
    observed_counts, edges = np.histogram(observed, bins=bins)
    fitted_counts, _ = np.histogram(fitted, bins=edges)
    widths = np.diff(edges)
    return {
        "x": (0.5 * (edges[:-1] + edges[1:])).tolist(),
        "observed": (observed_counts * observed_scale / widths).tolist(),
        "fitted": (fitted_counts * fitted_scale / widths).tolist(),
        "observed_count": int(observed.size),
        "fitted_count": int(fitted.size),
    }


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float,
) -> float | None:
    if values.size == 0 or weights.size != values.size:
        return None
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = np.maximum(weights[order], 0.0)
    total = float(np.sum(sorted_weights))
    if total <= 0.0:
        return None
    cumulative = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    return float(np.interp(float(quantile) * total, cumulative, sorted_values))


def _weighted_summary(
    values: np.ndarray, weights: np.ndarray, *, area_arcmin2: float,
    classification_variance: float = 0.0,
) -> dict[str, Any]:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    values, weights = values[valid], weights[valid]
    total = float(np.sum(weights))
    mean = float(np.sum(values * weights) / total) if total > 0.0 else None
    std = (
        float(np.sqrt(np.sum(weights * (values - mean) ** 2) / total))
        if total > 0.0 and mean is not None else None
    )
    sum_sq = float(np.sum(weights ** 2))
    return {
        "expected_count": total,
        "density_arcmin2": total / area_arcmin2 if area_arcmin2 > 0 else None,
        "mean": mean,
        "std": std,
        "p16": _weighted_quantile(values, weights, 0.16),
        "p50": _weighted_quantile(values, weights, 0.50),
        "p84": _weighted_quantile(values, weights, 0.84),
        "effective_n": total * total / sum_sq if sum_sq > 0.0 else 0.0,
        "classification_sigma_count": float(np.sqrt(max(classification_variance, 0.0))),
        "classification_sigma_density_arcmin2": (
            float(np.sqrt(max(classification_variance, 0.0)) / area_arcmin2)
            if area_arcmin2 > 0 else None
        ),
    }


def _weighted_histogram(
    values: np.ndarray, weights: np.ndarray, *, bins: np.ndarray,
    area_arcmin2: float,
) -> dict[str, Any]:
    counts, edges = np.histogram(values, bins=bins, weights=weights)
    widths = np.diff(edges)
    return {
        "x": (0.5 * (edges[:-1] + edges[1:])).tolist(),
        "observed": (counts / area_arcmin2 / widths).tolist(),
        "observed_count": int(values.size),
        "weighted_count": float(np.sum(weights)),
    }


def _fixed_width_edges(lower: float, upper: float, width: float) -> np.ndarray:
    """Histogram edges that stop at ``upper`` instead of adding an empty bin."""
    edges = np.arange(lower, upper + 0.5 * width, width, dtype=np.float64)
    if edges[-1] < upper - 1e-9:
        edges = np.append(edges, upper)
    else:
        edges[-1] = upper
    return edges


def fit_star_population(
    *, faint_limit: float | None = None, bright_limit: float | None = None,
) -> dict[str, Any]:
    """Fit counts and correlated Euclid colours from cached Gaia/Euclid rows."""
    faint = float(faint_limit or Config.STAR_MAG_FAINT)
    bright = float(bright_limit or Config.STAR_MAG_BRIGHT)
    gaia_rows = _read_rows(gaia_catalog_path())
    euclid_rows = _read_rows(euclid_catalog_path())
    try:
        meta = json.loads(gaia_catalog_meta_path().read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Gaia cone metadata is unavailable") from exc

    euclid_point_rows: list[tuple[dict[str, str], float]] = []
    missing_probability = 0
    invalid_probability = 0
    for row in euclid_rows:
        probability = _finite(row.get("point_like_prob"))
        if probability is None:
            missing_probability += 1
            continue
        if not 0.0 <= probability <= 1.0:
            invalid_probability += 1
            continue
        euclid_point_rows.append((row, probability))

    usable_gaia = [row for row in gaia_rows if _finite(row.get("g_mag")) is not None]
    by_id = {str(row["source_id"]): row for row in usable_gaia}
    matches: list[tuple[dict[str, str], dict[str, str]]] = []
    for euclid in euclid_rows:
        identifier = str(euclid.get("gaia_id") or "").strip()
        gaia = by_id.get(identifier)
        if gaia is None or euclid.get("type") != "star":
            continue
        values = [
            _finite(euclid.get(key))
            for key in ("mag_vis", "mag_y_e", "mag_j_e", "mag_h_e")
        ]
        if all(value is not None for value in values) and _finite(gaia.get("bp_rp")) is not None:
            matches.append((euclid, gaia))

    warnings: list[str] = []
    if len(matches) < 8:
        warnings.append("fewer than 8 clean Euclid-Gaia stellar matches")
    mapping: dict[str, list[float]] = {}
    residual_columns: list[np.ndarray] = []
    if matches:
        g = np.asarray([float(pair[1]["g_mag"]) for pair in matches])
        color = np.asarray([float(pair[1]["bp_rp"]) for pair in matches])
        design = np.column_stack([np.ones(len(matches)), color, g - 20.0])
        for key in ("mag_vis", "mag_y_e", "mag_j_e", "mag_h_e"):
            target = np.asarray([float(pair[0][key]) for pair in matches]) - g
            coefficients, residual = _robust_fit(design, target)
            mapping[key] = [float(value) for value in coefficients]
            residual_columns.append(residual)
        residual_matrix = np.column_stack(residual_columns)
        covariance = np.cov(residual_matrix, rowvar=False)
        covariance += np.eye(4) * 1e-4
    else:
        covariance = np.eye(4) * 0.04

    area_per_cone = math.pi * float(meta["radius_arcmin"]) ** 2
    density_counts: list[int] = []
    for cone_index in range(int(meta["cone_count"])):
        density_counts.append(sum(
            int(row.get("cone_index", -1)) == cone_index
            and row.get("central_selected_star") != "1"
            and float(row["g_mag"]) <= _GAIA_COUNT_LIMIT_MAG
            for row in usable_gaia
        ))
    bright_density = float(np.mean(density_counts) / area_per_cone)

    predicted_vis: list[float] = []
    comparison_gaia_vis: list[float] = []
    gaia_bright_by_cone: list[list[float]] = [
        [] for _ in range(int(meta["cone_count"]))
    ]
    if mapping:
        coeff = np.asarray(mapping["mag_vis"])
        for row in usable_gaia:
            bp_rp = _finite(row.get("bp_rp"))
            if bp_rp is None or row.get("central_selected_star") == "1":
                continue
            g_mag = float(row["g_mag"])
            vis_mag = g_mag + float(np.dot(coeff, [1.0, bp_rp, g_mag - 20.0]))
            predicted_vis.append(vis_mag)
            # Gaia supplies the bright side only. Keeping the transformed VIS
            # cut here makes Gaia and Euclid components disjoint at the splice;
            # G<=20.5 remains the normalization convention for Gaia counts.
            if (
                g_mag <= _GAIA_COUNT_LIMIT_MAG
                and bright <= vis_mag <= _GAIA_COUNT_LIMIT_MAG
            ):
                comparison_gaia_vis.append(vis_mag)
                cone_index = int(row.get("cone_index") or -1)
                if 0 <= cone_index < len(gaia_bright_by_cone) \
                        and vis_mag <= _GAIA_COUNT_LIMIT_MAG:
                    gaia_bright_by_cone[cone_index].append(vis_mag)
    count_mags = np.asarray(predicted_vis, dtype=np.float64)
    hist, edges = np.histogram(count_mags, bins=np.arange(14.0, 21.01, 0.5))
    centres = 0.5 * (edges[:-1] + edges[1:])
    selected = hist >= 2
    slope = 0.2
    if np.count_nonzero(selected) >= 4:
        slope = float(np.polyfit(centres[selected], np.log10(hist[selected]), 1)[0])
    slope = float(np.clip(slope, 0.02, 0.45))
    # Normalize to the directly counted Gaia bright side, then extrapolate one
    # continuous magnitude law to the configured faint limit.
    beta = slope * math.log(10.0)
    bright_integral = math.expm1(beta * (_GAIA_COUNT_LIMIT_MAG - bright))
    full_integral = math.expm1(beta * (faint - bright))
    # ``slope`` remains a compatibility diagnostic.  The active generator
    # samples the empirical distribution below whenever it is present.
    density = bright_density * full_integral / max(bright_integral, 1e-9)

    colors = np.asarray([
        float(row["bp_rp"]) for row in usable_gaia
        if _finite(row.get("bp_rp")) is not None
        and row.get("central_selected_star") != "1"
    ])
    temperatures = np.asarray([
        float(row["temperature_k"]) for row in usable_gaia
        if _finite(row.get("temperature_k")) is not None
        and row.get("central_selected_star") != "1"
    ])
    total_area = float(meta["area_arcmin2"])
    magnitude_bins = _fixed_width_edges(bright, faint, 0.5)
    euclid_faint_values: list[float] = []
    euclid_faint_weights: list[float] = []
    euclid_faint_by_cone: list[list[tuple[float, float]]] = [
        [] for _ in range(int(meta["cone_count"]))
    ]
    for row, probability in euclid_point_rows:
        magnitude = _finite(row.get("mag_vis"))
        if magnitude is None or not _GAIA_COUNT_LIMIT_MAG < magnitude <= faint:
            continue
        euclid_faint_values.append(magnitude)
        euclid_faint_weights.append(probability)
        cone_index = int(row.get("cone_index") or -1)
        if 0 <= cone_index < len(euclid_faint_by_cone):
            euclid_faint_by_cone[cone_index].append((magnitude, probability))

    gaia_values = np.asarray(comparison_gaia_vis, dtype=np.float64)
    gaia_weights = np.ones(gaia_values.size, dtype=np.float64)
    euclid_values = np.asarray(euclid_faint_values, dtype=np.float64)
    euclid_weights = np.asarray(euclid_faint_weights, dtype=np.float64)
    combined_values = np.concatenate([gaia_values, euclid_values])
    combined_weights = np.concatenate([gaia_weights, euclid_weights])
    combined_counts, _ = np.histogram(
        combined_values, bins=magnitude_bins, weights=combined_weights,
    )
    expected_count = float(np.sum(combined_weights))
    if expected_count > 0.0:
        density = expected_count / total_area
        cdf = np.concatenate([[0.0], np.cumsum(combined_counts) / expected_count])
    else:
        density = float("nan")
        cdf = np.zeros(magnitude_bins.size, dtype=np.float64)
    slope_bins = combined_counts > 0.0
    if np.count_nonzero(slope_bins) >= 4:
        slope = float(np.polyfit(
            0.5 * (magnitude_bins[:-1] + magnitude_bins[1:])[slope_bins],
            np.log10(combined_counts[slope_bins] / np.diff(magnitude_bins)[slope_bins]),
            1,
        )[0])
    slope = float(np.clip(slope, 0.02, 0.45))
    combined_summary = _weighted_summary(
        combined_values, combined_weights, area_arcmin2=total_area,
        classification_variance=float(np.sum(euclid_weights * (1.0 - euclid_weights))),
    )
    per_cone_density: list[float] = []
    for cone_index in range(int(meta["cone_count"])):
        count = float(len(gaia_bright_by_cone[cone_index]))
        count += float(sum(weight for _value, weight in euclid_faint_by_cone[cone_index]))
        per_cone_density.append(count / (total_area / int(meta["cone_count"])))
    per_cone = np.asarray(per_cone_density, dtype=np.float64)
    if len(colors) < 20:
        warnings.append("too few Gaia BP-RP measurements for a stable colour CDF")
    if len(temperatures) < 2:
        warnings.append("too few Gaia temperature estimates for stellar sampling")
    if not mapping:
        warnings.append("no Euclid-Gaia band mapping could be fitted")
    if not math.isfinite(density) or density <= 0:
        warnings.append("invalid probability-weighted point-source density")
    coverage_notes: list[str] = []
    if missing_probability:
        coverage_notes.append(
            f"excluded {missing_probability:,} Euclid rows without point-like probability"
        )
    if invalid_probability:
        coverage_notes.append(
            f"excluded {invalid_probability:,} Euclid rows with invalid point-like probability"
        )

    payload: dict[str, Any] = {
        "version": _STAR_POPULATION_VERSION,
        "kind": "star_population_fit",
        "valid": not warnings,
        "warnings": warnings,
        "coverage_notes": coverage_notes,
        "cone_provenance": {
            "count": int(meta["cone_count"]),
            "radius_arcmin": float(meta["radius_arcmin"]),
            "area_arcmin2": float(meta["area_arcmin2"]),
            "selection_seed": meta.get("euclid_cone_selection_seed"),
            "central_sources_excluded": int(meta["cone_count"]),
        },
        "population": {
            "density_arcmin2": density,
            "bright_gaia_density_arcmin2": bright_density,
            "bright_count_per_cone": density_counts,
            "magnitude_slope": slope,
            "mag_bright": bright,
            "mag_faint": faint,
            "magnitude_distribution": {
                "edges": magnitude_bins.tolist(),
                "cdf": cdf.tolist(),
                "splice_mag": _GAIA_COUNT_LIMIT_MAG,
                "gaia_bright_expected_count": float(gaia_values.size),
                "euclid_faint_expected_count": float(np.sum(euclid_weights)),
            },
            "weighted_statistics": combined_summary,
            "per_cone_density_arcmin2": per_cone_density,
            "per_cone_mean_density_arcmin2": float(np.mean(per_cone)) if per_cone.size else None,
            "per_cone_std_density_arcmin2": float(np.std(per_cone)) if per_cone.size else None,
        },
        "euclid_point_source_weights": {
            "rows": len(euclid_point_rows),
            "missing_probability_rows": missing_probability,
            "invalid_probability_rows": invalid_probability,
            "weight_sum": float(np.sum([weight for _row, weight in euclid_point_rows])),
            "classification_variance": float(np.sum(
                euclid_weights * (1.0 - euclid_weights)
            )),
            "selection": {
                "mag_min": bright,
                "mag_max": faint,
                "faint_component_min_exclusive": _GAIA_COUNT_LIMIT_MAG,
            },
        },
        "gaia": {
            "rows": len(usable_gaia),
            "bp_rp_quantiles": _quantiles(colors),
            "temperature_quantiles_k": _quantiles(temperatures),
        },
        "euclid_mapping": {
            "matched_stars": len(matches),
            "feature_order": ["intercept", "bp_rp", "g_minus_20"],
            "g_to_band_offset_coefficients": mapping,
            "band_order": ["VIS", "Y_E", "J_E", "H_E"],
            "residual_covariance": covariance.tolist(),
        },
    }
    if mapping and colors.size:
        model = EmpiricalStellarPrior.from_payload(payload)
        rng = np.random.default_rng(71033)
        sample_count = 10_000
        sample_mags = np.asarray([
            model.sample_magnitude(
                rng, slope=slope, m_bright=bright, m_faint=faint,
            )
            for _ in range(sample_count)
        ], dtype=np.float64)
        fitted_seds = [model.sample(rng, value) for value in sample_mags]
        fitted_band = {
            name: np.asarray([sed.magnitudes[name] for sed in fitted_seds])
            for name in ("VIS", "Y_E", "J_E", "H_E")
        }
        euclid_color_values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for color_key, left, right in (
            ("vis_y", "mag_vis", "mag_y_e"),
            ("y_j", "mag_y_e", "mag_j_e"),
            ("j_h", "mag_j_e", "mag_h_e"),
        ):
            values: list[float] = []
            weights: list[float] = []
            for row, probability in euclid_point_rows:
                first, second = _finite(row.get(left)), _finite(row.get(right))
                if first is not None and second is not None:
                    values.append(first - second)
                    weights.append(probability)
            euclid_color_values[color_key] = (
                np.asarray(values, dtype=np.float64),
                np.asarray(weights, dtype=np.float64),
            )
        diagnostics: dict[str, Any] = {
            "star_density_per_cone": {
                "x": list(range(1, len(density_counts) + 1)),
                "observed": per_cone_density,
                "fitted": [density] * len(per_cone_density),
                "label": "probability-weighted point-source density per cone",
                "unit": "point sources / arcmin²",
                "statistics": {
                    "mean": float(np.mean(per_cone)) if per_cone.size else None,
                    "std": float(np.std(per_cone)) if per_cone.size else None,
                    "p16": float(np.percentile(per_cone, 16)) if per_cone.size else None,
                    "p50": float(np.percentile(per_cone, 50)) if per_cone.size else None,
                    "p84": float(np.percentile(per_cone, 84)) if per_cone.size else None,
                },
            },
            "parameters": {},
        }
        magnitude_diagnostic = _weighted_histogram(
            combined_values, combined_weights, bins=magnitude_bins,
            area_arcmin2=total_area,
        )
        fitted_counts, _ = np.histogram(fitted_band["VIS"], bins=magnitude_bins)
        magnitude_diagnostic["fitted"] = (
            fitted_counts.astype(np.float64) * density / sample_count
            / np.diff(magnitude_bins)
        ).tolist()
        magnitude_diagnostic["fitted_count"] = sample_count
        magnitude_diagnostic["gaia_bright"] = _weighted_histogram(
            gaia_values, gaia_weights, bins=magnitude_bins,
            area_arcmin2=total_area,
        )["observed"]
        magnitude_diagnostic["euclid_weighted"] = _weighted_histogram(
            euclid_values, euclid_weights, bins=magnitude_bins,
            area_arcmin2=total_area,
        )["observed"]
        magnitude_diagnostic["observed_limit_mag"] = _GAIA_COUNT_LIMIT_MAG
        diagnostics["parameters"]["mag_vis"] = {
            **magnitude_diagnostic,
            "label": "point-source density versus VIS magnitude",
            "unit": "AB mag",
            "density_unit": "point sources / arcmin² / mag",
            "observed_label": "Gaia bright + Euclid probability-weighted faint",
            "gaia_bright_label": "same-footprint Gaia transformed to VIS",
            "euclid_weighted_label": "Euclid weighted by POINT_LIKE_PROB",
            "statistics": combined_summary,
            "extrapolation_note": (
                "Gaia contributes the bright side through G≤20.5; the faint side "
                "is the Euclid point-like probability-weighted population."
            ),
        }
        for key, fitted_values, label in (
            ("vis_y", fitted_band["VIS"] - fitted_band["Y_E"], "VIS − Y"),
            ("y_j", fitted_band["Y_E"] - fitted_band["J_E"], "Y − J"),
            ("j_h", fitted_band["J_E"] - fitted_band["H_E"], "J − H"),
        ):
            observed_values, observed_weights = euclid_color_values[key]
            if observed_values.size == 0:
                continue
            lo = min(float(np.percentile(observed_values, 1)), float(np.percentile(fitted_values, 1)))
            hi = max(float(np.percentile(observed_values, 99)), float(np.percentile(fitted_values, 99)))
            bins = np.linspace(lo, hi, 25)
            weighted = _weighted_histogram(
                observed_values, observed_weights, bins=bins,
                area_arcmin2=float(np.sum(observed_weights)),
            )
            fitted_counts, _ = np.histogram(fitted_values, bins=bins)
            widths = np.diff(bins)
            diagnostics["parameters"][key] = {
                **weighted,
                "fitted": (fitted_counts / sample_count / widths).tolist(),
                "fitted_count": sample_count,
                "label": label,
                "unit": "AB mag",
                "density_unit": "probability density",
                "observed_label": "Euclid point-like probability-weighted",
                "statistics": _weighted_summary(
                    observed_values, observed_weights, area_arcmin2=1.0,
                    classification_variance=float(np.sum(
                        observed_weights * (1.0 - observed_weights)
                    )),
                ),
            }
        fitted_colors = np.asarray([
            float(np.interp(rng.random(), np.linspace(0, 1, colors.size), np.sort(colors)))
            for _ in range(sample_count)
        ])
        diagnostics["parameters"]["bp_rp"] = {
            **_histogram(
                colors, fitted_colors,
                bins=np.linspace(float(np.min(colors)), float(np.max(colors)), 25),
                observed_scale=1.0 / len(colors), fitted_scale=1.0 / sample_count,
            ),
            "label": "Gaia BP − RP", "unit": "mag",
            "density_unit": "probability density",
            "statistics": _weighted_summary(
                colors, np.ones(colors.size), area_arcmin2=1.0,
            ),
        }
        if temperatures.size:
            fitted_temperature = np.asarray([sed.temperature_k for sed in fitted_seds])
            diagnostics["parameters"]["temperature_k"] = {
                **_histogram(
                    temperatures, fitted_temperature,
                    bins=np.linspace(float(np.min(temperatures)), float(np.max(temperatures)), 25),
                    observed_scale=1.0 / len(temperatures), fitted_scale=1.0 / sample_count,
                ),
                "label": "Gaia temperature", "unit": "K",
                "density_unit": "probability density",
                "statistics": _weighted_summary(
                    temperatures, np.ones(temperatures.size), area_arcmin2=1.0,
                ),
            }
        payload["diagnostics"] = diagnostics
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["fingerprint"] = hashlib.sha256(canonical.encode()).hexdigest()
    write_star_candidate(payload)
    return payload
