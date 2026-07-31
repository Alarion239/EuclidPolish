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
    if mapping:
        coeff = np.asarray(mapping["mag_vis"])
        for row in usable_gaia:
            bp_rp = _finite(row.get("bp_rp"))
            if bp_rp is None or row.get("central_selected_star") == "1":
                continue
            g_mag = float(row["g_mag"])
            vis_mag = g_mag + float(np.dot(coeff, [1.0, bp_rp, g_mag - 20.0]))
            predicted_vis.append(vis_mag)
            if g_mag <= _GAIA_COUNT_LIMIT_MAG:
                comparison_gaia_vis.append(vis_mag)
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
    if len(colors) < 20:
        warnings.append("too few Gaia BP-RP measurements for a stable colour CDF")
    if len(temperatures) < 2:
        warnings.append("too few Gaia temperature estimates for stellar sampling")
    if not mapping:
        warnings.append("no Euclid-Gaia band mapping could be fitted")
    if not math.isfinite(density) or density <= 0:
        warnings.append("invalid extrapolated stellar density")

    payload: dict[str, Any] = {
        "version": 1,
        "kind": "star_population_fit",
        "valid": not warnings,
        "warnings": warnings,
        "cone_provenance": {
            "count": int(meta["cone_count"]),
            "radius_arcmin": float(meta["radius_arcmin"]),
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
        beta_sample = slope * math.log(10.0)
        uniform = rng.random(sample_count)
        span = faint - bright
        sample_mags = bright + np.log1p(
            uniform * np.expm1(beta_sample * span)
        ) / beta_sample
        fitted_seds = [model.sample(rng, value) for value in sample_mags]
        fitted_band = {
            name: np.asarray([sed.magnitudes[name] for sed in fitted_seds])
            for name in ("VIS", "Y_E", "J_E", "H_E")
        }
        observed_band = {
            name: np.asarray([
                float(pair[0][key]) for pair in matches
            ])
            for name, key in (
                ("VIS", "mag_vis"), ("Y_E", "mag_y_e"),
                ("J_E", "mag_j_e"), ("H_E", "mag_h_e"),
            )
        }
        euclid_vis_lower_bound = np.asarray([
            value
            for row in euclid_rows
            if row.get("type") == "star"
            and (value := _finite(row.get("mag_vis"))) is not None
        ], dtype=np.float64)
        total_area = float(meta["area_arcmin2"])
        diagnostics: dict[str, Any] = {
            "star_density_per_cone": {
                "x": list(range(1, len(density_counts) + 1)),
                "observed": [count / area_per_cone for count in density_counts],
                "fitted": [bright_density] * len(density_counts),
                "label": "Gaia G≤20.5 star density",
                "unit": "stars / arcmin²",
            },
            "parameters": {},
        }
        magnitude_diagnostic = _histogram(
            np.asarray(comparison_gaia_vis, dtype=np.float64), fitted_band["VIS"],
            bins=_fixed_width_edges(bright, faint, 0.5),
            observed_scale=1.0 / total_area,
            fitted_scale=density / sample_count,
        )
        magnitude_centres = magnitude_diagnostic["x"]
        # G=20.5 is the density-normalisation boundary, not evidence that an
        # observed zero exists in every fainter VIS bin.  Nulls leave that
        # portion of the Gaia histogram explicitly unobserved in the plot.
        magnitude_diagnostic["observed"] = [
            value if centre <= _GAIA_COUNT_LIMIT_MAG else None
            for centre, value in zip(
                magnitude_centres, magnitude_diagnostic["observed"], strict=True,
            )
        ]
        euclid_counts, euclid_edges = np.histogram(
            euclid_vis_lower_bound,
            bins=_fixed_width_edges(bright, faint, 0.5),
        )
        magnitude_diagnostic["euclid_lower_bound"] = (
            euclid_counts / total_area / np.diff(euclid_edges)
        ).tolist()
        magnitude_diagnostic["euclid_lower_bound_count"] = int(
            euclid_vis_lower_bound.size
        )
        magnitude_diagnostic["observed_limit_mag"] = _GAIA_COUNT_LIMIT_MAG
        diagnostics["parameters"]["mag_vis"] = {
            **magnitude_diagnostic,
            "label": "stellar density versus VIS magnitude",
            "unit": "AB mag",
            "density_unit": "stars / arcmin² / mag",
            "observed_label": "same-footprint Gaia transformed to VIS",
            "euclid_lower_bound_label": "Euclid high-purity point-like lower bound",
            "extrapolation_note": (
                "The fitted prior beyond the Gaia G≤20.5 boundary is an "
                "extrapolation; Euclid point-like flags are high-purity but incomplete."
            ),
        }
        for key, observed_values, fitted_values, label in (
            ("vis_y", observed_band["VIS"] - observed_band["Y_E"],
             fitted_band["VIS"] - fitted_band["Y_E"], "VIS − Y"),
            ("y_j", observed_band["Y_E"] - observed_band["J_E"],
             fitted_band["Y_E"] - fitted_band["J_E"], "Y − J"),
            ("j_h", observed_band["J_E"] - observed_band["H_E"],
             fitted_band["J_E"] - fitted_band["H_E"], "J − H"),
        ):
            lo = min(float(np.percentile(observed_values, 1)), float(np.percentile(fitted_values, 1)))
            hi = max(float(np.percentile(observed_values, 99)), float(np.percentile(fitted_values, 99)))
            diagnostics["parameters"][key] = {
                **_histogram(
                    observed_values, fitted_values,
                    bins=np.linspace(lo, hi, 25),
                    observed_scale=1.0 / max(len(observed_values), 1),
                    fitted_scale=1.0 / sample_count,
                ),
                "label": label,
                "unit": "AB mag",
                "density_unit": "probability density",
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
            }
        payload["diagnostics"] = diagnostics
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["fingerprint"] = hashlib.sha256(canonical.encode()).hexdigest()
    write_star_candidate(payload)
    return payload
