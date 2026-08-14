"""Q1 PHZ number counts plus matched Gaia--Euclid stellar colours."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from euclid_polish.config import Config
from euclid_polish.population.magnitude_law import (
    StraightMagnitudeLaw,
    fit_shared_slope,
    fit_straight_region,
)
from euclid_polish.sky.generation.stellar_sed import EmpiricalStellarPrior
from euclid_polish.web.helpers.population_calibration import (
    star_candidate_path,
    write_star_candidate,
)
from euclid_polish.web.helpers.population_comparison import (
    FIELD_AREA_ARCMIN2,
    _synthetic_paths,
)
from euclid_polish.web.helpers.q1_star_counts import (
    q1_star_counts_path,
    read_q1_phz_star_counts,
)
from euclid_polish.web.helpers.q1_stellar_colors import (
    Q1_STELLAR_COLOR_FIELD_RADIUS_DEG,
    Q1_STELLAR_COLOR_FIELDS,
    Q1_STELLAR_COLOR_SAMPLE_VERSION,
    q1_gaia_color_catalog_path,
    q1_gaia_color_meta_path,
)
from euclid_polish.web.helpers.q1_stellar_colors import (
    q1_stellar_color_catalog_path as euclid_catalog_path,
)
from euclid_polish.web.helpers.q1_stellar_colors import (
    q1_stellar_color_meta_path as euclid_catalog_meta_path,
)

_GAIA_COUNT_LIMIT_MAG = 20.5
# Gaia (E)DR3 G-band AB and Vega zero points are 25.8010446445 and
# 25.6873668671 mag, respectively.  The archive's phot_g_mean_mag is Vega.
_GAIA_G_AB_MINUS_VEGA_MAG = 25.8010446445 - 25.6873668671
_GAIA_TAP_PROVIDER = "ARI Gaia TAP"
_STAR_POPULATION_VERSION = 6
_STAR_DISTRIBUTION_VERSION = 11
_GAIA_COUNT_FIT_BIN_WIDTH_MAG = 0.5


def gaia_catalog_path() -> Path:
    return q1_gaia_color_catalog_path()


def gaia_catalog_meta_path() -> Path:
    return q1_gaia_color_meta_path()


def star_distribution_path() -> Path:
    return (
        Path(Config.DATA_DIR) / "population_comparison"
        / "star_distribution.json"
    )


def _finite(value: Any) -> float | None:
    if np.ma.is_masked(value):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _require_current_gaia_field_sampling(
    meta: dict[str, Any], euclid_rows: list[dict[str, str]],
) -> None:
    """Reject legacy random-centre caches and incomplete fixed-Q1 samples."""
    del euclid_rows
    if (
        meta.get("sampling_kind")
        != "fixed_q1_magnitude_stratified_color_fields"
        or int(meta.get("version") or 0) != Q1_STELLAR_COLOR_SAMPLE_VERSION
        or int(meta.get("field_count") or 0) != len(Q1_STELLAR_COLOR_FIELDS)
        or meta.get("tap_provider") != _GAIA_TAP_PROVIDER
        or meta.get("query_mode") != "sync"
        or meta.get("random_centres") is not False
        or not math.isclose(
            float(meta.get("radius_deg") or 0.0),
            Q1_STELLAR_COLOR_FIELD_RADIUS_DEG,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(
            "Stellar colour cache is stale; press Query MER + PHZ to refresh "
            "the fixed Q1 magnitude-stratified sample"
        )
    expected = Q1_STELLAR_COLOR_FIELDS
    cached = meta.get("fields") or []
    if len(cached) != len(expected):
        raise ValueError("Fixed Q1 stellar-colour field provenance is incomplete")
    for (wanted_ra, wanted_dec, wanted_name), found in zip(
        expected, cached, strict=True,
    ):
        ra_delta = abs(wanted_ra - float(found["ra"]))
        ra_delta = min(ra_delta, 360.0 - ra_delta)
        if ra_delta > 1e-9 or not math.isclose(
            wanted_dec, float(found["dec"]),
            rel_tol=0.0, abs_tol=1e-9,
        ) or str(found.get("name")) != wanted_name:
            raise ValueError(
                "Stellar colour cache no longer matches the fixed Q1 fields; "
                "press Query MER + PHZ again"
            )


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


def _density_series(
    values: np.ndarray,
    bins: np.ndarray,
    *,
    area_arcmin2: float | None = None,
    total_density_arcmin2: float | None = None,
) -> list[float]:
    """Histogram values as an area density or a density-scaled model draw."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    counts, _ = np.histogram(finite, bins=bins)
    widths = np.diff(bins)
    if total_density_arcmin2 is not None:
        scale = float(total_density_arcmin2) / max(finite.size, 1)
    else:
        scale = 1.0 / max(float(area_arcmin2 or 0.0), 1e-20)
    return (counts.astype(np.float64) * scale / widths).tolist()


def _shared_color_edges(
    populations: list[np.ndarray], *, bin_count: int = 40,
) -> np.ndarray:
    finite = [
        values[np.isfinite(values)]
        for values in populations
        if values.size
    ]
    combined = np.concatenate(finite) if finite else np.asarray([], dtype=np.float64)
    if combined.size < 2:
        return np.linspace(-1.0, 1.0, bin_count + 1)
    lower, upper = np.quantile(combined, [0.005, 0.995])
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        lower, upper = float(np.min(combined)), float(np.max(combined))
    if upper <= lower:
        lower, upper = lower - 0.5, upper + 0.5
    padding = 0.03 * (upper - lower)
    return np.linspace(lower - padding, upper + padding, bin_count + 1)


def _stellar_density_comparison(
    euclid_rows: list[dict[str, str]],
    gaia_rows: list[dict[str, str]],
    projected: dict[str, Any],
    stellar_model: dict[str, Any],
    *,
    euclid_area_arcmin2: float,
    gaia_area_arcmin2: float,
    synthetic_rows: list[dict[str, str]] | None = None,
    synthetic_area_arcmin2: float = 0.0,
    sample_count: int = 50_000,
) -> dict[str, Any] | None:
    """Compare measured, Gaia-projected, and generator stellar densities."""
    if (
        euclid_area_arcmin2 <= 0.0 or gaia_area_arcmin2 <= 0.0
        or sample_count <= 0
    ):
        return None
    try:
        prior = EmpiricalStellarPrior.from_payload(stellar_model)
        model_density = float(stellar_model["population"]["density_arcmin2"])
        model_magnitude_law = StraightMagnitudeLaw.from_payload(
            stellar_model["population"]["magnitude_distribution"]
        )
        bright = float(stellar_model["population"]["mag_bright"])
        faint = float(stellar_model["population"]["mag_faint"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(model_density) or model_density <= 0.0 or faint <= bright:
        return None

    fingerprint = str(stellar_model.get("fingerprint") or "stellar-density")
    seed = int(hashlib.sha256(fingerprint.encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    model_vis = np.asarray([
        prior.sample_magnitude(
            rng, slope=0.0, m_bright=bright, m_faint=faint,
        )
        for _ in range(sample_count)
    ], dtype=np.float64)
    model_seds = [prior.sample(rng, magnitude) for magnitude in model_vis]
    model_bands = {
        name: np.asarray([sed.magnitudes[name] for sed in model_seds])
        for name in ("VIS", "Y_E", "J_E", "H_E")
    }

    euclid_fields = {
        "VIS": "mag_vis", "Y_E": "mag_y_e",
        "J_E": "mag_j_e", "H_E": "mag_h_e",
    }
    euclid_vis: list[float] = []
    euclid_color_rows: list[dict[str, float]] = []
    for row in euclid_rows:
        probability = _finite(row.get("point_like_prob"))
        vis = _finite(row.get("mag_vis"))
        if (
            row.get("type") != "star" or probability is None
            or probability < 0.9 or vis is None
        ):
            continue
        euclid_vis.append(vis)
        magnitudes = {
            name: _finite(row.get(field))
            for name, field in euclid_fields.items()
        }
        if all(value is not None for value in magnitudes.values()):
            euclid_color_rows.append({
                name: float(value) for name, value in magnitudes.items()
                if value is not None
            })

    gaia_vis = np.asarray(
        projected["matched"]["vis_mag"] + projected["unmatched"]["vis_mag"],
        dtype=np.float64,
    )
    gaia_projected = {
        key: np.asarray(
            projected["matched"]["colors"][key]
            + projected["unmatched"]["colors"][key],
            dtype=np.float64,
        )
        for key in ("vis_y", "vis_j", "vis_h")
    }
    gaia_colors = {
        **gaia_projected,
        "y_j": gaia_projected["vis_j"] - gaia_projected["vis_y"],
        "y_h": gaia_projected["vis_h"] - gaia_projected["vis_y"],
        "j_h": gaia_projected["vis_h"] - gaia_projected["vis_j"],
    }
    model_colors = {
        "vis_y": model_bands["VIS"] - model_bands["Y_E"],
        "vis_j": model_bands["VIS"] - model_bands["J_E"],
        "vis_h": model_bands["VIS"] - model_bands["H_E"],
        "y_j": model_bands["Y_E"] - model_bands["J_E"],
        "y_h": model_bands["Y_E"] - model_bands["H_E"],
        "j_h": model_bands["J_E"] - model_bands["H_E"],
    }
    euclid_colors = {
        "vis_y": np.asarray([row["VIS"] - row["Y_E"] for row in euclid_color_rows]),
        "vis_j": np.asarray([row["VIS"] - row["J_E"] for row in euclid_color_rows]),
        "vis_h": np.asarray([row["VIS"] - row["H_E"] for row in euclid_color_rows]),
        "y_j": np.asarray([row["Y_E"] - row["J_E"] for row in euclid_color_rows]),
        "y_h": np.asarray([row["Y_E"] - row["H_E"] for row in euclid_color_rows]),
        "j_h": np.asarray([row["J_E"] - row["H_E"] for row in euclid_color_rows]),
    }
    synthetic_bands = {
        name: np.asarray([
            value
            for row in (synthetic_rows or [])
            if str(row.get("type", "")).strip().lower() == "star"
            if (value := _finite(row.get(field))) is not None
        ], dtype=np.float64)
        for name, field in euclid_fields.items()
    }
    synthetic_complete = []
    for row in synthetic_rows or []:
        if str(row.get("type", "")).strip().lower() != "star":
            continue
        magnitudes = {
            name: _finite(row.get(field))
            for name, field in euclid_fields.items()
        }
        if all(value is not None for value in magnitudes.values()):
            synthetic_complete.append({
                name: float(value) for name, value in magnitudes.items()
                if value is not None
            })
    synthetic_colors = {
        "vis_y": np.asarray([row["VIS"] - row["Y_E"] for row in synthetic_complete]),
        "vis_j": np.asarray([row["VIS"] - row["J_E"] for row in synthetic_complete]),
        "vis_h": np.asarray([row["VIS"] - row["H_E"] for row in synthetic_complete]),
        "y_j": np.asarray([row["Y_E"] - row["J_E"] for row in synthetic_complete]),
        "y_h": np.asarray([row["Y_E"] - row["H_E"] for row in synthetic_complete]),
        "j_h": np.asarray([row["J_E"] - row["H_E"] for row in synthetic_complete]),
    }
    labels = {key: label for key, label, _left, _right in _STAR_COLOR_PAIRS}

    try:
        q1_counts = read_q1_phz_star_counts(bright=bright, faint=faint)
    except ValueError:
        q1_counts = None
    magnitude_edges = (
        np.asarray(q1_counts["edges"], dtype=np.float64)
        if q1_counts is not None
        else _fixed_width_edges(bright, faint, 0.5)
    )
    q1_density = (
        [float(item["density_arcmin2_mag"]) for item in q1_counts["bins"]]
        if q1_counts is not None else None
    )
    q1_point_source_density = (
        [
            float(item["point_source_density_arcmin2_mag"])
            for item in q1_counts["bins"]
        ]
        if q1_counts is not None else None
    )
    gaia_g_ab = np.asarray([
        g_mag + _GAIA_G_AB_MINUS_VEGA_MAG
        for row in gaia_rows
        if str(row.get("central_selected_star") or "0").strip() != "1"
        and (g_mag := _finite(row.get("g_mag"))) is not None
    ], dtype=np.float64)
    magnitude_diagnostics = (
        stellar_model["population"]["magnitude_distribution"].get(
            "fit_diagnostics", {}
        )
    )
    gaia_fit_diagnostics = magnitude_diagnostics.get("gaia") or {}
    q1_fit_diagnostics = magnitude_diagnostics.get("q1") or {}
    gaia_bin_width = float(
        gaia_fit_diagnostics.get("bin_width_mag")
        or _GAIA_COUNT_FIT_BIN_WIDTH_MAG
    )
    gaia_edges = _fixed_width_edges(bright, faint, gaia_bin_width)
    gaia_centres = 0.5 * (gaia_edges[:-1] + gaia_edges[1:])
    gaia_fit_density = None
    try:
        gaia_intercept = float(gaia_fit_diagnostics["intercept"])
        gaia_fit_density = np.power(
            10.0,
            model_magnitude_law.slope
            * gaia_centres
            + gaia_intercept,
        ).tolist()
    except (KeyError, TypeError, ValueError):
        pass
    parameters: dict[str, Any] = {
        "vis": {
            "label": "VIS and native Gaia G brightness",
            "x_label": "apparent magnitude [AB]",
            "x": (0.5 * (magnitude_edges[:-1] + magnitude_edges[1:])).tolist(),
            "x_domain": [bright, faint],
            "euclid": q1_density if q1_density is not None else _density_series(
                np.asarray(euclid_vis), magnitude_edges,
                area_arcmin2=euclid_area_arcmin2,
            ),
            "point_sources": q1_point_source_density,
            # This is the native Gaia G band on the AB system, with only the
            # release zero-point conversion. It is not projected into VIS.
            "gaia_x": gaia_centres.tolist(),
            "gaia": _density_series(
                gaia_g_ab, gaia_edges, area_arcmin2=gaia_area_arcmin2,
            ),
            "model": model_magnitude_law.density(
                0.5 * (magnitude_edges[:-1] + magnitude_edges[1:])
            ).tolist(),
            "synthetic": _density_series(
                synthetic_bands["VIS"], magnitude_edges,
                area_arcmin2=synthetic_area_arcmin2,
            ) if synthetic_area_arcmin2 > 0.0 else [],
            "gaia_fit": gaia_fit_density,
            "fit_ranges": {
                "q1": [
                    q1_fit_diagnostics.get("fit_bright"),
                    q1_fit_diagnostics.get("fit_faint"),
                ],
                "gaia": [
                    gaia_fit_diagnostics.get("fit_bright"),
                    gaia_fit_diagnostics.get("fit_faint"),
                ],
            },
        },
    }
    for key in ("vis_y", "vis_j", "vis_h", "y_j", "y_h", "j_h"):
        edges = _shared_color_edges([
            euclid_colors[key], gaia_colors[key], model_colors[key],
            synthetic_colors[key],
        ])
        parameters[key] = {
            "label": labels[key],
            "x_label": f"{labels[key]} [AB mag]",
            "x": (0.5 * (edges[:-1] + edges[1:])).tolist(),
            "x_domain": [float(edges[0]), float(edges[-1])],
            "euclid": _density_series(
                euclid_colors[key], edges,
                area_arcmin2=euclid_area_arcmin2,
            ),
            "gaia": _density_series(
                gaia_colors[key], edges,
                area_arcmin2=gaia_area_arcmin2,
            ),
            "model": _density_series(
                model_colors[key], edges, total_density_arcmin2=model_density,
            ),
            "synthetic": _density_series(
                synthetic_colors[key], edges,
                area_arcmin2=synthetic_area_arcmin2,
            ) if synthetic_area_arcmin2 > 0.0 else [],
        }
    return {
        "area_arcmin2": euclid_area_arcmin2,
        "gaia_area_arcmin2": gaia_area_arcmin2,
        "model_density_arcmin2": model_density,
        "model_sample_count": sample_count,
        "euclid_vis_count": len(euclid_vis),
        "q1_phz_expected_stars": (
            float(q1_counts["expected_stars"])
            if q1_counts is not None else None
        ),
        "q1_expected_point_sources": (
            float(q1_counts["expected_point_sources"])
            if q1_counts is not None else None
        ),
        "q1_selected_point_sources": (
            int(q1_counts["selected_point_sources"])
            if q1_counts is not None else None
        ),
        "q1_area_arcmin2": (
            float(q1_counts["footprint_area_arcmin2"])
            if q1_counts is not None else None
        ),
        "euclid_color_count": len(euclid_color_rows),
        "gaia_count": int(gaia_vis.size),
        "gaia_native_g_count": int(gaia_g_ab.size),
        "synthetic_area_arcmin2": synthetic_area_arcmin2 or None,
        "synthetic_star_count": int(synthetic_bands["VIS"].size),
        "synthetic_color_count": len(synthetic_complete),
        "parameters": parameters,
        "note": (
            (
                "The VIS curves show the Q1-wide sums of POINT_LIKE_PROB "
                "and PHZ_STAR_PROB, each divided by the 63.1 deg² "
                "deep-field footprint and magnitude-bin width. "
                if q1_counts is not None else
                "The Q1 count cache is not available; the VIS curve is "
                "only the local high-purity Euclid diagnostic. "
            )
            + "The magnitude panel also shows native Gaia G converted from "
            "the archive Vega magnitude to AB with the Gaia (E)DR3 zero-point "
            "offset and divided by the fixed Q1 Gaia-field area. Gaia and Q1 are "
            "fit over independently selected straight regions with one shared "
            "slope and separate intercepts; the Q1 intercept normalizes the "
            "12–25 VIS generator. Colour curves "
            "remain matched-field fit diagnostics; model curves are intrinsic "
            "generator draws without Euclid measurement noise. Magenta points are "
            "the actual stars stored in the current synthetic test and validation "
            "source catalogues, normalized by their rendered field area."
        ),
    }


_LATENT_NODE_COUNT = 17
_LATENT_COLOR_ORDER = ("vis_y", "y_j", "j_h")
_LATENT_NU = 4.0


def _positive_semidefinite_covariance(
    values: np.ndarray, *, floor: float,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    projected = (
        (eigenvectors * np.maximum(eigenvalues, float(floor)))
        @ eigenvectors.T
    )
    # Roundoff in the eigensystem reconstruction can leave an antisymmetric
    # component large enough for NumPy's covariance validity check to warn.
    return 0.5 * (projected + projected.T)


def _draw_zero_mean_gaussian(
    rng: np.random.Generator, covariance: np.ndarray,
) -> np.ndarray:
    """Draw from an explicitly projected PSD covariance."""
    covariance = _positive_semidefinite_covariance(covariance, floor=0.0)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    return eigenvectors @ (
        np.sqrt(np.maximum(eigenvalues, 0.0))
        * rng.standard_normal(covariance.shape[0])
    )


def _raw_measurement(row: dict[str, str], band: str) -> tuple[float, float] | None:
    flux = _finite(row.get(f"flux_{band}_aper_uJy"))
    error = _finite(row.get(f"fluxerr_{band}_aper_uJy"))
    if flux is None or error is None or error <= 0.0:
        return None
    return float(flux), float(error)


_STAR_COLOR_PAIRS = (
    ("vis_y", "VIS − Y", "mag_vis", "mag_y_e"),
    ("vis_j", "VIS − J", "mag_vis", "mag_j_e"),
    ("vis_h", "VIS − H", "mag_vis", "mag_h_e"),
    ("y_j", "Y − J", "mag_y_e", "mag_j_e"),
    ("y_h", "Y − H", "mag_y_e", "mag_h_e"),
    ("j_h", "J − H", "mag_j_e", "mag_h_e"),
)

_STAR_COLOR_PROJECTIONS = {
    "vis_y": np.asarray([1.0, 0.0, 0.0]),
    "vis_j": np.asarray([1.0, 1.0, 0.0]),
    "vis_h": np.asarray([1.0, 1.0, 1.0]),
    "y_j": np.asarray([0.0, 1.0, 0.0]),
    "y_h": np.asarray([0.0, 1.0, 1.0]),
    "j_h": np.asarray([0.0, 0.0, 1.0]),
}


def _distribution_source_signature() -> dict[str, int | None]:
    def modified(path: Path) -> int | None:
        try:
            return path.stat().st_mtime_ns
        except OSError:
            return None

    signature = {
        "euclid_mtime_ns": modified(euclid_catalog_path()),
        "euclid_meta_mtime_ns": modified(euclid_catalog_meta_path()),
        "gaia_mtime_ns": modified(gaia_catalog_path()),
        "gaia_meta_mtime_ns": modified(gaia_catalog_meta_path()),
        "q1_phz_star_counts_mtime_ns": modified(q1_star_counts_path()),
    }
    _records, synthetic_sources = _synthetic_paths()
    signature.update({
        f"synthetic_{path.name}_mtime_ns": modified(path)
        for path in synthetic_sources
    })
    return signature


def _star_distribution_from_rows(
    euclid_rows: list[dict[str, str]],
    gaia_rows: list[dict[str, str]],
    *,
    calibration_fingerprint: str | None,
    color_model: dict[str, Any] | None = None,
    stellar_model: dict[str, Any] | None = None,
    area_arcmin2: float | None = None,
    gaia_area_arcmin2: float | None = None,
    gaia_sampling: dict[str, Any] | None = None,
    synthetic_rows: list[dict[str, str]] | None = None,
    synthetic_area_arcmin2: float = 0.0,
) -> dict[str, Any]:
    """Build all six measured Euclid colours against matched Gaia BP−RP."""
    euclid_counterpart_ids = {
        str(row.get("gaia_id") or "").strip()
        for row in euclid_rows
        if str(row.get("gaia_id") or "").strip()
    }
    cmd_matched_bp_rp: list[float] = []
    cmd_matched_g: list[float] = []
    cmd_unmatched_bp_rp: list[float] = []
    cmd_unmatched_g: list[float] = []
    cmd_without_color = 0
    for row in gaia_rows:
        bp_rp_value = _finite(row.get("bp_rp"))
        g_value = _finite(row.get("g_mag"))
        if bp_rp_value is None or g_value is None:
            cmd_without_color += 1
            continue
        if str(row.get("source_id")) in euclid_counterpart_ids:
            cmd_matched_bp_rp.append(bp_rp_value)
            cmd_matched_g.append(g_value)
        else:
            cmd_unmatched_bp_rp.append(bp_rp_value)
            cmd_unmatched_g.append(g_value)
    cmd_bp_rp = np.asarray(
        cmd_matched_bp_rp + cmd_unmatched_bp_rp,
        dtype=np.float64,
    )
    cmd_g = np.asarray(cmd_matched_g + cmd_unmatched_g, dtype=np.float64)
    cmd_x_domain = (
        [float(value) for value in np.quantile(cmd_bp_rp, [0.005, 0.995])]
        if cmd_bp_rp.size else [0.0, 1.0]
    )
    cmd_g_domain = (
        [float(value) for value in np.quantile(cmd_g, [0.005, 0.995])]
        if cmd_g.size else [0.0, 1.0]
    )
    gaia_by_id = {
        str(row.get("source_id")): row
        for row in gaia_rows
        if _finite(row.get("bp_rp")) is not None
        and _finite(row.get("g_mag")) is not None
        and row.get("central_selected_star") != "1"
    }
    bp_rp: list[float] = []
    colors = {key: [] for key, _label, _left, _right in _STAR_COLOR_PAIRS}
    high_quality = 0
    pointlike_over_09 = 0
    g_to_vis_bp: list[float] = []
    g_to_vis_offsets: list[float] = []
    for row in euclid_rows:
        gaia = gaia_by_id.get(str(row.get("gaia_id") or "").strip())
        if gaia is None or row.get("type") != "star":
            continue
        measurements = [
            _raw_measurement(row, band) for band in ("vis", "y", "j", "h")
        ]
        if any(item is None for item in measurements):
            continue
        magnitudes = {
            key: _finite(row.get(key))
            for key in ("mag_vis", "mag_y_e", "mag_j_e", "mag_h_e")
        }
        if any(value is None for value in magnitudes.values()):
            continue
        bp_rp.append(float(gaia["bp_rp"]))
        for key, _label, left, right in _STAR_COLOR_PAIRS:
            colors[key].append(float(magnitudes[left] - magnitudes[right]))
        signal_to_noise = [
            abs(flux / error)
            for measurement in measurements
            for flux, error in [measurement]  # type: ignore[misc]
        ]
        is_high_quality = min(signal_to_noise) >= 5.0
        high_quality += int(is_high_quality)
        if is_high_quality:
            g_to_vis_bp.append(float(gaia["bp_rp"]))
            g_to_vis_offsets.append(
                float(magnitudes["mag_vis"]) - float(gaia["g_mag"])
            )
        probability = _finite(row.get("point_like_prob"))
        pointlike_over_09 += int(probability is not None and probability >= 0.9)

    x = np.asarray(bp_rp, dtype=np.float64)
    x_domain = (
        [float(value) for value in np.quantile(x, [0.005, 0.995])]
        if x.size else [0.0, 1.0]
    )
    fit_nodes = np.asarray(
        color_model.get("bp_rp_nodes", []) if color_model else [],
        dtype=np.float64,
    )
    fit_locus = np.asarray(
        color_model.get("locus_colors", []) if color_model else [],
        dtype=np.float64,
    )
    fit_covariance = np.asarray(
        color_model.get("intrinsic_color_covariance", []) if color_model else [],
        dtype=np.float64,
    )
    has_fit = (
        fit_nodes.ndim == 1
        and fit_nodes.size >= 2
        and fit_locus.shape == (fit_nodes.size, 3)
        and fit_covariance.shape == (3, 3)
        and np.all(np.isfinite(fit_nodes))
        and np.all(np.isfinite(fit_locus))
        and np.all(np.isfinite(fit_covariance))
    )
    fit_edges = np.asarray(
        color_model.get("bp_rp_edges", []) if color_model else [],
        dtype=np.float64,
    )
    g_to_vis_locus = np.asarray(
        color_model.get("g_to_vis_offset", []) if color_model else [],
        dtype=np.float64,
    )
    if (
        has_fit
        and g_to_vis_locus.shape != fit_nodes.shape
        and fit_edges.shape == (fit_nodes.size + 1,)
        and g_to_vis_offsets
    ):
        offset_bp = np.asarray(g_to_vis_bp, dtype=np.float64)
        offset_values = np.asarray(g_to_vis_offsets, dtype=np.float64)
        fallback_offset = float(np.median(offset_values))
        g_to_vis_locus = np.asarray([
            float(np.median(offset_values[
                (offset_bp >= fit_edges[index])
                & (offset_bp <= fit_edges[index + 1])
            ]))
            if np.any(
                (offset_bp >= fit_edges[index])
                & (offset_bp <= fit_edges[index + 1])
            ) else fallback_offset
            for index in range(fit_nodes.size)
        ])
    plot_payload: dict[str, Any] = {}
    for key, label, _left, _right in _STAR_COLOR_PAIRS:
        y = np.asarray(colors[key], dtype=np.float64)
        y_domain = (
            [float(value) for value in np.quantile(y, [0.005, 0.995])]
            if y.size else [0.0, 1.0]
        )
        correlation = (
            float(np.corrcoef(x, y)[0, 1])
            if x.size > 1 and np.std(x) > 0.0 and np.std(y) > 0.0 else None
        )
        trend_x: list[float] = []
        trend_y: list[float] = []
        edges = np.linspace(x_domain[0], x_domain[1], 19)
        for index in range(edges.size - 1):
            selected = (
                (x >= edges[index])
                & (x <= edges[index + 1] if index == edges.size - 2
                   else x < edges[index + 1])
            )
            if np.count_nonzero(selected) < 8:
                continue
            trend_x.append(float(0.5 * (edges[index] + edges[index + 1])))
            trend_y.append(float(np.median(y[selected])))
        fit = None
        if has_fit:
            projection = _STAR_COLOR_PROJECTIONS[key]
            center = fit_locus @ projection
            variance = float(projection @ fit_covariance @ projection)
            sigma = float(np.sqrt(max(variance, 0.0)))
            fit = {
                "x": fit_nodes.tolist(),
                "center": center.tolist(),
                "sigma": sigma,
                "one_sigma_low": (center - sigma).tolist(),
                "one_sigma_high": (center + sigma).tolist(),
                "two_sigma_low": (center - 2.0 * sigma).tolist(),
                "two_sigma_high": (center + 2.0 * sigma).tolist(),
            }
        plot_payload[key] = {
            "label": label,
            "values": y.tolist(),
            "pearson_r": correlation,
            "y_domain": y_domain,
            "trend": {"x": trend_x, "y": trend_y},
            "fit": fit,
        }

    projection_payload = None
    density_comparison = None
    if has_fit and g_to_vis_locus.shape == fit_nodes.shape:
        projection_colors = {
            key: fit_locus @ _STAR_COLOR_PROJECTIONS[key]
            for key in ("vis_y", "vis_j", "vis_h")
        }
        projected = {
            "matched": {"vis_mag": [], "colors": {
                key: [] for key in projection_colors
            }},
            "unmatched": {"vis_mag": [], "colors": {
                key: [] for key in projection_colors
            }},
        }
        for row in gaia_rows:
            bp_rp_value = _finite(row.get("bp_rp"))
            g_value = _finite(row.get("g_mag"))
            if bp_rp_value is None or g_value is None:
                continue
            group = (
                "matched"
                if str(row.get("source_id")) in euclid_counterpart_ids
                else "unmatched"
            )
            predicted_vis = g_value + float(np.interp(
                bp_rp_value, fit_nodes, g_to_vis_locus,
            ))
            projected[group]["vis_mag"].append(predicted_vis)
            for key, color_locus in projection_colors.items():
                projected[group]["colors"][key].append(float(np.interp(
                    bp_rp_value, fit_nodes, color_locus,
                )))
        projected_vis = np.asarray(
            projected["matched"]["vis_mag"]
            + projected["unmatched"]["vis_mag"],
            dtype=np.float64,
        )
        euclid_observed = {
            key: {"vis_mag": [], "color": []}
            for key in projection_colors
        }
        euclid_vis_for_domain: list[float] = []
        projection_magnitude_fields = {
            "vis_y": "mag_y_e",
            "vis_j": "mag_j_e",
            "vis_h": "mag_h_e",
        }
        for row in euclid_rows:
            vis_magnitude = _finite(row.get("mag_vis"))
            if row.get("type") != "star" or vis_magnitude is None:
                continue
            has_projection_color = False
            for key, other_field in projection_magnitude_fields.items():
                other_magnitude = _finite(row.get(other_field))
                if other_magnitude is None:
                    continue
                euclid_observed[key]["vis_mag"].append(vis_magnitude)
                euclid_observed[key]["color"].append(
                    vis_magnitude - other_magnitude
                )
                has_projection_color = True
            if has_projection_color:
                euclid_vis_for_domain.append(vis_magnitude)
        combined_vis = np.concatenate([
            projected_vis,
            np.asarray(euclid_vis_for_domain, dtype=np.float64),
        ])
        projection_payload = {
            **projected,
            "euclid_observed": euclid_observed,
            "vis_domain": (
                [float(value) for value in np.quantile(
                    combined_vis, [0.005, 0.995],
                )]
                if combined_vis.size else [0.0, 1.0]
            ),
            "colors": {
                key: {
                    "label": plot_payload[key]["label"],
                    "x_domain": [float(value) for value in np.quantile(
                        np.asarray(
                            projected["matched"]["colors"][key]
                            + projected["unmatched"]["colors"][key]
                            + euclid_observed[key]["color"],
                            dtype=np.float64,
                        ),
                        [0.005, 0.995],
                    )],
                    "sigma": plot_payload[key]["fit"]["sigma"],
                }
                for key in projection_colors
            },
            "note": (
                "Each point is a fit-derived central prediction: Gaia G is "
                "mapped to VIS and Gaia BP−RP is mapped to the three Euclid "
                "colours. The fixed-Q1 Euclid overlay shows measured catalogue "
                "stars with valid VIS and the comparison band. No random "
                "scatter or simulated noise is added to the Gaia projection."
            ),
        }
        if (
            stellar_model is not None and area_arcmin2 is not None
            and gaia_area_arcmin2 is not None
        ):
            density_comparison = _stellar_density_comparison(
                euclid_rows,
                gaia_rows,
                projected,
                stellar_model,
                euclid_area_arcmin2=area_arcmin2,
                gaia_area_arcmin2=gaia_area_arcmin2,
                synthetic_rows=synthetic_rows,
                synthetic_area_arcmin2=synthetic_area_arcmin2,
            )

    return {
        "version": _STAR_DISTRIBUTION_VERSION,
        "calibration_fingerprint": calibration_fingerprint,
        "source_signature": _distribution_source_signature(),
        "matched_stars": int(x.size),
        "high_quality_stars": int(high_quality),
        "pointlike_over_0_9": int(pointlike_over_09),
        "bp_rp": x.tolist(),
        "x_domain": x_domain,
        "colors": plot_payload,
        "gaia_cmd": {
            "cached_stars": len(gaia_rows),
            "plotted_stars": int(cmd_bp_rp.size),
            "without_color": int(cmd_without_color),
            "x_domain": cmd_x_domain,
            "g_domain": cmd_g_domain,
            "matched": {
                "bp_rp": cmd_matched_bp_rp,
                "g_mag": cmd_matched_g,
            },
            "unmatched": {
                "bp_rp": cmd_unmatched_bp_rp,
                "g_mag": cmd_unmatched_g,
            },
            "note": (
                "Matched means that the Gaia source has any cached Euclid "
                "catalogue counterpart. This is broader than the four-band "
                "high-S/N sample used to fit the stellar colours."
            ),
        },
        "euclid_projection": projection_payload,
        "density_comparison": density_comparison,
        "gaia_sampling": gaia_sampling,
        "axis_note": (
            "Axes show the central 99%; all stars are retained and outliers "
            "are clipped to the plot boundary."
        ),
        "fit_note": (
            "The centre follows the Gaia BP−RP locus fitted to all-band "
            "S/N ≥ 5 matches. The 1σ and 2σ bands show fitted intrinsic "
            "colour scatter after subtracting Euclid flux-error covariance."
        ) if has_fit else None,
    }


def _write_star_distribution(payload: dict[str, Any]) -> None:
    output = star_distribution_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")))
    os.replace(temporary, output)


def star_distribution_payload() -> dict[str, Any] | None:
    """Read the matched-colour cache, rebuilding it when catalogues changed."""
    try:
        candidate = json.loads(star_candidate_path().read_text())
    except (OSError, json.JSONDecodeError):
        candidate = None
    fingerprint = (
        str(candidate.get("fingerprint"))
        if isinstance(candidate, dict) and candidate.get("fingerprint") else None
    )
    try:
        cached = json.loads(star_distribution_path().read_text())
    except (OSError, json.JSONDecodeError):
        cached = None
    signature = _distribution_source_signature()
    if (
        isinstance(cached, dict)
        and cached.get("version") == _STAR_DISTRIBUTION_VERSION
        and cached.get("calibration_fingerprint") == fingerprint
        and cached.get("source_signature") == signature
    ):
        return cached
    if not euclid_catalog_path().is_file() or not gaia_catalog_path().is_file():
        return None
    try:
        catalogue_meta = json.loads(euclid_catalog_meta_path().read_text())
        area_arcmin2 = float(catalogue_meta.get("area_arcmin2") or 0.0)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        area_arcmin2 = 0.0
    try:
        gaia_meta = json.loads(gaia_catalog_meta_path().read_text())
        gaia_area_arcmin2 = float(gaia_meta.get("area_arcmin2") or 0.0)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        gaia_meta = None
        gaia_area_arcmin2 = 0.0
    _records, synthetic_paths = _synthetic_paths()
    synthetic_rows: list[dict[str, str]] = []
    synthetic_fields = 0
    for path in synthetic_paths:
        rows = _read_rows(path)
        synthetic_rows.extend(rows)
        synthetic_fields += len({
            int(row["field_index"])
            for row in rows
            if str(row.get("field_index", "")).strip()
        })
    payload = _star_distribution_from_rows(
        _read_rows(euclid_catalog_path()),
        _read_rows(gaia_catalog_path()),
        calibration_fingerprint=fingerprint,
        color_model=(
            candidate.get("color_model")
            if isinstance(candidate, dict) else None
        ),
        stellar_model=candidate if isinstance(candidate, dict) else None,
        area_arcmin2=area_arcmin2,
        gaia_area_arcmin2=gaia_area_arcmin2,
        gaia_sampling=gaia_meta,
        synthetic_rows=synthetic_rows,
        synthetic_area_arcmin2=synthetic_fields * FIELD_AREA_ARCMIN2,
    )
    _write_star_distribution(payload)
    return payload


def _gaia_bp_rp_sigma(row: dict[str, str]) -> float:
    bp_flux = _finite(row.get("bp_flux"))
    rp_flux = _finite(row.get("rp_flux"))
    bp_error = _finite(row.get("bp_flux_error"))
    rp_error = _finite(row.get("rp_flux_error"))
    if (
        bp_flux is None or rp_flux is None or bp_error is None
        or rp_error is None or bp_flux <= 0.0 or rp_flux <= 0.0
    ):
        return 0.03
    sigma = 1.0857362047581296 * math.sqrt(
        (bp_error / bp_flux) ** 2 + (rp_error / rp_flux) ** 2,
    )
    return float(max(sigma, 1e-4)) if math.isfinite(sigma) else 0.03


def _color_measurement_covariance(
    row: dict[str, str],
) -> np.ndarray | None:
    measurements = [_raw_measurement(row, band) for band in ("vis", "y", "j", "h")]
    if any(item is None for item in measurements):
        return None
    sigma_mag = np.asarray([
        1.0857362047581296 * abs(error / flux)
        if abs(flux) > 0.0 else np.inf
        for flux, error in measurements  # type: ignore[misc]
    ])
    if not np.all(np.isfinite(sigma_mag)):
        return None
    variances = sigma_mag ** 2
    return np.asarray([
        [variances[0] + variances[1], -variances[1], 0.0],
        [-variances[1], variances[1] + variances[2], -variances[2]],
        [0.0, -variances[2], variances[2] + variances[3]],
    ])


def _normalise_probability_rows(values: np.ndarray, alpha: float) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    values += float(alpha)
    totals = values.sum(axis=1, keepdims=True)
    return np.divide(
        values, totals, out=np.full_like(values, 1.0 / values.shape[1]),
        where=totals > 0.0,
    )


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = np.asarray(values, dtype=np.float64)
    shifted = shifted - np.max(shifted)
    result = np.exp(np.clip(shifted, -700.0, 0.0))
    total = float(np.sum(result))
    return result / total if total > 0.0 else np.full_like(result, 1.0 / result.size)


def _logsumexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    maximum = float(np.max(values))
    return maximum + float(np.log(np.sum(np.exp(values - maximum))))


def _flux_ratios(colors: np.ndarray) -> np.ndarray:
    vis_y, y_j, j_h = np.asarray(colors, dtype=np.float64)
    return np.power(10.0, 0.4 * np.asarray([
        0.0, vis_y, vis_y + y_j, vis_y + y_j + j_h,
    ]))


def _source_node_log_likelihood(
    row: dict[str, str],
    locus_colors: np.ndarray,
    bp_rp_nodes: np.ndarray,
    bp_rp: float | None,
    bp_rp_sigma: float | None,
) -> np.ndarray:
    measurements = [_raw_measurement(row, band) for band in ("vis", "y", "j", "h")]
    valid = np.asarray([item is not None for item in measurements])
    if np.count_nonzero(valid) < 2:
        return np.full(locus_colors.shape[0], -np.inf)
    flux = np.asarray([
        item[0] if item is not None else 0.0 for item in measurements
    ])
    error = np.asarray([
        item[1] if item is not None else 1.0 for item in measurements
    ])
    result = np.empty(locus_colors.shape[0], dtype=np.float64)
    for index, colors in enumerate(locus_colors):
        ratios = _flux_ratios(colors)
        weights = 1.0 / error[valid] ** 2
        amplitude = float(np.sum(flux[valid] * ratios[valid] * weights))
        denominator = float(np.sum(ratios[valid] ** 2 * weights))
        amplitude = max(amplitude / max(denominator, 1e-20), 0.0)
        residual = (flux[valid] - amplitude * ratios[valid]) / error[valid]
        score = -0.5 * (_LATENT_NU + 1.0) * float(
            np.sum(np.log1p(residual ** 2 / _LATENT_NU))
        )
        if bp_rp is not None and bp_rp_sigma is not None and bp_rp_sigma > 0:
            score -= 0.5 * ((bp_rp - bp_rp_nodes[index]) / bp_rp_sigma) ** 2
        result[index] = score
    return result


def _robust_color_covariance(
    residuals: np.ndarray, measurement_covariances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if residuals.shape[0] < 8:
        return np.eye(3, dtype=np.float64) * 0.02 ** 2, np.ones(residuals.shape[0], dtype=bool)
    center = np.median(residuals, axis=0)
    scale = np.maximum(
        1.4826 * np.median(np.abs(residuals - center), axis=0), 0.01,
    )
    keep = np.all(np.abs(residuals - center) <= 4.0 * scale, axis=1)
    clipped = residuals[keep] - np.median(residuals[keep], axis=0)
    covariance = np.cov(clipped, rowvar=False)
    covariance -= np.mean(measurement_covariances[keep], axis=0)
    return _positive_semidefinite_covariance(covariance, floor=0.01 ** 2), keep


def _weighted_color_summary(
    colors: np.ndarray, weights: np.ndarray,
) -> dict[str, Any]:
    if colors.size == 0 or weights.size != colors.shape[0]:
        return {"count": 0, "effective_weight": 0.0}
    weights = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    total = float(np.sum(weights))
    output: dict[str, Any] = {
        "count": int(colors.shape[0]),
        "effective_weight": total,
        "effective_n": float(total * total / max(np.sum(weights ** 2), 1e-20)),
    }
    for index, name in enumerate(_LATENT_COLOR_ORDER):
        values = colors[:, index]
        order = np.argsort(values)
        sorted_values = values[order]
        sorted_weights = weights[order]
        cumulative = np.cumsum(sorted_weights) - 0.5 * sorted_weights
        mean = float(np.sum(values * weights) / max(total, 1e-20))
        output[name] = {
            "mean": mean,
            "std": float(np.sqrt(np.sum(weights * (values - mean) ** 2) / max(total, 1e-20))),
            "p16": float(np.interp(0.16 * total, cumulative, sorted_values)),
            "p50": float(np.interp(0.50 * total, cumulative, sorted_values)),
            "p84": float(np.interp(0.84 * total, cumulative, sorted_values)),
        }
    return output


def _fit_straight_star_magnitude_law(
    gaia_rows: list[dict[str, str]],
    gaia_meta: dict[str, Any],
    q1_counts: dict[str, Any],
) -> tuple[StraightMagnitudeLaw, dict[str, Any]]:
    """Fit a common Gaia-G/Q1-VIS slope with Q1 setting the VIS level."""
    edges = np.asarray(q1_counts["edges"], dtype=np.float64)
    centres = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    gaia_area = float(gaia_meta["area_arcmin2"])
    q1_area = float(q1_counts["footprint_area_arcmin2"])
    gaia_magnitudes = np.asarray([
        float(value) + _GAIA_G_AB_MINUS_VEGA_MAG
        for row in gaia_rows
        if str(row.get("central_selected_star") or "0").strip() != "1"
        and (value := _finite(row.get("g_mag"))) is not None
    ], dtype=np.float64)
    # The fixed Gaia fields contain only a few thousand sources.  Searching
    # for a 2.5-mag straight region in 0.1-mag cells makes Poisson structure,
    # rather than the broad count law, decide whether activation succeeds.
    # Gaia informs the shared slope only, so fit it at a stable 0.5-mag
    # resolution while retaining the native Q1 0.1-mag bins and normalization.
    gaia_edges = np.arange(
        float(Config.STAR_MAG_BRIGHT),
        float(Config.STAR_MAG_FAINT) + 0.5 * _GAIA_COUNT_FIT_BIN_WIDTH_MAG,
        _GAIA_COUNT_FIT_BIN_WIDTH_MAG,
        dtype=np.float64,
    )
    if not math.isclose(
        float(gaia_edges[-1]), float(Config.STAR_MAG_FAINT),
        rel_tol=0.0, abs_tol=1e-9,
    ):
        raise ValueError("Gaia fit limits must divide into 0.5-mag brackets")
    gaia_centres = 0.5 * (gaia_edges[:-1] + gaia_edges[1:])
    gaia_widths = np.diff(gaia_edges)
    gaia_counts, _ = np.histogram(gaia_magnitudes, bins=gaia_edges)
    gaia_density = gaia_counts / (gaia_area * gaia_widths)
    gaia_sigma = np.sqrt(gaia_counts) / (gaia_area * gaia_widths)

    q1_expected = np.asarray([
        float(item["expected_stars"]) for item in q1_counts["bins"]
    ], dtype=np.float64)
    q1_density = q1_expected / (q1_area * widths)
    q1_sigma = np.asarray([
        math.sqrt(
            max(float(item["classified_rows"]), 0.0)
            + max(float(item["classification_variance"]), 0.0)
        ) / q1_area / float(width)
        for item, width in zip(q1_counts["bins"], widths, strict=True)
    ], dtype=np.float64)

    try:
        gaia_region = fit_straight_region(
            gaia_centres, gaia_density, gaia_sigma,
            minimum_span_mag=2.5, minimum_r_squared=0.99,
        )
    except ValueError as exc:
        raise ValueError(
            "Gaia G_AB counts have no 2.5-mag straight region at 0.5-mag "
            "resolution with R² >= 0.99"
        ) from exc
    try:
        q1_region = fit_straight_region(
            centres, q1_density, q1_sigma,
            minimum_span_mag=2.5, minimum_r_squared=0.99,
        )
    except ValueError as exc:
        raise ValueError(
            "Q1 PHZ_STAR_PROB VIS counts have no 2.5-mag straight region "
            "at 0.1-mag resolution with R² >= 0.99"
        ) from exc
    gaia_slice = slice(gaia_region.start, gaia_region.stop)
    q1_slice = slice(q1_region.start, q1_region.stop)
    slope, intercepts, covariance, r_squared, rms = fit_shared_slope([
        (
            gaia_centres[gaia_slice], gaia_density[gaia_slice],
            gaia_sigma[gaia_slice],
        ),
        (
            centres[q1_slice], q1_density[q1_slice],
            q1_sigma[q1_slice],
        ),
    ])
    q1_intercept = intercepts[1]
    law_covariance = (
        (float(covariance[0, 0]), float(covariance[0, 2])),
        (float(covariance[2, 0]), float(covariance[2, 2])),
    )
    law = StraightMagnitudeLaw(
        slope=slope,
        intercept=q1_intercept,
        mag_bright=float(Config.STAR_MAG_BRIGHT),
        mag_faint=float(Config.STAR_MAG_FAINT),
        fit_bright=float(centres[q1_region.start]),
        fit_faint=float(centres[q1_region.stop - 1]),
        covariance=law_covariance,
        r_squared=r_squared,
        rms_log10_density=rms,
        source=(
            "shared slope from native Gaia G_AB and Euclid Q1 PHZ; "
            "Q1 PHZ_STAR_PROB sets VIS normalization"
        ),
    )
    diagnostics = {
        "gaia": {
            "band": "Gaia G_AB",
            "bin_width_mag": _GAIA_COUNT_FIT_BIN_WIDTH_MAG,
            "fit_bright": float(gaia_centres[gaia_region.start]),
            "fit_faint": float(gaia_centres[gaia_region.stop - 1]),
            "r_squared": float(gaia_region.r_squared),
            "rms_log10_density": float(gaia_region.rms),
            "intercept": float(intercepts[0]),
        },
        "q1": {
            "band": "Euclid VIS",
            "bin_width_mag": float(np.median(widths)),
            "fit_bright": float(centres[q1_region.start]),
            "fit_faint": float(centres[q1_region.stop - 1]),
            "r_squared": float(q1_region.r_squared),
            "rms_log10_density": float(q1_region.rms),
            "intercept": float(q1_intercept),
        },
        "shared_slope": float(slope),
        "shared_r_squared": float(r_squared),
        "shared_rms_log10_density": float(rms),
    }
    return law, diagnostics


def _fit_star_population_latent() -> dict[str, Any]:
    """Fit PHZ number counts plus a Gaia-anchored stellar color locus."""
    faint = float(Config.STAR_MAG_FAINT)
    bright = float(Config.STAR_MAG_BRIGHT)
    splice = _GAIA_COUNT_LIMIT_MAG
    gaia_rows = _read_rows(gaia_catalog_path())
    euclid_rows = _read_rows(euclid_catalog_path())
    meta = json.loads(gaia_catalog_meta_path().read_text())
    _require_current_gaia_field_sampling(meta, euclid_rows)
    try:
        euclid_area = float(json.loads(
            euclid_catalog_meta_path().read_text()
        )["area_arcmin2"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        euclid_area = float(meta["area_arcmin2"])
    q1_counts = read_q1_phz_star_counts(bright=bright, faint=faint)
    magnitude_law, magnitude_fit_diagnostics = (
        _fit_straight_star_magnitude_law(gaia_rows, meta, q1_counts)
    )
    gaia_usable = [
        row for row in gaia_rows
        if _finite(row.get("bp_rp")) is not None
        and _finite(row.get("g_mag")) is not None
        and row.get("central_selected_star") != "1"
    ]
    by_id = {str(row.get("source_id")): row for row in gaia_usable}
    matched: list[tuple[dict[str, str], dict[str, str]]] = []
    for row in euclid_rows:
        gaia = by_id.get(str(row.get("gaia_id") or "").strip())
        if gaia is None or row.get("type") != "star":
            continue
        if all(_raw_measurement(row, band) is not None for band in ("vis", "y", "j", "h")):
            matched.append((row, gaia))
    high_quality: list[tuple[dict[str, str], dict[str, str]]] = []
    for row, gaia in matched:
        snr = [
            abs(flux / error)
            for band in ("vis", "y", "j", "h")
            if (measurement := _raw_measurement(row, band)) is not None
            for flux, error in [measurement]
        ]
        if len(snr) == 4 and min(snr) >= 5.0:
            high_quality.append((row, gaia))
    if len(high_quality) < 32:
        raise ValueError("too few raw-flux Gaia-Euclid matches for latent color fit")

    bp_values = np.asarray([float(row["bp_rp"]) for row in gaia_usable])
    quantiles = np.linspace(0.0, 1.0, _LATENT_NODE_COUNT + 1)
    bp_edges = np.quantile(bp_values, quantiles)
    bp_edges = np.maximum.accumulate(bp_edges)
    for index in range(1, bp_edges.size):
        if bp_edges[index] <= bp_edges[index - 1]:
            bp_edges[index] = bp_edges[index - 1] + 1e-6
    bp_nodes = 0.5 * (bp_edges[:-1] + bp_edges[1:])
    all_temperatures = [
        float(row["temperature_k"]) for row in gaia_usable
        if _finite(row.get("temperature_k")) is not None
    ]
    temperature_fallback = (
        float(np.median(all_temperatures)) if all_temperatures else 5800.0
    )
    temperature_nodes = np.asarray([
        float(np.median([
            float(row["temperature_k"])
            for row in gaia_usable
            if _finite(row.get("temperature_k")) is not None
            and bp_edges[index] <= float(row["bp_rp"]) <= bp_edges[index + 1]
        ]))
        if any(
            _finite(row.get("temperature_k")) is not None
            and bp_edges[index] <= float(row["bp_rp"]) <= bp_edges[index + 1]
            for row in gaia_usable
        ) else temperature_fallback
        for index in range(_LATENT_NODE_COUNT)
    ])

    match_bp = np.asarray([float(gaia["bp_rp"]) for _row, gaia in high_quality])
    match_colors = np.asarray([
        [
            float(row["mag_vis"]) - float(row["mag_y_e"]),
            float(row["mag_y_e"]) - float(row["mag_j_e"]),
            float(row["mag_j_e"]) - float(row["mag_h_e"]),
        ]
        for row, _gaia in high_quality
    ])
    locus = np.asarray([
        np.median(match_colors[
            (match_bp >= bp_edges[index]) & (match_bp <= bp_edges[index + 1])
        ], axis=0)
        if np.any((match_bp >= bp_edges[index]) & (match_bp <= bp_edges[index + 1]))
        else np.median(match_colors, axis=0)
        for index in range(_LATENT_NODE_COUNT)
    ])
    expected_color = np.column_stack([
        np.interp(match_bp, bp_nodes, locus[:, index])
        for index in range(3)
    ])
    measurement_covariances = np.asarray([
        _color_measurement_covariance(row)
        for row, _gaia in high_quality
    ])
    intrinsic_covariance, kept = _robust_color_covariance(
        match_colors - expected_color, measurement_covariances,
    )

    vis_g_bp = np.asarray([float(gaia["bp_rp"]) for row, gaia in high_quality])
    vis_g_offsets = np.asarray([
        float(row["mag_vis"]) - float(gaia["g_mag"])
        for row, gaia in high_quality
    ])
    offset_order = np.argsort(vis_g_bp)
    offset_bp = vis_g_bp[offset_order]
    offset_values = vis_g_offsets[offset_order]
    offset_locus = np.asarray([
        float(np.median(offset_values[
            (offset_bp >= bp_edges[index]) & (offset_bp <= bp_edges[index + 1])
        ]))
        if np.any((offset_bp >= bp_edges[index]) & (offset_bp <= bp_edges[index + 1]))
        else float(np.median(offset_values))
        for index in range(_LATENT_NODE_COUNT)
    ])

    magnitude_edges = _fixed_width_edges(bright, faint, 0.5)
    node_weights = np.full(
        (magnitude_edges.size - 1, _LATENT_NODE_COUNT),
        1.0 / _LATENT_NODE_COUNT,
        dtype=np.float64,
    )
    for row in gaia_usable:
        g_mag = _finite(row.get("g_mag"))
        bp_rp = _finite(row.get("bp_rp"))
        if g_mag is None or bp_rp is None or g_mag > splice:
            continue
        vis_mag = g_mag + float(np.interp(bp_rp, bp_nodes, offset_locus))
        if not bright <= vis_mag <= splice:
            continue
        bin_index = int(np.searchsorted(magnitude_edges, vis_mag, side="right") - 1)
        bin_index = max(0, min(bin_index, node_weights.shape[0] - 1))
        node_index = max(0, min(
            int(np.searchsorted(bp_edges, bp_rp, side="right") - 1),
            _LATENT_NODE_COUNT - 1,
        ))
        node_weights[bin_index, node_index] += 1.0
    node_weights = _normalise_probability_rows(node_weights, alpha=0.5)

    euclid_records: list[dict[str, Any]] = []
    missing_flux = 0
    missing_probability = 0
    invalid_probability = 0
    usable_probability_rows = 0
    for row in euclid_rows:
        probability = _finite(row.get("point_like_prob"))
        if probability is None:
            missing_probability += 1
            continue
        if not 0.0 <= probability <= 1.0:
            invalid_probability += 1
            continue
        usable_probability_rows += 1
        measurements = [_raw_measurement(row, band) for band in ("vis", "y", "j", "h")]
        if sum(item is not None for item in measurements) < 2:
            missing_flux += 1
            continue
        euclid_records.append({
            "row": row,
            "probability": probability,
            "bp_rp": float(by_id[str(row.get("gaia_id") or "")]["bp_rp"])
            if str(row.get("gaia_id") or "") in by_id else None,
            "bp_rp_sigma": _gaia_bp_rp_sigma(
                by_id[str(row.get("gaia_id") or "")]
            ) if str(row.get("gaia_id") or "") in by_id else 0.03,
        })

    # The flux likelihood is independent of the magnitude-bin mixture weights.
    # Cache it once; recomputing 17 node scores for every EM iteration makes a
    # large cached fixed-field sample needlessly expensive.
    for record in euclid_records:
        magnitude = _finite(record["row"].get("mag_vis"))
        record["likelihood"] = (
            _source_node_log_likelihood(
                record["row"], locus, bp_nodes,
                record["bp_rp"], record["bp_rp_sigma"],
            )
            if magnitude is not None and splice < magnitude <= faint
            else None
        )

    converged = False
    objective_change = float("inf")
    previous_objective: float | None = None
    for _iteration in range(50):
        accum = np.full_like(node_weights, 0.5)
        objective = 0.0
        for record in euclid_records:
            magnitude = _finite(record["row"].get("mag_vis"))
            if magnitude is None or not splice < magnitude <= faint:
                continue
            bin_index = int(np.searchsorted(magnitude_edges, magnitude, side="right") - 1)
            bin_index = max(0, min(bin_index, node_weights.shape[0] - 1))
            likelihood = record["likelihood"]
            if likelihood is None:
                continue
            log_scores = (
                np.log(np.maximum(node_weights[bin_index], 1e-20))
                + likelihood
            )
            objective += float(record["probability"]) * _logsumexp(log_scores)
            posterior = _softmax(log_scores)
            accum[bin_index] += float(record["probability"]) * posterior
        for index in range(node_weights.shape[0]):
            if float(np.sum(accum[index])) <= 20.0:
                nearest = min(
                    range(node_weights.shape[0]),
                    key=lambda other: abs(other - index)
                    if float(np.sum(accum[other])) > 20.0 else 10 ** 9,
                )
                accum[index] = accum[nearest]
        node_weights = _normalise_probability_rows(accum, alpha=0.0)
        if previous_objective is not None:
            objective_change = abs(objective - previous_objective) / max(
                abs(previous_objective), 1.0,
            )
        previous_objective = objective
        if objective_change < 1e-5:
            converged = True
            break

    latent_colors: list[np.ndarray] = []
    latent_weights: list[float] = []
    dirty_colors: list[np.ndarray] = []
    dirty_weights: list[float] = []
    predictive_colors: list[np.ndarray] = []
    predictive_weights: list[float] = []
    diagnostic_rng = np.random.default_rng(71591)
    for record in euclid_records:
        row = record["row"]
        magnitude = _finite(row.get("mag_vis"))
        if magnitude is None or not splice < magnitude <= faint:
            continue
        bin_index = max(0, min(
            int(np.searchsorted(magnitude_edges, magnitude, side="right") - 1),
            node_weights.shape[0] - 1,
        ))
        likelihood = record["likelihood"]
        if likelihood is None:
            continue
        posterior = _softmax(np.log(np.maximum(node_weights[bin_index], 1e-20)) + likelihood)
        latent_base = posterior @ locus + _draw_zero_mean_gaussian(
            diagnostic_rng, intrinsic_covariance,
        )
        latent_colors.append(latent_base)
        latent_weights.append(float(record["probability"]))
        if all(_finite(row.get(key)) is not None for key in ("mag_vis", "mag_y_e", "mag_j_e", "mag_h_e")):
            measured = np.asarray([
                float(row["mag_vis"]) - float(row["mag_y_e"]),
                float(row["mag_y_e"]) - float(row["mag_j_e"]),
                float(row["mag_j_e"]) - float(row["mag_h_e"]),
            ])
            dirty_colors.append(measured)
            dirty_weights.append(float(record["probability"]))
            measurement_covariance = _color_measurement_covariance(row)
            if measurement_covariance is not None:
                measurement_covariance = _positive_semidefinite_covariance(
                    measurement_covariance, floor=0.0,
                )
                predictive_covariance = _positive_semidefinite_covariance(
                    intrinsic_covariance + measurement_covariance, floor=0.0,
                )
                predictive_noise = _draw_zero_mean_gaussian(
                    diagnostic_rng, predictive_covariance,
                )
                predictive_colors.append(
                    latent_base + predictive_noise
                )
                predictive_weights.append(float(record["probability"]))
    magnitude_bins = np.asarray(q1_counts["edges"], dtype=np.float64)
    counts = np.asarray([
        float(item["expected_stars"]) for item in q1_counts["bins"]
    ], dtype=np.float64)
    classification_variance = float(sum(
        float(item["classification_variance"]) for item in q1_counts["bins"]
    ))
    area = float(q1_counts["footprint_area_arcmin2"])
    total_count = float(np.sum(counts))
    if total_count <= 0.0:
        raise ValueError("Q1 PHZ stellar counts have zero total probability weight")
    color_model = {
        "kind": "gaia_euclid_latent_locus_v1",
        "bp_rp_edges": bp_edges.tolist(),
        "bp_rp_nodes": bp_nodes.tolist(),
        "temperature_nodes_k": temperature_nodes.tolist(),
        "locus_colors": locus.tolist(),
        "g_to_vis_offset": offset_locus.tolist(),
        "intrinsic_color_covariance": intrinsic_covariance.tolist(),
        "magnitude_edges": magnitude_edges.tolist(),
        "magnitude_node_weights": node_weights.tolist(),
        "measurement_model": {
            "euclid_flux_likelihood": "student_t",
            "student_t_degrees_of_freedom": _LATENT_NU,
            "gaia_bp_rp_sigma_default_mag": 0.03,
            "gaia_euclid_flux_errors_independent": True,
        },
    }
    payload: dict[str, Any] = {
        "version": _STAR_POPULATION_VERSION,
        "kind": "star_population_fit",
        "valid": bool(converged and np.all(np.isfinite(intrinsic_covariance))),
        "warnings": [] if converged else ["latent node mixture did not converge"],
        "coverage_notes": [
            (
                f"normalized VIS counts with PHZ_STAR_PROB over "
                f"{float(q1_counts['footprint_area_deg2']):g} deg² of Q1"
            ),
            f"used {len(high_quality):,} high-S/N Gaia-Euclid matches for the locus",
            f"excluded {missing_flux:,} Euclid rows with fewer than two usable flux bands",
            f"probability coverage {usable_probability_rows:,}/{len(euclid_rows):,}; "
            f"missing {missing_probability:,}, invalid {invalid_probability:,}",
        ],
        "coverage": {
            "gaia_rows": len(gaia_rows),
            "gaia_usable_rows": len(gaia_usable),
            "matched_rows": len(matched),
            "high_quality_matched_rows": len(high_quality),
            "euclid_rows": len(euclid_rows),
            "usable_probability_rows": usable_probability_rows,
            "missing_probability_rows": missing_probability,
            "invalid_probability_rows": invalid_probability,
            "incomplete_flux_rows": missing_flux,
            "negative_flux_rows": int(sum(
                any(
                    (_finite(row.get(f"flux_{band}_aper_uJy")) or 0.0) < 0.0
                    for band in ("vis", "y", "j", "h")
                ) for row in euclid_rows
            )),
        },
        "color_sample_provenance": {
            "field_count": int(meta["field_count"]),
            "radius_deg": float(meta["radius_deg"]),
            "area_arcmin2": float(meta["area_arcmin2"]),
            "sampling_kind": meta.get("sampling_kind"),
            "fields": meta.get("fields"),
            "random_centres": meta.get("random_centres"),
            "role": "Gaia-Euclid color and temperature locus only",
        },
        "population_provenance": {
            "survey": q1_counts["survey"],
            "fields": q1_counts["fields"],
            "area_deg2": float(q1_counts["footprint_area_deg2"]),
            "area_arcmin2": area,
            "magnitude_field": q1_counts["magnitude_field"],
            "classification_field": q1_counts["classification_field"],
            "selection": q1_counts["selection"],
        },
        "population": {
            "density_arcmin2": magnitude_law.integrated_density(),
            "magnitude_slope": magnitude_law.slope,
            "mag_bright": bright,
            "mag_faint": faint,
            "magnitude_distribution": {
                **magnitude_law.to_payload(),
                "expected_count_per_bin": counts.tolist(),
                "phz_expected_count": total_count,
                "classification_variance": classification_variance,
                "count_cache_source": "Q1 PHZ_STAR_PROB",
                "fit_diagnostics": magnitude_fit_diagnostics,
            },
            "weighted_statistics": _weighted_summary(
                0.5 * (magnitude_bins[:-1] + magnitude_bins[1:]),
                counts,
                area_arcmin2=area,
                classification_variance=classification_variance,
            ),
        },
        "gaia": {
            "rows": len(gaia_usable),
            "bp_rp_quantiles": _quantiles(bp_values),
            "temperature_quantiles_k": _quantiles(np.asarray([
                float(row["temperature_k"]) for row in gaia_usable
                if _finite(row.get("temperature_k")) is not None
            ])),
        },
        "euclid_mapping": {
            "matched_stars": len(matched),
            "feature_order": ["intercept", "bp_rp", "g_minus_20"],
            "g_to_band_offset_coefficients": {
                "mag_vis": [float(np.median(vis_g_offsets)), 0.0, 0.0],
                "mag_y_e": [0.0, 0.0, 0.0],
                "mag_j_e": [0.0, 0.0, 0.0],
                "mag_h_e": [0.0, 0.0, 0.0],
            },
            "residual_covariance": np.eye(4).tolist(),
        },
        "classification_weighting": {
            "population_selection": "POINT_LIKE_PROB >= 0.9 over Q1",
            "population_star_weight": "PHZ_STAR_PROB over selected Q1 rows",
            "color_locus_star_weight": (
                "POINT_LIKE_PROB in the fixed Q1 matched sample"
            ),
            "missing_probability_rows": missing_probability,
            "invalid_probability_rows": invalid_probability,
            "usable_probability_rows": usable_probability_rows,
        },
        "color_model": color_model,
        "fit": {
            "latent_node_count": _LATENT_NODE_COUNT,
            "iterations": _iteration + 1,
            "converged": converged,
            "objective_change": objective_change,
            "matched_rows": len(matched),
            "high_quality_matched_rows": len(high_quality),
            "euclid_flux_rows": len(euclid_records),
            "usable_probability_rows": usable_probability_rows,
            "missing_probability_rows": missing_probability,
            "invalid_probability_rows": invalid_probability,
            "incomplete_flux_rows": missing_flux,
            "intrinsic_rows": int(np.count_nonzero(kept)),
            "selection": {
                "color_mixture_gaia_g_max": splice,
                "color_mixture_euclid_vis_min_exclusive": splice,
                "color_mixture_euclid_vis_max": faint,
                "locus_min_band_snr": 5.0,
            },
        },
        "fingerprint_inputs": {
            "euclid_catalog_version": Q1_STELLAR_COLOR_SAMPLE_VERSION,
            "euclid_rows": len(euclid_rows),
            "gaia_rows": len(gaia_rows),
            "q1_area_arcmin2": area,
            "q1_phz_star_counts": q1_counts,
            "gaia_sampling": {
                key: meta.get(key)
                for key in (
                    "version", "sampling_kind", "field_count", "radius_deg",
                    "area_deg2", "fields", "random_centres",
                )
            },
            "fit_version": "q1-phz-gaia-shared-straight-counts-latent-locus-v5",
            "selection": {
                "bright_limit": bright,
                "faint_limit": faint,
                "population_selection": "POINT_LIKE_PROB >= 0.9",
                "population_probability_field": "PHZ_STAR_PROB",
                "color_probability_field": "POINT_LIKE_PROB",
                "flux_fields": [
                    f"flux_{band}_aper_uJy" for band in ("vis", "y", "j", "h")
                ],
                "error_fields": [
                    f"fluxerr_{band}_aper_uJy" for band in ("vis", "y", "j", "h")
                ],
            },
        },
    }
    model = EmpiricalStellarPrior.from_payload(payload)
    rng = np.random.default_rng(71033)
    fitted = [
        model.sample(rng, model.sample_magnitude(
            rng, slope=0.15, m_bright=bright, m_faint=faint,
        )) for _ in range(10_000)
    ]
    fitted_colors = np.asarray([
        [
            sed.magnitudes["VIS"] - sed.magnitudes["Y_E"],
            sed.magnitudes["Y_E"] - sed.magnitudes["J_E"],
            sed.magnitudes["J_E"] - sed.magnitudes["H_E"],
        ] for sed in fitted
    ])
    magnitude_centres = 0.5 * (magnitude_bins[:-1] + magnitude_bins[1:])
    fitted_vis_density = magnitude_law.density(magnitude_centres)
    gaia_g_ab = np.asarray([
        float(value) + _GAIA_G_AB_MINUS_VEGA_MAG
        for row in gaia_rows
        if str(row.get("central_selected_star") or "0").strip() != "1"
        and (value := _finite(row.get("g_mag"))) is not None
    ], dtype=np.float64)
    gaia_fit = magnitude_fit_diagnostics["gaia"]
    gaia_bin_width = float(gaia_fit["bin_width_mag"])
    gaia_edges = _fixed_width_edges(bright, faint, gaia_bin_width)
    gaia_centres = 0.5 * (gaia_edges[:-1] + gaia_edges[1:])
    gaia_counts, _ = np.histogram(gaia_g_ab, bins=gaia_edges)
    gaia_density = (
        gaia_counts.astype(np.float64)
        / float(meta["area_arcmin2"])
        / np.diff(gaia_edges)
    )
    gaia_fitted_density = np.power(
        10.0,
        magnitude_law.slope * gaia_centres + float(gaia_fit["intercept"]),
    )
    latent_array = np.asarray(latent_colors)
    latent_weight_array = np.asarray(latent_weights)
    dirty_array = np.asarray(dirty_colors)
    dirty_weight_array = np.asarray(dirty_weights)
    predictive_array = np.asarray(predictive_colors)
    predictive_weight_array = np.asarray(predictive_weights)
    diagnostics: dict[str, Any] = {
        "stellar_density_by_magnitude": {
            "x": magnitude_centres.tolist(),
            "observed": (
                counts / area / np.diff(magnitude_bins)
            ).tolist(),
            "fitted": fitted_vis_density.tolist(),
            "gaia_x": gaia_centres.tolist(),
            "gaia_observed": gaia_density.tolist(),
            "gaia_fitted": gaia_fitted_density.tolist(),
            "fit_ranges": {
                "q1": [
                    float(magnitude_fit_diagnostics["q1"]["fit_bright"]),
                    float(magnitude_fit_diagnostics["q1"]["fit_faint"]),
                ],
                "gaia": [
                    float(gaia_fit["fit_bright"]),
                    float(gaia_fit["fit_faint"]),
                ],
            },
            "sampling_interval": [bright, faint],
            "label": "Q1 PHZ probability-weighted stellar density",
            "unit": "stars / arcmin² / mag",
            "x_label": "native survey magnitude [AB]",
        },
        "parameters": {},
    }
    for index, key in enumerate(_LATENT_COLOR_ORDER):
        observed_latent = latent_array[:, index] if latent_array.size else np.asarray([])
        dirty = dirty_array[:, index] if dirty_array.size else np.asarray([])
        fitted_values = fitted_colors[:, index]
        bins = np.linspace(
            min(float(np.percentile(fitted_values, 1)), float(np.percentile(observed_latent, 1)))
            if observed_latent.size else float(np.percentile(fitted_values, 1)),
            max(float(np.percentile(fitted_values, 99)), float(np.percentile(observed_latent, 99)))
            if observed_latent.size else float(np.percentile(fitted_values, 99)),
            25,
        )
        intrinsic_hist = _weighted_histogram(
            observed_latent, latent_weight_array, bins=bins,
            area_arcmin2=float(np.sum(latent_weight_array)),
        ) if observed_latent.size else {"x": (0.5 * (bins[:-1] + bins[1:])).tolist(), "observed": [0.0] * 24}
        dirty_hist = _weighted_histogram(
            dirty, dirty_weight_array, bins=bins,
            area_arcmin2=float(np.sum(dirty_weight_array)),
        ) if dirty.size else {"observed": [0.0] * 24}
        predictive_hist = _weighted_histogram(
            predictive_array[:, index], predictive_weight_array, bins=bins,
            area_arcmin2=float(np.sum(predictive_weight_array)),
        ) if predictive_array.size else {"observed": [0.0] * 24}
        fitted_hist, _ = np.histogram(fitted_values, bins=bins)
        widths = np.diff(bins)
        latent_summary = _weighted_color_summary(
            latent_array, latent_weight_array,
        )
        dirty_summary = _weighted_color_summary(dirty_array, dirty_weight_array)
        latent_stats = dict(latent_summary.get(key, {}))
        latent_stats.update({
            "expected_count": latent_summary.get("effective_weight"),
            "effective_n": latent_summary.get("effective_n"),
        })
        dirty_stats = dict(dirty_summary.get(key, {}))
        dirty_stats.update({
            "expected_count": dirty_summary.get("effective_weight"),
            "effective_n": dirty_summary.get("effective_n"),
        })
        parameters = {
            **intrinsic_hist,
            "fitted": (fitted_hist / max(len(fitted_values), 1) / widths).tolist(),
            "fitted_count": len(fitted_values),
            "label": key.replace("_", " "),
            "unit": "AB mag",
            "density_unit": "probability density",
            "observed_label": "Estimated true colours of observed stars",
            "dirty_observed": dirty_hist.get("observed", [0.0] * 24),
            "dirty_observed_label": "Raw Euclid catalogue colours",
            "posterior_predictive": predictive_hist.get("observed", [0.0] * 24),
            "posterior_predictive_label": (
                "Estimated colours with simulated Euclid noise"
            ),
            "statistics": latent_stats,
            "dirty_statistics": dirty_stats,
            "posterior_predictive_statistics": _weighted_color_summary(
                predictive_array, predictive_weight_array,
            ),
            "summary": latent_summary,
            "dirty_summary": dirty_summary,
            "x": (0.5 * (bins[:-1] + bins[1:])).tolist(),
        }
        diagnostics["parameters"][key] = parameters
    payload["diagnostics"] = diagnostics
    latent_summary = _weighted_color_summary(latent_array, latent_weight_array)
    predictive_summary = _weighted_color_summary(
        predictive_array, predictive_weight_array,
    )
    measured_summary = _weighted_color_summary(dirty_array, dirty_weight_array)
    quality_colors: dict[str, Any] = {}
    quality_pass = True
    for key in _LATENT_COLOR_ORDER:
        latent_stats = latent_summary.get(key, {})
        measured_stats = measured_summary.get(key, {})
        predictive_stats = predictive_summary.get(key, {})
        median_error = abs(
            float(predictive_stats.get("p50", 0.0))
            - float(measured_stats.get("p50", 0.0))
        ) if measured_stats and predictive_stats else float("inf")
        measured_width = float(measured_stats.get("p84", 0.0)) - float(measured_stats.get("p16", 0.0))
        predictive_width = float(predictive_stats.get("p84", 0.0)) - float(predictive_stats.get("p16", 0.0))
        width_ratio = predictive_width / measured_width if measured_width > 0 else float("inf")
        quality_colors[key] = {
            "posterior_predictive_median_error_mag": median_error,
            "posterior_predictive_width_ratio": width_ratio,
            "intrinsic_summary": latent_stats,
            "measured_summary": measured_stats,
            "posterior_predictive_summary": predictive_stats,
        }
        quality_pass &= median_error <= 0.08 and 0.80 <= width_ratio <= 1.25
    payload["posterior_predictive"] = {
        "colors": quality_colors,
        "negative_flux_rows_retained": int(sum(
            any(
                (_finite(record["row"].get(f"flux_{band}_aper_uJy")) or 0.0) < 0.0
                for band in ("vis", "y", "j", "h")
            ) for record in euclid_records
        )),
    }
    payload["quality_gates"] = {
        "global_median_error_max_mag": 0.08,
        "robust_width_ratio_range": [0.80, 1.25],
        "passed": bool(quality_pass and converged),
    }
    payload["valid"] = bool(payload["valid"] and quality_pass)
    payload["fingerprint_inputs"]["catalog_digest"] = hashlib.sha256(
        json.dumps({
            "euclid": euclid_rows,
            "gaia": gaia_rows,
            "q1_phz_star_counts": q1_counts,
        }, sort_keys=True, default=str, separators=(",", ":")).encode()
    ).hexdigest()
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["fingerprint"] = hashlib.sha256(canonical.encode()).hexdigest()
    write_star_candidate(payload)
    _write_star_distribution(_star_distribution_from_rows(
        euclid_rows,
        gaia_rows,
        calibration_fingerprint=payload["fingerprint"],
        color_model=payload["color_model"],
        stellar_model=payload,
        area_arcmin2=euclid_area,
        gaia_area_arcmin2=float(meta["area_arcmin2"]),
        gaia_sampling=meta,
    ))
    return payload


def fit_star_population(
    *, faint_limit: float | None = None, bright_limit: float | None = None,
) -> dict[str, Any]:
    """Fit the required flux-aware empirical stellar prior.

    The legacy exponential/blackbody fit is intentionally no longer selected
    implicitly: a population artifact must be tied to raw Euclid fluxes and
    errors so generation cannot silently change statistical models.
    """
    if faint_limit is not None or bright_limit is not None:
        raise ValueError(
            "custom stellar magnitude limits are not supported by the strict "
            "data-driven fit; refit the empirical catalog artifact"
        )
    try:
        rows = _read_rows(euclid_catalog_path())
        raw_keys = tuple(
            key
            for band in ("vis", "y", "j", "h")
            for key in (f"flux_{band}_aper_uJy", f"fluxerr_{band}_aper_uJy")
        )
        has_raw_flux = bool(rows) and all(
            all(key in row for key in raw_keys)
            for row in rows[: min(32, len(rows))]
        )
    except (OSError, csv.Error) as exc:
        raise ValueError(
            "cannot read the raw Euclid flux catalog required for stellar fitting"
        ) from exc
    if not has_raw_flux:
        raise ValueError(
            "stellar fitting requires flux_y_aper_uJy/fluxerr_y_aper_uJy "
            "and the corresponding VIS/J/H fields"
        )
    return _fit_star_population_latent()
