"""Q1 PHZ number counts plus Euclid-only stellar colours."""

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
    fit_straight_region,
)
from euclid_polish.sky.generation.stellar_sed import (
    STAR_COLOR_MODEL_KIND,
    EmpiricalStellarPrior,
    StarColorModel,
    build_star_color_model,
    deconvolved_draw,
    magnitude_errors,
)
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

# Gaia (E)DR3 G-band AB and Vega zero points are 25.8010446445 and
# 25.6873668671 mag, respectively.  The archive's phot_g_mean_mag is Vega.
_GAIA_G_AB_MINUS_VEGA_MAG = 25.8010446445 - 25.6873668671
_GAIA_TAP_PROVIDER = "ARI Gaia TAP"
_STAR_POPULATION_VERSION = 7
#: The law is Euclid alone; Gaia G counts are only overlaid for comparison.
_STAR_FIT_VERSION = "q1-phz-straight-counts-euclid-colors-v6"
_STAR_DISTRIBUTION_VERSION = 16
_GAIA_COUNT_FIT_BIN_WIDTH_MAG = 0.5


def gaia_catalog_path() -> Path:
    return q1_gaia_color_catalog_path()


def gaia_catalog_meta_path() -> Path:
    return q1_gaia_color_meta_path()


def star_distribution_path(*, include_training: bool = False) -> Path:
    return (
        Path(Config.DATA_DIR) / "population_comparison"
        / (
            "star_distribution_with_training.json"
            if include_training else "star_distribution.json"
        )
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
    stellar_model: dict[str, Any],
    *,
    euclid_area_arcmin2: float,
    gaia_area_arcmin2: float,
    synthetic_rows: list[dict[str, str]] | None = None,
    synthetic_area_arcmin2: float = 0.0,
    synthetic_scope: str = "test + validation",
    sample_count: int = 50_000,
) -> dict[str, Any] | None:
    """Compare measured and generator stellar densities.

    The VIS panel shows the Q1 counts, the fitted law, the generated stars
    and, for comparison, the native Gaia G_AB counts with their own
    straight-region fit (``gaia_x`` / ``gaia`` / ``gaia_fit``). The colour
    panels carry no Gaia series.
    """
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
        prior.sample_magnitude(rng)
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
    # The Q1 colours are measured: noisy, and only over the VIS range of the
    # Euclid colour sample. The model's colour draws and the generated
    # stars are compared the same way: restricted to that VIS window and
    # given the Q1 flux errors (forward noised), so the widths compare. The
    # VIS law panel stays intrinsic.
    donor_vis, donor_sigma = _star_noise_donors(euclid_rows)
    color_bands = model_bands
    color_model_density = model_density
    synthetic_color_bands = {
        name: np.asarray([row[name] for row in synthetic_complete], dtype=np.float64)
        for name in euclid_fields
    }
    if donor_vis.size >= _MIN_NOISE_DONORS:
        window = np.quantile(donor_vis, _COLOR_WINDOW_QUANTILES)
        in_window = (model_bands["VIS"] >= window[0]) & (model_bands["VIS"] <= window[1])
        color_bands = _forward_noise_magnitudes(
            {name: values[in_window] for name, values in model_bands.items()},
            donor_vis, donor_sigma, rng,
        )
        color_model_density = model_density * float(np.mean(in_window))
        synthetic_in = (
            (synthetic_color_bands["VIS"] >= window[0])
            & (synthetic_color_bands["VIS"] <= window[1])
        )
        synthetic_color_bands = _forward_noise_magnitudes(
            {name: values[synthetic_in] for name, values in synthetic_color_bands.items()},
            donor_vis, donor_sigma, rng,
        )
        model_color_noise: dict[str, Any] = {
            "applied": True, "donors": int(donor_vis.size),
            "vis_window": [round(float(window[0]), 2), round(float(window[1]), 2)],
        }
    else:
        model_color_noise = {
            "applied": False,
            "detail": (
                f"only {int(donor_vis.size)} Q1 colour stars carry flux errors in all "
                f"four bands (at least {_MIN_NOISE_DONORS} are needed)"
            ),
        }
    synthetic_colors = {
        "vis_y": synthetic_color_bands["VIS"] - synthetic_color_bands["Y_E"],
        "vis_j": synthetic_color_bands["VIS"] - synthetic_color_bands["J_E"],
        "vis_h": synthetic_color_bands["VIS"] - synthetic_color_bands["H_E"],
        "y_j": synthetic_color_bands["Y_E"] - synthetic_color_bands["J_E"],
        "y_h": synthetic_color_bands["Y_E"] - synthetic_color_bands["H_E"],
        "j_h": synthetic_color_bands["J_E"] - synthetic_color_bands["H_E"],
    }
    model_colors = {
        "vis_y": color_bands["VIS"] - color_bands["Y_E"],
        "vis_j": color_bands["VIS"] - color_bands["J_E"],
        "vis_h": color_bands["VIS"] - color_bands["H_E"],
        "y_j": color_bands["Y_E"] - color_bands["J_E"],
        "y_h": color_bands["Y_E"] - color_bands["H_E"],
        "j_h": color_bands["J_E"] - color_bands["H_E"],
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
    magnitude_diagnostics = (
        stellar_model["population"]["magnitude_distribution"].get(
            "fit_diagnostics", {}
        )
    )
    q1_fit_diagnostics = magnitude_diagnostics.get("q1") or {}
    # Native Gaia G_AB counts (Vega→AB) over the fixed Q1 Gaia fields, with
    # Gaia's own straight-region line, for comparison with the Q1 law.
    gaia_g_ab = np.asarray([
        float(value) + _GAIA_G_AB_MINUS_VEGA_MAG
        for row in gaia_rows
        if str(row.get("central_selected_star") or "0").strip() != "1"
        and (value := _finite(row.get("g_mag"))) is not None
    ], dtype=np.float64)
    gaia_fit_diagnostics = magnitude_diagnostics.get("gaia") or {}
    gaia_bin_width = float(
        gaia_fit_diagnostics.get("bin_width_mag") or _GAIA_COUNT_FIT_BIN_WIDTH_MAG
    )
    gaia_edges = _fixed_width_edges(bright, faint, gaia_bin_width)
    gaia_centres = 0.5 * (gaia_edges[:-1] + gaia_edges[1:])
    gaia_counts, _ = np.histogram(gaia_g_ab, bins=gaia_edges)
    gaia_density = (
        gaia_counts / (gaia_area_arcmin2 * np.diff(gaia_edges))
    ).tolist()
    # Gaia's own straight-region line (a comparison; it is not the law).
    gaia_intercept = _finite(gaia_fit_diagnostics.get("intercept"))
    gaia_slope = _finite(gaia_fit_diagnostics.get("slope"))
    gaia_fit = (
        np.power(10.0, gaia_slope * gaia_centres + gaia_intercept).tolist()
        if gaia_intercept is not None and gaia_slope is not None else None
    )
    parameters: dict[str, Any] = {
        "vis": {
            "label": "VIS brightness",
            "gaia_x": gaia_centres.tolist(),
            "gaia": gaia_density,
            "gaia_fit": gaia_fit,
            "x_label": "apparent magnitude [AB]",
            "x": (0.5 * (magnitude_edges[:-1] + magnitude_edges[1:])).tolist(),
            "x_domain": [bright, faint],
            "euclid": q1_density if q1_density is not None else _density_series(
                np.asarray(euclid_vis), magnitude_edges,
                area_arcmin2=euclid_area_arcmin2,
            ),
            "point_sources": q1_point_source_density,
            "model": model_magnitude_law.density(
                0.5 * (magnitude_edges[:-1] + magnitude_edges[1:])
            ).tolist(),
            "synthetic": _density_series(
                synthetic_bands["VIS"], magnitude_edges,
                area_arcmin2=synthetic_area_arcmin2,
            ) if synthetic_area_arcmin2 > 0.0 else [],
            "fit_ranges": {
                "q1": [
                    q1_fit_diagnostics.get("fit_bright"),
                    q1_fit_diagnostics.get("fit_faint"),
                ],
            },
        },
    }
    for key in ("vis_y", "vis_j", "vis_h", "y_j", "y_h", "j_h"):
        edges = _shared_color_edges([
            euclid_colors[key], model_colors[key], synthetic_colors[key],
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
            "model": _density_series(
                model_colors[key], edges, total_density_arcmin2=color_model_density,
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
        "model_color_noise": model_color_noise,
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
            + "The 12–25 VIS generator is the straight region of the Q1 "
            "PHZ-weighted counts alone. For comparison only, the magnitude "
            "panel also shows native Gaia G, converted from the archive Vega "
            "magnitude to AB with the Gaia (E)DR3 zero-point offset and divided "
            "by the fixed Q1 Gaia-field area, with its own straight-region "
            "line. Colour curves are fixed-field Euclid diagnostics; "
            + (
                "the model colour draws and the generated stars are taken over "
                "the VIS range of the Q1 colour sample and carry Q1 measurement "
                "noise, with flux errors borrowed from VIS-nearest Q1 stars. "
                if model_color_noise["applied"] else
                "the model colour draws are intrinsic, without Q1 measurement "
                "noise (too few Q1 stars carry flux errors). "
            )
            + "Magenta points are "
            "the actual stars stored in the selected synthetic "
            f"{synthetic_scope} source catalogues, normalized by their "
            "rendered field area."
        ),
    }


_COLOR_ORDER = ("vis_y", "y_j", "j_h")


def _raw_measurement(row: dict[str, str], band: str) -> tuple[float, float] | None:
    flux = _finite(row.get(f"flux_{band}_aper_uJy"))
    error = _finite(row.get(f"fluxerr_{band}_aper_uJy"))
    if flux is None or error is None or error <= 0.0:
        return None
    return float(flux), float(error)


#: Model colour draws borrow their measurement noise from a random star
#: among this many VIS-nearest Q1 colour stars (the noise donors).
_NOISE_DONOR_WINDOW = 16
#: Fewer donors than this leave the model colours intrinsic.
_MIN_NOISE_DONORS = 5
#: The colour panels compare over this VIS quantile range of the donors.
_COLOR_WINDOW_QUANTILES = (0.005, 0.995)
_AB_ZEROPOINT_UJY_MAG = 23.9
#: (model band, catalogue flux band, catalogue magnitude column).
_NOISE_BANDS = (
    ("VIS", "vis", "mag_vis"), ("Y_E", "y", "mag_y_e"),
    ("J_E", "j", "mag_j_e"), ("H_E", "h", "mag_h_e"),
)


def _star_noise_donors(
    rows: list[dict[str, str]],
) -> tuple[np.ndarray, np.ndarray]:
    """VIS magnitudes and per-band flux errors [µJy] of the Q1 colour stars.

    The donors are the rows of the colour sample (``type`` star,
    POINT_LIKE_PROB ≥ 0.9, all four magnitudes) that also carry a positive
    aperture flux and error in every band. Each aperture error is scaled by
    the star's own aperture correction (total ÷ aperture flux), so it sits
    on the flux scale of the magnitudes the colours are made from. Sorted by
    VIS magnitude; shapes ``(n,)`` and ``(n, 4)``.
    """
    vis: list[float] = []
    sigma: list[list[float]] = []
    for row in rows:
        probability = _finite(row.get("point_like_prob"))
        if row.get("type") != "star" or probability is None or probability < 0.9:
            continue
        magnitudes = [_finite(row.get(column)) for _b, _f, column in _NOISE_BANDS]
        measurements = [_raw_measurement(row, band) for _b, band, _c in _NOISE_BANDS]
        if any(value is None for value in magnitudes) or any(
            item is None or item[0] <= 0.0 for item in measurements
        ):
            continue
        sigma.append([
            measurement[1] * 10.0 ** (-0.4 * (magnitude - _AB_ZEROPOINT_UJY_MAG))
            / measurement[0]
            for magnitude, measurement in zip(magnitudes, measurements, strict=True)
            if magnitude is not None and measurement is not None
        ])
        vis.append(float(magnitudes[0] or 0.0))
    order = np.argsort(np.asarray(vis, dtype=np.float64), kind="stable")
    return (
        np.asarray(vis, dtype=np.float64)[order],
        np.asarray(sigma, dtype=np.float64).reshape(-1, len(_NOISE_BANDS))[order],
    )


def _forward_noise_magnitudes(
    magnitudes: dict[str, np.ndarray],
    donor_vis: np.ndarray,
    donor_sigma: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Model magnitudes as the Q1 photometry would measure them.

    Each draw borrows the four flux errors of a random donor among the
    ``_NOISE_DONOR_WINDOW`` VIS-nearest Q1 stars and gets independent
    Gaussian flux noise in each band. The absolute error is kept
    (background-limited photometry), so a draw fainter than every donor
    takes the faintest donors' errors. A band whose noised flux is not
    positive has no magnitude (NaN), as a Q1 non-detection has none. The
    input arrays are not modified.
    """
    vis = np.asarray(magnitudes["VIS"], dtype=np.float64)
    donors = int(donor_vis.size)
    window = min(_NOISE_DONOR_WINDOW, donors)
    nearest = np.searchsorted(donor_vis, vis)
    low = np.clip(nearest - window // 2, 0, donors - window)
    chosen = low + rng.integers(0, window, size=vis.size)
    noised: dict[str, np.ndarray] = {}
    for index, (band, _flux_band, _column) in enumerate(_NOISE_BANDS):
        flux = 10.0 ** (-0.4 * (
            np.asarray(magnitudes[band], dtype=np.float64) - _AB_ZEROPOINT_UJY_MAG
        ))
        flux = flux + donor_sigma[chosen, index] * rng.standard_normal(vis.size)
        positive = flux > 0.0
        noised[band] = np.where(
            positive,
            _AB_ZEROPOINT_UJY_MAG - 2.5 * np.log10(np.where(positive, flux, 1.0)),
            np.nan,
        )
    return noised


_STAR_COLOR_PAIRS = (
    ("vis_y", "VIS − Y", "mag_vis", "mag_y_e"),
    ("vis_j", "VIS − J", "mag_vis", "mag_j_e"),
    ("vis_h", "VIS − H", "mag_vis", "mag_h_e"),
    ("y_j", "Y − J", "mag_y_e", "mag_j_e"),
    ("y_h", "Y − H", "mag_y_e", "mag_h_e"),
    ("j_h", "J − H", "mag_j_e", "mag_h_e"),
)

def _distribution_source_signature(
    *, include_training: bool = False,
) -> dict[str, int | None]:
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
    _records, synthetic_sources = (
        _synthetic_paths(include_training=True)
        if include_training else _synthetic_paths()
    )
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
    synthetic_scope: str = "test + validation",
) -> dict[str, Any]:
    """Count the Euclid colour sample and compare stellar densities.

    The counts describe the Q1 point-source sample the stellar prior fits its
    colours on (Euclid alone; no Gaia match).
    """
    pointlike_over_09 = sum(
        int((probability := _finite(row.get("point_like_prob"))) is not None
            and probability >= 0.9)
        for row in euclid_rows
    )
    color_rows = int(_star_color_rows(euclid_rows)["weight"].size)
    has_fit = bool(
        color_model and color_model.get("kind") == STAR_COLOR_MODEL_KIND
    )
    density_comparison = None
    if (
        has_fit and stellar_model is not None
        and area_arcmin2 is not None and gaia_area_arcmin2 is not None
    ):
        density_comparison = _stellar_density_comparison(
            euclid_rows,
            gaia_rows,
            stellar_model,
            euclid_area_arcmin2=area_arcmin2,
            gaia_area_arcmin2=gaia_area_arcmin2,
            synthetic_rows=synthetic_rows,
            synthetic_area_arcmin2=synthetic_area_arcmin2,
            synthetic_scope=synthetic_scope,
        )

    return {
        "version": _STAR_DISTRIBUTION_VERSION,
        "calibration_fingerprint": calibration_fingerprint,
        "source_signature": _distribution_source_signature(),
        "color_sample_rows": len(euclid_rows),
        "color_rows_used": color_rows,
        "pointlike_over_0_9": int(pointlike_over_09),
        "density_comparison": density_comparison,
        "gaia_sampling": gaia_sampling,
    }


def _write_star_distribution(
    payload: dict[str, Any], *, include_training: bool = False,
) -> None:
    output = star_distribution_path(include_training=include_training)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")))
    os.replace(temporary, output)


#: In-process memo of the last computed distribution per ``include_training``
#: variant, keyed by (calibration fingerprint, source signature): a GET must
#: never write into ``data/``, so a stale on-disk cache is recomputed in memory
#: once per input state; only POST jobs (``persist=True``) write the cache.
_DISTRIBUTION_MEMO: dict[bool, tuple[tuple[Any, Any], dict[str, Any]]] = {}


def star_distribution_payload(
    *, include_training: bool = False, persist: bool = False,
) -> dict[str, Any] | None:
    """Read stellar diagnostics for current or catalogue-only all splits.

    Read-only unless ``persist`` (the fit job): a missing or stale cache is
    computed in memory and memoised for this process.
    """
    try:
        candidate = json.loads(star_candidate_path().read_text())
    except (OSError, json.JSONDecodeError):
        candidate = None
    fingerprint = (
        str(candidate.get("fingerprint"))
        if isinstance(candidate, dict) and candidate.get("fingerprint") else None
    )
    try:
        cached = json.loads(star_distribution_path(
            include_training=include_training,
        ).read_text())
    except (OSError, json.JSONDecodeError):
        cached = None
    signature = _distribution_source_signature(
        include_training=include_training,
    )
    if (
        isinstance(cached, dict)
        and cached.get("version") == _STAR_DISTRIBUTION_VERSION
        and cached.get("calibration_fingerprint") == fingerprint
        and cached.get("source_signature") == signature
    ):
        return cached
    memo_key = (fingerprint, json.dumps(signature, sort_keys=True, default=str))
    memo = _DISTRIBUTION_MEMO.get(include_training)
    if memo is not None and memo[0] == memo_key:
        if persist:
            _write_star_distribution(memo[1], include_training=include_training)
        return memo[1]
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
    _records, synthetic_paths = (
        _synthetic_paths(include_training=True)
        if include_training else _synthetic_paths()
    )
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
        synthetic_scope=(
            "training + test + validation" if include_training
            else "test + validation"
        ),
    )
    payload["source_signature"] = signature
    payload["synthetic_splits"] = [
        split for split in ("train", "test", "validate")
        if any(path.stem == f"sources_{split}" for path in synthetic_paths)
    ]
    payload["training_included"] = bool(
        include_training and "train" in payload["synthetic_splits"]
    )
    payload["training_catalog_only"] = payload["training_included"]
    _DISTRIBUTION_MEMO[include_training] = (memo_key, payload)
    if persist:
        _write_star_distribution(payload, include_training=include_training)
    return payload


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
    for index, name in enumerate(_COLOR_ORDER):
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
    """Fit the Q1 PHZ VIS count law; fit Gaia G separately to compare."""
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
    # for a 2.5-mag straight region in 0.1-mag cells would make Poisson
    # structure decide the comparison, so Gaia is fitted at a stable 0.5-mag
    # resolution; the Q1 law keeps its native 0.1-mag bins.
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
        q1_region = fit_straight_region(
            centres, q1_density, q1_sigma,
            minimum_span_mag=2.5, minimum_r_squared=0.99,
        )
    except ValueError as exc:
        raise ValueError(
            "Q1 PHZ_STAR_PROB VIS counts have no 2.5-mag straight region "
            "at 0.1-mag resolution with R² >= 0.99"
        ) from exc
    law = StraightMagnitudeLaw(
        slope=q1_region.slope,
        intercept=q1_region.intercept,
        mag_bright=float(Config.STAR_MAG_BRIGHT),
        mag_faint=float(Config.STAR_MAG_FAINT),
        fit_bright=float(centres[q1_region.start]),
        fit_faint=float(centres[q1_region.stop - 1]),
        covariance=(
            (float(q1_region.covariance[0, 0]), float(q1_region.covariance[0, 1])),
            (float(q1_region.covariance[1, 0]), float(q1_region.covariance[1, 1])),
        ),
        r_squared=q1_region.r_squared,
        rms_log10_density=q1_region.rms,
        source=(
            "straight region of the Euclid Q1 PHZ_STAR_PROB-weighted VIS "
            "counts; Gaia G_AB counts are a comparison only"
        ),
    )
    diagnostics: dict[str, Any] = {
        "q1": {
            "band": "Euclid VIS",
            "bin_width_mag": float(np.median(widths)),
            "fit_bright": float(centres[q1_region.start]),
            "fit_faint": float(centres[q1_region.stop - 1]),
            "slope": float(q1_region.slope),
            "intercept": float(q1_region.intercept),
            "r_squared": float(q1_region.r_squared),
            "rms_log10_density": float(q1_region.rms),
        },
        "gaia": None,
    }
    # Comparison: Gaia's own straight region and slope. It never enters the
    # law, so a Gaia sample without a clean straight region only loses the
    # comparison.
    try:
        gaia_region = fit_straight_region(
            gaia_centres, gaia_density, gaia_sigma,
            minimum_span_mag=2.5, minimum_r_squared=0.99,
        )
    except ValueError:
        return law, diagnostics
    slope_sigma = math.sqrt(
        max(float(gaia_region.covariance[0, 0]), 0.0)
        + max(float(q1_region.covariance[0, 0]), 0.0)
    )
    diagnostics["gaia"] = {
        "band": "Gaia G_AB",
        "role": "comparison only",
        "bin_width_mag": _GAIA_COUNT_FIT_BIN_WIDTH_MAG,
        "fit_bright": float(gaia_centres[gaia_region.start]),
        "fit_faint": float(gaia_centres[gaia_region.stop - 1]),
        "slope": float(gaia_region.slope),
        "intercept": float(gaia_region.intercept),
        "r_squared": float(gaia_region.r_squared),
        "rms_log10_density": float(gaia_region.rms),
        "slope_difference_sigma": (
            (float(gaia_region.slope) - float(q1_region.slope)) / slope_sigma
            if slope_sigma > 0.0 else None
        ),
    }
    return law, diagnostics


_COLOR_SNR_FLOOR = 5.0
_COLOR_BANDS = ("vis", "y", "j", "h")


def _star_color_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Select the Q1 point sources whose colours calibrate the stellar colours.

    A row needs POINT_LIKE_PROB ≥ 0.9, a PHZ star probability (its weight), a
    VIS PSF magnitude (the binning variable, as for the counts) and aperture
    fluxes at S/N ≥ 5 in all four bands, so its magnitude errors are small
    enough to linearise. Stars in the colour range are bright in NISP: the
    rows this cut removes (~20% at every magnitude) are almost all MER rows
    reporting absurd errors (tens to hundreds of µJy on a few-µJy-σ source).
    """
    magnitude: list[float] = []
    color: list[list[float]] = []
    mag_sigma: list[np.ndarray] = []
    weight: list[float] = []
    fluxes: list[list[float]] = []
    rejected = {
        "not_point_like": 0, "no_phz_weight": 0, "no_psf_magnitude": 0,
        "incomplete_flux": 0, "low_snr": 0,
    }
    for row in rows:
        probability = _finite(row.get("point_like_prob"))
        if probability is None or not 0.9 <= probability <= 1.0:
            rejected["not_point_like"] += 1
            continue
        star_weight = _finite(row.get("phz_star_prob"))
        if star_weight is None or not 0.0 < star_weight <= 1.0:
            rejected["no_phz_weight"] += 1
            continue
        psf_magnitude = _finite(row.get("mag_vis_psf"))
        if psf_magnitude is None:
            rejected["no_psf_magnitude"] += 1
            continue
        measurements = [_raw_measurement(row, band) for band in _COLOR_BANDS]
        if any(item is None for item in measurements):
            rejected["incomplete_flux"] += 1
            continue
        flux = np.asarray([item[0] for item in measurements if item is not None])
        error = np.asarray([item[1] for item in measurements if item is not None])
        if np.any(flux <= 0.0) or np.any(flux / error < _COLOR_SNR_FLOOR):
            rejected["low_snr"] += 1
            continue
        band_magnitude = _AB_ZEROPOINT_UJY_MAG - 2.5 * np.log10(flux)
        magnitude.append(psf_magnitude)
        color.append(list(-np.diff(band_magnitude)))
        mag_sigma.append(magnitude_errors(flux, error))
        weight.append(star_weight)
        fluxes.append([*flux, *error])
    return {
        "magnitude": np.asarray(magnitude, dtype=np.float64),
        "color": np.asarray(color, dtype=np.float64).reshape(-1, 3),
        "mag_sigma": np.asarray(mag_sigma, dtype=np.float64).reshape(-1, 4),
        "weight": np.asarray(weight, dtype=np.float64),
        # VIS/Y/J/H aperture fluxes then errors [µJy], for the re-noising check.
        "fluxes": np.asarray(fluxes, dtype=np.float64).reshape(-1, 8),
        "rejected": rejected,
    }


def _renoised_colors(
    colors: np.ndarray, fluxes: np.ndarray, rng: np.random.Generator,
) -> np.ndarray | None:
    """Colours of a star at ``colors`` measured with one row's flux errors.

    The star keeps the row's VIS flux; the NISP fluxes follow from the
    colours, every band gets its own Gaussian flux error, and the colours are
    remeasured (None if a noised flux is not positive).
    """
    vis_flux = fluxes[0]
    offsets = np.concatenate(([0.0], np.cumsum(colors)))
    true_flux = vis_flux * np.power(10.0, 0.4 * offsets)
    noised = true_flux + fluxes[4:] * rng.standard_normal(4)
    if np.any(noised <= 0.0):
        return None
    return -np.diff(-2.5 * np.log10(noised))


def _fit_star_population_euclid() -> dict[str, Any]:
    """Fit PHZ number counts plus Euclid-only point-source colours."""
    faint = float(Config.STAR_MAG_FAINT)
    bright = float(Config.STAR_MAG_BRIGHT)
    gaia_rows = _read_rows(gaia_catalog_path())
    euclid_rows = _read_rows(euclid_catalog_path())
    meta = json.loads(gaia_catalog_meta_path().read_text())
    euclid_meta = json.loads(euclid_catalog_meta_path().read_text())
    _require_current_gaia_field_sampling(meta, euclid_rows)
    if int(euclid_meta.get("version") or 0) != Q1_STELLAR_COLOR_SAMPLE_VERSION:
        raise ValueError(
            "Stellar colour cache is stale; press Query MER + PHZ to refresh "
            "the Euclid-only colour sample"
        )
    q1_counts = read_q1_phz_star_counts(bright=bright, faint=faint)
    magnitude_law, magnitude_fit_diagnostics = (
        _fit_straight_star_magnitude_law(gaia_rows, meta, q1_counts)
    )

    selected = _star_color_rows(euclid_rows)
    row_count = int(selected["weight"].size)
    if row_count < 200:
        raise ValueError(
            f"only {row_count} Q1 point sources are usable for stellar colours; "
            "at least 200 are required"
        )
    color_model = build_star_color_model(
        selected["magnitude"], selected["color"], selected["mag_sigma"],
        selected["weight"], mag_bright=bright, mag_faint=faint,
    )
    sampler = StarColorModel.from_payload(color_model)

    # Per observed row: its raw colours, one posterior draw of its true
    # colours (the same deconvolution generation uses) and that draw measured
    # again with the row's own flux errors, which must reproduce the raw
    # distribution.
    diagnostic_rng = np.random.default_rng(71591)
    latent_colors: list[np.ndarray] = []
    predictive_colors: list[np.ndarray] = []
    predictive_weights: list[float] = []
    for index in range(row_count):
        location, intrinsic = sampler.bin_moments[
            sampler.serving_bin(float(selected["magnitude"][index]))
        ]
        latent = deconvolved_draw(
            location, intrinsic, selected["color"][index],
            sampler.noise_covariance[index], diagnostic_rng,
        )
        latent_colors.append(latent)
        predictive = _renoised_colors(
            latent, selected["fluxes"][index], diagnostic_rng,
        )
        if predictive is not None:
            predictive_colors.append(predictive)
            predictive_weights.append(float(selected["weight"][index]))
    latent_weights = list(selected["weight"])
    dirty_colors = list(selected["color"])
    dirty_weights = list(selected["weight"])

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
    populated_bins = int(np.sum(
        np.asarray(color_model["bin_weight"]) >= color_model["min_bin_weight"]
    ))
    payload: dict[str, Any] = {
        "version": _STAR_POPULATION_VERSION,
        "kind": "star_population_fit",
        "valid": True,
        "warnings": [],
        "coverage_notes": [
            (
                f"normalized VIS counts with PHZ_STAR_PROB over "
                f"{float(q1_counts['footprint_area_deg2']):g} deg² of Q1"
            ),
            (
                f"colours from {row_count:,} Q1 point sources (no Gaia match); "
                f"{populated_bins} of {len(color_model['bin_source'])} 0.5-mag "
                "bins sampled on their own, the rest use the nearest one"
            ),
        ],
        "coverage": {
            "euclid_rows": len(euclid_rows),
            "color_rows": row_count,
            "color_weight": float(np.sum(selected["weight"])),
            "rejected_rows": selected["rejected"],
            "count_check_gaia_rows": len(gaia_rows),
        },
        "color_sample_provenance": {
            "version": int(euclid_meta["version"]),
            "field_count": int(euclid_meta["field_count"]),
            "radius_deg": float(euclid_meta["radius_deg"]),
            "area_arcmin2": float(euclid_meta["area_arcmin2"]),
            "sampling_kind": euclid_meta.get("sampling_kind"),
            "fields": euclid_meta.get("fields"),
            "random_centres": euclid_meta.get("random_centres"),
            "rows_per_bracket_limit": euclid_meta.get("rows_per_bracket_limit"),
            "selection": euclid_meta.get("selection"),
            "role": "Euclid point-source colours only",
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
        "classification_weighting": {
            "population_selection": "POINT_LIKE_PROB >= 0.9 over Q1",
            "population_star_weight": "PHZ_STAR_PROB over selected Q1 rows",
            "color_star_weight": "PHZ_STAR_PROB in the fixed Q1 colour sample",
        },
        "color_model": color_model,
        "fit": {
            "color_rows": row_count,
            "populated_bins": populated_bins,
            "selection": {
                "point_like_prob_min": 0.9,
                "color_band_snr_min": _COLOR_SNR_FLOOR,
                "binning_magnitude": "VIS PSF",
                "flux_fields": [
                    f"flux_{band}_aper_uJy" for band in _COLOR_BANDS
                ],
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
            "fit_version": _STAR_FIT_VERSION,
            "selection": {
                "bright_limit": bright,
                "faint_limit": faint,
                "population_selection": "POINT_LIKE_PROB >= 0.9",
                "population_probability_field": "PHZ_STAR_PROB",
                "color_probability_field": "PHZ_STAR_PROB",
                "flux_fields": [
                    f"flux_{band}_aper_uJy" for band in _COLOR_BANDS
                ],
                "error_fields": [
                    f"fluxerr_{band}_aper_uJy" for band in _COLOR_BANDS
                ],
            },
        },
    }
    model = EmpiricalStellarPrior.from_payload(payload)
    rng = np.random.default_rng(71033)
    fitted = [
        model.sample(rng, model.sample_magnitude(rng))
        for _ in range(10_000)
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
    gaia_bin_width = _GAIA_COUNT_FIT_BIN_WIDTH_MAG
    gaia_edges = _fixed_width_edges(bright, faint, gaia_bin_width)
    gaia_centres = 0.5 * (gaia_edges[:-1] + gaia_edges[1:])
    gaia_counts, _ = np.histogram(gaia_g_ab, bins=gaia_edges)
    gaia_density = (
        gaia_counts.astype(np.float64)
        / float(meta["area_arcmin2"])
        / np.diff(gaia_edges)
    )
    gaia_fitted_density = (
        np.power(
            10.0,
            float(gaia_fit["slope"]) * gaia_centres + float(gaia_fit["intercept"]),
        ).tolist()
        if gaia_fit is not None else None
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
            "gaia_fitted": gaia_fitted_density,
            "fit_ranges": {
                "q1": [
                    float(magnitude_fit_diagnostics["q1"]["fit_bright"]),
                    float(magnitude_fit_diagnostics["q1"]["fit_faint"]),
                ],
                "gaia": (
                    [float(gaia_fit["fit_bright"]), float(gaia_fit["fit_faint"])]
                    if gaia_fit is not None else None
                ),
            },
            "sampling_interval": [bright, faint],
            "label": "Q1 PHZ probability-weighted stellar density",
            "unit": "stars / arcmin² / mag",
            "x_label": "native survey magnitude [AB]",
        },
        "parameters": {},
    }
    for index, key in enumerate(_COLOR_ORDER):
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
    for key in _COLOR_ORDER:
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
        "low_snr_rows_rejected": int(selected["rejected"]["low_snr"]),
    }
    payload["quality_gates"] = {
        "global_median_error_max_mag": 0.08,
        "robust_width_ratio_range": [0.80, 1.25],
        "passed": bool(quality_pass),
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
    # Only the candidate is written here: the plot caches are written by
    # star_distribution_payload(persist=True), the fit job's next step, which
    # adds the generated stars and split keys every cached view carries.
    write_star_candidate(payload)
    return payload


def fit_star_population(
    *, faint_limit: float | None = None, bright_limit: float | None = None,
) -> dict[str, Any]:
    """Fit the required flux-aware empirical stellar prior.

    The legacy exponential/blackbody fit was deleted (2acd91f) and there is
    no fallback: a population artifact must be tied to raw Euclid fluxes and
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
    return _fit_star_population_euclid()
