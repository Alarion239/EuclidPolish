"""Six-variable corner plot: Euclid Q1 catalogue rows against model draws.

Variables: VIS 2FWHM magnitude, log10 SFR, log10 circularized Sérsic Rₑ, and
the three NISP/VIS colours. The lower triangle traces the Q1 rows the
colour+SFR forest is trained on (raw forced-photometry colours); the upper
triangle traces draws from the fitted joint model (deconvolved colours),
restricted to the Q1 VIS range so both describe the same population. Every
off-diagonal cell plots the column variable on x against the row variable
on y; the diagonal holds both samples' unit-area distributions.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import contourpy
import numpy as np
from scipy.ndimage import gaussian_filter

from euclid_polish.population.conditional_color_sfr import weighted_quantile
from euclid_polish.population.conditional_color_sfr_fit import (
    ColorSFRRows,
    read_color_sfr_rows,
)
from euclid_polish.population.euclid_galaxy_prior import ConditionalRadiusLaw
from euclid_polish.sky.generation.cosmos_tng_prior import (
    JointGalaxyPopulationPrior,
)

#: (key, axis label, unit) in row/column order.
CORNER_VARIABLES: tuple[tuple[str, str, str], ...] = (
    ("vis", "VIS 2FWHM", "AB mag"),
    ("log_sfr", "log₁₀ SFR", "M☉ yr⁻¹"),
    ("log_re", "log₁₀ Rₑ", "arcsec"),
    ("vis_minus_y", "VIS − Y", "AB mag"),
    ("y_minus_j", "Y − J", "AB mag"),
    ("j_minus_h", "J − H", "AB mag"),
)
CORNER_CONTOUR_MASS_FRACTIONS = (0.99, 0.95, 0.80, 0.50, 0.20)
CORNER_JOINT_BINS = 36
CORNER_DIAGONAL_BINS = 40
CORNER_SMOOTHING_SIGMA_BINS = 1.0
CORNER_MODEL_DRAWS = 6000
CORNER_MODEL_SEED = 20260914
CORNER_MIN_ROWS = 50
#: The pair explorer re-bins every pair finer and traces more levels; its
#: smoothing kernel keeps the corner's width in data units.
EXPLORER_BINS = 64
EXPLORER_SMOOTHING_SIGMA_BINS = (
    CORNER_SMOOTHING_SIGMA_BINS * EXPLORER_BINS / CORNER_JOINT_BINS
)
EXPLORER_CONTOUR_MASS_FRACTIONS = (0.99, 0.95, 0.80, 0.50, 0.20)
#: Plot windows span these weighted quantiles of the pooled samples; the
#: raw Q1 colours have noise-dominated tails that would otherwise flatten
#: every panel.
_WINDOW_QUANTILES = (0.01, 0.99)
#: Model draws are restricted to this Q1 VIS quantile range.
_VIS_QUANTILES = (0.005, 0.995)
_WINDOW_PADDING = 0.05
#: PHZ physical-parameter SFRs carry an unphysical tail down to log SFR ≈
#: −250 (about 5% of the SFR-valid Q1 rows fall below −5). Those values are
#: effectively zero SFR; the SFR window ignores them so it spans the
#: populated range, and they are reported as off-window.
CORNER_LOG_SFR_DISPLAY_FLOOR = -5.0


def mass_contour_thresholds(
    density: np.ndarray,
    cell_mass: np.ndarray,
    fractions: tuple[float, ...],
) -> list[tuple[float, float]]:
    """Density thresholds enclosing fixed fractions of joint population mass."""
    values = np.asarray(density, dtype=np.float64).ravel()
    mass = np.asarray(cell_mass, dtype=np.float64).ravel()
    keep = (
        np.isfinite(values) & np.isfinite(mass)
        & (values > 0.0) & (mass > 0.0)
    )
    values, mass = values[keep], mass[keep]
    if values.size < 2 or float(np.sum(mass)) <= 0.0:
        return []
    order = np.argsort(values)[::-1]
    cumulative = np.cumsum(mass[order]) / np.sum(mass)
    return [
        (
            fraction,
            float(values[order[min(
                int(np.searchsorted(cumulative, fraction)),
                len(order) - 1,
            )]]),
        )
        for fraction in fractions
    ]


def mass_fraction_contours(
    density: np.ndarray,
    cell_mass: np.ndarray,
    x_center: np.ndarray,
    y_center: np.ndarray,
    fractions: tuple[float, ...],
) -> list[dict[str, Any]]:
    """Trace plot-ready contours enclosing the requested mass fractions.

    ``density`` and ``cell_mass`` are indexed ``[x, y]``.
    """
    generator = contourpy.contour_generator(
        x=np.asarray(x_center, dtype=np.float64),
        y=np.asarray(y_center, dtype=np.float64),
        z=np.asarray(density, dtype=np.float64).T,
        corner_mask=True,
    )
    contours = []
    seen: set[float] = set()
    for mass_fraction, level in mass_contour_thresholds(
        density, cell_mass, fractions,
    ):
        # Sparse histograms can assign more than one enclosed-mass fraction to
        # the same density threshold.  One geometric line is sufficient; its
        # label retains every represented fraction.
        rounded = round(level, 12)
        if rounded in seen:
            continue
        seen.add(rounded)
        paths = []
        for raw_vertices in generator.lines(level):
            vertices = np.asarray(raw_vertices, dtype=np.float64)
            if vertices.ndim != 2 or vertices.shape[1] < 2:
                continue
            if vertices.shape[0] < 2:
                continue
            paths.append({
                "x": vertices[:, 0].astype(float).tolist(),
                "y": vertices[:, 1].astype(float).tolist(),
            })
        if paths:
            contours.append({
                "mass_fraction": mass_fraction,
                "level": level,
                "paths": paths,
            })
    return contours


def _q1_matrix(rows: ColorSFRRows) -> np.ndarray:
    """Per-row corner variables; NaN where a variable is not measured."""
    ratio = rows.ratio
    positive = ratio > 0.0
    log_ratio = np.where(
        positive, np.log10(np.where(positive, ratio, 1.0)), np.nan,
    )
    return np.column_stack((
        rows.magnitude,
        np.where(rows.sfr_valid, rows.log_sfr, np.nan),
        np.where(rows.resolved, rows.log_radius, np.nan),
        2.5 * log_ratio[:, 0],
        2.5 * (log_ratio[:, 1] - log_ratio[:, 0]),
        2.5 * (log_ratio[:, 2] - log_ratio[:, 1]),
    ))


def _model_matrix(
    candidate: dict[str, Any],
    vis_range: tuple[float, float],
    draws: int,
    seed: int,
) -> np.ndarray:
    """Draw from the fitted joint model inside the Q1 VIS range."""
    prior = JointGalaxyPopulationPrior({**candidate, "active": True})
    rng = np.random.default_rng(seed)
    samples = np.empty((draws, len(CORNER_VARIABLES)), dtype=np.float64)
    kept = 0
    for _ in range(50 * draws):
        if kept == draws:
            break
        geometry = prior.sample_geometry(rng)
        magnitude, _flux = prior.sample_brightness(
            rng, radius_arcsec=geometry.re_arcsec,
        )
        if not vis_range[0] <= magnitude <= vis_range[1]:
            continue
        color = prior.color_sfr_sampler.sample(
            magnitude, geometry.re_arcsec, rng,
        )
        samples[kept] = (
            magnitude,
            color.log_sfr,
            math.log10(geometry.re_arcsec),
            color.vis_minus_y,
            color.y_minus_j,
            color.j_minus_h,
        )
        kept += 1
    if kept < CORNER_MIN_ROWS:
        raise ValueError(
            "the fitted model produced too few draws inside the Q1 VIS range"
        )
    return samples[:kept]


def _window(
    q1_values: np.ndarray,
    q1_weight: np.ndarray,
    model_values: np.ndarray,
    *,
    floor: float = -np.inf,
) -> tuple[float, float]:
    """Pooled plot window from both samples' central quantiles above ``floor``."""
    bounds: list[float] = []
    finite = np.isfinite(q1_values) & (q1_values >= floor)
    if int(np.sum(finite)) >= CORNER_MIN_ROWS:
        bounds.extend(float(value) for value in weighted_quantile(
            q1_values[finite], q1_weight[finite], _WINDOW_QUANTILES,
        ))
    finite = np.isfinite(model_values) & (model_values >= floor)
    if int(np.sum(finite)) >= CORNER_MIN_ROWS:
        bounds.extend(float(value) for value in np.quantile(
            model_values[finite], _WINDOW_QUANTILES,
        ))
    if not bounds:
        return (0.0, 1.0)
    low, high = min(bounds), max(bounds)
    span = max(high - low, 1e-3)
    return (low - _WINDOW_PADDING * span, high + _WINDOW_PADDING * span)


def _outside_fraction(
    values: np.ndarray, weight: np.ndarray, window: tuple[float, float],
) -> float:
    """Weighted fraction of measured values that fall outside the window."""
    finite = np.isfinite(values)
    total = float(np.sum(weight[finite]))
    if total <= 0.0:
        return 0.0
    outside = finite & ((values < window[0]) | (values > window[1]))
    return float(np.sum(weight[outside])) / total


def _unit_area_histogram(
    values: np.ndarray, weight: np.ndarray, edges: np.ndarray,
) -> list[float]:
    finite = np.isfinite(values)
    counts, _ = np.histogram(values[finite], edges, weights=weight[finite])
    total = float(np.sum(counts * np.diff(edges)))
    if total <= 0.0:
        return [0.0] * (edges.size - 1)
    return [round(float(value), 6) for value in counts / total]


def _rounded_contours(contours: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "mass_fraction": contour["mass_fraction"],
            "paths": [
                {
                    "x": [round(value, 4) for value in path["x"]],
                    "y": [round(value, 4) for value in path["y"]],
                }
                for path in contour["paths"]
            ],
        }
        for contour in contours
    ]


def _joint_layer(
    matrix: np.ndarray,
    weight: np.ndarray,
    *,
    x_index: int,
    y_index: int,
    windows: list[tuple[float, float]],
    bins: int,
    sigma_bins: float,
    fractions: tuple[float, ...],
    include_density: bool = False,
) -> dict[str, Any]:
    """Smoothed joint density of two corner variables inside their windows.

    ``density`` (when requested) is indexed ``[x-bin, y-bin]`` and scaled to
    a peak of one; the bins are uniform, so it is proportional to cell mass.
    """
    x, y = matrix[:, x_index], matrix[:, y_index]
    x_window, y_window = windows[x_index], windows[y_index]
    inside = (
        np.isfinite(x) & np.isfinite(y)
        & (x >= x_window[0]) & (x <= x_window[1])
        & (y >= y_window[0]) & (y <= y_window[1])
    )
    count = int(np.sum(inside))
    layer: dict[str, Any] = {"rows": count, "contours": []}
    if include_density:
        layer["density"] = []
    if count < CORNER_MIN_ROWS:
        return layer
    mass, x_edges, y_edges = np.histogram2d(
        x[inside], y[inside],
        bins=bins,
        range=[x_window, y_window],
        weights=weight[inside],
    )
    smoothed = gaussian_filter(
        mass, sigma=sigma_bins, mode="constant", cval=0.0,
    )
    layer["contours"] = _rounded_contours(mass_fraction_contours(
        smoothed, smoothed,
        0.5 * (x_edges[:-1] + x_edges[1:]),
        0.5 * (y_edges[:-1] + y_edges[1:]),
        fractions,
    ))
    peak = float(np.max(smoothed))
    if include_density and peak > 0.0:
        layer["density"] = np.round(smoothed / peak, 3).tolist()
    return layer


def build_galaxy_corner(
    catalog_path: str | Path,
    candidate: dict[str, Any] | None,
    *,
    model_draws: int = CORNER_MODEL_DRAWS,
    seed: int = CORNER_MODEL_SEED,
) -> dict[str, Any]:
    """Build the 6×6 corner-plot payload (raises ``ValueError`` when blocked)."""
    if not candidate:
        raise ValueError("Fit the galaxy population model first.")
    try:
        radius_law = ConditionalRadiusLaw.from_payload(candidate["radius_law"])
    except (KeyError, TypeError) as exc:
        raise ValueError("The galaxy population model has no radius law") from exc
    rows = read_color_sfr_rows(catalog_path, radius_law=radius_law)
    if rows.weight.size < CORNER_MIN_ROWS:
        raise ValueError("The Euclid cache has too few colour-model rows")
    q1 = _q1_matrix(rows)
    q1_weight = rows.weight
    vis_low, vis_high = (float(value) for value in weighted_quantile(
        rows.magnitude, rows.weight, _VIS_QUANTILES,
    ))
    model = _model_matrix(
        candidate, (vis_low, vis_high), int(model_draws), int(seed),
    )
    model_weight = np.ones(model.shape[0], dtype=np.float64)

    vis_padding = _WINDOW_PADDING * (vis_high - vis_low)
    windows: list[tuple[float, float]] = []
    for index, (key, _label, _unit) in enumerate(CORNER_VARIABLES):
        if key == "vis":
            windows.append((vis_low - vis_padding, vis_high + vis_padding))
        else:
            windows.append(_window(
                q1[:, index], q1_weight, model[:, index],
                floor=(
                    CORNER_LOG_SFR_DISPLAY_FLOOR if key == "log_sfr"
                    else -np.inf
                ),
            ))
    diagonal = []
    for index, window in enumerate(windows):
        edges = np.linspace(window[0], window[1], CORNER_DIAGONAL_BINS + 1)
        diagonal.append({
            "edges": [float(value) for value in edges],
            "q1": _unit_area_histogram(q1[:, index], q1_weight, edges),
            "model": _unit_area_histogram(
                model[:, index], model_weight, edges,
            ),
        })
    samples = {"q1": (q1, q1_weight), "model": (model, model_weight)}
    cells = []
    count = len(CORNER_VARIABLES)
    for row in range(count):
        for col in range(count):
            if row == col:
                continue
            source = "q1" if row > col else "model"
            matrix, weight = samples[source]
            cells.append({
                "row": row, "col": col, "source": source,
                **_joint_layer(
                    matrix, weight, x_index=col, y_index=row,
                    windows=windows, bins=CORNER_JOINT_BINS,
                    sigma_bins=CORNER_SMOOTHING_SIGMA_BINS,
                    fractions=CORNER_CONTOUR_MASS_FRACTIONS,
                ),
            })
    keys = [key for key, _label, _unit in CORNER_VARIABLES]
    pairs = []
    for low in range(count):
        for high in range(low + 1, count):
            pair: dict[str, Any] = {
                "x": keys[low],
                "y": keys[high],
                "x_edges": np.linspace(
                    windows[low][0], windows[low][1], EXPLORER_BINS + 1,
                ).tolist(),
                "y_edges": np.linspace(
                    windows[high][0], windows[high][1], EXPLORER_BINS + 1,
                ).tolist(),
            }
            for source, (matrix, weight) in samples.items():
                pair[source] = _joint_layer(
                    matrix, weight, x_index=low, y_index=high,
                    windows=windows, bins=EXPLORER_BINS,
                    sigma_bins=EXPLORER_SMOOTHING_SIGMA_BINS,
                    fractions=EXPLORER_CONTOUR_MASS_FRACTIONS,
                    include_density=True,
                )
            pairs.append(pair)
    return {
        "available": True,
        "variables": [
            {
                "key": key, "label": label, "unit": unit,
                "domain": [float(window[0]), float(window[1])],
                "outside_fraction": {
                    "q1": round(_outside_fraction(
                        q1[:, index], q1_weight, window,
                    ), 4),
                    "model": round(_outside_fraction(
                        model[:, index], model_weight, window,
                    ), 4),
                },
            }
            for index, ((key, label, unit), window) in enumerate(zip(
                CORNER_VARIABLES, windows, strict=True,
            ))
        ],
        "contour_mass_fractions": list(CORNER_CONTOUR_MASS_FRACTIONS),
        "diagonal": diagonal,
        "cells": cells,
        "pairs": pairs,
        "explorer_contour_mass_fractions": list(
            EXPLORER_CONTOUR_MASS_FRACTIONS
        ),
        "q1_rows": int(rows.weight.size),
        "model_draws": int(model.shape[0]),
        "vis_range": [round(vis_low, 3), round(vis_high, 3)],
    }


def split_joint_pairs(
    corner: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate the explorer grids from the page-sized corner payload."""
    slim = {
        key: value for key, value in corner.items()
        if key not in ("pairs", "explorer_contour_mass_fractions")
    }
    sidecar = {
        "available": bool(corner.get("available")) and "pairs" in corner,
        "detail": corner.get("detail"),
        "variables": corner.get("variables", []),
        "diagonal": corner.get("diagonal", []),
        "pairs": corner.get("pairs", []),
        "contour_mass_fractions": corner.get(
            "explorer_contour_mass_fractions", [],
        ),
        "q1_rows": corner.get("q1_rows"),
        "model_draws": corner.get("model_draws"),
        "vis_range": corner.get("vis_range"),
    }
    return slim, sidecar


def _transposed(layer: dict[str, Any]) -> dict[str, Any]:
    return {
        "rows": layer["rows"],
        "density": [
            list(column) for column in zip(*layer["density"], strict=True)
        ],
        "contours": [
            {
                "mass_fraction": contour["mass_fraction"],
                "paths": [
                    {"x": path["y"], "y": path["x"]}
                    for path in contour["paths"]
                ],
            }
            for contour in layer["contours"]
        ],
    }


def orient_joint_pair(
    sidecar: dict[str, Any], x_key: str, y_key: str,
) -> dict[str, Any]:
    """One explorer view with ``x_key`` on x and ``y_key`` on y.

    Pairs are stored once (lower variable index on x); the reverse view is
    the transpose. Choosing the same variable twice returns its 1-D
    distributions. Unknown keys raise ``ValueError``.
    """
    if not sidecar.get("available"):
        return {
            "available": False,
            "detail": sidecar.get("detail")
            or "Rebuild cached plots to draw the joint distributions.",
        }
    variables = sidecar["variables"]
    keys = [variable["key"] for variable in variables]
    if x_key not in keys or y_key not in keys:
        raise ValueError(
            f"unknown variable; choose from {', '.join(keys)}"
        )
    x_index, y_index = keys.index(x_key), keys.index(y_key)
    view = {
        "available": True,
        "x": variables[x_index],
        "y": variables[y_index],
        "contour_mass_fractions": sidecar["contour_mass_fractions"],
        "q1_rows": sidecar["q1_rows"],
        "model_draws": sidecar["model_draws"],
        "vis_range": sidecar["vis_range"],
    }
    if x_index == y_index:
        return {
            **view, "kind": "marginal",
            "diagonal": sidecar["diagonal"][x_index],
        }
    low, high = sorted((x_index, y_index))
    pair = next(
        (
            candidate for candidate in sidecar["pairs"]
            if candidate["x"] == keys[low] and candidate["y"] == keys[high]
        ),
        None,
    )
    if pair is None:
        raise ValueError(f"no stored pair for {keys[low]} × {keys[high]}")
    if x_index == low:
        return {
            **view, "kind": "joint",
            "x_edges": pair["x_edges"], "y_edges": pair["y_edges"],
            "q1": pair["q1"], "model": pair["model"],
        }
    return {
        **view, "kind": "joint",
        "x_edges": pair["y_edges"], "y_edges": pair["x_edges"],
        "q1": _transposed(pair["q1"]), "model": _transposed(pair["model"]),
    }
