"""The Noise tab's payload: the sky-noise distribution scenes are drawn from.

Everything is computed from the committed Euclid Q1 level table
(``euclid_polish/sky/observation/mer_noise_levels.json``) and the generator's
defaults, so the tab needs no network or FASRC connection.
"""

from __future__ import annotations

import dataclasses
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from euclid_polish.config import Config
from euclid_polish.sky.observation.mer_noise_levels import (
    TABLE_PATH,
    load_mer_noise_levels,
)
from euclid_polish.sky.observation.noise import dithered_unit_noise
from euclid_polish.sky.observation.observation_simulator import (
    ObservationSimulatorConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
QUANTILES = {"p5": 0.05, "p16": 0.16, "median": 0.50, "p84": 0.84, "p95": 0.95}
BIN_WIDTH_DEX = 0.02
PIXEL_SCATTER_SHAPE = (512, 512)
# A seam is a straight-line depth step with both sides internally uniform. A
# field whose brightest sub-tile merely exceeds the rest is usually a source
# inflating the noise map, not a pointing boundary.
SEAM_STEP_MIN = 1.10
SEAM_UNIFORM_MAX = 1.10
SEAM_BIN_WIDTH = 0.025
SEAM_BIN_MAX = 1.60


def _quantiles(values: np.ndarray) -> dict[str, Any]:
    summary: dict[str, Any] = {"count": int(len(values)), "min": round(float(values.min()), 4)}
    for name, q in QUANTILES.items():
        summary[name] = round(float(np.quantile(values, q)), 4)
    summary["max"] = round(float(values.max()), 4)
    return summary


def log10_edges(low: float, high: float) -> np.ndarray:
    """Bin edges in log10(level), BIN_WIDTH_DEX wide, covering [low, high]."""
    start = math.floor(math.log10(low) / BIN_WIDTH_DEX)
    stop = math.ceil(math.log10(high) / BIN_WIDTH_DEX)
    return np.arange(start, stop + 1) * BIN_WIDTH_DEX


def jittered_counts(
    values: np.ndarray, edges: np.ndarray, low: float, high: float,
) -> np.ndarray:
    """Expected histogram of value × Uniform(low, high), one draw per value.

    Each product is uniform on [low·v, high·v], so a bin receives the fraction
    of that interval it overlaps. ``edges`` are in linear level units.
    """
    values = np.asarray(values, dtype=np.float64)[:, None]
    if high <= low:
        return np.histogram(values[:, 0] * low, edges)[0].astype(np.float64)
    overlap = (
        np.minimum(edges[None, 1:], high * values)
        - np.maximum(edges[None, :-1], low * values)
    )
    return (np.clip(overlap, 0.0, None) / ((high - low) * values)).sum(axis=0)


def pixel_scatter_ratio(band_name: str) -> float:
    """Single-pixel std of the unit noise field, whose large-area variance is one."""
    band = Config.get_band(band_name)
    field = dithered_unit_noise(PIXEL_SCATTER_SHAPE, band, np.random.default_rng(0))
    return float(np.std(field))


def seam_splits(side: int) -> list[np.ndarray]:
    """Every straight-line bipartition of a ``side``×``side`` sub-tile grid.

    Rows, columns and both diagonals, keeping only splits that leave at least a
    quarter of the field on each side — a pointing boundary crossing a scene.
    """
    yy, xx = np.mgrid[:side, :side]
    masks = []
    for k in range(1, side):
        masks += [yy < k, xx < k]
    for k in range(-(side - 2), side):
        masks += [(yy + xx) <= k, (yy - xx) <= k - (side - 2)]
    cells = side * side
    return [
        mask.reshape(-1) for mask in masks
        if cells // 4 <= int(mask.sum()) <= 3 * cells // 4
    ]


def seam_step(
    grid: list, splits: list[np.ndarray],
) -> tuple[float, float] | None:
    """Largest straight-line depth step inside one field, and its uniformity.

    Returns ``(step, scatter)`` where ``step`` is the ratio of the two sides'
    median levels (always >= 1) and ``scatter`` is how far the brighter tile of
    either side sits above that side's median. A real seam steps cleanly, so it
    has a large ``step`` and a ``scatter`` near one; a source gives both.
    """
    values = np.array(
        [np.nan if value is None else float(value) for value in grid],
        dtype=np.float64,
    )
    side = int(round(math.sqrt(values.size)))
    if side * side != values.size or np.isnan(values).sum() > side:
        return None
    best: tuple[float, float] | None = None
    for mask in splits:
        near, far = values[mask], values[~mask]
        near, far = near[~np.isnan(near)], far[~np.isnan(far)]
        if len(near) < 3 or len(far) < 3:
            continue
        median_near, median_far = float(np.median(near)), float(np.median(far))
        if median_near <= 0.0 or median_far <= 0.0:
            continue
        step = max(median_near / median_far, median_far / median_near)
        scatter = max(
            float(np.max(near)) / median_near, float(np.max(far)) / median_far,
        )
        if best is None or step > best[0]:
            best = (step, scatter)
    return best


def _within_field(table: dict[str, Any], bands: list[str]) -> dict[str, Any] | None:
    """Per-band seam statistics from the rows' sub-tile grids.

    ``None`` when the committed table predates ``sub_levels_e``.
    """
    rows = table.get("rows") or []
    if not rows or "sub_levels_e" not in rows[0]:
        return None
    side = int(round(math.sqrt(len(rows[0]["sub_levels_e"][bands[0]]))))
    splits = seam_splits(side)
    edges = np.arange(1.0, SEAM_BIN_MAX + 0.5 * SEAM_BIN_WIDTH, SEAM_BIN_WIDTH)
    by_band: dict[str, Any] = {}
    for band in bands:
        steps, seams = [], []
        for row in rows:
            measured = seam_step(row["sub_levels_e"][band], splits)
            if measured is None:
                continue
            step, scatter = measured
            steps.append(step)
            if step >= SEAM_STEP_MIN and scatter < SEAM_UNIFORM_MAX:
                seams.append(step)
        every, clean = np.array(steps), np.array(seams)
        by_band[band] = {
            "fields": int(every.size),
            "seam_count": int(clean.size),
            "seam_rate": round(float(clean.size / every.size), 4) if every.size else 0.0,
            "counts": np.histogram(
                np.clip(every, None, float(edges[-1]) - 1e-9), edges,
            )[0].tolist(),
            "steps": {
                "p50": round(float(np.quantile(clean, 0.5)), 3),
                "p90": round(float(np.quantile(clean, 0.9)), 3),
                "max": round(float(clean.max()), 3),
            } if clean.size else None,
        }
    return {
        "cutout_arcsec": round(
            float(table["cutout_pixels"]) * Config.VIS_PIXEL_SCALE_ARCSEC, 2,
        ),
        "sub_tile_arcsec": round(
            float(table["sub_tile_pixels"]) * Config.VIS_PIXEL_SCALE_ARCSEC, 2,
        ),
        "grid_side": side,
        "step_edges": [round(float(edge), 4) for edge in edges],
        "step_threshold": SEAM_STEP_MIN,
        "uniformity_threshold": SEAM_UNIFORM_MAX,
        "bands": by_band,
    }


def _generator_defaults() -> dict[str, Any]:
    defaults = {
        field.name: field.default
        for field in dataclasses.fields(ObservationSimulatorConfig)
    }
    varied = bool(defaults["add_noise_variation"])
    return {
        "noise_model": Config.NOISE_MODEL,
        "draws_measured_levels": bool(defaults["draw_mer_noise_levels"]),
        "scene_scale": [
            defaults["noise_global_scale_min"], defaults["noise_global_scale_max"],
        ] if varied else None,
        "region": {
            "probability": defaults["noise_region_probability"],
            "fraction": [
                defaults["noise_region_fraction_min"],
                defaults["noise_region_fraction_max"],
            ],
            "step": [
                defaults["noise_region_step_min"], defaults["noise_region_step_max"],
            ],
        } if varied else None,
    }


@lru_cache(maxsize=1)
def noise_levels_payload() -> dict[str, Any]:
    with TABLE_PATH.open(encoding="utf-8") as handle:
        table = json.load(handle)
    levels = load_mer_noise_levels()
    bands = list(levels.bands)
    values = levels.levels_e
    row_fields = np.array(levels.fields)
    fields = [name for name in table["fields"] if np.any(row_fields == name)]
    generator = _generator_defaults()
    low, high = generator["scene_scale"] or (1.0, 1.0)

    summary: dict[str, Any] = {}
    histograms: dict[str, Any] = {}
    for index, band in enumerate(bands):
        column = values[:, index]
        summary[band] = {
            **_quantiles(column),
            "pixel_scatter_ratio": round(pixel_scatter_ratio(band), 4),
        }
        edges = log10_edges(low * column.min(), high * column.max())
        linear_edges = 10.0 ** edges
        histograms[band] = {
            "log10_edges": [round(float(edge), 4) for edge in edges],
            "counts_by_field": {
                name: np.histogram(column[row_fields == name], linear_edges)[0].tolist()
                for name in fields
            },
            "jittered_counts": [
                round(float(count), 4)
                for count in jittered_counts(column, linear_edges, low, high)
            ],
        }

    return {
        "bands": bands,
        "source": {
            "release": table["release"],
            "archive": table["archive"],
            "description": table["description"],
            "units": table["units"],
            "retrieved_first": table["retrieved_first"],
            "retrieved_last": table["retrieved_last"],
            "tiles_attempted": int(table["tiles_attempted"]),
            "position_count": int(len(values)),
            "unobserved_tiles": len(table["rejected"]),
            "table_path": str(TABLE_PATH.relative_to(REPO_ROOT)),
        },
        "generator": generator,
        "summary": summary,
        "fields": [
            {
                "name": name,
                "positions": int(np.sum(row_fields == name)),
                "bands": {
                    band: _quantiles(values[row_fields == name, index])
                    for index, band in enumerate(bands)
                },
            }
            for name in fields
        ],
        "histograms": histograms,
        "within_field": _within_field(table, bands),
        "log_correlation": np.round(
            np.corrcoef(np.log(values), rowvar=False), 4,
        ).tolist(),
        "positions": [
            {
                "field": row["field"],
                "tile": row["tile"],
                "ra": row["ra"],
                "dec": row["dec"],
                "levels_e": row["levels_e"],
            }
            for row in table["rows"]
        ],
    }
