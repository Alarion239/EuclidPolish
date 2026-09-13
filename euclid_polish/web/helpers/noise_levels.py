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
            "scale": [
                defaults["noise_region_scale_min"], defaults["noise_region_scale_max"],
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
