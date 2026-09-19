"""Euclid Q1 sky noise levels: the per-scene noise-level distribution.

``mer_noise_levels.json`` is written by ``scripts/download_mer_noise_levels.py``
from Euclid's own MER noise (RMS) maps: one observed position in every
extragalactic Q1 tile, with the sky level in VIS, Y, J and H read at the same
position. Each level is the median over a 25.6" cutout — the size of one
generated scene, so the level is measured at the scale it is used. Drawing one
whole row per scene keeps how the bands' depths move together: Y, J and H
almost in lockstep, VIS only loosely.

Rows also carry ``sub_levels_e``, the 4x4 grid of 6.4" sub-tile levels inside
each cutout. Nothing in the generator reads it yet; it measures how the depth
varies within a single field (pointing seams), which the field-wide scale and
the regional strip stand for.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

TABLE_PATH = Path(__file__).with_name("mer_noise_levels.json")
KIND = "euclid_q1_mer_noise_levels"
VERSION = 1


@dataclass(frozen=True)
class MERNoiseLevels:
    """Measured sky noise levels, one row per sky position."""

    bands: tuple[str, ...]
    levels_e: np.ndarray
    fields: tuple[str, ...]

    def __post_init__(self) -> None:
        levels = np.asarray(self.levels_e, dtype=np.float64)
        if levels.ndim != 2 or levels.shape[1] != len(self.bands) or len(levels) < 1:
            raise ValueError(
                f"levels_e must have shape (positions, {len(self.bands)}), "
                f"got {levels.shape}"
            )
        if not np.all(np.isfinite(levels)) or np.any(levels <= 0.0):
            raise ValueError("noise levels must be finite and positive")
        if len(self.fields) != len(levels):
            raise ValueError("every row needs a field label")
        levels.setflags(write=False)
        object.__setattr__(self, "levels_e", levels)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> MERNoiseLevels:
        if payload.get("kind") != KIND or payload.get("version") != VERSION:
            raise ValueError(
                f"not a {KIND} v{VERSION} table: "
                f"kind={payload.get('kind')!r}, version={payload.get('version')!r}"
            )
        rows = payload["rows"]
        return cls(
            bands=tuple(payload["bands"]),
            levels_e=np.array([row["levels_e"] for row in rows], dtype=np.float64),
            fields=tuple(str(row["field"]) for row in rows),
        )

    def draw(self, rng: np.random.Generator) -> dict[str, float]:
        """Every band's level at one randomly chosen real sky position."""
        row = self.levels_e[int(rng.integers(len(self.levels_e)))]
        return {band: float(value) for band, value in zip(self.bands, row, strict=True)}

    def median(self, band: str) -> float:
        return float(np.median(self.levels_e[:, self.bands.index(band)]))


@lru_cache(maxsize=4)
def load_mer_noise_levels(path: str = str(TABLE_PATH)) -> MERNoiseLevels:
    """The committed level table (cached after the first read)."""
    with open(path, encoding="utf-8") as handle:
        return MERNoiseLevels.from_payload(json.load(handle))


__all__ = ["MERNoiseLevels", "TABLE_PATH", "load_mer_noise_levels"]
