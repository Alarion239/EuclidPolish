"""Which Euclid release the population calibration is measured on.

The galaxy/star population calibration (aperture counts, radius and FWHM laws,
the colour+SFR catalogue, star counts and colours) was built on Q1, and Q1 stays
the default. Setting ``EUCLID_POLISH_CALIBRATION_RELEASE=dr1`` points every one
of those archive queries at the DR1 Wide Survey instead:

* tables: ``dr1.*_wide_survey`` / ``dr1.phz_*_wide`` on the ESA archive (same
  columns as Q1's ``catalogue.*``);
* footprint: the frozen sample of interior Wide tiles in
  ``dr1_calibration_tiles.json`` (``tile_index IN (...)``, indexed on ESA), with
  area = tile count x the 30' x 30' core;
* outputs: ``<DATA_DIR>/population_comparison/dr1/``, so a DR1 run never touches
  the active Q1 artifacts or calibrations.

The release is read from the environment on every call, so one process (or one
test) can switch it.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from euclid_polish.config import Config
from euclid_polish.sky.observation.q1_fields import Q1_FIELD_CONES, Q1_FIELDS

RELEASE_ENV = "EUCLID_POLISH_CALIBRATION_RELEASE"
DR1_TILES_PATH = Path(__file__).with_name("dr1_calibration_tiles.json")

#: The released Q1 deep-field footprint (the Q1 count normalisation).
Q1_AREA_DEG2 = 63.1
#: Radius of each stellar-colour field (the Q1 sample uses three such cones).
COLOR_FIELD_RADIUS_DEG = 0.35
#: DR1 colour fields: cones on this many sample tiles. Interior tiles have
#: Wide neighbours, so a 0.35 deg cone around a centre stays inside the survey.
DR1_COLOR_FIELD_COUNT = 8
#: DR1 colour+SFR catalogue: 5' cones on this many sample tiles (Q1 used 24
#: cones on saved star positions).
DR1_POPULATION_CONE_COUNT = 24


@dataclass(frozen=True)
class CalibrationRelease:
    """Tables, footprint and output folder of one calibration release."""

    name: str
    survey: str
    tables: Mapping[str, str]
    area_deg2: float
    fields: tuple[str, ...]
    color_fields: tuple[tuple[float, float, str], ...]
    subdir: str = ""
    cones: tuple[tuple[float, float, float], ...] = ()
    tiles: tuple[str, ...] = ()
    irsa_tables: Mapping[str, str] = field(default_factory=dict)
    #: ESA's DR1 tables time out on cone + LEFT OUTER JOIN; the colour+SFR
    #: catalogue query then runs as INNER parts joined client-side.
    split_outer_joins: bool = False

    @property
    def area_arcmin2(self) -> float:
        return self.area_deg2 * 3600.0

    def table(self, kind: str) -> str:
        """The ESA table of ``kind`` (mer, morph, phz_class, photo_z, phys)."""
        return self.tables[kind]

    def footprint_predicate(self, alias: str = "mer") -> str:
        """ADQL clause restricting ``alias`` (a MER table) to the footprint."""
        if self.tiles:
            listed = ", ".join(self.tiles)
            return f"{alias}.tile_index IN ({listed})"
        clauses = [
            (
                f"CONTAINS(POINT('ICRS', {alias}.right_ascension, {alias}.declination), "
                f"CIRCLE('ICRS', {ra}, {dec}, {radius})) = 1"
            )
            for ra, dec, radius in self.cones
        ]
        return "(\n        " + "\n        OR ".join(clauses) + "\n      )"

    def footprint_text(self) -> str:
        if self.tiles:
            return f"{len(self.tiles)} sampled DR1 Wide tiles"
        words = {1: "one", 2: "two", 3: "three"}
        count = words.get(len(self.cones), str(len(self.cones)))
        return f"{count} {self.name.upper()} deep-field regions"

    def population_dir(self, data_dir: str | os.PathLike[str]) -> Path:
        base = Path(data_dir) / "population_comparison"
        return base / self.subdir if self.subdir else base


Q1 = CalibrationRelease(
    name="q1",
    survey="Euclid Q1 deep fields",
    tables={
        "mer": "catalogue.mer_catalogue",
        "morph": "catalogue.mer_morphology",
        "phz_class": "catalogue.phz_classification",
        "photo_z": "catalogue.phz_photo_z",
        "phys": "catalogue.phz_physical_parameters",
    },
    area_deg2=Q1_AREA_DEG2,
    fields=tuple(f.name for f in Q1_FIELDS),
    color_fields=tuple((f.ra, f.dec, f.name) for f in Q1_FIELDS),
    cones=Q1_FIELD_CONES,
    irsa_tables={
        "mer": "euclid_q1_mer_catalogue",
        "phz_class": "euclid_q1_phz_classification",
    },
)


@cache
def _dr1_tiles() -> dict:
    return json.loads(DR1_TILES_PATH.read_text())


def dr1_tile_rows() -> list[dict]:
    """The frozen DR1 sample tiles (tile, ra, dec, galactic_b, rows)."""
    return list(_dr1_tiles()["tiles"])


@cache
def _dr1() -> CalibrationRelease:
    payload = _dr1_tiles()
    tiles = payload["tiles"]
    if not tiles or not math.isfinite(float(payload["area_deg2"])):
        raise ValueError(f"{DR1_TILES_PATH.name} has no usable tiles")
    return CalibrationRelease(
        name="dr1",
        survey=f"Euclid DR1 Wide Survey ({len(tiles)}-tile sample)",
        tables={
            "mer": "dr1.mer_catalogue_wide_survey",
            "morph": "dr1.mer_morphology_wide_survey",
            "phz_class": "dr1.phz_classification_wide",
            "photo_z": "dr1.phz_photo_z_wide",
            "phys": "dr1.phz_physical_parameters_wide",
        },
        area_deg2=float(payload["area_deg2"]),
        fields=tuple(f"TILE{tile['tile']}" for tile in tiles),
        color_fields=tuple(
            (float(tile["ra"]), float(tile["dec"]), f"TILE{tile['tile']}")
            for tile in tiles[:DR1_COLOR_FIELD_COUNT]
        ),
        subdir="dr1",
        tiles=tuple(str(int(tile["tile"])) for tile in tiles),
        irsa_tables={
            "mer": "euclid_dr1_mer_catalogue_wide_survey",
            "phz_class": "euclid_dr1_phz_classification_wide",
        },
        split_outer_joins=True,
    )


def release(name: str) -> CalibrationRelease:
    key = name.strip().lower()
    if key in ("", "q1"):
        return Q1
    if key == "dr1":
        return _dr1()
    raise ValueError(f"unknown calibration release {name!r} (expected q1 or dr1)")


def active_release() -> CalibrationRelease:
    """The release named by ``EUCLID_POLISH_CALIBRATION_RELEASE`` (default Q1)."""
    return release(os.environ.get(RELEASE_ENV, "q1"))


def population_dir() -> Path:
    """``population_comparison`` folder of the active release."""
    return active_release().population_dir(Config.DATA_DIR)


def dr1_population_cone_centers() -> list[dict]:
    """Centres for the DR1 colour+SFR catalogue cones (sample tile centres)."""
    return [
        {"ra": float(tile["ra"]), "dec": float(tile["dec"]), "tile": str(tile["tile"])}
        for tile in dr1_tile_rows()[:DR1_POPULATION_CONE_COUNT]
    ]


__all__ = [
    "COLOR_FIELD_RADIUS_DEG",
    "DR1_TILES_PATH",
    "Q1",
    "Q1_AREA_DEG2",
    "RELEASE_ENV",
    "CalibrationRelease",
    "active_release",
    "dr1_population_cone_centers",
    "dr1_tile_rows",
    "population_dir",
    "release",
]
