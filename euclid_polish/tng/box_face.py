"""Atlas galaxies that TNG50-1's periodic box cuts.

The TNG50-SKIRT atlas did not unwrap particle positions across TNG50-1's
periodic box (35000 ckpc/h): a galaxy near a box face lost every star beyond
the face plane, in its dust-free frames as much as in its dusty ones. Where an
orientation sees the plane edge-on the galaxy's light ends on a straight line
- 229934, 16 kpc from x = 0, through its halo in O1, O4 and O5; 580250,
0.7 kpc away, through its disc (2026-10-20). Those galaxies are no donors.

``box_face_galaxies.json`` lists every atlas galaxy within ``reach_kpc`` of a
face with its distance to the nearest one (subhalo positions from the TNG
API, snapshot 99).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from euclid_polish.config import Config

TABLE_PATH = Path(__file__).with_name("box_face_galaxies.json")


@lru_cache(maxsize=1)
def _table() -> dict:
    return json.loads(TABLE_PATH.read_text(encoding="utf-8"))


def face_distances_kpc() -> dict[str, float]:
    """Subhalo id → distance (kpc) to the nearest box face, for the atlas
    galaxies within the table's reach (farther ones are not listed)."""
    return {gid: float(row["distance_kpc"]) for gid, row in _table()["galaxies"].items()}


def table_reach_kpc() -> float:
    """Every atlas galaxy closer than this to a face is in the table."""
    return float(_table()["reach_kpc"])


def cut_galaxies() -> frozenset[str]:
    """Galaxies closer to a box face than ``Config.Tng.MIN_BOX_FACE_DISTANCE_KPC``."""
    limit = Config.Tng.MIN_BOX_FACE_DISTANCE_KPC
    if limit > table_reach_kpc():
        raise ValueError(f"MIN_BOX_FACE_DISTANCE_KPC {limit:g} exceeds the table's reach "
                         f"({table_reach_kpc():g} kpc): re-query {TABLE_PATH.name}")
    return frozenset(gid for gid, distance in face_distances_kpc().items() if distance < limit)
