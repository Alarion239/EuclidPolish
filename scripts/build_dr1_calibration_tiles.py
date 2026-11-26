#!/usr/bin/env python3
"""Freeze the DR1 Wide Survey tile sample the population calibration queries.

Q1 is three deep fields, so its calibration queries three fixed cones. DR1 Wide
covers ~2,400 deg^2 in ~9,500 MER tiles (888M catalogue rows), far too much to
aggregate hundreds of times, so the DR1 calibration uses a seeded random sample
of *interior* Wide tiles. Interior means the tile has its full ring of Wide
neighbours (>= 6 tile centres within 0.75 deg), so its 30' x 30' core is fully
observed and the footprint area is the tile count times the core area, with no
coverage bias from the survey edge. Every sampled tile is then checked against
the ESA archive: it must have rows in ``dr1.mer_catalogue_wide_survey``.

The ESA catalogue assigns every object to the one tile whose core holds it, so
``tile_index IN (...)`` selects each sky position once (and ``tile_index`` is
indexed on ESA, unlike a long OR of cones).

    python scripts/build_dr1_calibration_tiles.py [--count 300] [--seed 2026]

Writes ``euclid_polish/sky/observation/dr1_calibration_tiles.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.esa.euclid import Euclid
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
INDEX_PATH = REPO / "services" / "cutout" / "index" / "dr1.json"
OUTPUT = REPO / "euclid_polish" / "sky" / "observation" / "dr1_calibration_tiles.json"
MER_TABLE = "dr1.mer_catalogue_wide_survey"
#: A MER tile core is 30' x 30'; the 32' mosaic adds a 1' overlap each side.
CORE_SIDE_DEG = 0.5
NEIGHBOUR_RADIUS_DEG = 0.75
MIN_NEIGHBOURS = 6
#: Tiles with fewer catalogue rows than this fraction of the sample median are
#: dropped as damaged/partial (a missing tile has none at all).
MIN_ROW_FRACTION = 0.5


def unit_vectors(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.c_[np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]


def chord(angle_deg: float) -> float:
    return 2.0 * math.sin(math.radians(angle_deg) / 2.0)


def interior_wide_tiles() -> tuple[list[dict], int]:
    index = json.loads(INDEX_PATH.read_text())
    columns = index["columns"]
    rows = [dict(zip(columns, row, strict=True)) for row in index["tiles"]]
    wide = [row for row in rows if int(row["deep"]) == 0]
    ra = np.array([float(row["ra"]) for row in wide])
    dec = np.array([float(row["dec"]) for row in wide])
    tree = cKDTree(unit_vectors(ra, dec))
    neighbours = [
        len(hits) - 1
        for hits in tree.query_ball_point(
            unit_vectors(ra, dec), chord(NEIGHBOUR_RADIUS_DEG),
        )
    ]
    galactic = SkyCoord(ra * u.deg, dec * u.deg).galactic
    return [
        {
            "tile": str(row["tile"]),
            "ra": round(float(row["ra"]), 7),
            "dec": round(float(row["dec"]), 7),
            "galactic_b": round(float(b), 3),
        }
        for row, count, b in zip(wide, neighbours, galactic.b.deg, strict=True)
        if count >= MIN_NEIGHBOURS
    ], len(wide)


def archive_rows(tiles: list[str]) -> dict[str, int]:
    listed = ", ".join(str(int(tile)) for tile in tiles)
    query = (
        f"SELECT mer.tile_index, COUNT(*) AS n FROM {MER_TABLE} AS mer "
        f"WHERE mer.tile_index IN ({listed}) GROUP BY mer.tile_index"
    )
    table = Euclid.launch_job_async(query).get_results()
    return {str(int(row["tile_index"])): int(row["n"]) for row in table}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    interior, wide_total = interior_wide_tiles()
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(interior))
    # Draw a margin so dropped tiles can be replaced without re-seeding.
    candidates = [interior[i] for i in order[: int(args.count * 1.2) + 10]]
    rows = archive_rows([tile["tile"] for tile in candidates])
    median = float(np.median(list(rows.values()))) if rows else 0.0
    kept, dropped = [], []
    for tile in candidates:
        n = rows.get(tile["tile"], 0)
        if n < MIN_ROW_FRACTION * median:
            dropped.append({**tile, "rows": n})
            continue
        if len(kept) < args.count:
            kept.append({**tile, "rows": n})
    if len(kept) < args.count:
        raise SystemExit(f"only {len(kept)} usable tiles among {len(candidates)} candidates")

    core_area = CORE_SIDE_DEG**2
    payload = {
        "kind": "euclid_dr1_wide_calibration_tiles",
        "version": 1,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": args.seed,
        "source_index": str(INDEX_PATH.relative_to(REPO)),
        "mer_table": MER_TABLE,
        "selection": (
            f"DR1 Wide (non-deep) MER tiles with >= {MIN_NEIGHBOURS} Wide tile centres "
            f"within {NEIGHBOUR_RADIUS_DEG} deg (interior), seeded random order; "
            f"tiles with < {MIN_ROW_FRACTION:g} x the median catalogue rows dropped"
        ),
        "wide_tiles": wide_total,
        "interior_wide_tiles": len(interior),
        "core_side_deg": CORE_SIDE_DEG,
        "core_area_deg2": core_area,
        "area_deg2": core_area * len(kept),
        "median_rows": median,
        "rows_total": int(sum(tile["rows"] for tile in kept)),
        "dropped": dropped,
        "tiles": kept,
    }
    OUTPUT.write_text(json.dumps(payload, indent=1) + "\n")
    b = np.abs([tile["galactic_b"] for tile in kept])
    print(f"wrote {OUTPUT.relative_to(REPO)}: {len(kept)} tiles, {payload['area_deg2']:.1f} deg², "
          f"{payload['rows_total']:,} rows; dropped {len(dropped)}; "
          f"|b| p5/50/95 = {np.percentile(b, 5):.1f}/{np.median(b):.1f}/{np.percentile(b, 95):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
