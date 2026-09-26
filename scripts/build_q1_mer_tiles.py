#!/usr/bin/env python
"""Build the committed Q1 MER tile table ``q1_mer_tiles.json``.

Input: the IRSA obscore VIS mosaic table cached by
``scripts/download_mer_noise_levels.py``
(``data/population_comparison/mer_noise_levels_64px/q1_vis_mosaics.csv``:
``obs_id, s_ra, s_dec, s_region, access_url``) and the committed noise table
``euclid_polish/sky/observation/mer_noise_levels.json``.

Output (``euclid_polish/sky/observation/q1_mer_tiles.json``): one row per
tile — ``tile``, ``ra``, ``dec``, ``polygon`` (4 ``[ra, dec]`` vertices, the
closing vertex dropped), ``field`` (position-derived via ``q1_field_for``),
``region`` (``field`` or ``"LDN1641"`` for the Galactic calibration cloud),
``levels_e`` (VIS/Y/J/H sky level where measured), ``level_position`` (where
it was measured) and ``rejected`` (the noise campaign's rejection reason).

Usage::

    python scripts/build_q1_mer_tiles.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from euclid_polish.sky.observation.q1_fields import (  # noqa: E402
    angular_separation_deg,
    q1_field_for,
)

DEFAULT_CSV = _PROJECT_ROOT / "data/population_comparison/mer_noise_levels_64px/q1_vis_mosaics.csv"
DEFAULT_NOISE = _PROJECT_ROOT / "euclid_polish/sky/observation/mer_noise_levels.json"
DEFAULT_OUT = _PROJECT_ROOT / "euclid_polish/sky/observation/q1_mer_tiles.json"
# The Q1 Galactic dust-cloud calibration field (not one of the deep fields).
LDN1641 = ("LDN1641", 85.7, -8.0, 3.0)


def parse_polygon(s_region: str) -> list[list[float]]:
    tokens = s_region.replace(",", " ").split()
    numbers = [float(t) for t in tokens if t.upper() not in {"POLYGON", "ICRS"}]
    points = [[numbers[i], numbers[i + 1]] for i in range(0, len(numbers) - 1, 2)]
    if len(points) > 3 and points[0] == points[-1]:
        points = points[:-1]
    return [[round(ra, 7), round(dec, 7)] for ra, dec in points]


def build(csv_path: Path, noise_path: Path) -> dict:
    noise = json.loads(noise_path.read_text(encoding="utf-8"))
    levels = {str(row["tile"]): row for row in noise.get("rows", [])}
    rejected = {str(row["tile"]): str(row.get("reason") or "rejected")
                for row in noise.get("rejected", [])}
    tiles = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            tile = str(row["obs_id"]).split("_")[0]
            ra, dec = float(row["s_ra"]), float(row["s_dec"])
            field = q1_field_for(ra, dec)
            name, lra, ldec, radius = LDN1641
            region = field or (name if angular_separation_deg(ra, dec, lra, ldec) <= radius else None)
            level = levels.get(tile)
            tiles.append({
                "tile": tile,
                "ra": round(ra, 7), "dec": round(dec, 7),
                "polygon": parse_polygon(row["s_region"]),
                "field": field, "region": region,
                "levels_e": None if level is None else [float(v) for v in level["levels_e"]],
                "level_position": (None if level is None
                                   else [round(float(level["ra"]), 6), round(float(level["dec"]), 6)]),
                "rejected": rejected.get(tile),
            })
    tiles.sort(key=lambda item: item["tile"])
    if any(not math.isfinite(v) for t in tiles for p in t["polygon"] for v in p):
        raise ValueError("non-finite polygon vertex")
    return {
        "kind": "q1_mer_tiles",
        "version": 1,
        "description": (
            "Euclid Q1 VIS MER mosaic footprints (IRSA obscore s_region) with "
            "position-derived Q1 field labels and the scene-sized MER sky "
            "levels of mer_noise_levels.json where measured."
        ),
        "release": noise.get("release", "Q1"),
        "source": "IRSA obscore euclid_DpdMerBksMosaic (energy_bandpassname='VIS')",
        "units": {"levels_e": noise.get("units", "")},
        "bands": list(noise.get("bands", ["VIS", "Y_E", "J_E", "H_E"])),
        "count": len(tiles),
        "tiles": tiles,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--noise", type=Path, default=DEFAULT_NOISE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    payload = build(args.csv, args.noise)
    args.out.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Wrote {payload['count']} tiles to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
