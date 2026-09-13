#!/usr/bin/env python3
"""Sample Euclid's own MER noise maps at one random position per Q1 tile.

Each Q1 MER tile ships a per-pixel noise (RMS) map next to its science image.
This script requests a small 64x64 server-side cutout of the VIS, Y, J and H
noise maps at one random observed position inside every extragalactic Q1
tile, takes the median over sky pixels, and converts it to stack electrons
per 0.1" pixel with the header MAGZERO. Tiles at field edges are partly
unobserved, so each tile has a few pre-drawn candidate positions; the first
candidate with coverage in all four bands is kept. The resulting table of four-band levels is the
noise-level distribution used by the synthetic generator.

Stages, run from the repository root:

    python scripts/download_mer_noise_levels.py plan
    python scripts/download_mer_noise_levels.py acquire --max-minutes 55
    python scripts/download_mer_noise_levels.py finalize

Positions are ordered so that every prefix of the run is spread over the
three fields in proportion to their tile counts, and over each field's area.
Results are appended one position at a time, so stopping early leaves a
representative sample and ``acquire`` resumes where it stopped.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import requests
from astropy.io import fits

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from euclid_polish.config import Config  # noqa: E402
from euclid_polish.photometry import (  # noqa: E402
    adu_per_s_to_electrons_factor,
    header_magzero,
)

IRSA = "https://irsa.ipac.caltech.edu"
USER_AGENT = "EuclidPolish-mer-noise-levels/1 (public science diagnostics)"
WORK_DIR = REPO / "data" / "population_comparison" / "mer_noise_levels"
TABLE_PATH = REPO / "euclid_polish" / "sky" / "observation" / "mer_noise_levels.json"
TILE_QUERY = (
    "SELECT obs_id, s_ra, s_dec, s_region FROM ivoa.obscore "
    "WHERE obs_collection='euclid_DpdMerBksMosaic' AND energy_bandpassname='VIS'"
)
NOISE_MAP_QUERY = (
    "SELECT uri, contentlength FROM euclid.artifact_euclid_q1 "
    "WHERE producttype='noise' AND uri LIKE 'ibe/data/euclid/q1/MER/%'"
)
# File-name band token -> generator band.
BAND_TOKENS = {"VIS": "VIS", "NIR-Y": "Y_E", "NIR-J": "J_E", "NIR-H": "H_E"}
BANDS = ("VIS", "Y_E", "J_E", "H_E")
# Approximate field centres, used only to label tiles. LDN1641 is a small
# Galactic dust-cloud calibration field and is not part of the sample.
FIELD_CENTRES = {
    "EDF-N": (269.7, 66.0),
    "EDF-F": (52.9, -28.1),
    "EDF-S": (61.2, -48.4),
    "LDN1641": (85.7, -8.0),
}
SAMPLED_FIELDS = ("EDF-N", "EDF-S", "EDF-F")
CUTOUT_PIXELS = 64
PIXEL_SCALE_ARCSEC = 0.1
SEED = 2026
POSITION_FRACTION = 0.8  # stay inside the central 80% of each tile's extent
CANDIDATES_PER_TILE = 4
# Euclid fills unobserved pixels with a placeholder near 1e16 archive units;
# real noise values are around 0.004 (VIS ADU/s) and a few electrons (NISP).
NO_COVERAGE_NATIVE = 1.0e6


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def tap_csv(session: requests.Session, query: str) -> list[dict[str, str]]:
    response = session.get(
        IRSA + "/TAP/sync",
        params={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query},
        timeout=(15, 300),
    )
    response.raise_for_status()
    return list(csv.DictReader(io.StringIO(response.text)))


def angular_separation_deg(ra1, dec1, ra2, dec2):
    r1, d1, r2, d2 = map(np.radians, (ra1, dec1, ra2, dec2))
    cosine = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(r1 - r2)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def nearest_field(ra: float, dec: float) -> str:
    return min(
        FIELD_CENTRES,
        key=lambda name: float(angular_separation_deg(ra, dec, *FIELD_CENTRES[name])),
    )


def polygon_half_extents(region: str, ra0: float, dec0: float) -> tuple[float, float]:
    """Half extents of a tile polygon in tangent-plane degrees (x east, y north)."""
    values = [float(v) for v in region.replace("POLYGON", "").replace("ICRS", "").split()]
    corners = np.asarray(values).reshape(-1, 2)
    dx = ((corners[:, 0] - ra0 + 180.0) % 360.0 - 180.0) * math.cos(math.radians(dec0))
    dy = corners[:, 1] - dec0
    return float(np.max(np.abs(dx))), float(np.max(np.abs(dy)))


def spread_order(points: np.ndarray, rng: np.random.Generator) -> list[int]:
    """Greedy farthest-point order: every prefix covers the whole area."""
    remaining = list(range(len(points)))
    order = [remaining.pop(int(rng.integers(len(remaining))))]
    distance = angular_separation_deg(points[:, 0], points[:, 1], *points[order[0]])
    while remaining:
        best = max(remaining, key=lambda i: distance[i])
        remaining.remove(best)
        order.append(best)
        distance = np.minimum(
            distance, angular_separation_deg(points[:, 0], points[:, 1], *points[best]),
        )
    return order


def interleave_fields(per_field: dict[str, list[dict]]) -> list[dict]:
    """Alternate fields so each prefix matches the fields' tile proportions."""
    totals = {name: len(items) for name, items in per_field.items() if items}
    taken = dict.fromkeys(totals, 0)
    out = []
    while len(out) < sum(totals.values()):
        name = min(
            (n for n in totals if taken[n] < totals[n]),
            key=lambda n: ((taken[n] + 1) / totals[n], SAMPLED_FIELDS.index(n)),
        )
        out.append(per_field[name][taken[name]])
        taken[name] += 1
    return out


def make_plan() -> dict:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    tiles = tap_csv(session, TILE_QUERY)
    noise_maps = tap_csv(session, NOISE_MAP_QUERY)
    urls: dict[str, dict[str, str]] = {}
    for row in noise_maps:
        name = row["uri"].rsplit("/", 1)[-1]
        for token, band in BAND_TOKENS.items():
            prefix = f"EUC_MER_MOSAIC-{token}-RMS_TILE"
            if name.startswith(prefix):
                tile = name[len(prefix):].split("-", 1)[0]
                urls.setdefault(tile, {})[band] = f"{IRSA}/{row['uri']}"
    rng = np.random.default_rng(SEED)
    per_field: dict[str, list[dict]] = {name: [] for name in SAMPLED_FIELDS}
    skipped = {"outside_sampled_fields": 0, "missing_noise_maps": 0}
    for row in sorted(tiles, key=lambda r: r["obs_id"]):
        tile = row["obs_id"].split("_", 1)[0]
        ra0, dec0 = float(row["s_ra"]), float(row["s_dec"])
        field = nearest_field(ra0, dec0)
        if field not in per_field:
            skipped["outside_sampled_fields"] += 1
            continue
        if set(urls.get(tile, {})) != set(BANDS):
            skipped["missing_noise_maps"] += 1
            continue
        half_x, half_y = polygon_half_extents(row["s_region"], ra0, dec0)
        candidates = []
        for _ in range(CANDIDATES_PER_TILE):
            dx, dy = rng.uniform(-POSITION_FRACTION, POSITION_FRACTION, 2) * (half_x, half_y)
            candidates.append({
                "ra": (ra0 + dx / math.cos(math.radians(dec0))) % 360.0,
                "dec": dec0 + dy,
            })
        per_field[field].append({
            "field": field,
            "tile": tile,
            "tile_ra": ra0,
            "tile_dec": dec0,
            "candidates": candidates,
            "noise_maps": urls[tile],
        })
    for field, items in per_field.items():
        points = np.array([[item["tile_ra"], item["tile_dec"]] for item in items])
        per_field[field] = [items[i] for i in spread_order(points, rng)]
    positions = interleave_fields(per_field)
    for index, item in enumerate(positions):
        item["order"] = index
    plan = {
        "kind": "euclid_q1_mer_noise_level_plan",
        "created_at": utc_now(),
        "seed": SEED,
        "cutout_pixels": CUTOUT_PIXELS,
        "position_fraction_of_tile_extent": POSITION_FRACTION,
        "candidates_per_tile": CANDIDATES_PER_TILE,
        "tile_query": TILE_QUERY,
        "noise_map_query": NOISE_MAP_QUERY,
        "tiles_listed": len(tiles),
        "skipped_tiles": skipped,
        "tiles_per_field": {name: len(items) for name, items in per_field.items()},
        "positions": positions,
    }
    (WORK_DIR / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    return plan


def level_from_cutout(content: bytes, band_name: str, url: str) -> dict:
    """Sky noise level of one noise-map cutout, in stack electrons per pixel."""
    with fits.open(io.BytesIO(content), memmap=False) as hdus:
        hdu = next((h for h in hdus if h.data is not None and h.data.ndim == 2), None)
        if hdu is None:
            raise ValueError("no 2-D image in cutout")
        if hdu.header.get("DATASETR") != "Q1_R1":
            raise ValueError(f"unexpected release {hdu.header.get('DATASETR')!r}")
        data = np.asarray(hdu.data, dtype=np.float64)
        magzero = header_magzero(hdu.header, source=f"{band_name} noise map {url}")
    observed = np.isfinite(data) & (data > 0.0) & (data < NO_COVERAGE_NATIVE)
    if observed.mean() < 0.5:
        raise ValueError(f"no coverage: only {observed.mean():.0%} of the cutout is observed")
    # Source photon noise raises the map around objects; the median over
    # pixels below five times the cutout median is the local sky level.
    first = float(np.median(data[observed]))
    sky = observed & (data < 5.0 * first)
    if sky.mean() < 0.5:
        raise ValueError(f"only {sky.mean():.0%} of the cutout is sky")
    factor = adu_per_s_to_electrons_factor(magzero, Config.get_band(band_name))
    values = data[sky] * factor
    return {
        "level_e": float(np.median(values)),
        "p16_e": float(np.percentile(values, 16)),
        "p84_e": float(np.percentile(values, 84)),
        "sky_fraction": float(sky.mean()),
        "magzero": magzero,
        "shape": list(data.shape),
    }


def fetch_band(session: requests.Session, url: str, band_name: str,
               candidate: dict) -> dict:
    """One noise-map cutout: its sky level, or the reason it has none."""
    size_deg = CUTOUT_PIXELS * PIXEL_SCALE_ARCSEC / 3600.0
    params = {"center": f"{candidate['ra']:.8f},{candidate['dec']:.8f}",
              "size": f"{size_deg:.10f}", "gzip": "false"}
    entry: dict = {"url": url}
    for attempt in range(3):
        try:
            t0 = time.monotonic()
            response = session.get(url, params=params, timeout=(15, 60))
            response.raise_for_status()
            entry["seconds"] = round(time.monotonic() - t0, 3)
            entry["bytes"] = len(response.content)
            entry["sha256"] = hashlib.sha256(response.content).hexdigest()
            entry.update(level_from_cutout(response.content, band_name, url))
            entry.pop("error", None)
            return entry
        except (requests.RequestException, OSError) as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            time.sleep(2.0 * (attempt + 1))
        except ValueError as exc:
            entry["error"] = str(exc)
            return entry
    return entry


def measure_tile(session: requests.Session, position: dict) -> dict:
    """Try the tile's candidates until one has coverage in all four bands."""
    record = {key: position[key] for key in ("order", "field", "tile")}
    record["attempts"] = []
    for index, candidate in enumerate(position["candidates"]):
        bands: dict = {}
        for band_name in BANDS:
            entry = fetch_band(session, position["noise_maps"][band_name], band_name, candidate)
            bands[band_name] = entry
            if "level_e" not in entry:
                break  # an unobserved spot costs one request, not four
        attempt = {"candidate": index, "ra": candidate["ra"], "dec": candidate["dec"],
                   "retrieved_at": utc_now(), "bands": bands}
        record["attempts"].append(attempt)
        if all("level_e" in bands.get(b, {}) for b in BANDS):
            record.update({"ra": candidate["ra"], "dec": candidate["dec"], "bands": bands})
            return record
    return record


def acquire(max_minutes: float, limit: int | None) -> None:
    plan = json.loads((WORK_DIR / "plan.json").read_text())
    results_path = WORK_DIR / "levels.jsonl"
    done = set()
    if results_path.exists():
        done = {json.loads(line)["order"] for line in results_path.read_text().splitlines() if line}
    todo = [p for p in plan["positions"] if p["order"] not in done]
    if limit is not None:
        todo = todo[:limit]
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    start = time.monotonic()
    deadline = start + 60.0 * max_minutes
    print(f"{len(done)} tiles already done; {len(todo)} to go; "
          f"hard stop after {max_minutes:g} min", flush=True)
    with results_path.open("a") as out:
        for count, position in enumerate(todo, start=1):
            if time.monotonic() >= deadline:
                print("time limit reached; stopping cleanly", flush=True)
                break
            record = measure_tile(session, position)
            out.write(json.dumps(record) + "\n")
            out.flush()
            if count % 20 == 0 or count == len(todo):
                elapsed = time.monotonic() - start
                rate = elapsed / count
                remaining = rate * (len(todo) - count)
                print(f"{count}/{len(todo)} tiles, {elapsed / 60:.1f} min elapsed, "
                      f"{rate:.1f} s/tile, about {remaining / 60:.1f} min left", flush=True)


def finalize() -> dict:
    plan = json.loads((WORK_DIR / "plan.json").read_text())
    records = [json.loads(line) for line in (WORK_DIR / "levels.jsonl").read_text().splitlines() if line]
    records.sort(key=lambda r: r["order"])
    rows, rejected = [], []
    for record in records:
        bands = record.get("bands", {})
        if all("level_e" in bands.get(b, {}) for b in BANDS):
            rows.append({
                "field": record["field"],
                "tile": record["tile"],
                "ra": round(record["ra"], 6),
                "dec": round(record["dec"], 6),
                "levels_e": [round(bands[b]["level_e"], 4) for b in BANDS],
            })
        else:
            last = record["attempts"][-1]["bands"] if record.get("attempts") else {}
            rejected.append({
                "order": record["order"],
                "tile": record["tile"],
                "reason": next((entry.get("error") for entry in last.values()
                                if "error" in entry), "no attempt"),
            })
    table = {
        "kind": "euclid_q1_mer_noise_levels",
        "version": 1,
        "description": (
            "Sky noise levels read from Euclid's Q1 MER noise (RMS) maps: one random "
            "position per extragalactic tile, the median of noise-map values below five "
            "times the cutout median in a 64x64 pixel (6.4 arcsec) cutout, in stack "
            "electrons per 0.1 arcsec pixel via each cutout's MAGZERO."
        ),
        "release": "Q1_R1",
        "archive": IRSA,
        "tile_query": plan["tile_query"],
        "noise_map_query": plan["noise_map_query"],
        "plan_seed": plan["seed"],
        "fields": list(SAMPLED_FIELDS),
        "bands": list(BANDS),
        "units": "stack electrons per 0.1 arcsec pixel",
        "retrieved_first": min(
            (a["retrieved_at"] for r in records for a in r.get("attempts", [])), default=None),
        "retrieved_last": max(
            (a["retrieved_at"] for r in records for a in r.get("attempts", [])), default=None),
        "tiles_planned": len(plan["positions"]),
        "tiles_attempted": len(records),
        "rejected": rejected,
        "rows": rows,
    }
    TABLE_PATH.write_text(json.dumps(table, indent=1) + "\n")
    levels = np.array([row["levels_e"] for row in rows])
    print(f"wrote {TABLE_PATH.relative_to(REPO)}: {len(rows)} rows, {len(rejected)} rejected")
    if len(rows):
        for i, band in enumerate(BANDS):
            q = np.percentile(levels[:, i], [5, 16, 50, 84, 95])
            print(f"  {band:4} p5 {q[0]:7.3f}  p16 {q[1]:7.3f}  median {q[2]:7.3f}  p84 {q[3]:7.3f}  p95 {q[4]:7.3f}")
    return table


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("plan", help="list tiles and noise maps on IRSA and freeze positions")
    acquire_parser = sub.add_parser("acquire", help="download noise-map cutouts, resuming")
    acquire_parser.add_argument("--max-minutes", type=float, default=55.0)
    acquire_parser.add_argument("--limit", type=int, default=None)
    sub.add_parser("finalize", help="write the level table used by the generator")
    args = parser.parse_args()
    if args.stage == "plan":
        plan = make_plan()
        print(f"{len(plan['positions'])} positions; tiles per field {plan['tiles_per_field']}; "
              f"skipped {plan['skipped_tiles']}")
        print("first 12 fields in order:", [p["field"] for p in plan["positions"][:12]])
    elif args.stage == "acquire":
        acquire(args.max_minutes, args.limit)
    else:
        finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
