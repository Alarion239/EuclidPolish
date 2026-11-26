#!/usr/bin/env python3
"""DR1 Wide twin of ``download_mer_noise_levels.py``: MER noise-map sky levels.

Same measurement as the Q1 table (one scene-sized 256x256 window of the VIS,
Y, J and H RMS maps per tile, median over sky pixels, MAGZERO -> stack
electrons per 0.1" pixel, plus the 4x4 sub-tile grid), on the frozen DR1 Wide
calibration tile sample (``euclid_polish/sky/observation/dr1_calibration_tiles.json``).

DR1 mosaics are cloud-hosted (``nasa-irsa-euclid-dr1`` on S3), where IRSA's
server-side cutouts are not offered, so each window is read with HTTP Range
requests straight from the uncompressed float32 mosaic, one request per row.
Where a tile has several runs the Wide Survey run is used (the newest one),
never a deep-field run (PPOID ``...DATA_RELEASE_DEEP...``), matching the
cutout server.

The output table is NOT read by the generator: the Q1 table stays the noise
model until a DR1 switch is decided (that bumps ``Config.NOISE_MODEL``).

    python scripts/download_mer_noise_levels_dr1.py plan
    python scripts/download_mer_noise_levels_dr1.py acquire --max-minutes 120
    python scripts/download_mer_noise_levels_dr1.py finalize
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import requests
from astropy.io import fits
from astropy.wcs import WCS

REPO = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from download_mer_noise_levels import (  # noqa: E402
    BANDS,
    CANDIDATES_PER_TILE,
    CUTOUT_PIXELS,
    POSITION_FRACTION,
    SEED,
    SUB_TILE_PIXELS,
    level_from_noise_map,
)

from euclid_polish.config import Config  # noqa: E402
from euclid_polish.photometry import header_magzero  # noqa: E402
from euclid_polish.sky.observation.calibration_release import (  # noqa: E402
    DR1_TILES_PATH,
    dr1_tile_rows,
)

BUCKET = "https://nasa-irsa-euclid-dr1.s3.us-east-1.amazonaws.com"
PREFIX = "dr1/MER"
USER_AGENT = "EuclidPolish-mer-noise-levels-dr1/1 (public science diagnostics)"
WORK_DIR = Path(Config.DATA_DIR) / "population_comparison" / "dr1" / "mer_noise_levels"
TABLE_PATH = REPO / "euclid_polish" / "sky" / "observation" / "mer_noise_levels_dr1.json"
#: generator band -> (bucket folder, file-name token of its RMS mosaic)
RMS_TOKENS = {
    "VIS": ("VIS", "MOSAIC-VIS-RMS"),
    "Y_E": ("NISP", "MOSAIC-NIR-Y-RMS"),
    "J_E": ("NISP", "MOSAIC-NIR-J-RMS"),
    "H_E": ("NISP", "MOSAIC-NIR-H-RMS"),
}
DEEP_PLAN = "DATA_RELEASE_DEEP"
#: Half of the 30' tile core in 0.1" pixels; windows stay inside its central 80%.
CORE_HALF_PIXELS = 9000
ROW_WORKERS = 32
RETRIES = 4
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
PROCESSED_RE = re.compile(r"_(\d{8}T\d{6}(?:\.\d+)?)Z_")
FITS_BLOCK = 2880

_local = threading.local()


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def session() -> requests.Session:
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
        _local.session.headers["User-Agent"] = USER_AGENT
    return _local.session


def get(url: str, **kwargs) -> requests.Response:
    """GET with retries on S3's transient 5xx (SlowDown) answers."""
    for attempt in range(RETRIES + 1):
        try:
            response = session().get(url, timeout=(15, 60), **kwargs)
            if response.status_code < 500 or attempt == RETRIES:
                response.raise_for_status()
                return response
        except requests.ConnectionError:
            if attempt == RETRIES:
                raise
        time.sleep(0.2 * 4**attempt)
    raise AssertionError("unreachable")


def list_keys(prefix: str) -> list[str]:
    keys, token = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        root = ElementTree.fromstring(get(BUCKET + "/", params=params).content)
        keys += [el.text for el in root.iter(f"{S3_NS}Key") if el.text]
        token = root.findtext(f"{S3_NS}NextContinuationToken")
        if root.findtext(f"{S3_NS}IsTruncated") != "true" or not token:
            return keys


def read_range(key: str, start: int, length: int) -> bytes:
    response = get(f"{BUCKET}/{key}", headers={"Range": f"bytes={start}-{start + length - 1}"})
    body = response.content
    if response.status_code == 200:  # server ignored Range
        body = body[start:start + length]
    if len(body) != length:
        raise OSError(f"{key}: wanted {length} bytes, got {len(body)}")
    return body


def read_header(key: str) -> tuple[fits.Header, int]:
    """The primary header and the byte offset of its data."""
    raw = b""
    while True:
        if len(raw) >= 40 * FITS_BLOCK:
            raise ValueError(f"{key}: no END card in the first {len(raw)} bytes")
        raw += read_range(key, len(raw), 2 * FITS_BLOCK)
        cards = [raw[i:i + 80] for i in range(0, len(raw), 80)]
        end = next((i for i, card in enumerate(cards) if card[:8] == b"END     "), None)
        if end is not None:
            header = fits.Header.fromstring(raw[: 80 * (end + 1)].decode("ascii"))
            return header, ((80 * end) // FITS_BLOCK + 1) * FITS_BLOCK


def processing_time(key: str) -> str:
    match = PROCESSED_RE.search(key.rsplit("/", 1)[-1])
    return match.group(1) if match else ""


def choose_rms(tile: str) -> dict[str, dict]:
    """Each band's RMS mosaic for ``tile``: Wide run, newest processing."""
    listings = {inst: list_keys(f"{PREFIX}/{tile}/{inst}/") for inst in ("VIS", "NISP")}
    chosen = {}
    for band, (inst, token) in RMS_TOKENS.items():
        keys = sorted(
            k for k in listings[inst]
            if k.endswith(".fits")
            and k.rsplit("/", 1)[-1].startswith(f"EUC_MER_{token}_TILE{tile}-")
        )
        if not keys:
            raise LookupError(f"tile {tile}: no {token} file")
        options = []
        for key in keys:
            header, offset = read_header(key)
            options.append({
                "key": key,
                "offset": offset,
                "header": header,
                "deep": DEEP_PLAN in str(header.get("PPOID", "")),
            })
        wide = [o for o in options if not o["deep"]]
        if not wide:
            raise LookupError(f"tile {tile}: only deep-field {token} runs")
        chosen[band] = max(wide, key=lambda o: processing_time(o["key"]))
        chosen[band]["runs"] = len(keys)
    return chosen


def read_window(option: dict, x0: int, y0: int) -> np.ndarray:
    header = option["header"]
    width, height = int(header["NAXIS1"]), int(header["NAXIS2"])
    if header.get("BITPIX") != -32 or header.get("NAXIS") != 2:
        raise ValueError(f"{option['key']}: expected a 2-D float32 image")
    n = CUTOUT_PIXELS
    if x0 < 0 or y0 < 0 or x0 + n > width or y0 + n > height:
        raise ValueError(f"window ({x0},{y0}) outside the {width}x{height} mosaic")

    def row(y: int) -> np.ndarray:
        start = option["offset"] + 4 * (y * width + x0)
        return np.frombuffer(read_range(option["key"], start, 4 * n), dtype=">f4")

    with ThreadPoolExecutor(ROW_WORKERS) as pool:
        data = np.vstack(list(pool.map(row, range(y0, y0 + n)))).astype(np.float64)
    bscale, bzero = float(header.get("BSCALE", 1.0)), float(header.get("BZERO", 0.0))
    return data * bscale + bzero


def make_plan() -> dict:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    positions = []
    for order, tile in enumerate(dr1_tile_rows()):
        offsets = rng.uniform(-POSITION_FRACTION, POSITION_FRACTION, (CANDIDATES_PER_TILE, 2))
        positions.append({
            "order": order,
            "field": "DR1-Wide",
            "tile": str(tile["tile"]),
            "tile_ra": tile["ra"],
            "tile_dec": tile["dec"],
            "galactic_b": tile["galactic_b"],
            # Offsets from the mosaic centre in units of the core half-width;
            # turned into pixels once the mosaic shape is known.
            "candidate_offsets": offsets.round(6).tolist(),
        })
    plan = {
        "kind": "euclid_dr1_mer_noise_level_plan",
        "created_at": utc_now(),
        "seed": SEED,
        "cutout_pixels": CUTOUT_PIXELS,
        "position_fraction_of_core": POSITION_FRACTION,
        "candidates_per_tile": CANDIDATES_PER_TILE,
        "tiles_source": str(DR1_TILES_PATH.relative_to(REPO)),
        "bucket": BUCKET,
        "positions": positions,
    }
    (WORK_DIR / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    return plan


def measure_tile(position: dict) -> dict:
    record = {key: position[key] for key in ("order", "field", "tile", "galactic_b")}
    record["attempts"] = []
    try:
        chosen = choose_rms(position["tile"])
    except (LookupError, ValueError, OSError, requests.RequestException) as exc:
        record["attempts"].append({"error": f"{type(exc).__name__}: {exc}",
                                   "retrieved_at": utc_now()})
        return record
    vis_header = chosen["VIS"]["header"]
    width, height = int(vis_header["NAXIS1"]), int(vis_header["NAXIS2"])
    half = min(CORE_HALF_PIXELS, min(width, height) // 2 - CUTOUT_PIXELS)
    wcs = WCS(vis_header)
    for index, (fx, fy) in enumerate(position["candidate_offsets"]):
        cx, cy = width / 2.0 + fx * half, height / 2.0 + fy * half
        x0, y0 = int(round(cx - CUTOUT_PIXELS / 2)), int(round(cy - CUTOUT_PIXELS / 2))
        ra, dec = (float(v) for v in wcs.pixel_to_world_values(cx, cy))
        bands: dict = {}
        for band in BANDS:
            option = chosen[band]
            entry = {"key": option["key"], "runs": option["runs"],
                     "datasetr": str(option["header"].get("DATASETR", ""))}
            try:
                t0 = time.monotonic()
                magzero = header_magzero(option["header"], source=f"{band} {option['key']}")
                entry.update(level_from_noise_map(read_window(option, x0, y0), band, magzero))
                entry["seconds"] = round(time.monotonic() - t0, 3)
            except ValueError as exc:
                entry["error"] = str(exc)
            except (OSError, requests.RequestException) as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
            bands[band] = entry
            if "level_e" not in entry:
                break
        attempt = {"candidate": index, "ra": ra % 360.0, "dec": dec, "x0": x0, "y0": y0,
                   "retrieved_at": utc_now(), "bands": bands}
        record["attempts"].append(attempt)
        if all("level_e" in bands.get(b, {}) for b in BANDS):
            record.update({"ra": ra % 360.0, "dec": dec, "bands": bands})
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
    start = time.monotonic()
    deadline = start + 60.0 * max_minutes
    print(f"{len(done)} tiles already done; {len(todo)} to go; "
          f"hard stop after {max_minutes:g} min", flush=True)
    with results_path.open("a") as out:
        for count, position in enumerate(todo, start=1):
            if time.monotonic() >= deadline:
                print("time limit reached; stopping cleanly", flush=True)
                break
            record = measure_tile(position)
            out.write(json.dumps(record) + "\n")
            out.flush()
            if count % 10 == 0 or count == len(todo):
                elapsed = time.monotonic() - start
                rate = elapsed / count
                print(f"{count}/{len(todo)} tiles, {elapsed / 60:.1f} min elapsed, "
                      f"{rate:.1f} s/tile, about {rate * (len(todo) - count) / 60:.1f} min left",
                      flush=True)


def finalize() -> dict:
    plan = json.loads((WORK_DIR / "plan.json").read_text())
    records = [json.loads(line) for line in (WORK_DIR / "levels.jsonl").read_text().splitlines()
               if line]
    records.sort(key=lambda r: r["order"])
    rows, rejected = [], []
    for record in records:
        bands = record.get("bands", {})
        if all("level_e" in bands.get(b, {}) for b in BANDS):
            rows.append({
                "field": record["field"],
                "tile": record["tile"],
                "galactic_b": record.get("galactic_b"),
                "ra": round(record["ra"], 6),
                "dec": round(record["dec"], 6),
                "levels_e": [round(bands[b]["level_e"], 4) for b in BANDS],
                "sub_levels_e": {b: bands[b].get("sub_levels_e") for b in BANDS},
                "datasetr": bands["VIS"].get("datasetr"),
            })
        else:
            last = record["attempts"][-1] if record.get("attempts") else {}
            reason = last.get("error") or next(
                (e.get("error") for e in last.get("bands", {}).values() if "error" in e),
                "no attempt")
            rejected.append({"order": record["order"], "tile": record["tile"], "reason": reason})
    stamps = [a["retrieved_at"] for r in records for a in r.get("attempts", [])]
    table = {
        "kind": "euclid_dr1_mer_noise_levels",
        "version": 1,
        "description": (
            "Sky noise levels read from Euclid's DR1 Wide Survey MER noise (RMS) maps: one "
            "random position per sampled interior Wide tile, the median of noise-map values "
            "below five times the window median in a 256x256 pixel (25.6 arcsec, one "
            "generated scene) window read from the cloud-hosted mosaic, in stack electrons per "
            "0.1 arcsec pixel via each mosaic's MAGZERO. Every row also carries sub_levels_e, "
            "the 4x4 grid of 64-pixel sub-tile levels. Not read by the generator."
        ),
        "cutout_pixels": int(plan.get("cutout_pixels", CUTOUT_PIXELS)),
        "sub_tile_pixels": int(SUB_TILE_PIXELS),
        "release": "DR1",
        "archive": BUCKET,
        "tiles_source": plan["tiles_source"],
        "plan_seed": plan["seed"],
        "fields": ["DR1-Wide"],
        "bands": list(BANDS),
        "units": "stack electrons per 0.1 arcsec pixel",
        "retrieved_first": min(stamps, default=None),
        "retrieved_last": max(stamps, default=None),
        "tiles_planned": len(plan["positions"]),
        "tiles_attempted": len(records),
        "rejected": rejected,
        "rows": rows,
    }
    TABLE_PATH.write_text(json.dumps(table, indent=1) + "\n")
    levels = np.array([row["levels_e"] for row in rows])
    print(f"wrote {TABLE_PATH.relative_to(REPO)}: {len(rows)} rows, {len(rejected)} rejected")
    for i, band in enumerate(BANDS):
        if len(rows):
            q = np.percentile(levels[:, i], [5, 16, 50, 84, 95])
            print(f"  {band:4} p5 {q[0]:7.3f}  p16 {q[1]:7.3f}  median {q[2]:7.3f}  "
                  f"p84 {q[3]:7.3f}  p95 {q[4]:7.3f}")
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("plan", help="freeze one candidate set per sampled DR1 Wide tile")
    acquire_parser = sub.add_parser("acquire", help="read noise-map windows, resuming")
    acquire_parser.add_argument("--max-minutes", type=float, default=120.0)
    acquire_parser.add_argument("--limit", type=int, default=None)
    sub.add_parser("finalize", help="write mer_noise_levels_dr1.json (not used by the generator)")
    args = parser.parse_args()
    if args.stage == "plan":
        plan = make_plan()
        print(f"{len(plan['positions'])} positions planned in {WORK_DIR}")
    elif args.stage == "acquire":
        acquire(args.max_minutes, args.limit)
    else:
        finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
