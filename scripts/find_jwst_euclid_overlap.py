#!/usr/bin/env python
"""Discover public JWST imaging observations overlapping Euclid mosaics.

The first pass queries the public Euclid ``sedm.mosaic_product`` table for VIS
mosaic centres, then uses MAST cone searches to find nearby JWST observations.
When MAST returns an ``s_region`` polygon, the script sends that polygon back
to the Euclid archive and uses ``mosaic_product.fov`` for an exact archive-side
footprint intersection.  Results and intermediate responses are cached so a
retry does not repeat completed archive queries.

No FITS images are downloaded by this script.  It writes a CSV and a JSON
manifest containing observation IDs, instruments, filters, URLs, footprints,
and the Euclid tile(s) that intersect each JWST observation.

Usage::

    python scripts/find_jwst_euclid_overlap.py
    python scripts/find_jwst_euclid_overlap.py --refresh
    python scripts/find_jwst_euclid_overlap.py --jwst-archive esa --out data/jwst_euclid_overlap/esa.csv

The repository's ``EuclidPolishEnv`` contains the optional archive clients::

    conda run -n EuclidPolishEnv python scripts/find_jwst_euclid_overlap.py

The default backend is MAST.  ``astroquery.esa.jwst`` is also supported with
``--jwst-archive esa``; ``both`` runs both services and keeps the backend in
the output key.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_DIRECT_INSTRUMENTS = {"NIRCAM", "MIRI", "NIRISS"}
_S_REGION_RE = re.compile(r"^\s*POLYGON(?:\s+ICRS)?\s+(.+?)\s*$", re.IGNORECASE)
_CSV_FIELDS = [
    "euclid_tile_index",
    "euclid_ra_deg",
    "euclid_dec_deg",
    "euclid_file_name",
    "jwst_archive",
    "jwst_observation_id",
    "jwst_obsid",
    "jwst_target_name",
    "jwst_proposal_id",
    "jwst_instrument",
    "jwst_filters",
    "jwst_data_rights",
    "jwst_exposure_time_s",
    "jwst_ra_deg",
    "jwst_dec_deg",
    "jwst_distance_deg",
    "jwst_data_url",
    "jwst_jpeg_url",
    "jwst_s_region",
    "footprint_status",
    "overlap_method",
]


def _text(value: Any) -> str:
    """Convert archive scalar values to stable JSON/CSV text."""
    if value is None:
        return ""
    try:
        import numpy as np

        if np.ma.is_masked(value):
            return ""
        value = value.item() if hasattr(value, "item") else value
    except (ImportError, TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text in {"--", "nan", "None", "null"} else text


def _float(value: Any) -> float | None:
    text = _text(value)
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _row_value(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return None


def _is_direct_imaging(row: Mapping[str, Any]) -> bool:
    """Keep direct NIRCam/MIRI/NIRISS imaging, excluding spectroscopy/WFSS."""
    instrument = _text(_row_value(row, "instrument_name", "instrument")).upper()
    exp_type = _text(_row_value(row, "exp_type", "exposure_type")).upper()
    if instrument.split("/")[0] not in _DIRECT_INSTRUMENTS:
        return False
    if "/IMAGE" in instrument or instrument.endswith("IMAGE"):
        return True
    if "IMAGE" in exp_type:
        return not any(token in exp_type for token in ("GRISM", "WFSS", "SOSS", "IFU", "SLIT"))
    return False


def _is_public(row: Mapping[str, Any]) -> bool:
    rights = _text(_row_value(row, "dataRights", "data_rights", "public", "data_rights_status")).upper()
    return not rights or rights in {"PUBLIC", "TRUE", "T", "1", "YES"}


def _jsonable(value: Any) -> Any:
    """Make Astropy/numpy archive values serialisable without losing metadata."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    try:
        import numpy as np

        if np.ma.is_masked(value):
            return None
        if isinstance(value, np.ndarray):
            return value.tolist()
        if hasattr(value, "item"):
            return _jsonable(value.item())
    except (ImportError, TypeError, ValueError):
        pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return _text(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_cache(path: Path, *, refresh: bool) -> Any | None:
    if refresh or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _cache_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _table_rows(table: Iterable[Any]) -> list[dict[str, Any]]:
    """Convert an Astropy table to ordinary dictionaries."""
    names = list(getattr(table, "colnames", []))
    return [{name: _jsonable(row[name]) for name in names} for row in table]


def _load_euclid_tiles(cache_dir: Path, *, refresh: bool) -> list[dict[str, Any]]:
    cache_path = cache_dir / "euclid_vis_mosaics.json"
    cached = _read_cache(cache_path, refresh=refresh)
    if cached is not None:
        return list(cached["rows"])

    try:
        from astroquery.esa.euclid import Euclid
    except ImportError as exc:
        raise RuntimeError(
            "astroquery is required; run this script in EuclidPolishEnv "
            "or install astroquery"
        ) from exc

    query = (
        "SELECT file_path, file_name, tile_index, instrument_name, filter_name, ra, dec "
        "FROM sedm.mosaic_product "
        "WHERE instrument_name = 'VIS' AND technique = 'IMAGE'"
    )
    Euclid.ROW_LIMIT = -1
    job = Euclid.launch_job_async(query)
    if job is None:
        raise RuntimeError("Euclid archive returned no job for the VIS mosaic query")
    rows = _table_rows(job.get_results())

    # Multiple product versions can describe one tile.  Keep one stable row;
    # the exact archive-side intersection below queries all matching products.
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        tile = _text(row.get("tile_index"))
        ra = _float(row.get("ra"))
        dec = _float(row.get("dec"))
        if tile and ra is not None and dec is not None:
            unique.setdefault(tile, row)
    output = list(unique.values())
    _write_json(cache_path, {"query": query, "rows": output})
    return output


def _angular_distance_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    dra = math.radians((ra1 - ra2 + 180) % 360 - 180)
    dec1_rad = math.radians(dec1)
    dec2_rad = math.radians(dec2)
    cos_angle = (
        math.sin(dec1_rad) * math.sin(dec2_rad)
        + math.cos(dec1_rad) * math.cos(dec2_rad) * math.cos(dra)
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_angle))))


def _euclid_tile_groups(tiles: list[dict[str, Any]], link_deg: float = 3.0) -> list[list[dict[str, Any]]]:
    """Group neighboring Euclid tiles so archive cones are not repeated per tile."""
    points = [(_float(tile.get("ra")), _float(tile.get("dec"))) for tile in tiles]
    groups: list[list[dict[str, Any]]] = []
    unseen = set(range(len(tiles)))
    while unseen:
        first = unseen.pop()
        pending = [first]
        indices = [first]
        while pending:
            current = pending.pop()
            ra1, dec1 = points[current]
            if ra1 is None or dec1 is None:
                continue
            linked = []
            for other in unseen:
                ra2, dec2 = points[other]
                if (
                    ra2 is not None
                    and dec2 is not None
                    and _angular_distance_deg(ra1, dec1, ra2, dec2) <= link_deg
                ):
                    linked.append(other)
            for other in linked:
                unseen.remove(other)
                pending.append(other)
                indices.append(other)
        groups.append([tiles[index] for index in indices])
    return sorted(groups, key=lambda group: _text(group[0].get("tile_index")))


def _scope_for_tiles(tiles: list[dict[str, Any]], radius_deg: float) -> dict[str, Any]:
    """Return a spherical-cap query covering a connected Euclid tile group."""
    import numpy as np

    vectors = []
    for tile in tiles:
        ra = math.radians(float(tile["ra"]))
        dec = math.radians(float(tile["dec"]))
        vectors.append((math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)))
    mean = np.asarray(vectors, dtype=float).mean(axis=0)
    center_ra = math.degrees(math.atan2(mean[1], mean[0])) % 360.0
    center_dec = math.degrees(math.atan2(mean[2], math.hypot(mean[0], mean[1])))
    extent = max(
        _angular_distance_deg(center_ra, center_dec, float(tile["ra"]), float(tile["dec"]))
        for tile in tiles
    )
    tile_ids = ",".join(sorted(_text(tile["tile_index"]) for tile in tiles))
    return {
        "center_ra": center_ra,
        "center_dec": center_dec,
        "query_radius_deg": extent + radius_deg,
        "tile_ids": tile_ids,
    }


def _split_euclid_groups(
    tiles: list[dict[str, Any]], radius_deg: float, max_cone_deg: float,
) -> list[list[dict[str, Any]]]:
    """Split connected fields until each archive cone is bounded."""
    pending = list(_euclid_tile_groups(tiles))
    output: list[list[dict[str, Any]]] = []
    while pending:
        group = pending.pop()
        scope = _scope_for_tiles(group, radius_deg)
        if scope["query_radius_deg"] <= max_cone_deg or len(group) <= 1:
            output.append(group)
            continue
        ra_values = [float(tile["ra"]) for tile in group]
        dec_values = [float(tile["dec"]) for tile in group]
        ra_span = (max(ra_values) - min(ra_values)) * math.cos(math.radians(float(scope["center_dec"])))
        dec_span = max(dec_values) - min(dec_values)
        if dec_span >= ra_span:
            ordered = sorted(group, key=lambda tile: float(tile["dec"]))
        else:
            ordered = sorted(group, key=lambda tile: float(tile["ra"]))
        midpoint = len(ordered) // 2
        pending.append(ordered[:midpoint])
        pending.append(ordered[midpoint:])
    return sorted(output, key=lambda group: _text(group[0].get("tile_index")))


def _mast_rows_for_scope(
    scope: Mapping[str, Any],
    *,
    cache_dir: Path,
    refresh: bool,
) -> list[dict[str, Any]]:
    cache_key = _cache_key(f"{scope['tile_ids']}:{scope['query_radius_deg']:.6f}")
    cache_path = cache_dir / "mast" / f"scope_{cache_key}.json"
    cached = _read_cache(cache_path, refresh=refresh)
    if cached is not None:
        return list(cached["rows"])

    try:
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        from astroquery.mast import Observations
    except ImportError as exc:
        raise RuntimeError(
            "astroquery and astropy are required for the MAST backend"
        ) from exc

    coord = SkyCoord(ra=float(scope["center_ra"]), dec=float(scope["center_dec"]), unit="deg", frame="icrs")
    table = Observations.query_criteria(
        coordinates=coord,
        radius=float(scope["query_radius_deg"]) * u.deg,
        obs_collection="JWST",
        intentType="science",
        dataproduct_type="image",
    )
    rows = _table_rows(table)
    _write_json(cache_path, {**scope, "rows": rows})
    return rows


def _esa_rows_for_scope(
    scope: Mapping[str, Any],
    *,
    cache_dir: Path,
    refresh: bool,
) -> list[dict[str, Any]]:
    cache_key = _cache_key(f"{scope['tile_ids']}:{scope['query_radius_deg']:.6f}")
    cache_path = cache_dir / "esa" / f"scope_{cache_key}.json"
    cached = _read_cache(cache_path, refresh=refresh)
    if cached is not None:
        return list(cached["rows"])

    try:
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        from astroquery.esa.jwst import Jwst
    except ImportError as exc:
        raise RuntimeError(
            "astroquery and astropy are required for the ESA-JWST backend"
        ) from exc

    coord = SkyCoord(ra=float(scope["center_ra"]), dec=float(scope["center_dec"]), unit="deg", frame="icrs")
    job = Jwst.cone_search(
        coordinate=coord, radius=float(scope["query_radius_deg"]) * u.deg, async_job=True,
    )
    table = job.get_results() if hasattr(job, "get_results") else job
    rows = _table_rows(table)
    _write_json(cache_path, {**scope, "rows": rows})
    return rows


def _polygon_from_s_region(value: Any) -> list[tuple[float, float]] | None:
    text = _text(value)
    match = _S_REGION_RE.match(text)
    if match is None:
        return None
    tokens = match.group(1).replace(",", " ").split()
    try:
        numbers = [float(token) for token in tokens]
    except ValueError:
        return None
    if len(numbers) < 6 or len(numbers) % 2:
        return None
    points = list(zip(numbers[::2], numbers[1::2], strict=True))
    if any(not math.isfinite(ra) or not math.isfinite(dec) or not -90 <= dec <= 90 for ra, dec in points):
        return None
    return points


def _euclid_exact_matches(
    jwst: Mapping[str, Any],
    *,
    cache_dir: Path,
    refresh: bool,
) -> list[dict[str, Any]] | None:
    """Ask Euclid whether its VIS ``fov`` intersects a JWST ``s_region``."""
    polygon = _polygon_from_s_region(_row_value(jwst, "s_region", "stc_s", "position_bounds_spoly"))
    if polygon is None:
        return None

    observation_id = _text(_row_value(jwst, "obs_id", "obsid", "observationid", "observation_id"))
    key = observation_id or repr(polygon)
    cache_path = cache_dir / "exact" / f"{_cache_key(key)}.json"
    cached = _read_cache(cache_path, refresh=refresh)
    if cached is not None:
        if cached.get("error"):
            return None
        return list(cached["rows"])

    try:
        from astroquery.esa.euclid import Euclid
    except ImportError as exc:
        raise RuntimeError("astroquery is required for exact Euclid footprint matching") from exc

    values = ", ".join(f"{ra:.10f}, {dec:.10f}" for ra, dec in polygon)
    query = (
        "SELECT file_path, file_name, tile_index, instrument_name, filter_name, ra, dec "
        "FROM sedm.mosaic_product "
        "WHERE instrument_name = 'VIS' AND technique = 'IMAGE' "
        f"AND INTERSECTS(mosaic_product.fov, POLYGON('ICRS', {values})) = 1"
    )
    try:
        Euclid.ROW_LIMIT = -1
        job = Euclid.launch_job_async(query)
        if job is None:
            return None
        rows = _table_rows(job.get_results())
    except Exception as exc:  # noqa: BLE001 — preserve candidate fallback on archive geometry errors
        _write_json(cache_path, {"query": query, "error": f"{type(exc).__name__}: {exc}", "rows": []})
        return None
    _write_json(cache_path, {"query": query, "rows": rows})
    return rows


def _jwst_identity(row: Mapping[str, Any], backend: str) -> str:
    observation_id = _text(_row_value(row, "obs_id", "obsid", "observationid", "observation_id"))
    archive_id = _text(_row_value(row, "ArchiveFileID", "archivefileid"))
    return f"{backend}:{observation_id or archive_id or repr(sorted(row.items()))}"


def _distance_deg(tile: Mapping[str, Any], jwst: Mapping[str, Any]) -> float | None:
    ra = _float(_row_value(jwst, "s_ra", "ra", "ra_deg", "target_ra"))
    dec = _float(_row_value(jwst, "s_dec", "dec", "dec_deg", "target_dec"))
    tile_ra = _float(tile.get("ra"))
    tile_dec = _float(tile.get("dec"))
    if None in {ra, dec, tile_ra, tile_dec}:
        return None
    return _angular_distance_deg(ra, dec, tile_ra, tile_dec)


def _output_row(
    tile: Mapping[str, Any],
    jwst: Mapping[str, Any],
    *,
    backend: str,
    status: str,
    method: str,
) -> dict[str, Any]:
    return {
        "euclid_tile_index": _text(tile.get("tile_index")),
        "euclid_ra_deg": _float(tile.get("ra")),
        "euclid_dec_deg": _float(tile.get("dec")),
        "euclid_file_name": _text(tile.get("file_name")),
        "jwst_archive": backend,
        "jwst_observation_id": _text(_row_value(jwst, "obs_id", "observationid", "observation_id")),
        "jwst_obsid": _text(_row_value(jwst, "obsid", "objID", "archivefileid")),
        "jwst_target_name": _text(_row_value(jwst, "target_name", "targetname", "sci_targname")),
        "jwst_proposal_id": _text(_row_value(jwst, "proposal_id", "sci_pep_id")),
        "jwst_instrument": _text(_row_value(jwst, "instrument_name", "instrument")),
        "jwst_filters": _text(_row_value(jwst, "filters", "filter", "filter_name")),
        "jwst_data_rights": _text(_row_value(jwst, "dataRights", "data_rights", "public")),
        "jwst_exposure_time_s": _float(_row_value(jwst, "t_exptime", "exposure_time")),
        "jwst_ra_deg": _float(_row_value(jwst, "s_ra", "ra", "ra_deg", "target_ra")),
        "jwst_dec_deg": _float(_row_value(jwst, "s_dec", "dec", "dec_deg", "target_dec")),
        "jwst_distance_deg": _distance_deg(tile, jwst),
        "jwst_data_url": _text(_row_value(jwst, "dataURL", "data_url", "access_url")),
        "jwst_jpeg_url": _text(_row_value(jwst, "jpegURL", "jpeg_url")),
        "jwst_s_region": _text(_row_value(jwst, "s_region", "stc_s", "position_bounds_spoly")),
        "footprint_status": status,
        "overlap_method": method,
    }


def discover(
    *,
    cache_dir: Path,
    radius_deg: float,
    jwst_archive: str,
    refresh: bool,
    public_only: bool,
    max_cone_deg: float,
    scope_start: int = 0,
    scope_limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tiles = _load_euclid_tiles(cache_dir, refresh=refresh)
    connected_groups = _euclid_tile_groups(tiles)
    groups = _split_euclid_groups(tiles, radius_deg, max_cone_deg)
    backends = ["mast", "esa"] if jwst_archive == "both" else [jwst_archive]
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    candidate_ids: set[str] = set()
    exact_ids: set[str] = set()

    scope_end = len(groups) if scope_limit is None else min(len(groups), scope_start + scope_limit)
    selected_groups = groups[scope_start:scope_end]
    for index, group in enumerate(selected_groups, start=scope_start + 1):
        scope = _scope_for_tiles(group, radius_deg)
        print(
            f"[{index}/{len(groups)}] Euclid field with {len(group)} tiles "
            f"at {scope['center_ra']:.3f}, {scope['center_dec']:.3f} "
            f"(cone {scope['query_radius_deg']:.2f} deg)"
        )
        for backend in backends:
            finder = _mast_rows_for_scope if backend == "mast" else _esa_rows_for_scope
            jwst_rows = finder(scope, cache_dir=cache_dir, refresh=refresh)
            for jwst in jwst_rows:
                if not _is_direct_imaging(jwst) or (public_only and not _is_public(jwst)):
                    continue
                jwst_key = (backend, _jwst_identity(jwst, backend))
                candidate_ids.add("|".join(jwst_key))
                exact = (
                    _euclid_exact_matches(jwst, cache_dir=cache_dir, refresh=refresh)
                    if backend == "mast" else None
                )
                if exact is not None:
                    exact_ids.add("|".join(jwst_key))
                    for exact_tile in exact:
                        tile_match = dict(exact_tile)
                        key = (backend, _jwst_identity(jwst, backend), _text(exact_tile.get("tile_index")))
                        rows[key] = _output_row(
                            tile_match, jwst, backend=backend,
                            status="exact_intersection", method="euclid_fov_vs_jwst_s_region",
                        )
                    continue

                # No usable JWST polygon: retain only nearby tile candidates,
                # not every tile in the field-level cone.
                for tile in group:
                    distance = _distance_deg(tile, jwst)
                    if distance is None or distance > radius_deg:
                        continue
                    key = (backend, _jwst_identity(jwst, backend), _text(tile.get("tile_index")))
                    rows[key] = _output_row(
                        tile, jwst, backend=backend,
                        status="candidate_only", method="tile_center_radius",
                    )

    result = sorted(
        rows.values(),
        key=lambda row: (
            row["jwst_archive"],
            row["euclid_tile_index"],
            row["jwst_observation_id"],
        ),
    )
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "jwst_archive": jwst_archive,
        "radius_deg": radius_deg,
        "public_only": public_only,
        "euclid_tile_count": len(tiles),
        "euclid_field_group_count": len(connected_groups),
        "euclid_query_scope_count": len(groups),
        "processed_scope_start": scope_start,
        "processed_scope_count": len(selected_groups),
        "partial": scope_start != 0 or scope_end != len(groups),
        "jwst_candidate_count": len(candidate_ids),
        "exact_intersection_observation_count": len(exact_ids),
        "result_count": len(result),
        "exact_intersections_require_mast_s_region": True,
    }
    return result, manifest


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("data/jwst_euclid_overlap"),
        help="cache directory for archive responses (default: %(default)s)",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="CSV output path (default: <cache-dir>/overlap.csv)",
    )
    parser.add_argument(
        "--json", dest="json_path", type=Path, default=None,
        help="JSON manifest path (default: <cache-dir>/overlap.json)",
    )
    parser.add_argument(
        "--radius-deg", type=float, default=0.55,
        help="candidate cone radius around each Euclid tile centre (default: %(default)s)",
    )
    parser.add_argument(
        "--max-cone-deg", type=float, default=3.0,
        help="maximum field-level archive cone radius before splitting (default: %(default)s)",
    )
    parser.add_argument(
        "--scope-start", type=int, default=0,
        help="zero-based query-scope index to start at (default: %(default)s)",
    )
    parser.add_argument(
        "--scope-limit", type=int, default=None,
        help="process at most this many query scopes (default: all)",
    )
    parser.add_argument(
        "--jwst-archive", choices=("mast", "esa", "both"), default="mast",
        help="JWST metadata service to query (default: %(default)s)",
    )
    parser.add_argument(
        "--include-nonpublic", action="store_true",
        help="include JWST rows not marked PUBLIC by the archive",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="ignore cached archive responses and query again",
    )
    args = parser.parse_args(argv)
    if args.radius_deg <= 0 or args.max_cone_deg <= 0:
        parser.error("--radius-deg and --max-cone-deg must be positive")
    if args.max_cone_deg <= args.radius_deg:
        parser.error("--max-cone-deg must exceed --radius-deg")
    if args.scope_start < 0 or (args.scope_limit is not None and args.scope_limit <= 0):
        parser.error("--scope-start must be nonnegative and --scope-limit must be positive")

    out_csv = args.out or args.cache_dir / "overlap.csv"
    out_json = args.json_path or args.cache_dir / "overlap.json"
    rows, manifest = discover(
        cache_dir=args.cache_dir,
        radius_deg=args.radius_deg,
        jwst_archive=args.jwst_archive,
        refresh=args.refresh,
        public_only=not args.include_nonpublic,
        max_cone_deg=args.max_cone_deg,
        scope_start=args.scope_start,
        scope_limit=args.scope_limit,
    )
    _write_csv(out_csv, rows)
    _write_json(out_json, {"manifest": manifest, "rows": rows})
    print(f"Wrote {len(rows)} overlap rows to {out_csv}")
    print(f"Wrote manifest to {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
