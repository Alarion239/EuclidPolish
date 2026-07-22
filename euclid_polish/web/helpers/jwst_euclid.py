"""Cached paired Euclid/JWST downloads and WCS registration.

The overlap discovery table is deliberately the input to this module.  A pair
is published only after both archive products have been downloaded, the JWST
image has been sampled on the Euclid image grid, and the display PNGs have
been written.  This keeps the WebUI cache useful after the archive session has
gone away and avoids presenting a half-complete field.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from euclid_polish.config import Config

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")
_IMAGE_SUFFIXES = (".fits", ".fits.gz", ".fit", ".fit.gz")
_JWST_FILTER = re.compile(r"(?:^|[-_])((?:f|F)\d{3,4}[a-zA-Z0-9]*)")
_OVERLAP_FILENAMES = (
    "esa.csv",
    "mast.csv",
    "overlap.csv",
    "esa_partial.csv",
    "mast_partial.csv",
)


def overlap_root() -> Path:
    """Return the configured, ignored cache root for overlap products."""
    return Path(Config.DATA_DIR) / "jwst_euclid_overlap"


def pair_root() -> Path:
    return overlap_root() / "paired_fields"


def coverage_scan_path() -> Path:
    return overlap_root() / "euclid_coverage_scan.json"


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if np.ma.is_masked(value):
            return ""
        value = value.item() if hasattr(value, "item") else value
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text in {"", "--", "nan", "None", "null"} else text


def _number(value: Any) -> float | None:
    text = _text(value)
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _safe(value: str, fallback: str = "unknown") -> str:
    cleaned = _SAFE.sub("-", _text(value)).strip("-._")
    return cleaned or fallback


def field_id(archive: str, tile_index: str, observation_id: str, size_arcsec: float) -> str:
    """Build a stable path-safe id for one archive pair and cutout size."""
    size_token = f"{size_arcsec:.1f}".rstrip("0").rstrip(".").replace(".", "p")
    base = f"{_safe(archive)}-{_safe(tile_index)}-{_safe(observation_id)}-s{size_token}"
    if len(base) <= 180:
        return base
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:12]
    return f"{_safe(archive)}-{_safe(tile_index)}-{digest}-s{size_token}"


def location_id(ra: float, dec: float, size_arcsec: float) -> str:
    """Stable id for one sky location, irrespective of its JWST products."""
    size_token = f"{size_arcsec:.1f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"location-ra{ra:.7f}-dec{dec:.7f}-s{size_token}".replace("+", "p")


def jwst_filter_name(row: Mapping[str, Any]) -> str:
    """Extract a human band name from discovery metadata or the product id."""
    for value in (row.get("jwst_filters"), row.get("jwst_observation_id")):
        text = _text(value)
        match = _JWST_FILTER.search(text)
        if match:
            return match.group(1).upper()
    return ""


def jwst_header_filter(header: Mapping[str, Any], fallback: str = "JWST") -> str:
    """Prefer the actual archive filter/pupil once a JWST FITS is downloaded."""
    for value in (header.get("FILTER"), header.get("PUPIL")):
        match = _JWST_FILTER.search(_text(value))
        if match:
            return match.group(1).upper()
    return fallback


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    try:
        if np.ma.is_masked(value):
            return None
        if hasattr(value, "item"):
            return _jsonable(value.item())
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return _text(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _row_value(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return None


def _normalise_row(row: Mapping[str, Any], archive: str | None = None) -> dict[str, Any]:
    result = {str(key): _jsonable(value) for key, value in row.items()}
    result["jwst_archive"] = _text(result.get("jwst_archive") or archive or "esa").lower()
    result["euclid_tile_index"] = _text(result.get("euclid_tile_index"))
    result["jwst_observation_id"] = _text(
        result.get("jwst_observation_id") or result.get("jwst_obsid")
    )
    return result


def _coverage_key(ra: float, dec: float) -> str:
    return f"{ra:.7f},{dec:.7f}"


def _load_coverage_scan() -> dict[str, Any]:
    path = coverage_scan_path()
    if not path.exists():
        return {"version": 2, "results": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": 2, "results": {}}
    if (
        not isinstance(payload, dict)
        or payload.get("version") != 2
        or not isinstance(payload.get("results"), dict)
    ):
        # Version 1 only tested catalog footprints.  It must not be reused as
        # evidence that a real cutout contains usable VIS pixels.
        return {"version": 2, "results": {}}
    return payload


def _coverage_scan_summary(
    scan: Mapping[str, Any], unique_count: int, keys: Iterable[str] | None = None,
) -> dict[str, Any]:
    results = scan.get("results", {})
    if not isinstance(results, Mapping):
        results = {}
    values = (
        [results[key] for key in keys if key in results]
        if keys is not None
        else list(results.values())
    )
    counts = {
        "covered_count": sum(
            result.get("status") == "covered"
            for result in values
            if isinstance(result, Mapping)
        ),
        "not_covered_count": sum(
            result.get("status") == "not_covered"
            for result in values
            if isinstance(result, Mapping)
        ),
        "error_count": sum(
            result.get("status") == "error"
            for result in values
            if isinstance(result, Mapping)
        ),
    }
    return {
        "checked_count": counts["covered_count"] + counts["not_covered_count"] + counts["error_count"],
        "unique_count": unique_count,
        **counts,
        "updated_utc": _text(scan.get("updated_utc")),
    }


def _coverage_for_row(row: Mapping[str, Any], scan: Mapping[str, Any]) -> Mapping[str, Any] | None:
    ra, dec = field_coordinates(row)
    if ra is None or dec is None:
        return None
    results = scan.get("results", {})
    result = results.get(_coverage_key(ra, dec)) if isinstance(results, Mapping) else None
    return result if isinstance(result, Mapping) else None


def overlap_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load cached discovery rows and a small source-status summary."""
    root = overlap_root()
    coverage_scan = _load_coverage_scan()
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    sources: list[str] = []
    for filename in _OVERLAP_FILENAMES:
        path = root / filename
        if not path.exists():
            continue
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for raw in reader:
                    row = _normalise_row(raw)
                    key = (
                        row["jwst_archive"],
                        row["euclid_tile_index"],
                        row["jwst_observation_id"],
                    )
                    if all(key):
                        rows.setdefault(key, row)
            sources.append(filename)
        except (OSError, csv.Error):
            continue

    manifest: dict[str, Any] = {}
    for name in ("esa_partial.json", "mast_partial.json", "overlap.json", "esa.json", "mast.json"):
        path = root / name
        if not path.exists():
            continue
        try:
            candidate = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(candidate, dict):
            manifest = candidate
            break

    output = []
    for row in rows.values():
        identifier = field_id(
            row["jwst_archive"], row["euclid_tile_index"], row["jwst_observation_id"], 30.0,
        )
        row = dict(row)
        row["field_id"] = identifier
        coverage = _coverage_for_row(row, coverage_scan)
        row["euclid_coverage_status"] = _text(coverage.get("status")) if coverage else "unchecked"
        row["euclid_coverage_tile_count"] = int(coverage.get("tile_count", 0)) if coverage else 0
        row["euclid_coverage_error"] = _text(coverage.get("error")) if coverage else ""
        manifest_path = pair_root() / identifier / "manifest.json"
        row["available"] = False
        if manifest_path.exists():
            try:
                cached_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cached_manifest = None
            row["available"] = (
                isinstance(cached_manifest, dict)
                and _cached_pair_is_usable(manifest_path.parent, cached_manifest)
            )
        output.append(row)
    output.sort(key=lambda row: (
        not row.get("available", False),
        row.get("footprint_status", "") != "exact_intersection",
        row.get("jwst_target_name", ""),
        row["euclid_tile_index"],
        row["jwst_observation_id"],
    ))
    position_keys = {
        _coverage_key(ra, dec)
        for row in output
        for ra, dec in [field_coordinates(row)]
        if ra is not None and dec is not None
    }
    status = {
        "source_files": sources,
        "partial": bool(manifest.get("partial", False)),
        "source_manifest": manifest,
        "count": len(output),
        "coverage_scan": _coverage_scan_summary(coverage_scan, len(position_keys), position_keys),
    }
    return output, status


def find_overlap_row(archive: str, tile_index: str, observation_id: str) -> dict[str, Any] | None:
    key = (archive.lower(), _text(tile_index), _text(observation_id))
    rows, _ = overlap_rows()
    return next(
        (row for row in rows if (
            row.get("jwst_archive"), row.get("euclid_tile_index"), row.get("jwst_observation_id"),
        ) == key),
        None,
    )


def _angular_separation_arcsec(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Small-angle separation, sufficient for grouping duplicate pointings."""
    ra0, dec0 = a
    ra1, dec1 = b
    return math.hypot(
        (ra0 - ra1) * math.cos(math.radians((dec0 + dec1) / 2.0)), dec0 - dec1,
    ) * 3600.0


def _location_components(rows: list[dict[str, Any]], link_arcsec: float = 1.0) -> list[list[int]]:
    """Connected components of duplicate JWST position rows within one arcsec."""
    parents = list(range(len(rows)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def join(left: int, right: int) -> None:
        left_root, right_root = root(left), root(right)
        if left_root != right_root:
            parents[right_root] = left_root

    coordinates = [field_coordinates(row) for row in rows]
    for index, coordinate in enumerate(coordinates):
        if coordinate[0] is None or coordinate[1] is None:
            continue
        for other_index, other in enumerate(coordinates[:index]):
            if other[0] is None or other[1] is None:
                continue
            if _angular_separation_arcsec(coordinate, other) <= link_arcsec:
                join(index, other_index)
    grouped: dict[int, list[int]] = {}
    for index in range(len(rows)):
        grouped.setdefault(root(index), []).append(index)
    return list(grouped.values())


def _location_products(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Choose one best calibrated JWST product for each filter at a location."""
    raw_rows = [dict(raw) for raw in rows]
    candidates: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in raw_rows:
        band = jwst_filter_name(row)
        if not band:
            continue  # detector-only archive rows duplicate a named filter product
        instrument = _text(row.get("jwst_instrument") or "JWST imaging")
        candidates.setdefault((instrument, band), []).append(row)

    # Some archive rows name a detector but omit the filter entirely.  Keep a
    # location selectable rather than silently dropping it; the downloaded
    # FITS header then supplies the real filter name in the saved manifest.
    if not candidates and raw_rows:
        fallback = raw_rows[0]
        instrument = _text(fallback.get("jwst_instrument") or "JWST imaging")
        candidates[(instrument, "JWST")] = raw_rows

    def rank(row: Mapping[str, Any]) -> tuple[int, int, str]:
        observation = _text(row.get("jwst_observation_id")).lower()
        # ESA ``oNNN`` products are field mosaics; detector readouts are a
        # fallback only when a mosaic is not in the discovery table.
        return (
            0 if re.search(r"(?:^|[-_])o\d+", observation) else 1,
            -int(_number(row.get("jwst_exposure_time_s")) or 0),
            observation,
        )

    selected = [min(options, key=rank) for options in candidates.values()]
    return sorted(selected, key=lambda row: (
        _text(row.get("jwst_instrument")), jwst_filter_name(row),
        _text(row.get("jwst_observation_id")),
    ))


def location_groups() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collapse archive products into selectable sky locations and JWST bands."""
    rows, status = overlap_rows()
    groups: list[dict[str, Any]] = []
    for component in _location_components(rows):
        source_rows = [rows[index] for index in component]
        products = _location_products(source_rows)
        if not products:
            continue
        representative = products[0]
        ra, dec = field_coordinates(representative)
        if ra is None or dec is None:
            continue
        identifier = location_id(ra, dec, 30.0)
        manifest_path = pair_root() / identifier / "manifest.json"
        available = False
        if manifest_path.exists():
            try:
                cached_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cached_manifest = None
            available = isinstance(cached_manifest, dict) and _cached_pair_is_usable(
                manifest_path.parent, cached_manifest,
            )
        group = dict(representative)
        group.update({
            "field_id": identifier,
            "jwst_product_count": len(products),
            "jwst_row_count": len(source_rows),
            "jwst_filters": ", ".join(jwst_filter_name(product) for product in products),
            "jwst_products": products,
            "available": available,
        })
        groups.append(group)
    groups.sort(key=lambda row: (
        not row.get("available", False),
        row.get("jwst_target_name", ""), row["field_id"],
    ))
    grouped_status = dict(status)
    grouped_status["product_count"] = len(rows)
    grouped_status["count"] = len(groups)
    return groups, grouped_status


def find_location_group(identifier: str) -> dict[str, Any] | None:
    groups, _ = location_groups()
    return next((group for group in groups if group.get("field_id") == identifier), None)


def _table_rows(table: Iterable[Any]) -> list[dict[str, Any]]:
    names = list(getattr(table, "colnames", []))
    return [{name: _jsonable(row[name]) for name in names} for row in table]


def euclid_tile(
    tile_index: str, row: Mapping[str, Any], *, refresh: bool = False,
) -> dict[str, Any]:
    """Resolve the Euclid product path, using the overlap cache first."""
    cached_path = overlap_root() / "euclid_vis_mosaics.json"
    if not refresh and cached_path.exists():
        try:
            payload = json.loads(cached_path.read_text(encoding="utf-8"))
            for candidate in payload.get("rows", []):
                if _text(candidate.get("tile_index")) == tile_index and candidate.get("file_path"):
                    return _jsonable(candidate)
        except (OSError, json.JSONDecodeError):
            pass

    try:
        from astroquery.esa.euclid import Euclid
    except ImportError as exc:
        raise RuntimeError("astroquery is required for Euclid downloads") from exc

    escaped = tile_index.replace("'", "''")
    query = (
        "SELECT file_path, file_name, tile_index, instrument_name, filter_name, ra, dec "
        "FROM sedm.mosaic_product WHERE instrument_name = 'VIS' AND technique = 'IMAGE' "
        f"AND tile_index = '{escaped}'"
    )
    Euclid.ROW_LIMIT = -1
    job = Euclid.launch_job_async(query)
    if job is None:
        raise RuntimeError(f"Euclid archive returned no query job for tile {tile_index}")
    rows = _table_rows(job.get_results())
    if not rows:
        raise RuntimeError(f"Euclid archive has no VIS mosaic product for tile {tile_index}")
    return rows[0]


def euclid_product_path(tile: Mapping[str, Any]) -> str:
    """Join the archive directory and filename into the cutout product path."""
    path = _text(tile.get("file_path"))
    filename = _text(tile.get("file_name"))
    if path and filename and not path.rstrip("/").endswith(f"/{filename}"):
        return f"{path.rstrip('/')}/{filename}"
    return path


def field_coordinates(row: Mapping[str, Any]) -> tuple[float | None, float | None]:
    """Use the JWST footprint center, falling back to the Euclid tile center."""
    ra = _number(row.get("jwst_ra_deg")) or _number(row.get("euclid_ra_deg"))
    dec = _number(row.get("jwst_dec_deg")) or _number(row.get("euclid_dec_deg"))
    return ra, dec


def euclid_tiles_covering(ra: float, dec: float, *, strict: bool = False) -> list[dict[str, Any]]:
    """Query Euclid for VIS mosaics whose archive footprint covers a point."""
    try:
        from astroquery.esa.euclid import Euclid
    except ImportError as exc:
        raise RuntimeError("astroquery is required for Euclid downloads") from exc

    query = (
        "SELECT file_path, file_name, tile_index, instrument_name, filter_name, ra, dec "
        "FROM sedm.mosaic_product WHERE instrument_name = 'VIS' AND technique = 'IMAGE' "
        f"AND INTERSECTS(mosaic_product.fov, CIRCLE('ICRS', {ra:.10f}, {dec:.10f}, 0.0003)) = 1"
    )
    Euclid.ROW_LIMIT = -1
    job = Euclid.launch_job_async(query)
    if job is None:
        if strict:
            raise RuntimeError("Euclid coverage query returned no TAP job")
        return []
    return _table_rows(job.get_results())


def _probe_euclid_tiles(
    euclid_client: Any,
    tiles: Iterable[Mapping[str, Any]],
    *,
    coordinate: Any,
    radius: Any,
    destination_dir: Path,
) -> tuple[dict[str, Any] | None, int, list[str]]:
    """Try real VIS cutouts and return the first one with non-zero pixels."""
    blank_count = 0
    errors: list[str] = []
    destination = destination_dir / "euclid_probe.fits"
    for tile in tiles:
        file_path = euclid_product_path(tile)
        tile_index = _text(tile.get("tile_index"))
        if not file_path or not tile_index:
            errors.append("coverage row has no downloadable VIS product")
            continue
        try:
            _download_euclid_cutout(
                euclid_client,
                file_path=file_path,
                tile_index=tile_index,
                coordinate=coordinate,
                radius=radius,
                destination=destination,
            )
            data, _, _, _ = _find_image(destination)
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append(f"{tile_index}: {exc}")
            continue
        if _has_signal(data):
            return dict(tile), blank_count, errors
        blank_count += 1
    return None, blank_count, errors


def scan_euclid_coverage(progress: Any = None) -> dict[str, Any]:
    """Check every unique JWST field center against the Euclid VIS footprint.

    Results are written after each archive query, so an interrupted scan can be
    resumed without repeating successful coverage checks.
    """
    rows, _ = overlap_rows()
    positions: dict[str, tuple[float, float]] = {}
    for row in rows:
        ra, dec = field_coordinates(row)
        if ra is not None and dec is not None:
            positions.setdefault(_coverage_key(ra, dec), (ra, dec))

    scan = _load_coverage_scan()
    scan["version"] = 2
    results = scan.setdefault("results", {})
    if not isinstance(results, dict):
        results = {}
        scan["results"] = results
    pending = [key for key in positions if not isinstance(results.get(key), Mapping)
               or results[key].get("status") == "error"]
    total = len(pending)
    done = 0
    if progress:
        progress(done, total, "checking Euclid VIS coverage")

    for key in pending:
        ra, dec = positions[key]
        checked_utc = datetime.now(UTC).isoformat()
        try:
            tiles = euclid_tiles_covering(ra, dec, strict=True)
            tile_records = [
                {
                    "tile_index": _text(tile.get("tile_index")),
                    "file_name": _text(tile.get("file_name")),
                    "file_path": _text(tile.get("file_path")),
                }
                for tile in tiles[:8]
            ]
            if not tiles:
                result = {
                    "status": "not_covered",
                    "tile_count": 0,
                    "reason": "no_metadata_footprint",
                    "tiles": [],
                    "ra_deg": ra,
                    "dec_deg": dec,
                    "checked_utc": checked_utc,
                }
            else:
                import astropy.units as u
                from astropy.coordinates import SkyCoord
                from astroquery.esa.euclid import Euclid

                probe_dir = Path(tempfile.mkdtemp(prefix=".coverage-probe-", dir=overlap_root()))
                try:
                    usable_tile, blank_count, probe_errors = _probe_euclid_tiles(
                        Euclid,
                        tiles,
                        coordinate=SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs"),
                        radius=15.0 * u.arcsec,
                        destination_dir=probe_dir,
                    )
                finally:
                    shutil.rmtree(probe_dir, ignore_errors=True)
                if usable_tile is not None:
                    result = {
                        "status": "covered",
                        "tile_count": len(tiles),
                        "usable_tile": {
                            "tile_index": _text(usable_tile.get("tile_index")),
                            "file_name": _text(usable_tile.get("file_name")),
                            "file_path": _text(usable_tile.get("file_path")),
                        },
                        "tiles": tile_records,
                        "ra_deg": ra,
                        "dec_deg": dec,
                        "checked_utc": checked_utc,
                    }
                elif probe_errors and not blank_count:
                    result = {
                        "status": "error",
                        "tile_count": len(tiles),
                        "tiles": tile_records,
                        "ra_deg": ra,
                        "dec_deg": dec,
                        "checked_utc": checked_utc,
                        "error": "; ".join(probe_errors[:3]),
                    }
                else:
                    result = {
                        "status": "not_covered",
                        "tile_count": len(tiles),
                        "reason": "blank_cutout",
                        "tiles": tile_records,
                        "ra_deg": ra,
                        "dec_deg": dec,
                        "checked_utc": checked_utc,
                    }
        except Exception as exc:  # noqa: BLE001 - continue checking other fields
            result = {
                "status": "error",
                "tile_count": 0,
                "ra_deg": ra,
                "dec_deg": dec,
                "checked_utc": checked_utc,
                "error": str(exc),
            }
        results[key] = result
        scan["updated_utc"] = checked_utc
        _write_json(coverage_scan_path(), scan)
        done += 1
        if progress:
            progress(done, total, f"checked {done}/{total} field centers")

    summary = _coverage_scan_summary(scan, len(positions), positions)
    summary["path"] = str(coverage_scan_path())
    return summary


def _find_image(path: Path) -> tuple[np.ndarray, Any, Any, str]:
    """Read the first 2-D image HDU and its celestial WCS."""
    from astropy.io import fits
    from astropy.wcs import WCS

    with fits.open(path, memmap=False) as hdul:
        primary_header = hdul[0].header.copy()
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            if data is None or np.ndim(data) != 2:
                continue
            image = np.asarray(data, dtype=np.float32)
            headers = [hdu.header, primary_header] if hdu is not hdul[0] else [hdu.header]
            for header in headers:
                try:
                    wcs = WCS(header).celestial
                    if wcs.has_celestial:
                        return image, header.copy(), wcs, hdu.name or "PRIMARY"
                except Exception:  # noqa: BLE001 - archive headers vary by instrument
                    continue
    raise ValueError(f"no 2-D image with celestial WCS found in {path.name}")


def _pixel_metadata(data: np.ndarray, wcs: Any, header: Any) -> dict[str, Any]:
    """Return compact product and pixel metadata for the viewer manifest."""
    try:
        from astropy.wcs.utils import proj_plane_pixel_scales

        scales = [float(abs(value) * 3600.0) for value in proj_plane_pixel_scales(wcs)[:2]]
    except Exception:  # noqa: BLE001 - some archive WCS headers omit a scale
        scales = []
    return {
        "shape": [int(value) for value in data.shape],
        "pixel_scale_arcsec": scales,
        "units": _text(header.get("BUNIT")),
        "instrument": _text(header.get("INSTRUME") or header.get("INSTRUMENT")),
        "detector": _text(header.get("DETECTOR")),
        "filter": _text(header.get("FILTER")),
        "pupil": _text(header.get("PUPIL")),
        "exposure_s": _number(header.get("EXPTIME") or header.get("EFFEXPTM")),
    }


def _native_sky_cutout(
    data: np.ndarray, wcs: Any, coordinate: Any, size_arcsec: float,
) -> tuple[np.ndarray, Any]:
    """Crop a field on its source WCS without changing its pixel scale."""
    import astropy.units as u
    from astropy.nddata import Cutout2D

    cutout = Cutout2D(
        data,
        position=coordinate,
        size=float(size_arcsec) * u.arcsec,
        wcs=wcs,
        mode="partial",
        fill_value=np.nan,
    )
    return np.asarray(cutout.data, np.float32), cutout.wcs


def _primary_image_header(source_header: Any, wcs: Any, product_name: str, history: str) -> Any:
    """Copy archive metadata into a WCS-correct primary image header."""
    header = source_header.copy()
    for key in ("XTENSION", "EXTNAME", "EXTVER", "PCOUNT", "GCOUNT", "THEAP"):
        header.pop(key, None)
    header.update(wcs.to_header(relax=True))
    header["SRCFILE"] = product_name[:68]
    header.add_history(history)
    return header


def _has_signal(data: np.ndarray) -> bool:
    finite = data[np.isfinite(data)]
    return finite.size > 0 and bool(np.any(finite != 0))


def _cached_pair_is_usable(directory: Path, manifest: Mapping[str, Any]) -> bool:
    """Reject old/partial caches, including successful-but-blank cutouts."""
    files = manifest.get("files", {})
    required = [
        directory / str(files.get("euclid", "euclid_vis.fits")),
        directory / str(files.get("jwst_native", "jwst_native.fits")),
        directory / str(files.get("euclid_png", "euclid_vis.png")),
        directory / str(files.get("jwst_png", "jwst_aligned.png")),
    ]
    if any(not path.is_file() or path.stat().st_size == 0 for path in required):
        return False
    try:
        euclid_data, _, _, _ = _find_image(required[0])
        jwst_data, _, _, _ = _find_image(required[1])
    except (OSError, ValueError):
        return False
    return _has_signal(euclid_data) and _has_signal(jwst_data)


def enrich_manifest_metadata(directory: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Add pixel metadata to older cached manifests when their FITS files exist."""
    result = dict(manifest)
    files = manifest.get("files", {})
    try:
        euclid_data, euclid_header, euclid_wcs, _ = _find_image(
            directory / str(files.get("euclid", "euclid_vis.fits")),
        )
        jwst_data, jwst_header, jwst_wcs, _ = _find_image(
            directory / str(files.get("jwst_native", "jwst_native.fits")),
        )
    except (OSError, ValueError):
        return result
    result.setdefault("euclid_product", result.get("euclid_file_name", ""))
    result["euclid_metadata"] = result.get(
        "euclid_metadata", _pixel_metadata(euclid_data, euclid_wcs, euclid_header),
    )
    result["jwst_metadata"] = result.get(
        "jwst_metadata", _pixel_metadata(jwst_data, jwst_wcs, jwst_header),
    )
    aligned_path = directory / str(files.get("jwst_aligned", "jwst_aligned_to_euclid.fits"))
    if aligned_path.is_file():
        try:
            aligned_data, aligned_header, aligned_wcs, _ = _find_image(aligned_path)
            result["aligned_metadata"] = result.get(
                "aligned_metadata", _pixel_metadata(aligned_data, aligned_wcs, aligned_header),
            )
        except (OSError, ValueError):
            pass
    return result


def align_to_target(data: np.ndarray, source_wcs: Any, target_wcs: Any, shape: tuple[int, int]) -> np.ndarray:
    """Sample a source image on the target WCS grid using bilinear pixels."""
    from scipy.ndimage import map_coordinates

    yy, xx = np.indices(shape, dtype=np.float64)
    sky = target_wcs.pixel_to_world(xx, yy)
    source_x, source_y = source_wcs.world_to_pixel(sky)
    coordinates = np.asarray([source_y, source_x])
    valid = map_coordinates(
        np.isfinite(data).astype(np.float32), coordinates, order=0, mode="constant", cval=0.0,
    ) > 0.5
    sampled = map_coordinates(
        np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0),
        coordinates, order=1, mode="constant", cval=0.0,
    ).astype(np.float32)
    sampled[~valid] = np.nan
    return sampled


def _write_display_png(data: np.ndarray, path: Path, accent: tuple[float, float, float]) -> dict[str, float]:
    from PIL import Image

    finite = data[np.isfinite(data)]
    if finite.size == 0:
        normalized = np.zeros(data.shape, dtype=np.float32)
        lo, hi = 0.0, 0.0
    else:
        lo, hi = np.percentile(finite, [0.5, 99.7])
        if not math.isfinite(float(hi)) or hi <= lo:
            hi = lo + 1.0
        normalized = np.clip((np.nan_to_num(data, nan=lo) - lo) / (hi - lo), 0.0, 1.0)
    normalized = np.arcsinh(8.0 * normalized) / np.arcsinh(8.0)
    rgb = np.stack([normalized * channel for channel in accent], axis=-1)
    Image.fromarray(np.asarray(np.clip(rgb * 255, 0, 255), dtype=np.uint8), mode="RGB").save(path)
    return {"display_min": float(lo), "display_max": float(hi)}


def _aligned_primary_header(source_header: Any, product_name: str) -> Any:
    """Make an archive image header safe for a new primary HDU."""
    header = source_header.copy()
    # Euclid image extensions can carry EXTNAME/EXTVER and extension-only
    # structural cards.  Some products contain a non-string EXTNAME; copying
    # that card into a new PrimaryHDU makes Astropy reject the aligned file.
    for key in ("XTENSION", "EXTNAME", "EXTVER", "PCOUNT", "GCOUNT", "THEAP"):
        header.pop(key, None)
    header["ALIGN"] = "JWST-EUCLID"
    header["SRCFILE"] = product_name[:68]
    header.add_history("JWST resampled onto the Euclid VIS cutout WCS")
    return header


def _copy_downloaded(source: Any, destination: Path) -> None:
    if isinstance(source, (list, tuple)):
        source = source[0] if source else None
    if source is None:
        raise RuntimeError(f"archive returned no local path for {destination.name}")
    source_path = Path(str(source))
    if not source_path.exists():
        raise RuntimeError(f"archive reported missing local file: {source_path}")
    shutil.copy2(source_path, destination)


def _is_readable_fits(path: Path) -> bool:
    """Return whether an archive output is a non-empty, readable FITS file."""
    try:
        if not path.is_file() or path.stat().st_size < 2880:
            return False
        from astropy.io import fits

        with fits.open(path, memmap=False):
            return True
    except (OSError, ValueError):
        return False


def _copy_valid_fits(source: Any, destination: Path) -> None:
    """Copy the first readable FITS path returned by an archive client."""
    candidates = source if isinstance(source, (list, tuple)) else [source]
    for candidate in candidates:
        if candidate is None:
            continue
        source_path = Path(str(candidate))
        if not _is_readable_fits(source_path):
            continue
        if source_path.resolve() != destination.resolve():
            shutil.copy2(source_path, destination)
        return
    raise RuntimeError("archive returned no readable FITS file")


def _download_euclid_cutout(
    euclid_client: Any,
    *,
    file_path: str,
    tile_index: str,
    coordinate: Any,
    radius: Any,
    destination: Path,
) -> None:
    """Download a Euclid cutout, recovering extracted files from bad placeholders."""
    last_error = "archive returned no readable FITS file"
    for attempt in range(2):
        with __import__("contextlib").suppress(OSError):
            destination.unlink()
        result = euclid_client.get_cutout(
            file_path=file_path,
            instrument="VIS",
            id=tile_index,
            coordinate=coordinate,
            radius=radius,
            output_file=str(destination),
            verbose=True,
        )
        if _is_readable_fits(destination):
            return
        try:
            _copy_valid_fits(result, destination)
            if _is_readable_fits(destination):
                return
        except RuntimeError as error:
            last_error = str(error)
        if result is None:
            last_error = "cutout client returned no file (archive request or network failed)"
        if attempt == 0:
            print("Euclid returned an invalid cutout file; retrying once")
    raise RuntimeError(f"Euclid archive did not return a readable VIS cutout: {last_error}")


def _choose_jwst_product(rows: Iterable[Mapping[str, Any]]) -> str:
    """Prefer a calibrated/resampled 2-D image over ramps and metadata."""
    names = []
    for row in rows:
        name = _text(_row_value(row, "filename", "file_name", "productFilename", "product_filename"))
        if name.lower().endswith(_IMAGE_SUFFIXES):
            names.append(name)
    if not names:
        raise RuntimeError("JWST archive returned no FITS science image product")
    def rank(name: str) -> tuple[int, int, str]:
        lower = name.lower()
        if "_i2d" in lower:
            priority = 0
        elif "_drz" in lower or "_drc" in lower:
            priority = 1
        elif "_cal" in lower:
            priority = 2
        elif "_rate" in lower or "_uncal" in lower:
            priority = 4
        else:
            priority = 3
        return priority, len(name), name
    return sorted(names, key=rank)[0]


def _download_jwst_esa(observation_id: str, destination: Path) -> str:
    from astroquery.esa.jwst import Jwst

    products = Jwst.get_product_list(
        observation_id=observation_id, cal_level=3, product_type="science",
    )
    product_name = _choose_jwst_product(_table_rows(products))
    # astroquery's ESA JWST client writes the requested filename in the
    # process working directory; copy it immediately into our transaction.
    downloaded = Jwst.get_product(file_name=product_name)
    _copy_downloaded(downloaded, destination)
    with __import__("contextlib").suppress(OSError):
        Path(str(downloaded)).unlink()
    return product_name


def _download_jwst_mast(observation_id: str, destination: Path) -> str:
    from astroquery.mast import Observations

    products = Observations.get_product_list(observation_id)
    rows = _table_rows(products)
    product_name = _choose_jwst_product(rows)
    selected = [
        row for row in products
        if _text(_row_value(row, "productFilename", "product_filename")) == product_name
    ]
    downloaded = Observations.download_products(
        selected or products[:1], download_dir=str(destination.parent), mrp_only=False,
    )
    paths = [Path(str(value)) for value in downloaded["Local Path"] if Path(str(value)).exists()]
    if not paths:
        raise RuntimeError("MAST returned no local JWST product path")
    shutil.copy2(paths[0], destination)
    return product_name


def _download_jwst(archive: str, observation_id: str, destination: Path) -> str:
    if archive == "mast":
        return _download_jwst_mast(observation_id, destination)
    if archive == "esa":
        return _download_jwst_esa(observation_id, destination)
    raise ValueError(f"unsupported JWST archive {archive!r}")


def download_and_align_pair(
    row: Mapping[str, Any],
    *,
    size_arcsec: float = 30.0,
    progress: Any | None = None,
) -> dict[str, Any]:
    """Download one Euclid field and all selected JWST filters atomically."""
    product_rows = [dict(product) for product in row.get("jwst_products", [])]
    if not product_rows:
        product_rows = [dict(row)]
    primary_row = product_rows[0]
    archive = _text(primary_row.get("jwst_archive") or "esa").lower()
    tile_index = _text(row.get("euclid_tile_index"))
    observation_id = _text(primary_row.get("jwst_observation_id") or primary_row.get("jwst_obsid"))
    ra, dec = field_coordinates(row)
    if not tile_index or not observation_id or ra is None or dec is None:
        raise ValueError("pair row needs Euclid tile, JWST observation id, and Euclid coordinates")
    if not 1.0 <= float(size_arcsec) <= 120.0:
        raise ValueError("size_arcsec must be between 1 and 120 arcsec")

    identifier = _text(row.get("field_id")) or field_id(
        archive, tile_index, observation_id, float(size_arcsec),
    )
    final_dir = pair_root() / identifier
    manifest_path = final_dir / "manifest.json"
    if manifest_path.exists():
        try:
            cached_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached_manifest = None
        if isinstance(cached_manifest, dict) and _cached_pair_is_usable(final_dir, cached_manifest):
            return cached_manifest
        # A previous archive response can be formally valid FITS but contain
        # only zeros when the requested center missed the tile footprint.
        shutil.rmtree(final_dir, ignore_errors=True)

    pair_root().mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{identifier}.", dir=pair_root()))
    try:
        temporary_dir.mkdir(parents=True, exist_ok=True)
        if progress:
            progress(1, len(product_rows) + 3, "checking Euclid VIS coverage")
        tile = euclid_tile(tile_index, row)
        file_path = euclid_product_path(tile)
        if not file_path:
            raise RuntimeError(f"Euclid tile {tile_index} has no downloadable file_path")

        import astropy.units as u
        from astropy.coordinates import SkyCoord
        from astropy.io import fits

        euclid_path = temporary_dir / "euclid_vis.fits"
        from astroquery.esa.euclid import Euclid

        if progress:
            progress(2, len(product_rows) + 3, "downloading Euclid VIS cutout")
        coordinate = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")
        radius = (float(size_arcsec) / 2.0) * u.arcsec
        covering_tiles = euclid_tiles_covering(ra, dec, strict=True)
        if not covering_tiles:
            raise RuntimeError("Euclid VIS has no archive footprint covering this JWST location")
        candidate_tiles: list[dict[str, Any]] = [dict(candidate) for candidate in covering_tiles]
        candidate_tiles.append(dict(tile))
        seen_paths = {file_path}
        alternatives_loaded = False
        discovery_errors: list[str] = []
        last_error = "archive returned no readable FITS file"
        blank_seen = False
        selected: tuple[dict[str, Any], str, np.ndarray, Any, Any, str] | None = None
        candidate_index = 0
        while candidate_index < len(candidate_tiles):
            candidate_tile = candidate_tiles[candidate_index]
            candidate_path = euclid_product_path(candidate_tile)
            candidate_tile_index = _text(candidate_tile.get("tile_index")) or tile_index
            if candidate_index > 0 and progress:
                progress(
                    2, len(product_rows) + 3,
                    f"trying alternate Euclid VIS product {candidate_index + 1}",
                )
            try:
                if not candidate_path:
                    raise RuntimeError("candidate has no downloadable file_path")
                _download_euclid_cutout(
                    Euclid,
                    file_path=candidate_path,
                    tile_index=candidate_tile_index,
                    coordinate=coordinate,
                    radius=radius,
                    destination=euclid_path,
                )
                candidate_data, candidate_header, candidate_wcs, candidate_hdu = _find_image(euclid_path)
                if _has_signal(candidate_data):
                    selected = (
                        candidate_tile,
                        candidate_path,
                        candidate_data,
                        candidate_header,
                        candidate_wcs,
                        candidate_hdu,
                    )
                    break
                blank_seen = True
            except (OSError, RuntimeError, ValueError) as exc:
                last_error = str(exc)

            if not alternatives_loaded and candidate_index == 0:
                alternatives_loaded = True
                # Refresh the original product metadata, then ask the archive
                # for every VIS product whose footprint covers this position.
                try:
                    fresh_tile = euclid_tile(tile_index, row, refresh=True)
                    fresh_path = euclid_product_path(fresh_tile)
                    if fresh_path and fresh_path not in seen_paths:
                        candidate_tiles.append(dict(fresh_tile))
                        seen_paths.add(fresh_path)
                except Exception as exc:  # noqa: BLE001 - retain the original cutout error
                    discovery_errors.append(f"metadata refresh: {exc}")
                try:
                    covering_tiles = euclid_tiles_covering(ra, dec, strict=True)
                    for covering_tile in covering_tiles:
                        covering_path = euclid_product_path(covering_tile)
                        if covering_path and covering_path not in seen_paths:
                            candidate_tiles.append(dict(covering_tile))
                            seen_paths.add(covering_path)
                except Exception as exc:  # noqa: BLE001 - retain the original cutout error
                    discovery_errors.append(f"coverage query: {exc}")
            candidate_index += 1

        if selected is None:
            if blank_seen:
                raise RuntimeError(
                    "Euclid returned a readable VIS cutout, but it contained no non-zero pixels; "
                    "the catalog footprint match is not a usable overlap for this JWST field"
                )
            details = last_error
            if discovery_errors:
                details = f"{details}; {'; '.join(discovery_errors[:2])}"
            raise RuntimeError(
                f"Euclid archive did not return a readable VIS cutout after trying "
                f"{len(candidate_tiles)} candidate product(s): {details}"
            )

        tile, file_path, euclid_data, euclid_header, euclid_wcs, euclid_hdu = selected

        jwst_bands: list[dict[str, Any]] = []
        jwst_errors: list[str] = []
        primary_data = primary_header = primary_wcs = primary_product = None
        primary_hdu = "PRIMARY"
        for product_index, product_row in enumerate(product_rows, start=1):
            product_archive = _text(product_row.get("jwst_archive") or "esa").lower()
            product_observation = _text(
                product_row.get("jwst_observation_id") or product_row.get("jwst_obsid"),
            )
            band = jwst_filter_name(product_row) or f"JWST{product_index}"
            band_key = _safe(band.lower())
            if progress:
                progress(
                    product_index + 2, len(product_rows) + 3,
                    f"downloading JWST {band} ({product_index}/{len(product_rows)})",
                )
            try:
                download_path = temporary_dir / f"jwst_download_{band_key}.fits"
                product_name = _download_jwst(product_archive, product_observation, download_path)
                jwst_data, jwst_header, jwst_wcs, jwst_hdu = _find_image(download_path)
                band = jwst_header_filter(jwst_header, band)
                band_key = _safe(band.lower())
                jwst_native, jwst_native_wcs = _native_sky_cutout(
                    jwst_data, jwst_wcs, coordinate, float(size_arcsec),
                )
                if not _has_signal(jwst_native):
                    raise RuntimeError("native-grid cutout has no non-zero pixels at this location")
                jwst_native_header = _primary_image_header(
                    jwst_header, jwst_native_wcs, product_name,
                    "JWST native-grid cutout; no resampling onto the Euclid grid",
                )
                relative_path = "jwst_native.fits" if not jwst_bands else f"jwst/{band_key}.fits"
                output_path = temporary_dir / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fits.PrimaryHDU(data=jwst_native, header=jwst_native_header).writeto(
                    output_path, overwrite=True, output_verify="silentfix",
                )
                metadata = _pixel_metadata(jwst_native, jwst_native_wcs, jwst_native_header)
                jwst_bands.append({
                    "key": f"jwst{len(jwst_bands)}",
                    "filter": band,
                    "archive": product_archive,
                    "observation_id": product_observation,
                    "product": product_name,
                    "instrument": _text(product_row.get("jwst_instrument")) or "JWST imaging",
                    "file": relative_path,
                    "metadata": metadata,
                    "native_is_field_cutout": True,
                })
                if primary_data is None:
                    primary_data = jwst_native
                    primary_header = jwst_native_header
                    primary_wcs = jwst_native_wcs
                    primary_product, primary_hdu = product_name, jwst_hdu
            except Exception as exc:  # noqa: BLE001 - preserve other filters when one product is unavailable
                jwst_errors.append(f"{band}: {exc}")
        if not jwst_bands or primary_data is None or primary_header is None or primary_wcs is None:
            detail = "; ".join(jwst_errors[:3]) or "no usable JWST science product"
            raise RuntimeError(f"no JWST filter could be downloaded for this location: {detail}")

        euclid_png = temporary_dir / "euclid_vis.png"
        jwst_png = temporary_dir / "jwst_native.png"
        euclid_display = _write_display_png(euclid_data, euclid_png, (0.40, 0.76, 1.0))
        jwst_display = _write_display_png(primary_data, jwst_png, (1.0, 0.69, 0.28))
        if progress:
            progress(len(product_rows) + 3, len(product_rows) + 3, "publishing complete location")

        manifest = {
            "version": 3,
            "field_id": identifier,
            "jwst_archive": archive,
            "jwst_observation_id": observation_id,
            "jwst_product": primary_product,
            "jwst_instrument": _text(primary_row.get("jwst_instrument")) or "JWST imaging",
            "jwst_filters": ", ".join(entry["filter"] for entry in jwst_bands),
            "jwst_bands": jwst_bands,
            "jwst_download_errors": jwst_errors,
            "euclid_tile_index": tile_index,
            "euclid_file_name": _text(tile.get("file_name") or row.get("euclid_file_name")),
            "euclid_product": _text(tile.get("file_name") or row.get("euclid_file_name")),
            "target_name": _text(row.get("jwst_target_name")),
            "ra_deg": ra,
            "dec_deg": dec,
            "size_arcsec": float(size_arcsec),
            "shape": list(euclid_data.shape),
            "jwst_native_is_field_cutout": True,
            "euclid_hdu": euclid_hdu,
            "jwst_hdu": primary_hdu,
            "euclid_metadata": _pixel_metadata(euclid_data, euclid_wcs, euclid_header),
            "jwst_metadata": _pixel_metadata(primary_data, primary_wcs, primary_header),
            "alignment": {
                "method": "native WCS cutout",
                "target_grid": "each instrument native pixel grid",
                "source_units": _text(primary_header.get("BUNIT")) or "archive header not specified",
                "target_units": _text(euclid_header.get("BUNIT")) or "archive header not specified",
            },
            "display": {"euclid": euclid_display, "jwst": jwst_display},
            "files": {
                "euclid": "euclid_vis.fits",
                "jwst_native": "jwst_native.fits",
                "euclid_png": "euclid_vis.png",
                "jwst_png": "jwst_native.png",
            },
            "source_row": _jsonable(dict(primary_row)),
            "source_products": _jsonable(product_rows),
        }
        _write_json(temporary_dir / "manifest.json", manifest)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary_dir, final_dir)
        return manifest
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def run_starfull_pair_inference(
    identifier: str, *, progress: Any | None = None,
) -> dict[str, Any]:
    """Run the fitted STARFULL combiner on matching four-band Euclid cutouts.

    JWST is deliberately absent from the model input: it remains a native-grid
    reference image in the viewer, while the production model expects Euclid
    VIS+Y+J+H on its own common LR grid.
    """
    pair_dir = pair_root() / _safe(identifier)
    manifest_file = pair_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("saved JWST × Euclid field not found") from exc
    if not isinstance(manifest, dict) or not _cached_pair_is_usable(pair_dir, manifest):
        raise RuntimeError("saved JWST × Euclid field is incomplete or invalid")
    ra = _number(manifest.get("ra_deg"))
    dec = _number(manifest.get("dec_deg"))
    size_arcsec = _number(manifest.get("size_arcsec"))
    if ra is None or dec is None or size_arcsec is None:
        raise RuntimeError("saved field has no usable sky coordinate or angular size")

    from astropy.io import fits

    from euclid_polish.catalog.downloader import fetch_cutout_at
    from euclid_polish.ensemble import EnsembleModel, default_ensemble_dir
    from euclid_polish.eval.combiner import ACTIVE_COMBINER_KINDS, COMBINER_MODELS, load_combiner
    from euclid_polish.photometry import adu_per_s_to_electrons_factor

    inference_dir = pair_dir / "starfull_inference"
    raw_dir = inference_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    side = max(1, int(round(size_arcsec / Config.VIS_PIXEL_SCALE_ARCSEC)))
    # Request extra source pixels for every band.  Y/J/H are re-sampled to
    # the VIS WCS below; without this guard band a sub-pixel WCS offset leaves
    # an otherwise avoidable NaN rim around the model input.
    registration_padding = 8
    source_side = side + registration_padding
    padded_size_arcsec = source_side * float(Config.VIS_PIXEL_SCALE_ARCSEC)
    from astropy.coordinates import SkyCoord

    coordinate = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")
    bands: list[np.ndarray] = []
    vis_header = None
    vis_data = None
    vis_wcs = None
    for index, band_name in enumerate(Config.LR_INPUT_BAND_NAMES):
        raw_path = raw_dir / f"{band_name}_padded.fits"
        if not _is_readable_fits(raw_path):
            if progress:
                progress(index + 1, 8, f"downloading Euclid {band_name} input")
            ok, error = fetch_cutout_at(
                ra=ra, dec=dec, band_name=band_name, output_file=str(raw_path),
                cutout_size_vis_pixels=source_side,
            )
            if not ok:
                raise RuntimeError(f"{band_name} input unavailable: {error}")
        else:
            if progress:
                progress(index + 1, 8, f"reusing Euclid {band_name} input")
        data, header, wcs, _ = _find_image(raw_path)
        if band_name == "VIS":
            registered, registered_wcs = _native_sky_cutout(
                data, wcs, coordinate, size_arcsec,
            )
            vis_data, vis_wcs = registered, registered_wcs
        else:
            if vis_data is None or vis_wcs is None:
                raise RuntimeError("VIS must be available before registering Euclid NISP bands")
            source_data, source_wcs = _native_sky_cutout(
                data, wcs, coordinate, padded_size_arcsec,
            )
            registered = align_to_target(
                source_data, source_wcs, vis_wcs, vis_data.shape,
            )
            registered_wcs = vis_wcs
        if registered.shape != (side, side):
            raise RuntimeError(
                f"{band_name} registered input has shape {registered.shape}; expected {(side, side)}"
            )
        if not _has_signal(registered):
            raise RuntimeError(f"{band_name} input has no usable pixels at the selected field")
        if not np.all(np.isfinite(registered)):
            raise RuntimeError(
                f"{band_name} cannot be registered fully onto the VIS footprint; "
                "archive coverage is incomplete"
            )
        band = Config.get_band(band_name)
        bands.append(registered * adu_per_s_to_electrons_factor(
            float(header.get("MAGZERO", band.sim_zeropoint_e)), band,
        ))
        if band_name == "VIS":
            vis_header = _primary_image_header(
                header, registered_wcs, raw_path.name, "Euclid STARFULL VIS reference input cutout",
            )
    if vis_header is None:
        raise RuntimeError("VIS header missing from STARFULL input")
    lr_cube = np.stack(bands, axis=-1).astype(np.float32)
    lr_path = inference_dir / "euclid_lr_vis_y_j_h.fits"
    vis_header["BANDS"] = (",".join(Config.LR_INPUT_BAND_NAMES), "input channel order")
    vis_header["REGWCS"] = ("VIS", "all Euclid input bands registered to VIS WCS")
    vis_header.add_history(
        "Y_E, J_E, and H_E sampled bilinearly onto the VIS native WCS before STARFULL inference",
    )
    fits.PrimaryHDU(np.moveaxis(lr_cube, -1, 0), header=vis_header).writeto(
        lr_path, overwrite=True, output_verify="silentfix",
    )

    if progress:
        progress(5, 8, "running active STARFULL members")
    ensemble = EnsembleModel(default_ensemble_dir(), starless=False)
    labels = list(ensemble.member_labels)
    if not labels:
        raise RuntimeError("no active STARFULL ensemble members")
    members = ensemble.member_arrays(lr_cube)
    if progress:
        progress(6, 8, "applying fitted STARFULL combiner")
    regime_dir = Path(Config.VIS_DIR) / "ensemble" / "starfull"
    selected_kind = None
    selected_combiner = None
    for kind in ACTIVE_COMBINER_KINDS:
        combiner = load_combiner(
            str(regime_dir), member_labels=labels,
            artifact_dir=COMBINER_MODELS[kind].artifact_dir,
        )
        if combiner is not None:
            selected_kind = kind
            selected_combiner = combiner
            break
    if selected_kind is None or selected_combiner is None:
        raise RuntimeError("no fitted STARFULL combiner is available")
    starfull = np.asarray(selected_combiner.apply_field(members), np.float32)
    sr_path = inference_dir / "starfull_combiner.fits"
    sr_header = vis_header.copy()
    scale = max(1, int(round(starfull.shape[0] / lr_cube.shape[0])))
    for key in ("CRPIX1", "CRPIX2"):
        if key in sr_header:
            sr_header[key] = (float(sr_header[key]) - 1.0) * scale + 1.0
    for key in ("CDELT1", "CDELT2"):
        if key in sr_header:
            sr_header[key] = float(sr_header[key]) / scale
    for key in ("CD1_1", "CD1_2", "CD2_1", "CD2_2"):
        if key in sr_header:
            sr_header[key] = float(sr_header[key]) / scale
    sr_header["SRCFILE"] = lr_path.name
    sr_header["SRMODE"] = "STARFULL"
    fits.PrimaryHDU(np.moveaxis(starfull, -1, 0), header=sr_header).writeto(
        sr_path, overwrite=True, output_verify="silentfix",
    )
    if progress:
        progress(7, 8, "publishing STARFULL inference")
    manifest["inference"] = {
        "mode": "starfull",
        "combiner_kind": selected_kind,
        "combiner_label": COMBINER_MODELS[selected_kind].label,
        "member_labels": labels,
        "pixel_scale_arcsec": float(Config.DEFAULT_PIXEL_SCALE),
        "input_registration": {
            "reference_band": "VIS",
            "method": "bilinear WCS sampling",
            "source_padding_vis_pixels": registration_padding,
        },
        "shape": [int(value) for value in starfull.shape],
        "files": {
            "lr": str(lr_path.relative_to(pair_dir)),
            "starfull": str(sr_path.relative_to(pair_dir)),
        },
    }
    _write_json(manifest_file, manifest)
    if progress:
        progress(8, 8, "STARFULL inference complete")
    return manifest


__all__ = [
    "align_to_target",
    "download_and_align_pair",
    "euclid_tile",
    "field_id",
    "find_overlap_row",
    "find_location_group",
    "jwst_filter_name",
    "location_groups",
    "location_id",
    "overlap_rows",
    "overlap_root",
    "pair_root",
    "run_starfull_pair_inference",
]
