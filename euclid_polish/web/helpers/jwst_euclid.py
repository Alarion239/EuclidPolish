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
from pathlib import Path
from typing import Any

import numpy as np

from euclid_polish.config import Config

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")
_IMAGE_SUFFIXES = (".fits", ".fits.gz", ".fit", ".fit.gz")
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


def overlap_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load cached discovery rows and a small source-status summary."""
    root = overlap_root()
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
        row["available"] = (pair_root() / identifier / "manifest.json").exists()
        output.append(row)
    output.sort(key=lambda row: (
        not row.get("available", False),
        row.get("footprint_status", "") != "exact_intersection",
        row.get("jwst_target_name", ""),
        row["euclid_tile_index"],
        row["jwst_observation_id"],
    ))
    status = {
        "source_files": sources,
        "partial": bool(manifest.get("partial", False)),
        "source_manifest": manifest,
        "count": len(output),
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
    """Download and register one Euclid/JWST field as an atomic cache entry."""
    archive = _text(row.get("jwst_archive") or "esa").lower()
    tile_index = _text(row.get("euclid_tile_index"))
    observation_id = _text(row.get("jwst_observation_id") or row.get("jwst_obsid"))
    ra = _number(row.get("euclid_ra_deg"))
    dec = _number(row.get("euclid_dec_deg"))
    if not tile_index or not observation_id or ra is None or dec is None:
        raise ValueError("pair row needs Euclid tile, JWST observation id, and Euclid coordinates")
    if not 1.0 <= float(size_arcsec) <= 120.0:
        raise ValueError("size_arcsec must be between 1 and 120 arcsec")

    identifier = field_id(archive, tile_index, observation_id, float(size_arcsec))
    final_dir = pair_root() / identifier
    manifest_path = final_dir / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    pair_root().mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{identifier}.", dir=pair_root()))
    try:
        temporary_dir.mkdir(parents=True, exist_ok=True)
        if progress:
            progress(1, 5, "resolving Euclid VIS tile")
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
            progress(2, 5, "downloading Euclid VIS cutout")
        coordinate = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")
        radius = (float(size_arcsec) / 2.0) * u.arcsec
        try:
            _download_euclid_cutout(
                Euclid,
                file_path=file_path,
                tile_index=tile_index,
                coordinate=coordinate,
                radius=radius,
                destination=euclid_path,
            )
        except RuntimeError as first_error:
            # The overlap cache can outlive an archive data release. Resolve
            # the tile once more after a failed cutout, but do not repeat the
            # same stale request when the archive returned the same path.
            try:
                fresh_tile = euclid_tile(tile_index, row, refresh=True)
            except Exception as refresh_error:  # noqa: BLE001 - retain the useful cutout error
                raise RuntimeError(
                    f"{first_error}; refreshing Euclid tile metadata also failed: "
                    f"{type(refresh_error).__name__}: {refresh_error}"
                ) from first_error
            fresh_path = euclid_product_path(fresh_tile)
            if not fresh_path or fresh_path == file_path:
                raise RuntimeError(
                    f"{first_error}; cached Euclid product path was still current"
                ) from first_error
            if progress:
                progress(2, 5, "retrying Euclid VIS cutout with refreshed metadata")
            _download_euclid_cutout(
                Euclid,
                file_path=fresh_path,
                tile_index=tile_index,
                coordinate=coordinate,
                radius=radius,
                destination=euclid_path,
            )
            tile = fresh_tile

        jwst_path = temporary_dir / "jwst_native.fits"
        if progress:
            progress(3, 5, f"downloading JWST {archive.upper()} image")
        product_name = _download_jwst(archive, observation_id, jwst_path)

        euclid_data, euclid_header, euclid_wcs, euclid_hdu = _find_image(euclid_path)
        jwst_data, jwst_header, jwst_wcs, jwst_hdu = _find_image(jwst_path)
        if progress:
            progress(4, 5, "registering JWST image on Euclid WCS")
        aligned = align_to_target(jwst_data, jwst_wcs, euclid_wcs, euclid_data.shape)
        aligned_path = temporary_dir / "jwst_aligned_to_euclid.fits"
        aligned_header = _aligned_primary_header(euclid_header, product_name)
        fits.PrimaryHDU(data=aligned, header=aligned_header).writeto(
            aligned_path, overwrite=True, output_verify="silentfix",
        )

        euclid_png = temporary_dir / "euclid_vis.png"
        jwst_png = temporary_dir / "jwst_aligned.png"
        euclid_display = _write_display_png(euclid_data, euclid_png, (0.40, 0.76, 1.0))
        jwst_display = _write_display_png(aligned, jwst_png, (1.0, 0.69, 0.28))
        if progress:
            progress(5, 5, "publishing complete paired field")

        manifest = {
            "version": 1,
            "field_id": identifier,
            "jwst_archive": archive,
            "jwst_observation_id": observation_id,
            "jwst_product": product_name,
            "jwst_instrument": _text(row.get("jwst_instrument")) or "JWST imaging",
            "jwst_filters": _text(row.get("jwst_filters")),
            "euclid_tile_index": tile_index,
            "euclid_file_name": _text(tile.get("file_name") or row.get("euclid_file_name")),
            "target_name": _text(row.get("jwst_target_name")),
            "ra_deg": ra,
            "dec_deg": dec,
            "size_arcsec": float(size_arcsec),
            "shape": list(euclid_data.shape),
            "euclid_hdu": euclid_hdu,
            "jwst_hdu": jwst_hdu,
            "alignment": {
                "method": "bilinear WCS remap",
                "target_grid": "Euclid VIS cutout",
                "source_units": _text(jwst_header.get("BUNIT")) or "archive header not specified",
                "target_units": _text(euclid_header.get("BUNIT")) or "archive header not specified",
            },
            "display": {"euclid": euclid_display, "jwst": jwst_display},
            "files": {
                "euclid": "euclid_vis.fits",
                "jwst_native": "jwst_native.fits",
                "jwst_aligned": "jwst_aligned_to_euclid.fits",
                "euclid_png": "euclid_vis.png",
                "jwst_png": "jwst_aligned.png",
            },
            "source_row": _jsonable(dict(row)),
        }
        _write_json(temporary_dir / "manifest.json", manifest)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary_dir, final_dir)
        return manifest
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


__all__ = [
    "align_to_target",
    "download_and_align_pair",
    "euclid_tile",
    "field_id",
    "find_overlap_row",
    "overlap_rows",
    "overlap_root",
    "pair_root",
]
