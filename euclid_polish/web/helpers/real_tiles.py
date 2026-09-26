"""One store over every REAL Euclid product the console can run models on.

A *real tile* is a four-band Euclid LR image on a celestial grid:
``RealTile{source, id, ra, dec, lr_e (H, W, 4) electrons, wcs_header, extras}``.
Sources (``SOURCES``):

==========  =================================================================
``nexus``   NEXUS × Euclid 255² tiles (registered LR FITS; JWST F200W native
            cutout); id ``<filter>-<source index:04d>``
``tile``    user-cached 25.6″ four-band tiles (:func:`cache_tile`): 256² VIS
            grid, NISP registered onto the VIS WCS; id ``ra…_dec…``; stored
            under ``<EUCLID_INFERENCE_DIR>/real_tiles/<id>/``
``field``   legacy 100-tile real fields (every manifest, not only the latest);
            id ``<field id>-<tile:03d>``
``archive`` the 220 multipoint archive fields (ADU/s → e⁻ via MAGZERO); id
            ``<sample:03d>``
``eval``    real evaluation objects (lens candidates, galaxies); id = the
            object's ``out_subdir``
``poster``  the poster target FITS (``poster/*_results.fits``); TAN WCS built
            from ``RA``/``DEC``/``PIXSCALE`` (north up); id = file stem
``pair``    saved JWST × Euclid pairs (four-band once
            :func:`jwst_euclid.pair_lr_input` ran; VIS-only before); id = pair id
==========  =================================================================

Listing is cheap (manifests + headers, no pixels); :func:`get_tile` loads the
LR. Field labels always come from the position (:func:`q1_field_for`). Ids
never contain ``/`` or ``,`` (experiments address tiles as ``source/id``).

**Model outputs of a tile** (:func:`tile_outputs`, :func:`load_output`) are
the C9 output store (:mod:`model_catalog`) merged with the SRs the pre-C9
pipelines wrote next to their inputs — the NEXUS whole-field inference
(``tiles/starfull_combiner_NNNN.fits``), pair inference
(``starfull_inference/<slug>.fits``) and the poster's ``SR_*`` HDUs — read in
place (never copied) as *legacy* outputs keyed by the spec their recorded
identity names. Their WCS is always rebuilt from the tile's LR grid (LR WCS
×2), so an SR file written with an older header convention is still served
on the right grid.
"""

from __future__ import annotations

import contextlib
import csv
import json
import os
import re
import time
import warnings
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.nddata.utils import NoOverlapError, PartialOverlapError
from astropy.wcs import WCS

from euclid_polish.config import Config
from euclid_polish.photometry import adu_per_s_to_electrons_factor, header_magzero
from euclid_polish.sky.observation import q1_mer_tiles
from euclid_polish.sky.observation.q1_fields import q1_field_for
from euclid_polish.web.helpers import archive_fields, jwst_euclid, model_catalog, real_field

SOURCES = ("nexus", "tile", "field", "archive", "eval", "poster", "pair")
SOURCE_INFO: dict[str, dict[str, str]] = {
    "nexus": {"label": "NEXUS × Euclid tiles",
              "description": "255² Euclid tiles covering the NEXUS F200W mosaic, with JWST."},
    "tile": {"label": "Cached 25.6″ tiles",
             "description": "Four-band Euclid tiles cached anywhere in Q1 from the atlas."},
    "field": {"label": "Legacy real fields",
              "description": "256″ real fields cut into 100 tiles (legacy inference)."},
    "archive": {"label": "Archive fields",
                "description": "220 multipoint 25.6″ archive samples (Synthetic–Real)."},
    "eval": {"label": "Evaluation objects",
             "description": "Real lens candidates and galaxies of the catalogue evaluation."},
    "poster": {"label": "Poster target",
               "description": "The 102.4″ poster galaxy LR (and its saved SR runs)."},
    "pair": {"label": "JWST × Euclid pairs",
             "description": "Downloaded JWST × Euclid comparison pairs."},
}
BAND_NAMES = tuple(Config.LR_INPUT_BAND_NAMES)
TILE_SIDE = 256
TILE_PADDING = 8
TILE_SIZE_ARCSEC = TILE_SIDE * float(Config.VIS_PIXEL_SCALE_ARCSEC)
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
_POSTER_GLOB = "*_results.fits"


class RealTileError(Exception):
    """A real tile is unknown (404) or cannot serve what was asked (4xx)."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class TileEntry:
    """One real tile as listed (no pixels)."""

    source: str
    id: str
    label: str
    ra: float | None
    dec: float | None
    shape: tuple[int, int] | None = None
    pixscale: float = float(Config.VIS_PIXEL_SCALE_ARCSEC)
    bands: tuple[str, ...] = BAND_NAMES
    has_jwst: bool = False
    polygon: list[list[float]] | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def field(self) -> str | None:
        if self.ra is None or self.dec is None:
            return None
        return q1_field_for(self.ra, self.dec)

    @property
    def model_ready(self) -> bool:
        return tuple(self.bands) == BAND_NAMES

    @property
    def ref(self) -> str:
        return f"{self.source}/{self.id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source, "id": self.id, "ref": self.ref, "label": self.label,
            "ra": self.ra, "dec": self.dec, "field": self.field,
            "shape": list(self.shape) if self.shape else None,
            "pixscale": self.pixscale, "bands": list(self.bands),
            "model_ready": self.model_ready,
            "tiers": ["lr"] + (["jwst"] if self.has_jwst else []),
            "has_jwst": self.has_jwst, "polygon": self.polygon,
            # ``legacy_outputs`` is served as the ``models`` rows (card, list,
            # viewer); repeating it per listed tile would double the payload.
            "extras": _jsonable({key: value for key, value in self.extras.items()
                                 if key != "legacy_outputs"}),
        }


@dataclass
class RealTile:
    """A real tile with its LR pixels."""

    entry: TileEntry
    lr_e: np.ndarray                     # (H, W, C) electrons, C = len(entry.bands)
    wcs_header: fits.Header | None

    @property
    def source(self) -> str:
        return self.entry.source

    @property
    def id(self) -> str:
        return self.entry.id

    @property
    def ra(self) -> float | None:
        return self.entry.ra

    @property
    def dec(self) -> float | None:
        return self.entry.dec

    @property
    def extras(self) -> dict[str, Any]:
        return self.entry.extras


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def check_source(source: str) -> str:
    if source not in SOURCES:
        raise RealTileError(404, f"unknown real-tile source {source!r}")
    return source


def check_id(identifier: str) -> str:
    if not _ID.fullmatch(str(identifier or "")):
        raise RealTileError(404, f"bad real-tile id {identifier!r}")
    return str(identifier)


# ---------------------------------------------------------------------------
# headers / WCS (cached by path + mtime)
# ---------------------------------------------------------------------------

_HEADERS: OrderedDict[tuple[str, int, str], fits.Header | None] = OrderedDict()


def _header(path: Path | str, hdu: int | str = 0) -> fits.Header | None:
    try:
        stamp = os.stat(path).st_mtime_ns
    except OSError:
        return None
    key = (os.fspath(path), stamp, str(hdu))
    if key in _HEADERS:
        _HEADERS.move_to_end(key)
        return _HEADERS[key]
    try:
        header: fits.Header | None = fits.getheader(path, hdu)
    except (OSError, IndexError, KeyError, ValueError):
        header = None
    _HEADERS[key] = header
    while len(_HEADERS) > 2048:
        _HEADERS.popitem(last=False)
    return header


def _celestial(header: fits.Header | None) -> WCS | None:
    if header is None:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wcs = WCS(header).celestial
    except Exception:  # noqa: BLE001 - heterogeneous archive headers
        return None
    return wcs if wcs.has_celestial else None


def _centre(wcs: WCS | None, shape: tuple[int, int]) -> tuple[float | None, float | None]:
    if wcs is None:
        return None, None
    ra, dec = wcs.pixel_to_world_values((shape[1] - 1) / 2.0, (shape[0] - 1) / 2.0)
    return float(ra) % 360.0, float(dec)


def wcs_only_header(header: fits.Header | None) -> fits.Header | None:
    """A clean 2-D celestial header (CD form) of ``header``'s WCS."""
    wcs = _celestial(header)
    if wcs is None:
        return None
    matrix = np.asarray(wcs.pixel_scale_matrix, np.float64)
    out = fits.Header()
    out["CTYPE1"], out["CTYPE2"] = str(wcs.wcs.ctype[0]), str(wcs.wcs.ctype[1])
    out["CRVAL1"], out["CRVAL2"] = float(wcs.wcs.crval[0]), float(wcs.wcs.crval[1])
    out["CRPIX1"], out["CRPIX2"] = float(wcs.wcs.crpix[0]), float(wcs.wcs.crpix[1])
    out["CD1_1"], out["CD1_2"] = float(matrix[0, 0]), float(matrix[0, 1])
    out["CD2_1"], out["CD2_2"] = float(matrix[1, 0]), float(matrix[1, 1])
    out["RADESYS"] = "ICRS"
    return out


def shifted_header(header: fits.Header | None, dx: float, dy: float) -> fits.Header | None:
    """WCS of a crop whose pixel (0, 0) is source pixel ``(dx, dy)``."""
    out = wcs_only_header(header)
    if out is None:
        return None
    out["CRPIX1"] = float(out["CRPIX1"]) - float(dx)
    out["CRPIX2"] = float(out["CRPIX2"]) - float(dy)
    return out


def poster_wcs_header(primary: Mapping[str, Any], shape: tuple[int, int]) -> fits.Header:
    """North-up TAN WCS of a poster FITS: ``RA``/``DEC`` at the grid centre,
    ``PIXSCALE`` (″/px) — the poster files carry no WCS of their own."""
    scale = float(primary.get("PIXSCALE") or primary.get("LRPIX") or Config.VIS_PIXEL_SCALE_ARCSEC)
    header = fits.Header()
    header["CTYPE1"], header["CTYPE2"] = "RA---TAN", "DEC--TAN"
    header["CRVAL1"], header["CRVAL2"] = float(primary["RA"]), float(primary["DEC"])
    header["CRPIX1"] = (shape[1] + 1) / 2.0
    header["CRPIX2"] = (shape[0] + 1) / 2.0
    header["CD1_1"], header["CD1_2"] = -scale / 3600.0, 0.0
    header["CD2_1"], header["CD2_2"] = 0.0, scale / 3600.0
    header["RADESYS"] = "ICRS"
    header["WCSNOTE"] = "constructed from RA/DEC/PIXSCALE (north up)"
    return header


def _footprint(header: fits.Header | None, shape: tuple[int, int] | None) -> list[list[float]] | None:
    if header is None or shape is None:
        return None
    return jwst_euclid.header_footprint(header, shape)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _fits_cube(path: Path, hdu: int = 0) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=False) as hdul:
        image = hdul[hdu]
        data = np.asarray(image.data, np.float32)
        header = image.header.copy()
    if data.ndim == 3:
        data = np.moveaxis(data, 0, -1)
    elif data.ndim == 2:
        data = data[..., None]
    return np.ascontiguousarray(data), header


# ---------------------------------------------------------------------------
# legacy SR products of the pre-C9 pipelines
# ---------------------------------------------------------------------------

#: ``combiner_kind`` of a legacy record → the spec it is an output of.
_LEGACY_KIND_SPECS = {
    model_catalog.PRODUCTION_KIND: model_catalog.SPEC_PRODUCTION,
    model_catalog.RBF_KIND: model_catalog.SPEC_RBF,
    "mean_explicit_members": model_catalog.SPEC_MEAN,     # poster "new4" runs
}


def _spec_kind(spec: str) -> str:
    if spec.startswith(model_catalog.MEMBER_PREFIX):
        return "member"
    if spec.startswith(model_catalog.GATE_PREFIX):
        return "gate"
    return spec                                   # production | mean | rbf


def legacy_spec(record: Mapping[str, Any]) -> str | None:
    """The model spec a legacy SR record is an output of: its recorded
    ``spec`` (records since C9), else its ``combiner_kind``; ``None`` when
    neither names a C9 spec."""
    raw = record.get("spec")
    if raw:
        with contextlib.suppress(ValueError):
            return model_catalog.canonical_spec(str(raw))
    return _LEGACY_KIND_SPECS.get(str(record.get("combiner_kind") or ""))


def _mtime_iso(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
    except OSError:
        return None


def legacy_record(inference: Mapping[str, Any] | None, *, directory: Path, origin: str,
                  ) -> dict[str, Any] | None:
    """One pre-C9 SR (a NEXUS tile / pair ``inference`` record) as an
    output-store style sidecar, or ``None`` when its file is missing.

    ``fingerprint`` is the recorded ``spec_fingerprint`` or, for older
    records, rebuilt from the recorded identity (combiner artifact hash +
    member checkpoint fingerprints) with the catalogue formula
    (:func:`model_catalog.spec_fingerprint`), so its state is decided exactly
    like a store output's.
    """
    if not isinstance(inference, Mapping):
        return None
    files = inference.get("files") if isinstance(inference.get("files"), Mapping) else {}
    relative = files.get("starfull")
    if not isinstance(relative, str) or not relative or not (directory / relative).is_file():
        return None
    spec = legacy_spec(inference)
    fps = [None if value is None else str(value)
           for value in inference.get("member_fingerprints") or []]
    labels = [str(value) for value in inference.get("member_labels") or []]
    fingerprint = inference.get("spec_fingerprint") or None
    if fingerprint is None and spec is not None:
        fingerprint = model_catalog.spec_fingerprint(
            _spec_kind(spec), combiner_kind=inference.get("combiner_kind"),
            combiner_fingerprint=inference.get("combiner_fingerprint"),
            member_labels=labels, member_fingerprints=fps)
    return {
        "spec": spec, "slug": model_catalog.spec_slug(spec) if spec else None,
        "kind": _spec_kind(spec) if spec else None,
        "label": str(inference.get("combiner_label") or spec or "legacy SR"),
        "fingerprint": fingerprint,
        "member_labels": labels, "member_fingerprints": fps,
        "member_count": len(labels) or len(fps),
        "combiner_kind": inference.get("combiner_kind"),
        "combiner_fingerprint": inference.get("combiner_fingerprint"),
        "identity": {key: inference.get(key) for key in (
            "combiner_kind", "combiner_fingerprint", "member_fingerprints")},
        "shape": inference.get("shape"), "file": relative,
        "path": str(directory / relative), "created": _mtime_iso(directory / relative),
        "origin": origin, "legacy": True, "experiment_id": None, "lr_sha": None,
    }


def _legacy_outputs(*records: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    return {record["spec"]: record for record in records
            if record is not None and record.get("spec")}


# ---------------------------------------------------------------------------
# nexus
# ---------------------------------------------------------------------------

def _nexus_manifests() -> list[tuple[str, dict[str, Any]]]:
    root = jwst_euclid.nexus_field_root()
    if not root.is_dir():
        return []
    out = []
    for directory in sorted(root.iterdir(), key=lambda path: path.name):
        manifest = jwst_euclid._read_nexus_field_manifest(directory.name) if directory.is_dir() else None
        if manifest and isinstance(manifest.get("tiles"), list):
            out.append((directory.name, manifest))
    return out


def _nexus_entries() -> list[TileEntry]:
    entries = []
    for field_id, manifest in _nexus_manifests():
        directory = jwst_euclid.nexus_field_root() / field_id
        polygons = jwst_euclid.nexus_tile_polygons(field_id)
        for position, tile in enumerate(manifest["tiles"]):
            if not isinstance(tile, Mapping):
                continue
            lr_file = tile.get("lr_file")
            has_lr = isinstance(lr_file, str) and (directory / lr_file).is_file()
            jwst_file = tile.get("jwst_file")
            has_jwst = isinstance(jwst_file, str) and (directory / jwst_file).is_file()
            legacy = legacy_record(tile.get("inference"), directory=directory,
                                   origin="nexus-field")
            identifier = jwst_euclid.nexus_tile_id(manifest, tile, position)
            index = int(tile.get("source_index", tile.get("index", position)))
            entries.append(TileEntry(
                source="nexus", id=identifier,
                label=f"NEXUS {manifest.get('filter') or ''} tile {index:04d}".strip(),
                ra=jwst_euclid._number(tile.get("ra_deg")),
                dec=jwst_euclid._number(tile.get("dec_deg")),
                shape=(jwst_euclid._NEXUS_EUCLID_TILE_SIDE,) * 2,
                bands=BAND_NAMES if has_lr else ("VIS",),
                has_jwst=has_jwst, polygon=polygons.get(position),
                extras={
                    "field_id": field_id, "position": position, "source_index": index,
                    "filter": manifest.get("filter"),
                    "lr_file": lr_file if has_lr else None,
                    "vis_file": jwst_euclid._nexus_euclid_files(tile).get("VIS"),
                    "jwst_file": jwst_file if has_jwst else None,
                    "euclid_tile_index": tile.get("euclid_tile_index"),
                    "legacy_sr": legacy,
                    "legacy_outputs": _legacy_outputs(legacy),
                },
            ))
    return entries


def _nexus_load(entry: TileEntry) -> tuple[np.ndarray, fits.Header | None]:
    directory = jwst_euclid.nexus_field_root() / str(entry.extras["field_id"])
    source = entry.extras.get("lr_file") or entry.extras.get("vis_file")
    if not source:
        raise RealTileError(404, f"{entry.ref} has no cached Euclid LR")
    cube, header = _fits_cube(directory / str(source))
    if cube.shape[-1] == 1:                      # VIS-only (ADU/s) cache
        cube = cube * adu_per_s_to_electrons_factor(
            header_magzero(header, source="NEXUS VIS"), Config.get_band("VIS"))
    return cube, header


# ---------------------------------------------------------------------------
# tile (user-cached 25.6" tiles)
# ---------------------------------------------------------------------------

def tiles_root() -> Path:
    return Path(Config.EUCLID_INFERENCE_DIR) / "real_tiles"


def real_tile_id(ra: float, dec: float) -> str:
    return real_field.field_id(float(ra), float(dec))


def _tile_entries() -> list[TileEntry]:
    root = tiles_root()
    if not root.is_dir():
        return []
    entries = []
    for directory in sorted(root.iterdir(), key=lambda path: path.name):
        manifest = _read_json(directory / "manifest.json") if directory.is_dir() else None
        if not manifest or not (directory / "lr_e.npy").is_file():
            continue
        entries.append(TileEntry(
            source="tile", id=directory.name,
            label=f"Tile {float(manifest['ra']):.5f}, {float(manifest['dec']):+.5f}",
            ra=float(manifest["ra"]), dec=float(manifest["dec"]),
            shape=tuple(manifest.get("shape") or (TILE_SIDE, TILE_SIDE))[:2],
            polygon=manifest.get("polygon"),
            extras={"euclid_tile_index": manifest.get("euclid_tile_index"),
                    "created": manifest.get("created"), "size_arcsec": manifest.get("size_arcsec")},
        ))
    return entries


def _tile_load(entry: TileEntry) -> tuple[np.ndarray, fits.Header | None]:
    directory = tiles_root() / entry.id
    cube = np.load(directory / "lr_e.npy").astype(np.float32)
    return cube, wcs_only_header(_header(directory / "lr.fits"))


# ---------------------------------------------------------------------------
# field (legacy real fields)
# ---------------------------------------------------------------------------

def _field_entries() -> list[TileEntry]:
    entries = []
    for manifest in real_field.list_fields():
        field_id = str(manifest.get("field_id") or "")
        if not _ID.fullmatch(field_id):
            continue
        directory = real_field.field_dir(field_id)
        header = _header(directory / "original_stack.fits")
        tile = int(manifest.get("tile_size", real_field.TILE_SIZE) or real_field.TILE_SIZE)
        side = int(manifest.get("grid_side", real_field.GRID_SIDE) or real_field.GRID_SIDE)
        count = int(manifest.get("count", side * side) or 0)
        for index in range(count):
            row, col = divmod(index, side)
            tile_header = shifted_header(header, col * tile, row * tile)
            wcs = _celestial(tile_header)
            ra, dec = _centre(wcs, (tile, tile))
            if not (directory / "cubes" / f"lr_{index:03d}.npy").is_file():
                continue
            entries.append(TileEntry(
                source="field", id=f"{field_id}-{index:03d}",
                label=f"Field {field_id} · tile {index + 1:03d} (row {row + 1}, col {col + 1})",
                ra=ra, dec=dec, shape=(tile, tile),
                polygon=_footprint(tile_header, (tile, tile)),
                extras={"field_id": field_id, "tile": index,
                        "legacy_members": len(manifest.get("member_labels") or []),
                        "legacy_combiners": list(manifest.get("combiner_kinds") or [])},
            ))
    return entries


def _field_load(entry: TileEntry) -> tuple[np.ndarray, fits.Header | None]:
    field_id, index = str(entry.extras["field_id"]), int(entry.extras["tile"])
    directory = real_field.field_dir(field_id)
    cube = np.load(directory / "cubes" / f"lr_{index:03d}.npy").astype(np.float32)
    manifest = real_field._read_manifest(field_id) or {}
    tile = int(manifest.get("tile_size", real_field.TILE_SIZE) or real_field.TILE_SIZE)
    side = int(manifest.get("grid_side", real_field.GRID_SIDE) or real_field.GRID_SIDE)
    row, col = divmod(index, side)
    return cube, shifted_header(_header(directory / "original_stack.fits"), col * tile, row * tile)


# ---------------------------------------------------------------------------
# archive
# ---------------------------------------------------------------------------

def _archive_fields() -> list[archive_fields.ArchiveField]:
    if not archive_fields.manifest_path().is_file():
        return []
    try:
        return list(archive_fields.iter_fields())
    except (archive_fields.ArchiveFieldError, OSError, ValueError, KeyError, TypeError):
        return []


def _archive_entries() -> list[TileEntry]:
    entries = []
    for sample in _archive_fields():
        header = _header(sample.path, 1)
        entries.append(TileEntry(
            source="archive", id=f"{sample.sample_id:03d}",
            label=(f"Archive {archive_fields.position_field(sample)} · pointing "
                   f"{sample.source_sample_id + 1} · {sample.position_name}"),
            ra=float(sample.ra), dec=float(sample.dec),
            shape=(archive_fields.TILE_SIZE, archive_fields.TILE_SIZE),
            polygon=_footprint(header, (archive_fields.TILE_SIZE, archive_fields.TILE_SIZE)),
            extras={"sample_id": sample.sample_id, "parent_id": sample.parent_id,
                    "position_name": sample.position_name, "stored_field": sample.field,
                    "path": str(sample.path)},
        ))
    return entries


def _archive_load(entry: TileEntry) -> tuple[np.ndarray, fits.Header | None]:
    sample_id = int(entry.extras["sample_id"])
    sample = next((item for item in _archive_fields() if item.sample_id == sample_id), None)
    if sample is None:
        raise RealTileError(404, f"archive sample {sample_id} is unavailable")
    try:
        cube = archive_fields.load_field(sample)      # ADU/s → e⁻ (MAGZERO) per band
    except archive_fields.ArchiveFieldError as exc:
        raise RealTileError(415, str(exc)) from exc
    return cube, wcs_only_header(_header(sample.path, 1))


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

def _eval_rows() -> list[dict[str, str]]:
    path = Path(Config.EVAL_RESULTS_DIR) / "manifest.csv"
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _eval_entries() -> list[TileEntry]:
    root = Path(Config.EVAL_RESULTS_DIR)
    entries = []
    for row in _eval_rows():
        if str(row.get("ok", "")).lower() != "true":
            continue
        sub = str(row.get("out_subdir") or row.get("id") or "")
        ra, dec = _finite(row.get("ra")), _finite(row.get("dec"))
        if not _ID.fullmatch(sub) or ra is None or dec is None:
            continue                          # synthetic objects have no sky position
        path = root / sub / "original_stack.fits"
        if not path.is_file():
            continue
        header = _header(path)
        shape = None
        if header is not None and header.get("NAXIS", 0) >= 2:
            shape = (int(header["NAXIS2"]), int(header["NAXIS1"]))
        grade = str(row.get("grade") or "").strip()
        entries.append(TileEntry(
            source="eval", id=sub,
            label=f"{row.get('id') or sub}" + (f" · {grade}" if grade else ""),
            ra=ra, dec=dec, shape=shape, polygon=_footprint(header, shape),
            extras={"grade": grade,
                    "kind": "lens" if grade in {"A", "B", "C"} else "galaxy",
                    "flux_ratio_sr_over_lr": _finite(row.get("flux_ratio_sr_over_lr")),
                    # The evaluation SR records no model identity: listed, not a tier.
                    "legacy_sr": ({"file": "SR.fits", "origin": "evaluation", "spec": None,
                                   "identity": None, "fingerprint": None, "legacy": True}
                                  if (root / sub / "SR.fits").is_file() else None)},
        ))
    return entries


def _eval_load(entry: TileEntry) -> tuple[np.ndarray, fits.Header | None]:
    path = Path(Config.EVAL_RESULTS_DIR) / entry.id / "original_stack.fits"
    cube, header = _fits_cube(path)
    return cube, wcs_only_header(header)


# ---------------------------------------------------------------------------
# poster
# ---------------------------------------------------------------------------

def poster_root() -> Path:
    return Path(__file__).resolve().parents[3] / "poster"


def _poster_entries() -> list[TileEntry]:
    root = poster_root()
    if not root.is_dir():
        return []
    entries = []
    for path in sorted(root.glob(_POSTER_GLOB)):
        primary = _header(path, 0)
        lr = _header(path, 1)
        if primary is None or lr is None or "RA" not in primary or "DEC" not in primary:
            continue
        shape = (int(lr.get("NAXIS2", 0)), int(lr.get("NAXIS1", 0)))
        header = poster_wcs_header(primary, shape)
        identifier = re.sub(r"[^A-Za-z0-9._-]+", "-", path.stem.removesuffix("_results"))
        legacy = _poster_record(path, primary)
        entries.append(TileEntry(
            source="poster", id=identifier,
            label=f"Poster {identifier}", ra=float(primary["RA"]), dec=float(primary["DEC"]),
            shape=shape, pixscale=float(primary.get("PIXSCALE") or Config.VIS_PIXEL_SCALE_ARCSEC),
            polygon=_footprint(header, shape),
            extras={"file": path.name, "wcs_constructed": True,
                    "legacy_sr": legacy, "legacy_outputs": _legacy_outputs(legacy)},
        ))
    return entries


def _poster_load(entry: TileEntry) -> tuple[np.ndarray, fits.Header | None]:
    path = poster_root() / str(entry.extras["file"])
    with fits.open(path, memmap=False) as hdul:
        primary = hdul[0].header.copy()
        planes = [np.asarray(hdul[f"LR_{band}"].data, np.float32) for band in BAND_NAMES]
    cube = np.stack(planes, axis=-1)
    return cube, poster_wcs_header(primary, cube.shape[:2])


def _poster_record(path: Path, primary: Mapping[str, Any]) -> dict[str, Any] | None:
    """The SR saved inside a poster FITS (``SR_*`` HDUs) as a legacy record.

    The poster run recorded only ``COMB_KIND`` and ``N_MEMBER`` — no member
    checkpoint or artifact identity — so its fingerprint is unknown: it can
    never be *current*, only an older output of the spec its kind names.
    """
    if _header(path, f"SR_{BAND_NAMES[0]}") is None:
        return None
    kind = str(primary.get("COMB_KIND") or "")
    count = primary.get("N_MEMBER")
    spec = legacy_spec({"combiner_kind": kind})
    return {
        "spec": spec, "slug": model_catalog.spec_slug(spec) if spec else None,
        "kind": _spec_kind(spec) if spec else None,
        "label": f"Poster SR · {kind or 'unknown combiner'}"
                 + (f" ({count} members)" if count else ""),
        "fingerprint": None, "member_labels": [], "member_fingerprints": [],
        "member_count": count, "combiner_kind": kind or None, "combiner_fingerprint": None,
        "identity": None, "shape": None, "file": path.name, "hdu": "SR_*",
        "path": str(path), "created": _mtime_iso(path), "origin": "poster",
        "legacy": True, "experiment_id": None, "lr_sha": None,
    }


def poster_legacy_sr(entry: TileEntry) -> tuple[np.ndarray, fits.Header]:
    """The SR saved inside a poster FITS (``SR_*`` HDUs) with its WCS."""
    path = poster_root() / str(entry.extras["file"])
    with fits.open(path, memmap=False) as hdul:
        primary = hdul[0].header.copy()
        planes = [np.asarray(hdul[f"SR_{band}"].data, np.float32) for band in BAND_NAMES]
    cube = np.stack(planes, axis=-1)
    lr_shape = entry.shape or (cube.shape[0] // 2, cube.shape[1] // 2)
    return cube, model_catalog.sr_header(poster_wcs_header(primary, lr_shape))


# ---------------------------------------------------------------------------
# pair
# ---------------------------------------------------------------------------

def _pair_entries() -> list[TileEntry]:
    entries = []
    for manifest in jwst_euclid.saved_pairs():
        identifier = str(manifest.get("field_id") or "")
        if not _ID.fullmatch(identifier):
            continue
        directory = jwst_euclid.pair_root() / identifier
        lr_input = (manifest.get("lr_input") or {}).get("file") or \
            "starfull_inference/euclid_lr_vis_y_j_h.fits"
        has_lr = (directory / lr_input).is_file()
        vis = str((manifest.get("files") or {}).get("euclid") or "euclid_vis.fits")
        header = _header(directory / (lr_input if has_lr else vis))
        shape = None
        if header is not None and header.get("NAXIS", 0) >= 2:
            shape = (int(header["NAXIS2"]), int(header["NAXIS1"]))
        production = legacy_record(manifest.get("inference"), directory=directory, origin="pair")
        extra_runs = manifest.get("model_inference")
        others = [legacy_record(record, directory=directory, origin="pair")
                  for record in (extra_runs.values() if isinstance(extra_runs, Mapping) else [])]
        entries.append(TileEntry(
            source="pair", id=identifier,
            label=str(manifest.get("target_name") or identifier),
            ra=jwst_euclid._number(manifest.get("ra_deg")),
            dec=jwst_euclid._number(manifest.get("dec_deg")),
            shape=shape, bands=BAND_NAMES if has_lr else ("VIS",), has_jwst=True,
            polygon=_footprint(header, shape),
            extras={"lr_file": lr_input if has_lr else None, "vis_file": vis,
                    "jwst_bands": [dict(band) for band in manifest.get("jwst_bands") or []]
                    or [{"filter": manifest.get("jwst_filters") or "JWST",
                         "file": (manifest.get("files") or {}).get("jwst_native")}],
                    "size_arcsec": manifest.get("size_arcsec"),
                    "legacy_sr": production,
                    "legacy_outputs": _legacy_outputs(production, *others)},
        ))
    return entries


def _pair_load(entry: TileEntry) -> tuple[np.ndarray, fits.Header | None]:
    directory = jwst_euclid.pair_root() / entry.id
    source = entry.extras.get("lr_file") or entry.extras.get("vis_file")
    cube, header = _fits_cube(directory / str(source))
    if cube.shape[-1] == 1:
        cube = cube * adu_per_s_to_electrons_factor(
            header_magzero(header, source="pair VIS"), Config.get_band("VIS"))
    return cube, wcs_only_header(header)


# ---------------------------------------------------------------------------
# the store
# ---------------------------------------------------------------------------

_LISTERS: dict[str, Callable[[], list[TileEntry]]] = {
    "nexus": _nexus_entries, "tile": _tile_entries, "field": _field_entries,
    "archive": _archive_entries, "eval": _eval_entries, "poster": _poster_entries,
    "pair": _pair_entries,
}
_LOADERS: dict[str, Callable[[TileEntry], tuple[np.ndarray, fits.Header | None]]] = {
    "nexus": _nexus_load, "tile": _tile_load, "field": _field_load,
    "archive": _archive_load, "eval": _eval_load, "poster": _poster_load,
    "pair": _pair_load,
}


#: ``source`` → (monotonic time, entries); listings are re-read after
#: ``LIST_TTL_S`` or an explicit :func:`invalidate` (writers call it).
_LIST_CACHE: dict[str, tuple[float, list[TileEntry]]] = {}
LIST_TTL_S = 10.0


def invalidate(source: str | None = None) -> None:
    """Forget cached listings (all sources, or one)."""
    if source is None:
        _LIST_CACHE.clear()
    else:
        _LIST_CACHE.pop(source, None)


def list_entries(source: str) -> list[TileEntry]:
    """Every tile of one source (cheap: no pixels; memoised ``LIST_TTL_S``)."""
    check_source(source)
    cached = _LIST_CACHE.get(source)
    now = time.monotonic()
    if cached is not None and now - cached[0] < LIST_TTL_S:
        return list(cached[1])
    entries = _LISTERS[source]()
    _LIST_CACHE[source] = (now, entries)
    return list(entries)


def get_entry(source: str, identifier: str) -> TileEntry:
    check_id(identifier)
    for entry in list_entries(source):
        if entry.id == identifier:
            return entry
    raise RealTileError(404, f"unknown real tile {source}/{identifier}")


def get_tile(source: str, identifier: str, *, entry: TileEntry | None = None) -> RealTile:
    """Load one tile's LR (electrons, ``(H, W, C)``) and its WCS header."""
    entry = entry or get_entry(source, identifier)
    cube, header = _LOADERS[entry.source](entry)
    return RealTile(entry=entry, lr_e=np.ascontiguousarray(cube, np.float32),
                    wcs_header=wcs_only_header(header) if header is not None else None)


def parse_refs(raw: str | list[str]) -> list[tuple[str, str]]:
    """``source/id`` references (comma list) → validated pairs, order kept."""
    items = raw.split(",") if isinstance(raw, str) else list(raw)
    out: list[tuple[str, str]] = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        source, _, identifier = text.partition("/")
        pair = (check_source(source), check_id(identifier))
        if pair not in out:
            out.append(pair)
    return out


def jwst_planes(entry: TileEntry) -> list[dict[str, Any]]:
    """Native JWST images of a tile: ``[{band, data (H, W), header, unit}]``."""
    if entry.source == "nexus" and entry.extras.get("jwst_file"):
        directory = jwst_euclid.nexus_field_root() / str(entry.extras["field_id"])
        data, header = _fits_cube(directory / str(entry.extras["jwst_file"]))
        return [{"band": str(entry.extras.get("filter") or "JWST"), "data": data[..., 0],
                 "header": header, "unit": "MJy/sr"}]
    if entry.source == "pair":
        directory = jwst_euclid.pair_root() / entry.id
        planes = []
        for band in entry.extras.get("jwst_bands") or []:
            relative = band.get("file")
            if not relative or not (directory / str(relative)).is_file():
                continue
            data, header = _fits_cube(directory / str(relative))
            planes.append({"band": str(band.get("filter") or "JWST"), "data": data[..., 0],
                           "header": header, "unit": "MJy/sr"})
        return planes
    return []


# ---------------------------------------------------------------------------
# model outputs of a tile: the C9 store merged with legacy SR products
# ---------------------------------------------------------------------------

def _pick(store: Mapping[str, Any] | None, legacy: Mapping[str, Any] | None,
          current: str | None) -> Mapping[str, Any] | None:
    """The store output wins unless only the legacy SR is current."""
    if store is None or legacy is None:
        return store if store is not None else legacy
    if current and store.get("fingerprint") != current and legacy.get("fingerprint") == current:
        return legacy
    return store


def tile_outputs(entry: TileEntry, current: Mapping[str, str | None] | None = None
                 ) -> dict[str, dict[str, Any]]:
    """``{spec: sidecar}`` of every model output of a tile: the output store
    plus the tile's legacy SRs (``legacy: true``, read in place). Where both
    hold a spec the store wins, unless ``current`` (``{spec: fingerprint}``)
    shows that only the legacy SR is current."""
    store = model_catalog.list_outputs(entry.source, entry.id)
    legacy = entry.extras.get("legacy_outputs") or {}
    out: dict[str, dict[str, Any]] = {}
    for spec in list(store) + [spec for spec in legacy if spec not in store]:
        chosen = _pick(store.get(spec), legacy.get(spec), (current or {}).get(spec))
        if chosen is not None:
            out[spec] = {"legacy": False, **dict(chosen)}
    return out


def lr_header(entry: TileEntry) -> fits.Header | None:
    """The celestial WCS of a tile's LR grid, without loading its pixels
    where the source allows (NEXUS / pair headers, poster construction)."""
    if entry.source in ("nexus", "pair"):
        directory = (jwst_euclid.nexus_field_root() / str(entry.extras["field_id"])
                     if entry.source == "nexus" else jwst_euclid.pair_root() / entry.id)
        relative = entry.extras.get("lr_file") or entry.extras.get("vis_file")
        return wcs_only_header(_header(directory / str(relative))) if relative else None
    if entry.source == "poster":
        primary = _header(poster_root() / str(entry.extras["file"]), 0)
        if primary is None or entry.shape is None:
            return None
        return poster_wcs_header(primary, entry.shape)
    return get_tile(entry.source, entry.id, entry=entry).wcs_header


def _legacy_cube(entry: TileEntry, record: Mapping[str, Any]) -> np.ndarray:
    if record.get("origin") == "poster":
        return poster_legacy_sr(entry)[0]
    cube, _header_unused = _fits_cube(Path(str(record["path"])))
    return cube


def load_output(entry: TileEntry, spec: str, *, current: str | None = None
                ) -> tuple[np.ndarray, fits.Header, dict[str, Any]]:
    """``(SR cube (2H, 2W, C) electrons, SR header, sidecar)`` of one spec on
    a tile — the store output or the legacy SR (:func:`tile_outputs` rules,
    ``current`` = the spec's fingerprint now). A legacy SR's header is the
    LR WCS ×2 (:func:`model_catalog.sr_header`), never the file's own.
    :class:`FileNotFoundError` when the spec has no output on the tile."""
    spec = model_catalog.canonical_spec(spec)
    store = model_catalog.list_outputs(entry.source, entry.id).get(spec)
    legacy = (entry.extras.get("legacy_outputs") or {}).get(spec)
    chosen = _pick(store, legacy, current)
    if chosen is None:
        raise FileNotFoundError(f"no {spec} output for {entry.ref}")
    if not chosen.get("legacy"):
        cube, header, meta = model_catalog.load_output(entry.source, entry.id, spec)
        return cube, header, {"legacy": False, **meta}
    cube = _legacy_cube(entry, chosen)
    header = model_catalog.sr_header(lr_header(entry))
    header["BUNIT"] = ("electron", "SR electrons per SR pixel")
    header["SPEC"] = (spec[:68], "model spec")
    header["LEGACYSR"] = (str(chosen.get("file") or "")[:68], "pre-C9 SR file (read in place)")
    return cube, header, dict(chosen)


def sources_payload() -> dict[str, Any]:
    """``GET /api/real/sources``."""
    out = []
    for source in SOURCES:
        try:
            entries = list_entries(source)
            reason = None
        except Exception as exc:  # noqa: BLE001 - one broken source must not hide the others
            entries, reason = [], f"{type(exc).__name__}: {exc}"
        out.append({
            "id": source, **SOURCE_INFO[source], "count": len(entries),
            "model_ready": sum(entry.model_ready for entry in entries),
            "has_jwst": any(entry.has_jwst for entry in entries),
            "ready": bool(entries), "reason": reason,
        })
    return {"sources": out}


# ---------------------------------------------------------------------------
# caching a new 25.6" tile anywhere in Q1
# ---------------------------------------------------------------------------

def _vis_grid(path: Path, ra: float, dec: float, side: int) -> tuple[np.ndarray, fits.Header, WCS]:
    data, header, wcs, _name = jwst_euclid._find_image(path)
    try:
        cutout = Cutout2D(data, position=SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs"),
                          size=(side, side), wcs=wcs, mode="strict")
    except (NoOverlapError, PartialOverlapError, ValueError) as exc:
        raise RuntimeError(f"VIS cutout cannot hold the {side}² grid: {exc}") from exc
    source = header.copy()
    for key in [key for key in source if re.fullmatch(r"(CD|PC)\d_\d|CDELT\d", key)]:
        del source[key]
    tile_header = jwst_euclid._primary_image_header(
        source, cutout.wcs, path.name, f"Euclid VIS archive cutout cropped to the {side}x{side} grid")
    return np.ascontiguousarray(cutout.data, np.float32), tile_header, cutout.wcs


def cache_tile(ra: float, dec: float, *,
               progress: Callable[[int, int, str], None] | None = None) -> dict[str, Any]:
    """Cache a 25.6″ four-band real tile at ``(ra, dec)``.

    Q1 coverage is checked first against the committed MER polygons; each
    band comes from the tile whose polygon contains the point
    (:func:`jwst_euclid.fetch_q1_cutout`, guard-banded by 8 px), VIS is cropped
    to the exact 256² grid centred on the position and Y/J/H are registered
    onto its WCS (the NEXUS tile machinery), all in electrons (MAGZERO).
    Writes ``lr_e.npy`` (256, 256, 4) float32, ``lr.fits`` (4, 256, 256) with
    the VIS celestial WCS, ``raw/<band>.fits`` and ``manifest.json``. Raw
    bands already on disk are reused.
    """
    ra, dec = float(ra) % 360.0, float(dec)
    q1_tile = jwst_euclid.choose_q1_tile(ra, dec, TILE_SIZE_ARCSEC)
    if q1_tile is None:
        raise ValueError(f"({ra:.5f}, {dec:+.5f}) is outside the Euclid Q1 footprint")
    identifier = real_tile_id(ra, dec)
    directory = tiles_root() / identifier
    raw_dir = directory / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    total = len(BAND_NAMES) + 2
    tick = progress or (lambda *_a: None)
    raw_files: dict[str, str] = {}
    for index, band_name in enumerate(BAND_NAMES):
        path = raw_dir / f"{band_name}.fits"
        raw_files[band_name] = str(path.relative_to(directory))
        if jwst_euclid._is_readable_fits(path):
            tick(index, total, f"reusing Euclid {band_name}")
            continue
        tick(index, total, f"downloading Euclid {band_name} (tile {q1_tile.tile})")
        ok, error = jwst_euclid.fetch_q1_cutout(
            ra=ra, dec=dec, band_name=band_name, output_file=str(path),
            cutout_size_vis_pixels=TILE_SIDE + TILE_PADDING)
        if not ok or not jwst_euclid._is_readable_fits(path):
            with contextlib.suppress(OSError):
                path.unlink()
            raise RuntimeError(f"{band_name} cutout unavailable: {error or 'unreadable file'}")
    tick(len(BAND_NAMES), total, "registering NISP onto the VIS grid")
    vis_data, vis_header, vis_wcs = _vis_grid(raw_dir / "VIS.fits", ra, dec, TILE_SIDE)
    planes: list[np.ndarray] = []
    for band_name in BAND_NAMES:
        path = raw_dir / f"{band_name}.fits"
        data, header, wcs, _name = jwst_euclid._find_image(path)
        registered = vis_data if band_name == "VIS" else jwst_euclid.align_to_target(
            data, wcs, vis_wcs, vis_data.shape)
        if not jwst_euclid._has_signal(registered):
            raise RuntimeError(f"{band_name} has no usable pixels at this position")
        if not np.all(np.isfinite(registered)):
            raise RuntimeError(f"{band_name} does not fully cover the {TILE_SIZE_ARCSEC:.1f}″ tile")
        planes.append(registered * adu_per_s_to_electrons_factor(
            header_magzero(header, source=f"{band_name} tile"), Config.get_band(band_name)))
    cube = np.stack(planes, axis=-1).astype(np.float32)
    header = vis_header.copy()
    header["BANDS"] = (",".join(BAND_NAMES), "input channel order")
    header["REGWCS"] = ("VIS", "all Euclid input bands registered to VIS WCS")
    header["BUNIT"] = ("electron", "stack electrons per LR pixel")
    header["Q1TILE"] = (q1_tile.tile, "Q1 MER tile containing the position")
    temporary = directory / f".lr.{os.getpid()}.tmp.fits"
    fits.PrimaryHDU(np.moveaxis(cube, -1, 0), header=header).writeto(
        temporary, overwrite=True, output_verify="silentfix")
    os.replace(temporary, directory / "lr.fits")
    temporary_npy = directory / f".lr_e.{os.getpid()}.tmp.npy"
    np.save(temporary_npy, cube)
    os.replace(temporary_npy, directory / "lr_e.npy")
    manifest = {
        "id": identifier, "source": "tile", "ra": ra, "dec": dec,
        "field": q1_field_for(ra, dec), "euclid_tile_index": q1_tile.tile,
        "size_arcsec": TILE_SIZE_ARCSEC, "shape": [int(v) for v in cube.shape],
        "bands": list(BAND_NAMES), "unit": "e-",
        "polygon": jwst_euclid.grid_footprint(vis_wcs, vis_data.shape),
        "created": datetime.now(UTC).isoformat(),
        "files": {"lr_npy": "lr_e.npy", "lr_fits": "lr.fits", "raw": raw_files},
        "registration": {"reference_band": "VIS", "method": "bilinear WCS sampling",
                         "source_padding_vis_pixels": TILE_PADDING},
    }
    jwst_euclid._write_json(directory / "manifest.json", manifest)
    invalidate("tile")
    tick(total, total, f"cached real tile {identifier}")
    return manifest


def disk_usage(entry: TileEntry) -> dict[str, int]:
    """Bytes a tile occupies: its own cache (``tile`` source), its C9 model
    outputs, its member-SR cache and its legacy SR files (NEXUS / pair
    inference; the poster SR lives inside the poster FITS and is not
    counted); ``total_bytes`` sums them."""
    def size(path: Path) -> int:
        if not path.exists():
            return 0
        if path.is_file():
            return path.stat().st_size
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())

    own = size(tiles_root() / entry.id) if entry.source == "tile" else 0
    outputs = size(model_catalog.output_dir(entry.source, entry.id))
    cache = size(model_catalog.member_cache_dir(entry.source, entry.id))
    legacy = sum(size(Path(str(record["path"])))
                 for record in (entry.extras.get("legacy_outputs") or {}).values()
                 if record.get("origin") != "poster")
    return {"tile_bytes": own, "output_bytes": outputs, "cache_bytes": cache,
            "legacy_bytes": legacy, "total_bytes": own + outputs + cache + legacy}


def entries_containing(ra: float, dec: float, *, sources: tuple[str, ...] = SOURCES
                       ) -> list[TileEntry]:
    """Real tiles whose footprint polygon contains a sky position."""
    out = []
    for source in sources:
        with contextlib.suppress(Exception):
            for entry in list_entries(source):
                if entry.polygon and q1_mer_tiles.point_in_polygon(ra, dec, entry.polygon):
                    out.append(entry)
    return out


__all__ = [
    "BAND_NAMES",
    "RealTile",
    "RealTileError",
    "SOURCES",
    "SOURCE_INFO",
    "TileEntry",
    "cache_tile",
    "check_id",
    "check_source",
    "disk_usage",
    "entries_containing",
    "get_entry",
    "get_tile",
    "invalidate",
    "jwst_planes",
    "legacy_record",
    "legacy_spec",
    "list_entries",
    "load_output",
    "lr_header",
    "parse_refs",
    "poster_legacy_sr",
    "poster_root",
    "poster_wcs_header",
    "real_tile_id",
    "sources_payload",
    "tile_outputs",
    "tiles_root",
    "wcs_only_header",
]
