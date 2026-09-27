"""Persist and render publication-ready selections from the cutout viewer.

Saved viewer results are deliberately science-first bundles.  Every selected
tier is reloaded through :mod:`viewer_data`, cropped in raw detector/image
units, and written as FITS before the bundle directory is atomically renamed
into place.  Display settings are provenance only: they never alter FITS
pixels.

Crops are matched on the sky: when the tiers carry a celestial WCS (C6
``info["wcs"]``), the selection centre is resolved on its source tier
(``selection.source_tier``, else the first tier with a WCS) and mapped
through every other tier's WCS, exactly as the viewer's lens does, and each
saved FITS keeps its crop's WCS (CRPIX shifted by the crop offset).

The Figures workspace (W-Figures) also manages the bundles: a user label
(:func:`rename_result`), deletion (:func:`delete_result`), thumbnails
(:func:`panel_request` + :func:`render_panel_png`), FITS downloads and named
grid layouts (:func:`list_layouts`, :func:`save_layout`,
:func:`delete_layout`) stored beside the bundles in ``grid_layouts.json``.
"""
from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import shutil
import tempfile
import threading
import warnings
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image as PILImage

from euclid_polish.config import Config
from euclid_polish.web.helpers import paths, viewer_data

SCHEMA_VERSION = 1

# The publication renderer intentionally has one absolute transfer.  Unlike a
# percentile stretch, the same number of input electrons always maps to the
# same output intensity across tiers and saved results.
ASINH_KNEE_E = 100.0
ASINH_WHITE_E = 30.0 * ASINH_KNEE_E

MAX_TIERS = 4
MAX_CROP_SIDE_PIXELS = 4096
MAX_CROP_PIXELS = MAX_CROP_SIDE_PIXELS**2
MAX_ANGULAR_SIDE_ARCSEC = 120.0
MAX_GRID_RESULTS = 12
MAX_GRID_ROWS = 16
MIN_GRID_DPI = 72
MAX_GRID_DPI = 600
DEFAULT_GRID_DPI = 300

A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
GRID_GAP_MM = 4.0
MIN_ROW_TITLE_TRACK_MM = 12.0
MIN_COLUMN_TITLE_TRACK_MM = 10.0
MAX_GRID_OUTPUT_PIXELS = 36_000_000
MAX_GRID_SOURCE_BYTES = 512 * 1024 * 1024
MAX_GRID_RENDER_BYTES = 384 * 1024 * 1024
_GRID_CANVAS_BYTES_PER_PIXEL = 8
_SHA_CACHE_MAX = 64

LOGICAL_TIER_ORDER = ("dirty", "sr", "hr", "jwst")
DISPLAY_MODES = ("VIS", "H_E", "VIS_H", "native")

_RESULT_ID = re.compile(r"^vr-[0-9a-f]{24}$")
_LAYOUT_ID = re.compile(r"^gl-[0-9a-f]{12}$")
_COLLECTION = re.compile(r"^[A-Za-z0-9-]{1,64}$")
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9._+-]{1,220}$")
_SAFE_SIMPLE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MEMBERS = re.compile(r"^[0-9]+(?:,[0-9]+)*$")
_BAND_NAME = re.compile(r"^[A-Za-z0-9_+-]{1,40}$")
# ``real`` collection model specs (C9): production, mean, rbf,
# member:member_<N>, gate:<variant>; a comma list of at most 64.
_MODEL_SPEC = re.compile(r"^[A-Za-z0-9._+-]{1,64}(?::[A-Za-z0-9._+-]{1,64})?$")
_TIER_KEY = re.compile(r"^[A-Za-z0-9._+:-]{1,120}$")
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")

#: A user label (rename) and a grid-layout name.
MAX_LABEL_CHARS = 120
MAX_LAYOUT_NAME_CHARS = 80
MAX_LAYOUTS = 200
LAYOUTS_FILE = "grid_layouts.json"
#: Thumbnail sides accepted by ``panel.png?size=`` (never upsampled).
MIN_THUMB_SIDE = 8
MAX_THUMB_SIDE = 2048
#: The thumbnail recipe of a result: the first of these it supports.
THUMBNAIL_PREFERENCE = (
    "sr:VIS_H", "dirty:VIS_H", "hr:VIS_H", "sr:VIS", "dirty:VIS", "hr:VIS", "jwst:native",
)
#: The WCS keywords a tier must carry for sky-matched crops (C6 CD form).
_WCS_KEYS = ("CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
             "CD1_1", "CD1_2", "CD2_1", "CD2_2")

_LAYOUT_LOCK = threading.Lock()

_PARAMS_BY_COLLECTION: dict[str, frozenset[str]] = {
    "sky": frozenset({"subset", viewer_data.BHR_FWHM_PARAM}),
    "cutouts": frozenset(),
    "evaluation": frozenset({viewer_data.BHR_FWHM_PARAM}),
    "ensemble": frozenset({"mode", "members", viewer_data.BHR_FWHM_PARAM}),
    "archive-fields": frozenset(),
    "real-field": frozenset({"field"}),
    "jwst-euclid": frozenset({"jwst_band"}),
    "nexus-field": frozenset({"field"}),
    "psfs": frozenset({"psf_warp", "psf_warp_seed"}),
    # C9 real tiles: ``m:<spec>`` tiers are the SR panel (one per result).
    "real": frozenset({"source", "models", "jwst_band"}),
}

_ALIASES = {
    "dirty": "dirty",
    "lr": "dirty",
    "real": "dirty",
    "original": "dirty",
    "original_stack": "dirty",
    "sr": "sr",
    "hr": "hr",
    "jwst": "jwst",
}

_REAL_COLLECTIONS = frozenset({
    "archive-fields", "cutouts", "real-field", "jwst-euclid", "nexus-field", "real",
})


def _logical_alias(tier_key: str) -> str | None:
    """The stable logical tier of a viewer tier key (lower-cased), or None.
    ``m:<spec>`` (the ``real`` collection's model outputs) is the SR panel."""
    if tier_key in _ALIASES:
        return _ALIASES[tier_key]
    if tier_key.startswith("m:") and len(tier_key) > 2:
        return "sr"
    return None

_SHA_CACHE: OrderedDict[tuple[str, int, int, int, int, int], str] = OrderedDict()


@dataclass(frozen=True)
class GridGeometry:
    """One A4 image-table layout expressed entirely in physical millimetres."""

    rows: int
    columns: int
    dpi: int
    gap_mm: float
    outer_padding_mm: float
    row_title_track_mm: float
    column_title_track_mm: float
    cell_side_mm: float
    grid_left_mm: float
    grid_bottom_mm: float
    page_width_pixels: int
    page_height_pixels: int
    cell_side_pixels: int

    def panel_bounds_mm(self, row: int, column: int) -> tuple[float, float, float, float]:
        """Return ``left, bottom, width, height`` for a top-indexed cell."""
        left = self.grid_left_mm + column * (self.cell_side_mm + self.gap_mm)
        bottom = self.grid_bottom_mm + (self.rows - 1 - row) * (
            self.cell_side_mm + self.gap_mm
        )
        return left, bottom, self.cell_side_mm, self.cell_side_mm


class ViewerResultError(Exception):
    """A client-visible saved-result validation or rendering error."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = int(code)


def results_root() -> Path:
    """Resolve the output root at call time so tests and live jobs can retarget it.

    ``EUCLID_POLISH_RESULTS_DIR`` is an explicit final-root override.  Without
    it, the current data-root environment (which may have changed since Config
    was imported) wins, followed by ``Config.DATA_DIR``.
    """
    explicit = os.environ.get("EUCLID_POLISH_RESULTS_DIR")
    if explicit:
        return Path(explicit).expanduser()
    data_root = os.environ.get("EUCLID_POLISH_DATA_DIR") or os.fspath(Config.DATA_DIR)
    return Path(data_root).expanduser() / "viewer_results"


def _json_object(value: Any, name: str) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ViewerResultError(400, f"{name} must be valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ViewerResultError(400, f"{name} must be an object")
    return {str(key): item for key, item in value.items()}


def _tier_list(value: Any) -> list[str]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ViewerResultError(400, "tiers must be valid JSON") from exc
        else:
            value = [part.strip() for part in stripped.split(",") if part.strip()]
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray, str)):
        raise ViewerResultError(400, "tiers must be a list")
    tiers = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ViewerResultError(400, "every tier must be a non-empty string")
        tiers.append(item.strip())
    if not tiers or len(tiers) > MAX_TIERS:
        raise ViewerResultError(400, f"select between 1 and {MAX_TIERS} tiers")
    return tiers


def request_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Decode the common JSON/form wire representation into one mapping."""
    payload = {str(key): value for key, value in raw.items()}
    for name in ("params", "selection", "display"):
        payload[name] = _json_object(payload.get(name, {}), name)
    payload["tiers"] = _tier_list(payload.get("tiers"))
    return payload


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ViewerResultError(400, f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ViewerResultError(400, f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ViewerResultError(400, f"{name} must be a finite number")
    return number


def _safe_params(collection: str, raw: Any) -> dict[str, str]:
    params = _json_object(raw, "params")
    allowed = _PARAMS_BY_COLLECTION.get(collection)
    if allowed is None:
        raise ViewerResultError(404, "unknown viewer collection")
    unknown = set(params) - allowed
    if unknown:
        raise ViewerResultError(400, f"unsupported viewer parameter: {sorted(unknown)[0]}")

    clean: dict[str, str] = {}
    for key, raw_value in params.items():
        if not isinstance(raw_value, (str, int, float)) or isinstance(raw_value, bool):
            raise ViewerResultError(400, f"viewer parameter {key} must be scalar")
        value = str(raw_value).strip()
        if not value or "/" in value or "\\" in value or "\x00" in value or ".." in value:
            raise ViewerResultError(400, f"viewer parameter {key} is invalid")
        if key == "members":
            if not _MEMBERS.fullmatch(value):
                raise ViewerResultError(400, "members must be a comma-separated index list")
            members = [int(part) for part in value.split(",")]
            if len(members) != len(set(members)):
                raise ViewerResultError(400, "members contains duplicate indices")
        elif key == viewer_data.BHR_FWHM_PARAM:
            fwhm = _finite_number(value, f"viewer parameter {key}")
            if not 0.0 <= fwhm <= viewer_data.BHR_FWHM_MAX_ARCSEC:
                raise ViewerResultError(400, f"viewer parameter {key} is outside its range")
        elif key == "psf_warp":
            if value not in {"0", "1"}:
                raise ViewerResultError(400, "psf_warp must be 0 or 1")
        elif key == "psf_warp_seed":
            try:
                seed = int(value)
            except ValueError as exc:
                raise ViewerResultError(400, "psf_warp_seed must be a uint32") from exc
            if not 0 <= seed <= 2**32 - 1:
                raise ViewerResultError(400, "psf_warp_seed must be a uint32")
        elif key in {"subset", "mode", "source"}:
            if not _SAFE_SIMPLE.fullmatch(value):
                raise ViewerResultError(400, f"viewer parameter {key} is invalid")
        elif key == "models":
            specs = [part.strip() for part in value.split(",")]
            if (not specs or len(specs) > 64
                    or any(not _MODEL_SPEC.fullmatch(spec) for spec in specs)):
                raise ViewerResultError(400, "models must be a comma list of model specs")
            value = ",".join(specs)
        elif not _SAFE_TOKEN.fullmatch(value):
            raise ViewerResultError(400, f"viewer parameter {key} is invalid")
        clean[key] = value
    return clean


def _selection(raw: Any) -> dict[str, Any]:
    value = _json_object(raw, "selection")
    u = _finite_number(value.get("u"), "selection.u")
    v = _finite_number(value.get("v"), "selection.v")
    if not 0.0 <= u <= 1.0 or not 0.0 <= v <= 1.0:
        raise ViewerResultError(400, "selection center must lie in the unit square")

    angular_raw = value.get("angular_side_arcsec", value.get("angularSide"))
    relative_raw = value.get("relative_side", value.get("relativeSide"))
    fallback_safe = value.get(
        "relative_fallback_safe", value.get("relativeFallbackSafe", False),
    ) is True
    # The tier the selection was made on (the viewer's ``sourceTier``): its
    # (u, v) is resolved there and mapped to the other tiers through the WCS.
    source_raw = value.get("source_tier", value.get("sourceTier"))
    source_tier: dict[str, str] = {}
    if source_raw is not None:
        if not isinstance(source_raw, str) or not _TIER_KEY.fullmatch(source_raw.strip()):
            raise ViewerResultError(400, "selection.source_tier is invalid")
        source_tier = {"source_tier": source_raw.strip()}

    if angular_raw is not None:
        angular = _finite_number(angular_raw, "selection.angular_side_arcsec")
        if not 0.0 < angular <= MAX_ANGULAR_SIDE_ARCSEC:
            raise ViewerResultError(
                400,
                f"selection angular side must be in (0, {MAX_ANGULAR_SIDE_ARCSEC:g}] arcsec",
            )
        return {
            "u": u,
            "v": v,
            "mode": "angular",
            "angular_side_arcsec": angular,
            "relative_side": None,
            "relative_fallback_safe": fallback_safe,
            **source_tier,
        }

    if not fallback_safe:
        raise ViewerResultError(
            400,
            "selection needs an angular side; relative fallback must be explicitly marked safe",
        )
    relative = _finite_number(relative_raw, "selection.relative_side")
    if not 0.0 < relative <= 1.0:
        raise ViewerResultError(400, "selection relative side must be in (0, 1]")
    return {
        "u": u,
        "v": v,
        "mode": "relative",
        "angular_side_arcsec": None,
        "relative_side": relative,
        "relative_fallback_safe": True,
        **source_tier,
    }


def _display(raw: Any) -> dict[str, Any]:
    """Keep a small, finite display-provenance block without applying it."""
    value = _json_object(raw, "display")
    unknown = set(value) - {"color", "mode", "layout", "knee", "gain", "transfers"}
    if unknown:
        raise ViewerResultError(400, f"unsupported display field: {sorted(unknown)[0]}")
    out: dict[str, Any] = {"applied_to_fits": False}
    for key in ("color", "mode", "layout"):
        if key not in value:
            continue
        item = value[key]
        if not isinstance(item, str) or not _SAFE_TOKEN.fullmatch(item):
            raise ViewerResultError(400, f"display.{key} is invalid")
        out[key] = item
    for key, maximum in (("knee", 1.0e12), ("gain", 1.0e6)):
        if key not in value:
            continue
        item = _finite_number(value[key], f"display.{key}")
        if not 0.0 < item <= maximum:
            raise ViewerResultError(400, f"display.{key} is outside the supported range")
        out[key] = item

    if "transfers" in value:
        transfers = _json_object(value["transfers"], "display.transfers")
        if len(transfers) > 8:
            raise ViewerResultError(400, "display.transfers has too many groups")
        clean_transfers: dict[str, dict[str, float]] = {}
        for group, settings_raw in transfers.items():
            if not _SAFE_SIMPLE.fullmatch(group):
                raise ViewerResultError(400, "display transfer group is invalid")
            settings = _json_object(settings_raw, f"display.transfers.{group}")
            if set(settings) - {"knee", "gain"}:
                raise ViewerResultError(400, "display transfer settings are invalid")
            clean: dict[str, float] = {}
            for key, maximum in (("knee", 1.0e12), ("gain", 1.0e6)):
                if key in settings:
                    item = _finite_number(settings[key], f"display.transfers.{group}.{key}")
                    if not 0.0 < item <= maximum:
                        raise ViewerResultError(400, "display transfer is outside the supported range")
                    clean[key] = item
            clean_transfers[group] = clean
        out["transfers"] = clean_transfers
    return out


def _source_tiers(meta: Mapping[str, Any], requested: Sequence[str]) -> list[tuple[str, str]]:
    advertised = [tier for tier in meta.get("tiers", []) if isinstance(tier, Mapping)]
    key_map = {str(tier.get("key")): tier for tier in advertised if tier.get("key") is not None}
    lower_map: dict[str, str] = {}
    for key in key_map:
        if key.lower() in lower_map and lower_map[key.lower()] != key:
            continue
        lower_map[key.lower()] = key

    object_tiers: set[str] | None = None
    objects = meta.get("objects")
    index = int(meta.get("_selected_index", -1))
    if isinstance(objects, list) and 0 <= index < len(objects):
        obj = objects[index]
        if isinstance(obj, Mapping) and isinstance(obj.get("tiers"), list):
            object_tiers = {str(item).lower() for item in obj["tiers"]}

    resolved: list[tuple[str, str]] = []
    source_seen: set[str] = set()
    logical_seen: set[str] = set()
    for raw in requested:
        source = raw if raw in key_map else lower_map.get(raw.lower())
        if source is None:
            raise ViewerResultError(400, f"tier {raw} is not advertised by this collection")
        lowered = source.lower()
        if lowered == "morph":
            raise ViewerResultError(400, "the client-animated morph tier cannot be saved")
        if lowered in source_seen:
            raise ViewerResultError(400, "duplicate tiers are not allowed")
        source_seen.add(lowered)
        logical = _logical_alias(lowered)
        if logical is None:
            raise ViewerResultError(400, f"tier {source} has no publication-result alias")
        if logical in logical_seen:
            raise ViewerResultError(400, f"multiple tiers resolve to logical tier {logical}")
        logical_seen.add(logical)
        if bool(key_map[source].get("disabled")):
            raise ViewerResultError(400, f"tier {source} is disabled")
        if object_tiers is not None and lowered not in object_tiers:
            raise ViewerResultError(404, f"tier {source} is unavailable for this object")
        resolved.append((source, logical))
    return resolved


def _bands(info: Mapping[str, Any], meta: Mapping[str, Any], channels: int) -> list[str]:
    raw = info.get("bands") or list(meta.get("band_names", []))[:channels]
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != channels:
        raise ViewerResultError(415, "cube band metadata does not match its channel count")
    bands = [str(item) for item in raw]
    if any(not _BAND_NAME.fullmatch(item) for item in bands) or len(set(bands)) != len(bands):
        raise ViewerResultError(415, "cube band metadata is invalid")
    return bands


def _crop_bounds(
    shape: tuple[int, int, int],
    pixscale: float,
    selection: Mapping[str, Any],
    center: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Pixel bounds of the square crop. ``center`` is the continuous crop
    centre on this tier (pixel ``i`` spans ``[i, i+1)``); default the
    selection's normalised ``(u, v)`` on this tier's grid."""
    height, width, _channels = shape
    if height <= 0 or width <= 0:
        raise ViewerResultError(415, "cube has an empty spatial axis")
    if selection["mode"] == "angular":
        side_float = float(selection["angular_side_arcsec"]) / pixscale
        side = int(round(side_float))
    else:
        side = int(round(float(selection["relative_side"]) * min(height, width)))
    if side < 1:
        raise ViewerResultError(400, "selection is smaller than one source pixel")
    if side > min(height, width):
        raise ViewerResultError(400, "selection is larger than one or more source cubes")
    if side > MAX_CROP_SIDE_PIXELS or side * side > MAX_CROP_PIXELS:
        raise ViewerResultError(413, "selection crop is too large")

    if center is None:
        center_x = float(selection["u"]) * width
        center_y = float(selection["v"]) * height
    else:
        center_x, center_y = center
    if not (math.isfinite(center_x) and math.isfinite(center_y)):
        raise ViewerResultError(422, "the selection centre is not on every tier's sky")
    # Snap float noise from the WCS round trip so an exact .5 never flips.
    x0 = int(round(round(center_x - side / 2.0, 6)))
    y0 = int(round(round(center_y - side / 2.0, 6)))
    if x0 < 0 or y0 < 0 or x0 + side > width or y0 + side > height:
        raise ViewerResultError(
            422,
            "selection center cannot support the requested matched crop in every tier",
        )
    return {
        "x0": x0,
        "x1": x0 + side,
        "y0": y0,
        "y1": y0 + side,
        "side_pixels": side,
        "actual_angular_side_arcsec": side * pixscale,
    }


def _safe_object(meta: Mapping[str, Any], index: int) -> dict[str, Any] | None:
    objects = meta.get("objects")
    if not isinstance(objects, list) or not 0 <= index < len(objects):
        return None
    obj = objects[index]
    if not isinstance(obj, Mapping):
        return None
    out: dict[str, Any] = {}
    for key in ("label", "grade", "subdir"):
        value = obj.get(key)
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
    # The stable object id (C6) and, for real objects, the sky position and
    # the C9 tile reference — so a saved result can reopen its source.
    for key in ("id", "ref", "field"):
        value = obj.get(key)
        if isinstance(value, str) and value and len(value) <= 256:
            out[key] = value
    for key in ("ra", "dec"):
        value = obj.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            out[key] = float(value)
    if isinstance(obj.get("tiers"), list):
        out["tiers"] = [str(item) for item in obj["tiers"]]
    return out


def _clean_wcs(raw: Any) -> dict[str, Any] | None:
    """A tier's C6 WCS keywords, or None when absent/malformed."""
    if not isinstance(raw, Mapping):
        return None
    out: dict[str, Any] = {}
    for key in _WCS_KEYS:
        value = raw.get(key)
        if key.startswith("CTYPE"):
            if not isinstance(value, str) or not value.strip():
                return None
            out[key] = value
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                return None
            out[key] = float(value)
    if out["CD1_1"] * out["CD2_2"] - out["CD1_2"] * out["CD2_1"] == 0.0:
        return None
    return out


def _wcs_object(keywords: Mapping[str, Any]) -> WCS | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wcs = WCS(fits.Header(dict(keywords))).celestial
    except Exception:  # noqa: BLE001 - malformed keywords
        return None
    return wcs if wcs.naxis == 2 and wcs.has_celestial else None


def _write_fits(
    path: Path,
    cube: np.ndarray,
    *,
    logical: str,
    source_tier: str,
    bands: Sequence[str],
    pixscale: float,
    wcs: Mapping[str, Any] | None = None,
) -> None:
    """``wcs`` = the crop's own celestial WCS keywords (FITS 1-based, axis 1
    = column) or None; axis 3 of the written cube is the band."""
    data = np.moveaxis(np.ascontiguousarray(cube, dtype=np.float32), -1, 0)
    hdu = fits.PrimaryHDU(data=data)
    hdu.header["LOGTIER"] = (logical[:68], "Stable viewer-result tier alias")
    hdu.header["SRCTIER"] = (source_tier[:68], "Source viewer tier key")
    hdu.header["PIXSCALE"] = (float(pixscale), "arcsec / pixel")
    if wcs:
        for key in _WCS_KEYS:
            hdu.header[key] = wcs[key]
        hdu.header["WCSKEEP"] = (True, "Celestial WCS of this crop (CRPIX shifted)")
    else:
        hdu.header["WCSKEEP"] = (False, "Source WCS was not available through viewer API")
    hdu.header["DSPAPPL"] = (False, "Display transfer has not been applied")
    for index, band in enumerate(bands):
        hdu.header[f"BAND{index}"] = (str(band), "Channel name")
    hdu.writeto(path, overwrite=False, checksum=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _load_cube_for_save(
    collection: str,
    index: int,
    source_tier: str,
    logical: str,
    params: dict[str, str],
    meta: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        cube, info = viewer_data.get_cube(collection, index, source_tier, params)
    except viewer_data.ViewerError as exc:
        raise ViewerResultError(exc.code, str(exc)) from exc
    array = np.asarray(cube, dtype=np.float32)
    if array.ndim != 3 or array.shape[-1] < 1:
        raise ViewerResultError(415, f"tier {source_tier} did not return an HWC cube")
    pixscale = _finite_number(info.get("pixscale"), f"tier {source_tier} pixel scale")
    if pixscale <= 0.0:
        raise ViewerResultError(415, f"tier {source_tier} has no positive pixel scale")
    display_scale_raw = info.get("display_scale")
    display_scale = None
    if display_scale_raw is not None:
        candidate = _finite_number(display_scale_raw, f"tier {source_tier} display scale")
        if candidate <= 0.0:
            raise ViewerResultError(415, f"tier {source_tier} has an invalid display scale")
        display_scale = candidate
    return {
        "source_tier": source_tier,
        "logical": logical,
        "cube": np.ascontiguousarray(array),
        "bands": _bands(info, meta, array.shape[-1]),
        "pixscale": pixscale,
        "label": str(info.get("label") or source_tier),
        "asinh": float(info.get("asinh", ASINH_KNEE_E)),
        "display_scale": display_scale,
        "direct_rgb": bool(info.get("direct_rgb")),
        "transfer_group": str(info.get("transfer_group") or "default"),
        "wcs": _clean_wcs(info.get("wcs")),
    }


def _matched_bounds(
    loaded: Sequence[dict[str, Any]], selection: Mapping[str, Any],
) -> dict[str, float] | None:
    """Set ``item["bounds"]`` for every tier, matched on the sky.

    The selection centre is the (u, v) of its source tier (``source_tier``,
    else the first tier with a WCS); it reaches every other tier with a WCS
    through both WCSs (continuous pixel ``x`` ↔ 0-based centre ``x − 0.5``,
    as the viewer's ``mapPoint``). Tiers without a WCS — or every tier when
    the source tier has none — use the normalised (u, v). Returns the sky
    centre ``{ra, dec}`` (deg) or None."""
    for item in loaded:
        item["wcs_object"] = _wcs_object(item["wcs"]) if item.get("wcs") else None
    wanted = str(selection.get("source_tier") or "").lower()
    if wanted:
        anchor = next((item for item in loaded
                       if item["source_tier"].lower() == wanted), None)
        if anchor is not None and anchor["wcs_object"] is None:
            anchor = None
    else:
        anchor = next((item for item in loaded if item["wcs_object"] is not None), None)

    sky: tuple[float, float] | None = None
    if anchor is not None:
        height, width = anchor["cube"].shape[:2]
        fx, fy = float(selection["u"]) * width, float(selection["v"]) * height
        ra, dec = anchor["wcs_object"].pixel_to_world_values(fx - 0.5, fy - 0.5)
        if math.isfinite(float(ra)) and math.isfinite(float(dec)):
            sky = (float(ra), float(dec))

    for item in loaded:
        center = None
        if sky is not None and item is not anchor and item["wcs_object"] is not None:
            px, py = item["wcs_object"].world_to_pixel_values(sky[0], sky[1])
            center = (float(px) + 0.5, float(py) + 0.5)
        item["bounds"] = _crop_bounds(item["cube"].shape, item["pixscale"], selection, center)
    return {"ra": sky[0] % 360.0, "dec": sky[1]} if sky is not None else None


def _require_native_f200w_request(
    collection: str,
    params: Mapping[str, str],
    resolved: Sequence[tuple[str, str]],
) -> None:
    """Reject the paired-field viewer's derived default before loading it.

    ``jwst-euclid`` defaults ``jwst_band`` to ``colour`` in the live viewer.
    That is a derived, registered RGB display product, not the native NEXUS
    F200W reference promised by the publication recipe.  Silently replacing
    it here would make the saved provenance disagree with what the user saw.
    """
    if not any(logical == "jwst" for _source, logical in resolved):
        return
    if collection == "jwst-euclid" and str(params.get("jwst_band") or "").upper() != "F200W":
        raise ViewerResultError(
            422,
            "JWST publication results require explicit native F200W; "
            "select F200W instead of the derived/default colour or temperature view",
        )


def _require_loaded_native_f200w(loaded: Sequence[Mapping[str, Any]]) -> None:
    """Verify the source loader actually returned one non-derived F200W plane."""
    for item in loaded:
        if item.get("logical") != "jwst":
            continue
        if item.get("bands") != ["F200W"] or bool(item.get("direct_rgb")):
            raise ViewerResultError(
                422,
                "JWST publication results require an exact native F200W plane; "
                "derived colour, temperature, and other filters cannot be labelled NEXUS F200W",
            )


def save_result(raw_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate, stage, and atomically publish one saved viewer selection."""
    payload = request_payload(raw_payload)
    collection = payload.get("collection")
    if not isinstance(collection, str) or not _COLLECTION.fullmatch(collection):
        raise ViewerResultError(400, "collection is invalid")
    index_raw = payload.get("index")
    if isinstance(index_raw, bool):
        raise ViewerResultError(400, "index must be an integer")
    try:
        index = int(index_raw)
    except (TypeError, ValueError) as exc:
        raise ViewerResultError(400, "index must be an integer") from exc
    if index < 0:
        raise ViewerResultError(400, "index must be non-negative")

    params = _safe_params(collection, payload["params"])
    selection = _selection(payload["selection"])
    display = _display(payload["display"])
    try:
        meta = viewer_data.get_meta(collection, params)
    except viewer_data.ViewerError as exc:
        raise ViewerResultError(exc.code, str(exc)) from exc
    count = meta.get("count")
    if not isinstance(count, int) or not 0 <= index < count:
        raise ViewerResultError(404, "viewer object index is unavailable")
    meta_for_tiers = dict(meta)
    meta_for_tiers["_selected_index"] = index
    resolved = _source_tiers(meta_for_tiers, payload["tiers"])
    _require_native_f200w_request(collection, params, resolved)
    loaded = [
        _load_cube_for_save(collection, index, source, logical, params, meta)
        for source, logical in resolved
    ]
    _require_loaded_native_f200w(loaded)

    center = _matched_bounds(loaded, selection)
    bounds: dict[str, dict[str, Any]] = {item["logical"]: item["bounds"] for item in loaded}

    source = {
        "collection": collection,
        "regime": "real" if collection in _REAL_COLLECTIONS else "synthetic",
        "index": index,
        "params": params,
        "object": _safe_object(meta, index),
        "viewer_tiers": [item["source_tier"] for item in loaded],
    }
    root = results_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=root))
    except OSError as exc:
        raise ViewerResultError(500, "could not create the viewer-results store") from exc

    published = False
    try:
        files: dict[str, dict[str, Any]] = {}
        for item in loaded:
            logical = item["logical"]
            b = item["bounds"]
            crop = np.ascontiguousarray(
                item["cube"][b["y0"]:b["y1"], b["x0"]:b["x1"], :],
                dtype=np.float32,
            )
            filename = f"{logical}.fits"
            path = staging / filename
            crop_wcs = (viewer_data.shifted_wcs_keywords(item["wcs"], dx=b["x0"], dy=b["y0"])
                        if item.get("wcs") else None)
            _write_fits(
                path,
                crop,
                logical=logical,
                source_tier=item["source_tier"],
                bands=item["bands"],
                pixscale=item["pixscale"],
                wcs=crop_wcs,
            )
            files[logical] = {
                "filename": filename,
                "sha256": _sha256(path),
                "shape_hwc": list(crop.shape),
                "dtype": "float32",
                "bands": item["bands"],
                "pixscale_arcsec": item["pixscale"],
                "source_tier": item["source_tier"],
                "source_label": item["label"],
                "source_asinh": item["asinh"],
                "display_scale": item["display_scale"],
                "direct_rgb": item["direct_rgb"],
                "transfer_group": item["transfer_group"],
                "wcs": crop_wcs is not None,
            }
        wcs_tiers = [logical for logical, entry in files.items() if entry["wcs"]]

        identity = {
            "schema_version": SCHEMA_VERSION,
            "source": source,
            "selection": selection,
            "bounds": bounds,
            "files": {
                key: {
                    "sha256": value["sha256"],
                    "bands": value["bands"],
                    "pixscale_arcsec": value["pixscale_arcsec"],
                    "source_tier": value["source_tier"],
                }
                for key, value in files.items()
            },
            "display": display,
        }
        identity_sha256 = _canonical_sha256(identity)
        result_id = f"vr-{identity_sha256[:24]}"
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "id": result_id,
            "identity_sha256": identity_sha256,
            "created_utc": datetime.now(UTC).isoformat(),
            "source": source,
            "selection": selection,
            "bounds": bounds,
            "bands": {key: value["bands"] for key, value in files.items()},
            "pixscale_arcsec": {
                key: value["pixscale_arcsec"] for key, value in files.items()
            },
            "logical_tiers": list(files),
            "files": files,
            "display": display,
            "rendering": {
                "transfer": "asinh_absolute",
                "knee_e": ASINH_KNEE_E,
                "white_e": ASINH_WHITE_E,
                "vis_h_false_colour": "VIS cyan; H_E amber",
            },
            "wcs_preserved": bool(files) and len(wcs_tiers) == len(files),
            "wcs_tiers": wcs_tiers,
            "center": center,
        }
        _write_manifest(staging / "manifest.json", manifest)

        final = root / result_id
        if final.exists():
            existing = _load_complete_manifest(
                final, expected_id=result_id, verify_hashes=True,
            )
            if existing is None or existing.get("identity_sha256") != identity_sha256:
                raise ViewerResultError(409, "saved-result identifier collision")
            return _result_summary(existing)
        try:
            os.rename(staging, final)
        except FileExistsError:
            existing = _load_complete_manifest(
                final, expected_id=result_id, verify_hashes=True,
            )
            if existing is None or existing.get("identity_sha256") != identity_sha256:
                raise ViewerResultError(409, "saved-result identifier collision") from None
            return _result_summary(existing)
        published = True
        return _result_summary(manifest)
    except ViewerResultError:
        raise
    except Exception as exc:
        raise ViewerResultError(500, "could not save the viewer result") from exc
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)


def _safe_bundle_file(directory: Path, filename: Any) -> Path | None:
    if not isinstance(filename, str) or not re.fullmatch(r"[a-z]+\.fits", filename):
        return None
    candidate = (directory / filename).resolve()
    try:
        candidate.relative_to(directory.resolve())
    except ValueError:
        return None
    return candidate


def _load_complete_manifest(
    directory: Path,
    *,
    expected_id: str,
    verify_hashes: bool = False,
) -> dict[str, Any] | None:
    try:
        root = results_root().resolve()
        resolved_directory = directory.resolve()
    except OSError:
        return None
    if directory.is_symlink() or resolved_directory.parent != root:
        return None
    try:
        manifest = json.loads((resolved_directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict) or manifest.get("complete") is not True:
        return None
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("id") != expected_id:
        return None
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        return None
    for logical, entry in files.items():
        if logical not in LOGICAL_TIER_ORDER or not isinstance(entry, dict):
            return None
        path = _safe_bundle_file(resolved_directory, entry.get("filename"))
        digest = entry.get("sha256")
        if path is None or not path.is_file() or not isinstance(digest, str):
            return None
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            return None
        if verify_hashes:
            try:
                actual = _cached_sha256(path)
            except ViewerResultError:
                return None
            if not hmac.compare_digest(actual, digest):
                return None
    return manifest


def get_result(result_id: str) -> dict[str, Any]:
    if not _RESULT_ID.fullmatch(result_id):
        raise ViewerResultError(404, "saved viewer result not found")
    manifest = _load_complete_manifest(results_root() / result_id, expected_id=result_id)
    if manifest is None:
        raise ViewerResultError(404, "saved viewer result not found")
    return manifest


def get_result_summary(result_id: str) -> dict[str, Any]:
    """Return the public, path-free summary for one complete saved result."""
    return _result_summary(get_result(result_id))


def _supported_recipes(manifest: Mapping[str, Any]) -> list[dict[str, str]]:
    files = manifest.get("files", {})
    recipes = []
    for logical in LOGICAL_TIER_ORDER:
        entry = files.get(logical) if isinstance(files, Mapping) else None
        if not isinstance(entry, Mapping):
            continue
        bands = set(entry.get("bands", []))
        modes = []
        if "VIS" in bands:
            modes.append("VIS")
        if "H_E" in bands:
            modes.append("H_E")
        if {"VIS", "H_E"} <= bands:
            modes.append("VIS_H")
        # The real-data A4 preset is specifically a NEXUS F200W comparison.
        # Do not silently relabel another JWST filter (or an approximate RGB
        # composite) as F200W.
        if logical == "jwst" and bands == {"F200W"} and len(entry.get("bands", [])) == 1:
            modes.append("native")
        for mode in modes:
            recipes.append({
                "tier": logical,
                "mode": mode,
                "key": f"{logical}:{mode}",
                "label": _recipe_label(logical, mode),
            })
    return recipes


def _default_label(source: Mapping[str, Any]) -> str:
    obj = source.get("object")
    label = obj.get("label") if isinstance(obj, Mapping) else None
    if not isinstance(label, str) or not label.strip():
        label = f"{source.get('collection', 'viewer')} {source.get('index', '')}".strip()
    return label


_PUBLIC_FILE_KEYS = (
    "filename", "shape_hwc", "bands", "pixscale_arcsec", "source_tier", "source_label",
    "display_scale", "direct_rgb", "transfer_group",
)


def _public_files(manifest: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], int, dict[str, str]]:
    """``({logical: public entry}, bundle bytes, {logical: inspect path})``
    — no checksums, no absolute paths."""
    files = manifest.get("files")
    directory = results_root() / str(manifest.get("id") or "")
    out: dict[str, dict[str, Any]] = {}
    inspect: dict[str, str] = {}
    total = 0
    if not isinstance(files, Mapping):
        return out, total, inspect
    for logical in LOGICAL_TIER_ORDER:
        entry = files.get(logical)
        if not isinstance(entry, Mapping):
            continue
        public = {key: entry.get(key) for key in _PUBLIC_FILE_KEYS if key in entry}
        public["wcs"] = entry.get("wcs") is True
        out[logical] = public
        path = _safe_bundle_file(directory, entry.get("filename"))
        if path is not None:
            with contextlib.suppress(OSError):
                total += path.stat().st_size
            inspect[logical] = paths._safe_relpath(os.fspath(path))
    return out, total, inspect


def _thumbnail_recipe(recipe_keys: Sequence[str]) -> str | None:
    for key in THUMBNAIL_PREFERENCE:
        if key in recipe_keys:
            return key
    return recipe_keys[0] if recipe_keys else None


def _result_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    raw_source = manifest.get("source", {})
    source = raw_source if isinstance(raw_source, Mapping) else {}
    default_label = _default_label(source)
    user_label = manifest.get("label")
    label = user_label.strip() if isinstance(user_label, str) and user_label.strip() else default_label
    recipe_options = _supported_recipes(manifest)
    recipes = [recipe["key"] for recipe in recipe_options]
    regime = source.get("regime") if isinstance(source, Mapping) else None
    if regime not in {"real", "synthetic"}:
        regime = "real" if source.get("collection") in _REAL_COLLECTIONS else "synthetic"
    files, total_bytes, inspect_paths = _public_files(manifest)
    wcs_tiers = [tier for tier in LOGICAL_TIER_ORDER if files.get(tier, {}).get("wcs")]
    center = manifest.get("center")
    if not (isinstance(center, Mapping) and all(
            isinstance(center.get(key), (int, float)) for key in ("ra", "dec"))):
        center = None
    return {
        "id": manifest["id"],
        "created_utc": manifest.get("created_utc"),
        "label": label,
        "default_label": default_label,
        "regime": regime,
        "source": source,
        "selection": manifest.get("selection"),
        "logical_tiers": manifest.get("logical_tiers", []),
        "bands": manifest.get("bands", {}),
        "pixscale_arcsec": manifest.get("pixscale_arcsec", {}),
        "recipes": recipes,
        "recipe_options": recipe_options,
        "thumbnail": _thumbnail_recipe(recipes),
        "files": files,
        "bytes": total_bytes,
        "inspect_paths": inspect_paths,
        "display": manifest.get("display", {}),
        "wcs_preserved": bool(files) and len(wcs_tiers) == len(files),
        "wcs_tiers": wcs_tiers,
        "center": dict(center) if center else None,
    }


def list_results() -> dict[str, Any]:
    root = results_root()
    summaries = []
    if root.is_dir():
        for directory in sorted(root.iterdir(), key=lambda path: path.name):
            if not directory.is_dir() or not _RESULT_ID.fullmatch(directory.name):
                continue
            manifest = _load_complete_manifest(
                directory, expected_id=directory.name, verify_hashes=True,
            )
            if manifest is not None:
                summaries.append(_result_summary(manifest))
    summaries.sort(key=lambda item: str(item.get("created_utc") or ""), reverse=True)
    return {
        "schema_version": SCHEMA_VERSION,
        "axis_defaults": {"columns": "results", "rows": "recipes"},
        "limits": {
            "max_results": MAX_GRID_RESULTS,
            "max_rows": MAX_GRID_ROWS,
        },
        "supported": {
            "logical_tiers": list(LOGICAL_TIER_ORDER),
            "modes": list(DISPLAY_MODES),
            "transfer": {
                "kind": "asinh_absolute",
                "knee_e": ASINH_KNEE_E,
                "white_e": ASINH_WHITE_E,
            },
            "dpi": {
                "preview": 120,
                "default": DEFAULT_GRID_DPI,
                "min": MIN_GRID_DPI,
                "max": MAX_GRID_DPI,
            },
        },
        "results": summaries,
    }


def _result_file(manifest: Mapping[str, Any], logical: str) -> tuple[Path, Mapping[str, Any]]:
    if logical not in LOGICAL_TIER_ORDER:
        raise ViewerResultError(400, "tier must be dirty, sr, hr, or jwst")
    entry = manifest.get("files", {}).get(logical)
    if not isinstance(entry, Mapping):
        raise ViewerResultError(404, f"saved result has no {logical} tier")
    directory = results_root() / str(manifest["id"])
    path = _safe_bundle_file(directory, entry.get("filename"))
    if path is None or not path.is_file():
        raise ViewerResultError(404, "saved FITS panel is unavailable")
    expected_digest = entry.get("sha256")
    if not isinstance(expected_digest, str) or not hmac.compare_digest(
        _cached_sha256(path), expected_digest,
    ):
        raise ViewerResultError(409, "saved FITS checksum does not match its manifest")
    return path, entry


def _cached_sha256(path: Path) -> str:
    """Hash a FITS file once per stable stat identity."""
    try:
        stat = path.stat()
    except OSError as exc:
        raise ViewerResultError(404, "saved FITS panel is unavailable") from exc
    key = (
        os.fspath(path.resolve()),
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )
    cached = _SHA_CACHE.get(key)
    if cached is not None:
        _SHA_CACHE.move_to_end(key)
        return cached
    try:
        digest = _sha256(path)
    except OSError as exc:
        raise ViewerResultError(404, "saved FITS panel is unavailable") from exc
    _SHA_CACHE[key] = digest
    if len(_SHA_CACHE) > _SHA_CACHE_MAX:
        _SHA_CACHE.popitem(last=False)
    return digest


def _read_saved_cube(path: Path, entry: Mapping[str, Any]) -> np.ndarray:
    try:
        # These bundles are written by this module as unscaled float32 primary
        # arrays, so a read-only memmap avoids retaining the entire FITS cube
        # while the grid renderer samples the one or two requested planes.
        data = fits.getdata(path, memmap=True)
    except (OSError, ValueError) as exc:
        raise ViewerResultError(415, "saved FITS panel is unreadable") from exc
    array = np.asarray(data)
    if array.ndim == 2:
        cube = array[..., None]
    elif array.ndim == 3:
        cube = np.moveaxis(array, 0, -1)
    else:
        raise ViewerResultError(415, "saved FITS panel has an invalid shape")
    expected = entry.get("shape_hwc")
    if not isinstance(expected, list) or list(cube.shape) != expected:
        raise ViewerResultError(415, "saved FITS shape disagrees with its manifest")
    return cube


def _absolute_asinh(values: np.ndarray, *, display_scale: float = 1.0) -> np.ndarray:
    clean = np.nan_to_num(
        np.asarray(values, dtype=np.float64) * display_scale,
        nan=0.0,
        posinf=ASINH_WHITE_E,
        neginf=0.0,
    )
    norm = math.asinh(ASINH_WHITE_E / ASINH_KNEE_E)
    return np.clip(
        np.arcsinh(np.clip(clean, 0.0, None) / ASINH_KNEE_E) / norm,
        0.0,
        1.0,
    ).astype(np.float32)


def _vis_h_false_colour(vis: np.ndarray, h_band: np.ndarray) -> np.ndarray:
    """Map absolute-transfer VIS to cyan-blue and H_E to amber-red."""
    red = h_band + 0.08 * vis
    green = 0.54 * (vis + h_band)
    blue = vis + 0.08 * h_band
    rgb = np.stack((red, green, blue), axis=-1)
    return np.power(np.clip(rgb, 0.0, 1.0), 0.92).astype(np.float32)


def _resample_plane(values: np.ndarray, target_side_pixels: int | None) -> np.ndarray:
    """Resample one raw plane before any RGB/display allocation."""
    plane = np.asarray(values, dtype=np.float32)
    if target_side_pixels is None or plane.shape == (target_side_pixels, target_side_pixels):
        return plane
    if target_side_pixels < 1:
        raise ViewerResultError(413, "grid cells are too small to render")
    image = PILImage.fromarray(plane)
    resized = image.resize(
        (target_side_pixels, target_side_pixels),
        resample=PILImage.Resampling.LANCZOS,
    )
    return np.asarray(resized, dtype=np.float32)


def _rgb_uint8(rgb: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def _render_panel_modes(
    manifest: Mapping[str, Any],
    logical: str,
    modes: Sequence[str],
    *,
    target_side_pixels: int | None = None,
) -> dict[str, np.ndarray]:
    """Render several modes from one FITS read, returning compact RGB uint8."""
    requested = list(dict.fromkeys(modes))
    if not requested or any(mode not in DISPLAY_MODES for mode in requested):
        raise ViewerResultError(400, "mode must be VIS, H_E, VIS_H, or native")
    path, entry = _result_file(manifest, logical)
    cube = _read_saved_cube(path, entry)
    bands = [str(item) for item in entry.get("bands", [])]
    display_scale_raw = entry.get("display_scale")
    display_scale = (
        float(display_scale_raw)
        if isinstance(display_scale_raw, (int, float)) and float(display_scale_raw) > 0.0
        else 1.0
    )
    plane_cache: dict[int, np.ndarray] = {}

    def plane(index: int) -> np.ndarray:
        if index not in plane_cache:
            plane_cache[index] = _resample_plane(cube[..., index], target_side_pixels)
        return plane_cache[index]

    rendered: dict[str, np.ndarray] = {}
    for mode in requested:
        if mode in {"VIS", "H_E"}:
            if mode not in bands:
                raise ViewerResultError(404, f"saved tier has no {mode} band")
            gray = _absolute_asinh(
                plane(bands.index(mode)), display_scale=display_scale,
            )
            rendered[mode] = _rgb_uint8(np.repeat(gray[..., None], 3, axis=-1))
            continue
        if mode == "VIS_H":
            if "VIS" not in bands or "H_E" not in bands:
                raise ViewerResultError(404, "saved tier has no VIS-H_E band pair")
            vis = _absolute_asinh(
                plane(bands.index("VIS")), display_scale=display_scale,
            )
            h_band = _absolute_asinh(
                plane(bands.index("H_E")), display_scale=display_scale,
            )
            rendered[mode] = _rgb_uint8(_vis_h_false_colour(vis, h_band))
            continue
        if cube.shape[-1] == 1:
            gray = _absolute_asinh(plane(0), display_scale=display_scale)
            rendered[mode] = _rgb_uint8(np.repeat(gray[..., None], 3, axis=-1))
            continue
        if bool(entry.get("direct_rgb")) and cube.shape[-1] >= 3:
            rgb = np.stack([plane(index) for index in range(3)], axis=-1)
            rgb = np.clip(np.nan_to_num(rgb * display_scale), 0.0, None)
            intensity = np.mean(rgb, axis=-1)
            luminance = _absolute_asinh(intensity)
            rescale = np.divide(
                luminance,
                intensity,
                out=np.zeros_like(luminance),
                where=intensity > 1.0e-30,
            )
            rendered[mode] = _rgb_uint8(rgb * rescale[..., None])
            continue
        raise ViewerResultError(400, "native mode requires one band or a direct-RGB cube")
    return rendered


@dataclass(frozen=True)
class PanelRequest:
    """One resolved ``panel.png`` request (validated before any render)."""

    manifest: Mapping[str, Any]
    logical: str
    mode: str
    size: int | None
    etag: str


def panel_request(result_id: str, tier: str = "", mode: str = "",
                  size: Any = None) -> PanelRequest:
    """Validate a panel request. Empty ``tier``/``mode`` = the result's
    thumbnail recipe; ``size`` = the thumbnail side (px, never upsampled).
    The ETag is content-addressed (the id hashes the saved FITS)."""
    manifest = get_result(result_id)
    logical, clean_mode = tier.strip().lower(), mode.strip()
    if not logical and not clean_mode:
        recipe = _thumbnail_recipe([item["key"] for item in _supported_recipes(manifest)])
        if recipe is None:
            raise ViewerResultError(404, "saved result has no renderable panel")
        logical, clean_mode = recipe.split(":", 1)
    side: int | None = None
    if size not in (None, ""):
        try:
            side = int(str(size).strip())
        except ValueError as exc:
            raise ViewerResultError(400, "size must be an integer") from exc
        if not MIN_THUMB_SIDE <= side <= MAX_THUMB_SIDE:
            raise ViewerResultError(
                400, f"size must be from {MIN_THUMB_SIDE} to {MAX_THUMB_SIDE} pixels")
    identity = str(manifest.get("identity_sha256") or manifest["id"])[:24]
    etag = f"{identity}-{logical}-{clean_mode}-{side or 0}-r1"
    return PanelRequest(manifest, logical, clean_mode, side, etag)


def render_panel_png(request: PanelRequest) -> bytes:
    """PNG of one saved panel; a ``size`` smaller than the crop downsamples
    it (LANCZOS on the raw plane, before the transfer), a larger one does not
    upsample (the browser scales it, pixelated)."""
    target = None
    if request.size is not None:
        _path, entry = _result_file(request.manifest, request.logical)
        height, width, _channels = _manifest_shape(entry)
        if request.size < min(height, width):
            target = request.size
    rgb = _render_panel_modes(
        request.manifest, request.logical, [request.mode], target_side_pixels=target,
    )[request.mode]
    output = BytesIO()
    PILImage.fromarray(rgb, mode="RGB").save(output, format="PNG")
    return output.getvalue()


def render_panel(result_id: str, tier: str, mode: str, size: Any = None) -> bytes:
    return render_panel_png(panel_request(result_id, tier, mode, size))


def result_fits(result_id: str, logical: str) -> tuple[Path, str]:
    """``(verified FITS path, download name)`` of one saved tier."""
    manifest = get_result(result_id)
    path, _entry = _result_file(manifest, logical.strip().lower())
    return path, f"{manifest['id']}_{logical.strip().lower()}.fits"


# ---------------------------------------------------------------------------
# managing saved results (W-Figures): user label, delete
# ---------------------------------------------------------------------------

def _clean_label(value: Any, *, name: str, maximum: int, allow_empty: bool) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise ViewerResultError(400, f"{name} must be a string")
    text = " ".join(value.split()) if not _CONTROL_CHARS.search(value) else None
    if text is None:
        raise ViewerResultError(400, f"{name} must not contain control characters")
    if not text and not allow_empty:
        raise ViewerResultError(400, f"{name} must not be empty")
    if len(text) > maximum:
        raise ViewerResultError(400, f"{name} must be at most {maximum} characters")
    return text


def _atomic_write_json(path: Path, value: Any) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def rename_result(result_id: str, label: Any) -> dict[str, Any]:
    """Set (or, with an empty label, clear) a result's user label. The id is
    content-addressed and does not change; the label is not part of it."""
    manifest = dict(get_result(result_id))
    clean = _clean_label(label, name="label", maximum=MAX_LABEL_CHARS, allow_empty=True)
    if clean and clean != _default_label(manifest.get("source") or {}):
        manifest["label"] = clean
    else:
        manifest.pop("label", None)
    try:
        _atomic_write_json(results_root() / str(manifest["id"]) / "manifest.json", manifest)
    except OSError as exc:
        raise ViewerResultError(500, "could not rename the viewer result") from exc
    return _result_summary(manifest)


def delete_result(result_id: str) -> str:
    """Delete one complete bundle (validated: an id under the results root,
    no symlink). It leaves the listing atomically (renamed first), is then
    removed, and is dropped from every saved grid layout."""
    manifest = get_result(result_id)
    root = results_root()
    directory = root / str(manifest["id"])
    doomed = root / f".deleting-{manifest['id']}-{secrets.token_hex(4)}"
    try:
        os.rename(directory, doomed)
    except OSError as exc:
        raise ViewerResultError(500, "could not delete the viewer result") from exc
    shutil.rmtree(doomed, ignore_errors=True)
    with contextlib.suppress(ViewerResultError, OSError):
        _prune_layouts({str(manifest["id"])})
    return str(manifest["id"])


# ---------------------------------------------------------------------------
# saved grid layouts
# ---------------------------------------------------------------------------

def _layouts_path() -> Path:
    return results_root() / LAYOUTS_FILE


def _read_layouts() -> list[dict[str, Any]]:
    try:
        raw = json.loads(_layouts_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, json.JSONDecodeError) as exc:
        raise ViewerResultError(500, "the saved grid layouts are unreadable") from exc
    items = raw.get("layouts") if isinstance(raw, Mapping) else None
    if not isinstance(items, list):
        return []
    return [dict(item) for item in items
            if isinstance(item, Mapping) and isinstance(item.get("id"), str)
            and _LAYOUT_ID.fullmatch(item["id"])]


def _write_layouts(layouts: list[dict[str, Any]]) -> None:
    root = results_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(_layouts_path(), {"schema_version": 1, "layouts": layouts})
    except OSError as exc:
        raise ViewerResultError(500, "could not save the grid layouts") from exc


def _prune_layouts(removed_ids: set[str]) -> None:
    with _LAYOUT_LOCK:
        layouts = _read_layouts()
        changed = False
        for layout in layouts:
            kept = [item for item in layout.get("results", []) if item not in removed_ids]
            if kept != layout.get("results", []):
                layout["results"] = kept
                changed = True
        if changed:
            _write_layouts(layouts)


def _id_list(value: Any, name: str) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",") if part.strip()]
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise ViewerResultError(400, f"{name} must be a list")
    return [str(item).strip() for item in value]


def list_layouts() -> dict[str, Any]:
    with _LAYOUT_LOCK:
        layouts = _read_layouts()
    layouts.sort(key=lambda item: str(item.get("updated_utc") or ""), reverse=True)
    return {"layouts": layouts}


def save_layout(payload: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    """Create or update a named grid layout: ``{name, results, rows, regime?,
    id?}``. An existing ``id`` — or the same name, any case — is updated in
    place. Returns ``(layout, created)``."""
    name = _clean_label(payload.get("name"), name="name", maximum=MAX_LAYOUT_NAME_CHARS,
                        allow_empty=False)
    result_ids = _id_list(payload.get("results"), "results")
    if len(result_ids) > MAX_GRID_RESULTS:
        raise ViewerResultError(400, f"a layout holds at most {MAX_GRID_RESULTS} results")
    if len(set(result_ids)) != len(result_ids):
        raise ViewerResultError(400, "a layout lists each result once")
    for result_id in result_ids:
        try:
            get_result(result_id)
        except ViewerResultError as exc:
            raise ViewerResultError(400, f"unknown saved result: {result_id}") from exc
    rows = _id_list(payload.get("rows"), "rows")
    if not rows:
        raise ViewerResultError(400, "a layout needs at least one row recipe")
    if len(rows) > MAX_GRID_ROWS:
        raise ViewerResultError(400, f"a layout holds at most {MAX_GRID_ROWS} rows")
    recipes = [":".join(_parse_recipe(row)) for row in rows]
    regime = payload.get("regime")
    if regime not in (None, "", "real", "synthetic"):
        raise ViewerResultError(400, "regime must be real or synthetic")
    layout_id = str(payload.get("id") or "").strip()
    if layout_id and not _LAYOUT_ID.fullmatch(layout_id):
        raise ViewerResultError(400, "layout id is invalid")
    now = datetime.now(UTC).isoformat()
    with _LAYOUT_LOCK:
        layouts = _read_layouts()
        existing = next((item for item in layouts if layout_id and item["id"] == layout_id), None)
        if existing is None and layout_id:
            raise ViewerResultError(404, "grid layout not found")
        if existing is None:
            existing = next((item for item in layouts
                             if str(item.get("name", "")).casefold() == name.casefold()), None)
        created = existing is None
        if created:
            if len(layouts) >= MAX_LAYOUTS:
                raise ViewerResultError(409, f"at most {MAX_LAYOUTS} grid layouts; delete one first")
            existing = {"id": f"gl-{secrets.token_hex(6)}", "created_utc": now}
            layouts.append(existing)
        existing.update({
            "name": name, "results": result_ids, "rows": recipes,
            "regime": regime or None, "updated_utc": now,
        })
        _write_layouts(layouts)
    return dict(existing), created


def delete_layout(layout_id: str) -> str:
    if not _LAYOUT_ID.fullmatch(layout_id or ""):
        raise ViewerResultError(404, "grid layout not found")
    with _LAYOUT_LOCK:
        layouts = _read_layouts()
        kept = [item for item in layouts if item["id"] != layout_id]
        if len(kept) == len(layouts):
            raise ViewerResultError(404, "grid layout not found")
        _write_layouts(kept)
    return layout_id


def _recipe_label(logical: str, mode: str) -> str:
    if logical == "jwst" and mode == "native":
        return "NEXUS F200W"
    suffix = {"dirty": "Dirty", "sr": "SR", "hr": "HR", "jwst": "JWST"}[logical]
    prefix = {"VIS": "VIS", "H_E": "H_E", "VIS_H": "VIS + H_E", "native": "Native"}[mode]
    return f"{prefix} {suffix}"


def _parse_recipe(value: str) -> tuple[str, str]:
    if not isinstance(value, str) or value.count(":") != 1:
        raise ViewerResultError(400, "grid row must have tier:mode form")
    logical, mode = (part.strip() for part in value.split(":", 1))
    logical = logical.lower()
    if logical not in LOGICAL_TIER_ORDER or mode not in DISPLAY_MODES:
        raise ViewerResultError(400, "grid row has an unsupported tier or mode")
    return logical, mode


def _default_grid_recipes(manifests: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    supported_by_result = [
        {(recipe["tier"], recipe["mode"]) for recipe in _supported_recipes(manifest)}
        for manifest in manifests
    ]
    supported = set.intersection(*supported_by_result) if supported_by_result else set()
    preferred = [
        ("dirty", "VIS"),
        ("dirty", "H_E"),
        ("dirty", "VIS_H"),
        ("sr", "VIS"),
        ("sr", "H_E"),
        ("sr", "VIS_H"),
        ("hr", "VIS"),
        ("hr", "H_E"),
        ("hr", "VIS_H"),
        ("jwst", "native"),
    ]
    return [recipe for recipe in preferred if recipe in supported]


def _grid_geometry(rows: int, columns: int, dpi: int) -> GridGeometry:
    """Fit square panels on A4 using one exact physical gap/padding value.

    The row- and column-title tracks absorb the spare dimension.  Consequently
    every panel remains square, every inter-cell gap is ``GRID_GAP_MM``, and
    the complete table has exactly that same padding at all four page edges.
    """
    if rows < 1 or columns < 1:
        raise ViewerResultError(400, "grid needs at least one row and one result")
    width_for_cells = (
        A4_WIDTH_MM
        - 2.0 * GRID_GAP_MM
        - MIN_ROW_TITLE_TRACK_MM
        - (columns - 1) * GRID_GAP_MM
    )
    height_for_cells = (
        A4_HEIGHT_MM
        - 2.0 * GRID_GAP_MM
        - MIN_COLUMN_TITLE_TRACK_MM
        - (rows - 1) * GRID_GAP_MM
    )
    side = min(width_for_cells / columns, height_for_cells / rows)
    if not math.isfinite(side) or side <= 0.0:
        raise ViewerResultError(413, "grid has too many rows or results for A4")

    row_title_track = (
        A4_WIDTH_MM
        - 2.0 * GRID_GAP_MM
        - columns * side
        - (columns - 1) * GRID_GAP_MM
    )
    column_title_track = (
        A4_HEIGHT_MM
        - 2.0 * GRID_GAP_MM
        - rows * side
        - (rows - 1) * GRID_GAP_MM
    )
    if (
        row_title_track + 1.0e-8 < MIN_ROW_TITLE_TRACK_MM
        or column_title_track + 1.0e-8 < MIN_COLUMN_TITLE_TRACK_MM
    ):
        raise ViewerResultError(413, "grid title tracks do not fit on A4")

    page_width_pixels = int(round(A4_WIDTH_MM / 25.4 * dpi))
    page_height_pixels = int(round(A4_HEIGHT_MM / 25.4 * dpi))
    cell_side_pixels = int(round(side / 25.4 * dpi))
    if cell_side_pixels < 1:
        raise ViewerResultError(413, "grid cells are too small to render")
    return GridGeometry(
        rows=rows,
        columns=columns,
        dpi=dpi,
        gap_mm=GRID_GAP_MM,
        outer_padding_mm=GRID_GAP_MM,
        row_title_track_mm=row_title_track,
        column_title_track_mm=column_title_track,
        cell_side_mm=side,
        grid_left_mm=GRID_GAP_MM + row_title_track,
        grid_bottom_mm=GRID_GAP_MM,
        page_width_pixels=page_width_pixels,
        page_height_pixels=page_height_pixels,
        cell_side_pixels=cell_side_pixels,
    )


def _manifest_shape(entry: Mapping[str, Any]) -> tuple[int, int, int]:
    raw = entry.get("shape_hwc")
    if (
        not isinstance(raw, list)
        or len(raw) != 3
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in raw)
    ):
        raise ViewerResultError(415, "saved FITS shape metadata is invalid")
    height, width, channels = raw
    return height, width, channels


def _grid_request_budget(
    manifests: Sequence[Mapping[str, Any]],
    recipes: Sequence[tuple[str, str]],
    geometry: GridGeometry,
) -> dict[str, int]:
    """Validate page, source-I/O, and peak working-memory pixel budgets."""
    page_pixels = geometry.page_width_pixels * geometry.page_height_pixels
    if page_pixels > MAX_GRID_OUTPUT_PIXELS:
        raise ViewerResultError(413, "requested A4 raster exceeds the output-pixel budget")

    unique_sources: dict[tuple[str, str], tuple[int, int, int]] = {}
    unique_panels: set[tuple[str, str, str]] = set()
    for manifest in manifests:
        result_id = str(manifest.get("id") or "")
        files = manifest.get("files", {})
        if not isinstance(files, Mapping):
            raise ViewerResultError(415, "saved result file metadata is invalid")
        for logical, mode in recipes:
            entry = files.get(logical)
            if not isinstance(entry, Mapping):
                raise ViewerResultError(404, f"saved result has no {logical} tier")
            unique_sources[(result_id, logical)] = _manifest_shape(entry)
            unique_panels.add((result_id, logical, mode))

    source_bytes = sum(
        height * width * channels * np.dtype(np.float32).itemsize
        for height, width, channels in unique_sources.values()
    )
    if source_bytes > MAX_GRID_SOURCE_BYTES:
        raise ViewerResultError(413, "grid source FITS exceed the request I/O budget")

    largest_source_plane_bytes = max(
        height * width * np.dtype(np.float32).itemsize
        for height, width, _channels in unique_sources.values()
    )
    panel_cache_bytes = (
        len(unique_panels) * geometry.cell_side_pixels**2 * 3 * np.dtype(np.uint8).itemsize
    )
    canvas_working_bytes = page_pixels * _GRID_CANVAS_BYTES_PER_PIXEL
    # One source plane is converted from the read-only FITS memmap at a time.
    # Forty bytes/output pixel covers the largest transient path: two stretched
    # float planes, a float RGB composite, and transfer/resize temporaries.
    panel_working_bytes = geometry.cell_side_pixels**2 * 40
    estimated_peak_bytes = (
        canvas_working_bytes
        + panel_cache_bytes
        + largest_source_plane_bytes
        + panel_working_bytes
    )
    if estimated_peak_bytes > MAX_GRID_RENDER_BYTES:
        raise ViewerResultError(413, "grid exceeds the server render-memory budget")
    return {
        "page_pixels": page_pixels,
        "source_bytes": source_bytes,
        "panel_cache_bytes": panel_cache_bytes,
        "estimated_peak_bytes": estimated_peak_bytes,
    }


def _grid_panel_cache(
    manifests: Sequence[Mapping[str, Any]],
    recipes: Sequence[tuple[str, str]],
    target_side_pixels: int,
) -> dict[tuple[str, str, str], np.ndarray]:
    """Render every unique panel once at exactly its output-cell resolution."""
    manifests_by_id = {str(manifest["id"]): manifest for manifest in manifests}
    modes_by_source: OrderedDict[tuple[str, str], list[str]] = OrderedDict()
    for manifest in manifests:
        result_id = str(manifest["id"])
        for logical, mode in recipes:
            key = (result_id, logical)
            modes = modes_by_source.setdefault(key, [])
            if mode not in modes:
                modes.append(mode)

    cache: dict[tuple[str, str, str], np.ndarray] = {}
    for (result_id, logical), modes in modes_by_source.items():
        rendered = _render_panel_modes(
            manifests_by_id[result_id],
            logical,
            modes,
            target_side_pixels=target_side_pixels,
        )
        for mode, rgb in rendered.items():
            cache[(result_id, logical, mode)] = rgb
    return cache


def render_grid(
    result_ids: Sequence[str],
    rows: Sequence[str],
    output_format: str,
    dpi: Any = DEFAULT_GRID_DPI,
) -> bytes:
    """Render results as columns and tier/mode recipes as rows."""
    if output_format not in {"png", "pdf"}:
        raise ViewerResultError(400, "grid format must be png or pdf")
    if isinstance(dpi, bool):
        raise ViewerResultError(400, "grid dpi must be an integer")
    try:
        render_dpi = int(dpi)
    except (TypeError, ValueError) as exc:
        raise ViewerResultError(400, "grid dpi must be an integer") from exc
    if str(dpi).strip() != str(render_dpi) or not MIN_GRID_DPI <= render_dpi <= MAX_GRID_DPI:
        raise ViewerResultError(
            400,
            f"grid dpi must be an integer from {MIN_GRID_DPI} to {MAX_GRID_DPI}",
        )
    ids = list(result_ids)
    if not ids:
        ids = [item["id"] for item in list_results()["results"]]
    if not ids:
        raise ViewerResultError(404, "no saved viewer results are available")
    if len(ids) > MAX_GRID_RESULTS:
        raise ViewerResultError(413, f"a grid supports at most {MAX_GRID_RESULTS} results")
    manifests = [get_result(result_id) for result_id in ids]

    recipes = [_parse_recipe(row) for row in rows] if rows else _default_grid_recipes(manifests)
    if not recipes:
        raise ViewerResultError(400, "no supported grid recipes are available")
    if len(recipes) > MAX_GRID_ROWS:
        raise ViewerResultError(413, f"a grid supports at most {MAX_GRID_ROWS} rows")
    supported_by_result = [
        {(recipe["tier"], recipe["mode"]) for recipe in _supported_recipes(manifest)}
        for manifest in manifests
    ]
    supported_intersection = set.intersection(*supported_by_result)
    if any(recipe not in supported_intersection for recipe in recipes):
        raise ViewerResultError(400, "every grid recipe must be available for every result")

    geometry = _grid_geometry(len(recipes), len(manifests), render_dpi)
    _grid_request_budget(manifests, recipes, geometry)
    panel_cache = _grid_panel_cache(manifests, recipes, geometry.cell_side_pixels)

    nrows = len(recipes)
    figure = Figure(figsize=(A4_WIDTH_MM / 25.4, A4_HEIGHT_MM / 25.4), facecolor="white")
    FigureCanvasAgg(figure)

    for row_index, (logical, mode) in enumerate(recipes):
        for col_index, manifest in enumerate(manifests):
            left, bottom, width, height = geometry.panel_bounds_mm(row_index, col_index)
            axis = figure.add_axes([
                left / A4_WIDTH_MM,
                bottom / A4_HEIGHT_MM,
                width / A4_WIDTH_MM,
                height / A4_HEIGHT_MM,
            ])
            axis.set_axis_off()
            rgb = panel_cache[(str(manifest["id"]), logical, mode)]
            axis.imshow(rgb, origin="upper", interpolation="nearest", aspect="equal")

    grid_top_mm = (
        geometry.grid_bottom_mm
        + nrows * geometry.cell_side_mm
        + (nrows - 1) * geometry.gap_mm
    )
    title_font_size = max(5.0, min(9.0, geometry.cell_side_mm * 0.18))
    for index, manifest in enumerate(manifests):
        label = str(_result_summary(manifest)["label"])
        if len(label) > 42:
            label = f"{label[:39]}..."
        figure.text(
            (
                geometry.grid_left_mm
                + index * (geometry.cell_side_mm + geometry.gap_mm)
                + geometry.cell_side_mm / 2.0
            ) / A4_WIDTH_MM,
            (grid_top_mm + geometry.column_title_track_mm / 2.0) / A4_HEIGHT_MM,
            label,
            ha="center",
            va="center",
            fontsize=title_font_size,
            color="#111111",
        )
    for index, (logical, mode) in enumerate(recipes):
        _left, bottom, _width, height = geometry.panel_bounds_mm(index, 0)
        figure.text(
            (geometry.outer_padding_mm + geometry.row_title_track_mm / 2.0) / A4_WIDTH_MM,
            (bottom + height / 2.0) / A4_HEIGHT_MM,
            _recipe_label(logical, mode),
            ha="center",
            va="center",
            rotation=90,
            fontsize=title_font_size,
            color="#111111",
        )

    output = BytesIO()
    figure.savefig(
        output,
        format=output_format,
        dpi=render_dpi,
        facecolor="white",
        edgecolor="none",
    )
    body = output.getvalue()
    figure.clear()
    panel_cache.clear()
    return body
