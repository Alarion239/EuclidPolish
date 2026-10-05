"""Header-first FITS inspection for the Files workspace (the former Inspect
workspace, spec §8.6).

Nothing here reads pixels unless asked:

* :func:`file_summary` — every HDU from its header alone (shape, dtype, unit,
  celestial WCS with centre / scale / footprint, bands, table columns) plus
  the *band groups* (``VIS``/``Y_E``/``J_E``/``H_E`` image HDUs sharing a
  prefix and a shape, e.g. the poster's ``LR_*`` / ``SR_*``) and the
  provenance stamp of the primary header.
* :func:`read_plane` — one 2-D plane of any image HDU of any dimensionality,
  memory-mapped and binned for display (block mean, or a strided sample for
  inputs above :data:`BLOCK_MEAN_MAX_PIXELS`), so a 2560² stack or a 30k²
  mosaic never loads whole. ``BSCALE``/``BZERO``/``BLANK`` are applied.
* :func:`plane_stats` / :func:`vector_series` — statistics + histogram of a
  plane (a strided sample above :data:`MAX_STATS_PIXELS`), and a 1-D image
  HDU as a plottable series.
* :func:`table_page` / :func:`table_stats` — paged (optionally sorted) rows
  of a table HDU and per-column statistics.
* :func:`provenance` — the PROVID stamp, the co-located ``<id>.<kind>.json``
  sidecars describing this file, and the records of its producing run and
  parents (``data/_prov``, checkpoint ``provenance.json``).

Plane / binning geometry: served pixel ``j`` (0-based) is centred on source
pixel ``bin·j + offset`` (``offset = (bin − 1)/2`` for a block mean, ``bin//2``
for a strided sample), so the served grid's WCS is the source WCS with
``CRPIX → (CRPIX − 1 − offset)/bin + 1`` and the pixel matrix ×``bin``
(:func:`binned_wcs`).
"""
from __future__ import annotations

import contextlib
import glob
import gzip
import json
import math
import os
import re
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from euclid_polish.config import Config
from euclid_polish.provenance.fits import read_stamp_cards
from euclid_polish.web.helpers.paths import _safe_relpath

BAND_NAMES: tuple[str, ...] = tuple(Config.LR_INPUT_BAND_NAMES)

#: Auto-bin a plane whose longer side exceeds this for display.
MAX_VIEW_SIDE = 2048
#: Never serve a plane whose longer side exceeds this (the bin grows).
MAX_OUTPUT_SIDE = 4096
#: Block-mean binning reads the whole plane; above this many source pixels a
#: strided sample is served instead (reads ~1/bin² of a memory-mapped file).
BLOCK_MEAN_MAX_PIXELS = 4096 * 4096
#: A gzip-compressed file or tile-compressed HDU must be decompressed to read
#: a plane: refuse planes larger than this (download the file instead).
MAX_DECOMPRESS_BYTES = 512 * 2**20
#: A ``.gz`` file larger than this lists only its primary HDU (finding the
#: next header means decompressing everything before it).
GZIP_SCAN_BYTES = 256 * 2**20
#: Statistics use a strided sample of planes larger than this.
MAX_STATS_PIXELS = 4096 * 4096
#: The viewer offers at most this many planes of one HDU.
MAX_PLANES = 10_000
#: At most this many HDUs are listed.
MAX_HDUS = 2000
TABLE_PAGE_MAX = 2000
TABLE_STATS_MAX_ROWS = 2_000_000
TABLE_UNIQUE_MAX_ROWS = 200_000
CARD_VALUE_MAX = 240
VECTOR_MAX_POINTS = 4000
HIST_BINS = 128
_PERCENTILES = (0.1, 1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0, 99.9)

_BITPIX_DTYPE = {8: "uint8", 16: ">i2", 32: ">i4", 64: ">i8", -32: ">f4", -64: ">f8"}
_BAND_SUFFIX = re.compile(r"^(?P<prefix>.*?)(?P<band>VIS|Y_E|J_E|H_E)$", re.IGNORECASE)
#: Bare NISP letters (``Y``, ``LR_J``, ``NISP_H``, ``LR_NIR_Y``): upper case
#: and the whole name or after a separator, so ``DEPTH`` is no band.
_NIR_SUFFIX = re.compile(r"^(?P<prefix>(?:.*?[_\- ])??)(?:NISP_|NIR_)?(?P<band>[YJH])$")
_FILTER_NIR = re.compile(r"^(?:NISP|NIR)?[_\- ]?(?P<band>[YJH])(?:_E)?$")
_SIDECAR = re.compile(r"^([0-9a-f]{8})\.([a-z0-9_]+)\.json$")
_PROV_ID = re.compile(r"^[0-9a-f]{8}$")


class InspectError(Exception):
    """A request the inspector cannot serve; ``code`` is the HTTP status."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------------------
# Opening
# ---------------------------------------------------------------------------

def _is_gzip(path: str) -> bool:
    return path.lower().endswith(".gz")


def _large_gzip(path: str) -> bool:
    """A ``.gz`` too large to scan: astropy sizes a gzip stream by
    decompressing all of it, so only its primary header is read (by hand)."""
    try:
        return _is_gzip(path) and os.path.getsize(path) > GZIP_SCAN_BYTES
    except OSError:
        return False


def _gzip_primary_header(path: str) -> fits.Header:
    """The primary header of a gzip FITS, read block by block (no sizing)."""
    blocks: list[bytes] = []
    try:
        with gzip.open(path, "rb") as fh:
            for _ in range(10_000):                        # ≤ 28.8 MB of header
                block = fh.read(2880)
                if len(block) < 2880:
                    raise InspectError(422, "unreadable FITS: truncated primary header")
                blocks.append(block)
                cards = [block[i:i + 80] for i in range(0, 2880, 80)]
                if any(card[:8] == b"END     " for card in cards):
                    break
            else:
                raise InspectError(422, "unreadable FITS: no END card in the primary header")
    except (OSError, EOFError) as exc:
        raise InspectError(422, f"unreadable FITS: {exc}") from exc
    try:
        return fits.Header.fromstring(b"".join(blocks).decode("ascii", "replace"))
    except (ValueError, TypeError) as exc:
        raise InspectError(422, f"unreadable FITS header: {exc}") from exc


def _open(path: str) -> fits.HDUList:
    """Lazy, memory-mapped (unless gzip), unscaled — headers cost nothing."""
    try:
        return fits.open(path, memmap=not _is_gzip(path), lazy_load_hdus=True,
                         do_not_scale_image_data=True)
    except (OSError, ValueError, TypeError) as exc:
        raise InspectError(422, f"unreadable FITS: {exc}") from exc


def _hdu_at(hdul: fits.HDUList, index: int) -> Any:
    try:
        return hdul[index]
    except IndexError as exc:
        raise InspectError(404, f"no HDU {index} in this file") from exc
    except (OSError, ValueError) as exc:
        raise InspectError(422, f"HDU {index} is unreadable: {exc}") from exc


class _PrimaryHeader:
    """A primary header standing in for its HDU (shape from ``NAXISn``) — the
    large-gzip path, where astropy would decompress the file to size it."""

    kind_name = "PrimaryHDU"
    name = "PRIMARY"
    ver = 1

    def __init__(self, header: fits.Header):
        self.header = header

    @property
    def shape(self) -> tuple[int, ...]:
        n = int(self.header.get("NAXIS", 0) or 0)
        return tuple(int(self.header.get(f"NAXIS{k}", 0) or 0) for k in range(n, 0, -1))


def _is_image(hdu: Any) -> bool:
    return isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU, _PrimaryHeader))


def _is_table(hdu: Any) -> bool:
    return isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)) and not isinstance(hdu, fits.CompImageHDU)


# ---------------------------------------------------------------------------
# Header summaries
# ---------------------------------------------------------------------------

def _card_value(value: Any) -> str:
    text = "" if isinstance(value, fits.card.Undefined) else str(value)
    return text if len(text) <= CARD_VALUE_MAX else text[:CARD_VALUE_MAX - 1] + "…"


def header_cards(header: fits.Header) -> list[tuple[str, str, str]]:
    """``(keyword, value, comment)`` per card, values as display strings."""
    return [(str(card.keyword), _card_value(card.value), str(card.comment))
            for card in header.cards]


def _image_shape(hdu: Any) -> tuple[int, ...]:
    try:
        return tuple(int(n) for n in (hdu.shape or ()))
    except (TypeError, ValueError, KeyError):
        return ()


def _image_dtype(header: fits.Header) -> str | None:
    return _BITPIX_DTYPE.get(int(header.get("BITPIX", 0) or 0))


def _plane_axes(shape: tuple[int, ...]) -> tuple[int, ...]:
    return shape[:-2] if len(shape) >= 2 else ()


def plane_count(shape: tuple[int, ...]) -> int:
    """How many 2-D planes an image of numpy ``shape`` holds (0 below 2-D)."""
    if len(shape) < 2:
        return 0
    return int(np.prod(_plane_axes(shape), dtype=np.int64)) if len(shape) > 2 else 1


def plane_index(shape: tuple[int, ...], plane: int) -> tuple[int, ...]:
    """The leading (numpy-order) index of plane ``plane``."""
    axes = _plane_axes(shape)
    if not axes:
        return ()
    return tuple(int(i) for i in np.unravel_index(int(plane), axes))


def _bunit(header: fits.Header, primary: fits.Header | None) -> str:
    value = header.get("BUNIT")
    if value in (None, "") and primary is not None:
        value = primary.get("BUNIT")
    return str(value or "").strip()


def _band_list(header: fits.Header, primary: fits.Header | None, n: int) -> tuple[list[str] | None, bool]:
    """Band names of an ``n``-plane cube: the ``BANDS`` card (own, else the
    primary's), else the four Euclid bands for an unlabelled 4-plane cube
    (``assumed``); ``None`` when the planes are not bands."""
    for source in (header, primary):
        raw = (source or {}).get("BANDS")
        if raw:
            names = [b.strip() for b in str(raw).split(",") if b.strip()]
            if len(names) == n and all(b in BAND_NAMES for b in names):
                return names, False
    if n == len(BAND_NAMES):
        return list(BAND_NAMES), True
    return None, False


def _band_suffix(name: str) -> tuple[str, str] | None:
    """``(prefix, Euclid band)`` of an HDU name ending in a band: ``LR_VIS``,
    ``SR_J_E``, or a bare NISP letter (``Y``, ``LR_NISP_H``) → ``Y_E`` ..."""
    match = _BAND_SUFFIX.match(name or "")
    if match:
        band = match.group("band").upper()
        return (match.group("prefix"), band) if band in BAND_NAMES else None
    match = _NIR_SUFFIX.match(name or "")
    if match:
        band = f"{match.group('band')}_E"
        return (match.group("prefix"), band) if band in BAND_NAMES else None
    return None


def _band_of_filter(value: object) -> str | None:
    """A ``FILTER`` card value as a Euclid band (``VIS``, ``Y_E``, ``NISP_Y``, ``J``)."""
    filt = str(value or "").strip().upper()
    if filt in BAND_NAMES:
        return filt
    match = _FILTER_NIR.match(filt)
    if match:
        band = f"{match.group('band')}_E"
        return band if band in BAND_NAMES else None
    return None


def _band_of_2d(header: fits.Header, name: str) -> str | None:
    """The Euclid band of a 2-D image HDU (``FILTER`` card or name suffix)."""
    band = _band_of_filter(header.get("FILTER"))
    if band:
        return band
    suffix = _band_suffix(name)
    return suffix[1] if suffix else None


def celestial_wcs(header: fits.Header, shape2d: tuple[int, int],
                  primary: fits.Header | None = None) -> tuple[WCS | None, bool]:
    """``(celestial WCS, constructed)`` of an image HDU, or ``(None, False)``.

    A header without a celestial WCS whose primary carries ``RA``/``DEC`` and
    a ``PIXSCALE`` (own or the primary's) gets a north-up TAN centred on the
    grid (``constructed=True``; the poster FITS convention).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            wcs = WCS(header).celestial
            ctype = [str(c) for c in wcs.wcs.ctype]
            if wcs.has_celestial and wcs.naxis == 2 and all(ctype) and np.all(
                    np.isfinite(wcs.pixel_scale_matrix)):
                return wcs, False
        except Exception:  # noqa: BLE001 - heterogeneous archive headers
            pass
        ra = (primary or {}).get("RA", header.get("RA"))
        dec = (primary or {}).get("DEC", header.get("DEC"))
        scale = header.get("PIXSCALE") or (primary or {}).get("PIXSCALE")
        try:
            ra_f, dec_f, scale_f = float(ra), float(dec), float(scale)
        except (TypeError, ValueError):
            return None, False
        if not all(math.isfinite(v) for v in (ra_f, dec_f, scale_f)) or scale_f <= 0:
            return None, False
        ny, nx = shape2d
        built = WCS(naxis=2)
        built.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        built.wcs.crval = [ra_f, dec_f]
        built.wcs.crpix = [(nx + 1) / 2.0, (ny + 1) / 2.0]
        built.wcs.cd = np.array([[-scale_f / 3600.0, 0.0], [0.0, scale_f / 3600.0]])
        return built, True


def binned_wcs(wcs: WCS | None, factor: int, offset: float) -> WCS | None:
    """The WCS of a grid whose pixel ``j`` is centred on source pixel
    ``factor·j + offset`` (0-based): ``CRPIX → (CRPIX − 1 − offset)/factor + 1``,
    pixel matrix ×``factor``."""
    if wcs is None or (factor == 1 and offset == 0):
        return wcs
    out = wcs.deepcopy()
    out.wcs.crpix = (np.asarray(out.wcs.crpix, dtype=np.float64) - 1.0 - offset) / factor + 1.0
    if out.wcs.has_cd():
        out.wcs.cd = np.asarray(out.wcs.cd, dtype=np.float64) * factor
    else:
        out.wcs.cdelt = np.asarray(out.wcs.cdelt, dtype=np.float64) * factor
    out.wcs.set()
    return out


def wcs_summary(wcs: WCS, ny: int, nx: int, constructed: bool) -> dict[str, Any] | None:
    """Centre, scale, extent and footprint of an ``ny × nx`` grid."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ra, dec = wcs.pixel_to_world_values((nx - 1) / 2.0, (ny - 1) / 2.0)
            cx = [-0.5, nx - 0.5, nx - 0.5, -0.5]
            cy = [-0.5, -0.5, ny - 0.5, ny - 0.5]
            cra, cdec = wcs.pixel_to_world_values(cx, cy)
            matrix = np.asarray(wcs.pixel_scale_matrix, dtype=np.float64)
        except Exception:  # noqa: BLE001
            return None
    scale = math.sqrt(abs(float(np.linalg.det(matrix)))) * 3600.0
    ra_f, dec_f = float(ra) % 360.0, float(dec)
    if not (math.isfinite(ra_f) and math.isfinite(dec_f) and math.isfinite(scale) and scale > 0):
        return None
    pairs = zip(np.atleast_1d(cra), np.atleast_1d(cdec), strict=False)
    corners = [[float(a) % 360.0, float(d)] for a, d in pairs
               if math.isfinite(float(a)) and math.isfinite(float(d))]
    return {
        "ctype": [str(c) for c in wcs.wcs.ctype],
        "ra": ra_f, "dec": dec_f,
        "pixscale_arcsec": scale,
        "width_arcsec": nx * scale, "height_arcsec": ny * scale,
        "fov_deg": max(nx, ny) * scale / 3600.0,
        "corners": corners,
        "constructed": bool(constructed),
    }


def _table_columns(hdu: Any) -> list[dict[str, Any]]:
    out = []
    try:
        columns = hdu.columns
    except (KeyError, ValueError, AttributeError):
        return out
    for column in columns:
        fmt = str(column.format)
        out.append({
            "name": str(column.name), "format": fmt,
            "unit": str(column.unit) if column.unit else None,
            "dim": str(column.dim) if column.dim else None,
            "null": column.null if isinstance(column.null, (int, float, str)) else None,
        })
    return out


def _stamp(header: fits.Header) -> dict[str, Any] | None:
    try:
        stamp = read_stamp_cards(header)
    except (ValueError, TypeError, KeyError):
        return None
    return stamp.to_dict() if stamp is not None else None


def hdu_summary(hdu: Any, index: int, primary: fits.Header | None, *,
                cards: bool = False) -> dict[str, Any]:
    """One HDU from its header (no pixels): see the module docstring."""
    header = hdu.header
    name = str(hdu.name or f"HDU{index}")
    out: dict[str, Any] = {
        "hdu_index": index, "index": index, "name": name,
        "ver": int(getattr(hdu, "ver", 1) or 1),
        "kind": getattr(hdu, "kind_name", type(hdu).__name__),
        "type": "other", "shape": None, "dtype": None, "ndim": 0, "planes": 0,
        "plane_axes": [], "bunit": _bunit(header, primary), "wcs": None,
        "bands": None, "bands_assumed": False, "band": None,
        "compressed": isinstance(hdu, fits.CompImageHDU), "size_bytes": 0,
        "viewable": False, "reason": None,
    }
    if _is_image(hdu):
        shape = _image_shape(hdu)
        dtype = _image_dtype(header)
        out.update(shape=list(shape) if shape else None, dtype=dtype, ndim=len(shape))
        if shape and dtype:
            out["size_bytes"] = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
        bscale, bzero = header.get("BSCALE"), header.get("BZERO")
        if bscale not in (None, 1, 1.0) or bzero not in (None, 0, 0.0):
            out["scaling"] = {"bscale": float(bscale or 1.0), "bzero": float(bzero or 0.0)}
        if not shape:
            out["type"] = "empty"
            out["reason"] = "no data"
        elif len(shape) == 1:
            out["type"] = "vector"
            out["reason"] = "1-D data: shown as a plot"
        else:
            out["type"] = "image"
            planes = plane_count(shape)
            out.update(planes=planes, plane_axes=list(_plane_axes(shape)), viewable=True)
            ny, nx = shape[-2], shape[-1]
            wcs, constructed = celestial_wcs(header, (ny, nx), primary)
            if wcs is not None:
                out["wcs"] = wcs_summary(wcs, ny, nx, constructed)
            if len(shape) == 3:
                bands, assumed = _band_list(header, primary, shape[0])
                out.update(bands=bands, bands_assumed=assumed)
            elif len(shape) == 2:
                out["band"] = _band_of_2d(header, name)
            if out["compressed"] and out["size_bytes"] // max(planes, 1) > MAX_DECOMPRESS_BYTES:
                out.update(viewable=False, reason="compressed plane too large to decompress for display")
    elif _is_table(hdu):
        out["type"] = "table"
        out["columns"] = _table_columns(hdu)
        out["nrows"] = int(header.get("NAXIS2", 0) or 0)
        out["ncols"] = len(out["columns"])
        out["size_bytes"] = int(header.get("NAXIS1", 0) or 0) * out["nrows"]
        out["viewable"] = True
    if cards:
        out["cards"] = header_cards(header)
    return out


def band_groups(hdus: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sets of 2-D image HDUs named ``<prefix><band>`` (or with a ``FILTER``
    band) covering all four Euclid bands with one shape → one colour cube."""
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for hdu in hdus:
        if hdu.get("type") != "image" or hdu.get("ndim") != 2:
            continue
        suffix = _band_suffix(hdu["name"])
        band = hdu.get("band")
        if not band:
            continue
        prefix = suffix[0] if suffix else ""
        groups.setdefault(prefix.upper(), {}).setdefault(band, hdu)
    out = []
    for prefix, members in groups.items():
        if set(members) != set(BAND_NAMES):
            continue
        shapes = {tuple(members[b]["shape"]) for b in BAND_NAMES}
        if len(shapes) != 1:
            continue
        first = members[BAND_NAMES[0]]
        label = prefix.rstrip("_- ") or "bands"
        out.append({
            "id": f"b:{prefix}", "prefix": prefix, "label": f"{label} · 4-band colour",
            "hdus": [members[b]["index"] for b in BAND_NAMES], "bands": list(BAND_NAMES),
            "shape": list(first["shape"]), "wcs": first.get("wcs"),
            "bunit": first.get("bunit", ""),
        })
    return out


def file_summary(path: str, *, cards: bool = False) -> dict[str, Any]:
    """Every HDU's header summary + band groups + provenance stamp."""
    if _large_gzip(path):
        primary = _gzip_primary_header(path)
        row = hdu_summary(_PrimaryHeader(primary), 0, None, cards=cards)
        rows = [row]
        return {"hdus": rows, "band_groups": band_groups(rows),
                "scan_truncated": bool(primary.get("EXTEND", False)), "stamp": _stamp(primary)}
    hdus: list[dict[str, Any]] = []
    truncated = False
    with _open(path) as hdul:
        primary_hdu = _hdu_at(hdul, 0)
        primary = primary_hdu.header
        index = 0
        while True:
            if index >= MAX_HDUS:
                truncated = True
                break
            try:
                hdu = hdul[index]
            except IndexError:
                break
            except (OSError, ValueError) as exc:
                hdus.append({"hdu_index": index, "index": index, "name": f"HDU{index}",
                             "kind": "unreadable", "type": "other", "viewable": False,
                             "reason": str(exc), "shape": None, "dtype": None})
                truncated = True
                break
            hdus.append(hdu_summary(hdu, index, primary if index else None, cards=cards))
            index += 1
    return {"hdus": hdus, "band_groups": band_groups(hdus), "scan_truncated": truncated,
            "stamp": _stamp(primary)}


# ---------------------------------------------------------------------------
# Planes
# ---------------------------------------------------------------------------

@dataclass
class Plane:
    """One served 2-D plane (row 0 = FITS y = 1) and how it maps to the source."""

    data: np.ndarray
    bin: int
    offset: float
    method: str                       # "none" | "mean" | "stride"
    full_shape: tuple[int, int]
    shape: tuple[int, ...]            # the HDU's numpy shape
    index: tuple[int, ...]            # leading index of this plane
    header: fits.Header
    primary: fits.Header | None
    name: str


def _physical(raw: np.ndarray, header: fits.Header) -> np.ndarray:
    """``raw · BSCALE + BZERO`` as float32 with ``BLANK`` → NaN."""
    arr = np.asarray(raw)
    out = arr.astype(np.float32)
    bscale = float(header.get("BSCALE", 1.0) or 1.0)
    bzero = float(header.get("BZERO", 0.0) or 0.0)
    if bscale != 1.0 or bzero != 0.0:
        out = (arr.astype(np.float64) * bscale + bzero).astype(np.float32)
    blank = header.get("BLANK")
    if blank is not None and arr.dtype.kind in "iu":
        out[arr == blank] = np.nan
    return out


def choose_bin(ny: int, nx: int, requested: int | None, max_side: int) -> int:
    """The bin: ``requested`` (≥ 1) or the smallest one fitting ``max_side``,
    never below what :data:`MAX_OUTPUT_SIDE` needs."""
    need = max(1, math.ceil(max(ny, nx) / max(1, MAX_OUTPUT_SIDE)))
    auto = max(1, math.ceil(max(ny, nx) / max(1, max_side)))
    chosen = int(requested) if requested and requested > 0 else auto
    return max(chosen, need)


def read_plane(path: str, hdu_index: int, plane: int = 0, *, bin: int | None = None,
               max_side: int | None = None, sample: bool = False) -> Plane:
    """Plane ``plane`` of image HDU ``hdu_index``, binned for display.

    ``max_side`` (default :data:`MAX_VIEW_SIDE`) sets the auto bin;
    ``sample=True`` always uses a strided sample when binning (unbiased pixel
    statistics). 404 no such HDU/plane, 413 compressed plane too large, 415
    not an image (or 1-D).
    """
    max_side = MAX_VIEW_SIDE if max_side is None else max_side
    if _large_gzip(path):
        _check_large_gzip_plane(path, hdu_index)
    with _open(path) as hdul:
        primary = hdul[0].header if hdu_index else None
        hdu = _hdu_at(hdul, hdu_index)
        if not _is_image(hdu):
            raise InspectError(415, f"HDU {hdu_index} is not an image")
        shape = _image_shape(hdu)
        if len(shape) < 2:
            raise InspectError(415, f"HDU {hdu_index} has no 2-D image data (shape {list(shape)})")
        n_planes = plane_count(shape)
        if plane < 0 or plane >= n_planes:
            raise InspectError(404, f"plane {plane} is outside 0…{n_planes - 1}")
        ny, nx = shape[-2], shape[-1]
        header = hdu.header
        dtype = _image_dtype(header) or ">f4"
        plane_bytes = ny * nx * np.dtype(dtype).itemsize
        tile_compressed = isinstance(hdu, fits.CompImageHDU)
        if (tile_compressed or _is_gzip(path)) and plane_bytes > MAX_DECOMPRESS_BYTES:
            raise InspectError(413, (
                f"this plane is {plane_bytes / 2**20:.0f} MB compressed data — too large to "
                "decompress for display; download the file instead"))
        index = plane_index(shape, plane)
        factor = choose_bin(ny, nx, bin, max_side)
        try:
            source = hdu.section if tile_compressed else hdu.data
            if source is None:
                raise InspectError(404, f"HDU {hdu_index} has no data")
            if factor == 1:
                raw = np.asarray(source[index + (slice(None), slice(None))])
                data, offset, method = _physical(raw, header), 0.0, "none"
            elif ny * nx <= BLOCK_MEAN_MAX_PIXELS and not sample:
                hc, wc = (ny // factor) * factor, (nx // factor) * factor
                raw = np.asarray(source[index + (slice(0, hc), slice(0, wc))])
                phys = _physical(raw, header).reshape(hc // factor, factor, wc // factor, factor)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    data = np.nanmean(phys, axis=(1, 3)).astype(np.float32)
                offset, method = (factor - 1) / 2.0, "mean"
            else:
                start = factor // 2
                raw = np.asarray(source[index + (slice(start, None, factor), slice(start, None, factor))])
                data, offset, method = _physical(raw, header), float(start), "stride"
        except InspectError:
            raise
        except (OSError, ValueError, TypeError, MemoryError) as exc:
            raise InspectError(422, f"cannot read HDU {hdu_index}: {exc}") from exc
        return Plane(
            data=np.ascontiguousarray(data, dtype=np.float32), bin=factor, offset=offset,
            method=method, full_shape=(ny, nx), shape=shape, index=index,
            header=header.copy(), primary=primary.copy() if primary is not None else None,
            name=str(hdu.name or f"HDU{hdu_index}"),
        )


def _check_large_gzip_plane(path: str, hdu_index: int) -> None:
    """Refuse (413) what a large gzip file cannot serve cheaply, from its
    primary header alone: extensions, and planes over the decompress limit."""
    if hdu_index != 0:
        raise InspectError(413, "extensions of a gzip file this large are not scanned; "
                                "download the file instead")
    header = _gzip_primary_header(path)
    naxis = int(header.get("NAXIS", 0) or 0)
    if naxis >= 2:
        item = np.dtype(_image_dtype(header) or ">f4").itemsize
        plane_bytes = int(header.get("NAXIS1", 0)) * int(header.get("NAXIS2", 0)) * item
        if plane_bytes > MAX_DECOMPRESS_BYTES:
            raise InspectError(413, (
                f"this plane is {plane_bytes / 2**20:.0f} MB compressed data — too large to "
                "decompress for display; download the file instead"))


def plane_label(shape: tuple[int, ...], plane: int, bands: list[str] | None = None) -> str:
    """``"VIS"`` (a band plane), ``"plane 3"`` (3-D), ``"[1, 2]"`` (N-D)."""
    index = plane_index(shape, plane)
    if not index:
        return "image"
    if len(index) == 1:
        if bands and index[0] < len(bands):
            return bands[index[0]]
        return f"plane {index[0]}"
    return "[" + ", ".join(str(i) for i in index) + "]"


def _finite_or_none(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def array_stats(values: np.ndarray, *, bins: int = HIST_BINS) -> dict[str, Any]:
    """Counts, moments, robust σ, percentiles and a histogram of an array."""
    flat = np.asarray(values, dtype=np.float64).ravel()
    finite_mask = np.isfinite(flat)
    finite = flat[finite_mask]
    out: dict[str, Any] = {
        "n": int(flat.size), "n_finite": int(finite.size),
        "n_nan": int(np.isnan(flat).sum()),
        "n_posinf": int(np.isposinf(flat).sum()), "n_neginf": int(np.isneginf(flat).sum()),
        "n_zero": int((finite == 0).sum()), "n_negative": int((finite < 0).sum()),
        "min": None, "max": None, "mean": None, "std": None, "median": None,
        "mad_std": None, "sum": None, "percentiles": {}, "histogram": None,
    }
    if not finite.size:
        return out
    pct = np.percentile(finite, _PERCENTILES)
    median = float(np.median(finite))
    out.update(
        min=float(finite.min()), max=float(finite.max()), mean=float(finite.mean()),
        std=float(finite.std()), median=median,
        mad_std=float(1.4826 * np.median(np.abs(finite - median))),
        sum=float(finite.sum()),
        percentiles={f"{p:g}": float(v) for p, v in zip(_PERCENTILES, pct, strict=True)},
    )
    lo, hi = float(pct[0]), float(pct[-1])
    if not hi > lo:
        lo, hi = float(finite.min()), float(finite.max())
    if not hi > lo:
        hi = lo + 1.0
    counts, edges = np.histogram(finite, bins=bins, range=(lo, hi))
    out["histogram"] = {
        "edges": [float(e) for e in edges], "counts": [int(c) for c in counts],
        "below": int((finite < lo).sum()), "above": int((finite > hi).sum()),
    }
    return out


def plane_stats(path: str, hdu_index: int, plane: int = 0) -> dict[str, Any]:
    """Statistics of one plane at full resolution (a strided sample above
    :data:`MAX_STATS_PIXELS`: ``sampled`` names the stride)."""
    side = int(math.isqrt(MAX_STATS_PIXELS))
    served = read_plane(path, hdu_index, plane, max_side=side, sample=True)
    stats = array_stats(served.data)
    stats.update(
        hdu=hdu_index, plane=plane, index=list(served.index), shape=list(served.full_shape),
        sampled=served.bin if served.bin > 1 else None,
    )
    if served.bin > 1:
        stats["sum"] = None       # a sample's sum is not the plane's
    return stats


def vector_series(path: str, hdu_index: int) -> dict[str, Any]:
    """A 1-D image HDU as ``{x, y}`` (≤ :data:`VECTOR_MAX_POINTS`, strided) + stats."""
    with _open(path) as hdul:
        hdu = _hdu_at(hdul, hdu_index)
        if not _is_image(hdu) or len(_image_shape(hdu)) != 1:
            raise InspectError(415, f"HDU {hdu_index} is not a 1-D image")
        n = _image_shape(hdu)[0]
        step = max(1, math.ceil(n / VECTOR_MAX_POINTS))
        try:
            raw = np.asarray(hdu.data[::step])
        except (OSError, ValueError, TypeError) as exc:
            raise InspectError(422, f"cannot read HDU {hdu_index}: {exc}") from exc
        values = _physical(raw, hdu.header)
    x = (np.arange(values.size) * step).tolist()
    y = [_finite_or_none(v) for v in values.tolist()]
    stats = array_stats(values)
    stats.update(hdu=hdu_index, n_points=n, step=step, series={"x": x, "y": y})
    return stats


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _column_kind(column: np.ndarray) -> str:
    arr = np.asarray(column)
    if arr.dtype.kind == "O":
        return "array"
    if arr.ndim > 1:
        return "array"
    if arr.dtype.kind == "b":
        return "bool"
    if arr.dtype.kind in "iuf":
        return "numeric"
    return "text"


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("ascii", "replace").rstrip()
    return str(value).rstrip()


def _cells(column: np.ndarray, kind: str) -> list[Any]:
    """JSON-safe cells of a (row-selected) column: non-finite floats → None,
    bytes decoded, arrays ≤ 16 values as lists, larger ones as a preview."""
    if kind == "numeric":
        arr = np.asarray(column)
        if arr.dtype.kind == "f":
            return [v if math.isfinite(v) else None for v in arr.astype(np.float64).tolist()]
        return arr.tolist()
    if kind == "bool":
        return [bool(v) for v in np.asarray(column).tolist()]
    if kind == "text":
        return [_text(v) for v in np.asarray(column).tolist()]
    out: list[Any] = []
    for value in column:
        arr = np.asarray(value)
        if arr.dtype.kind in "iufb" and arr.size <= 16:
            out.append([_finite_or_none(v) if arr.dtype.kind == "f" else v for v in arr.ravel().tolist()])
        elif arr.dtype.kind in "iuf" and arr.size:
            head = ", ".join(f"{float(v):.4g}" for v in arr.ravel()[:3])
            out.append(f"[{head}, … ×{arr.size}]")
        else:
            out.append(_text(value) if arr.ndim == 0 else f"[{arr.size} values]")
    return out


def _table_hdu(hdul: fits.HDUList, hdu_index: int) -> Any:
    hdu = _hdu_at(hdul, hdu_index)
    if not _is_table(hdu):
        raise InspectError(415, f"HDU {hdu_index} is not a table")
    return hdu


def _sort_order(column: np.ndarray, desc: bool) -> np.ndarray:
    """Stable order; NaN / empty always last (either direction)."""
    arr = np.asarray(column)
    if arr.dtype.kind == "f":
        nan = ~np.isfinite(arr)
        idx = np.flatnonzero(~nan)
        order = idx[np.argsort(arr[idx], kind="stable")]
        if desc:
            order = order[::-1]
        return np.concatenate([order, np.flatnonzero(nan)])
    if arr.dtype.kind in "SU":
        arr = np.char.lower(np.char.strip(arr.astype("U")))
    order = np.argsort(arr, kind="stable")
    return order[::-1] if desc else order


def table_page(path: str, hdu_index: int, *, offset: int = 0, limit: int = 200,
               sort: str | None = None, desc: bool = False) -> dict[str, Any]:
    """Rows ``offset … offset+limit`` (optionally sorted by one column)."""
    limit = max(1, min(int(limit), TABLE_PAGE_MAX))
    offset = max(0, int(offset))
    with _open(path) as hdul:
        hdu = _table_hdu(hdul, hdu_index)
        columns = _table_columns(hdu)
        try:
            data = hdu.data
        except (OSError, ValueError, TypeError) as exc:
            raise InspectError(422, f"cannot read table HDU {hdu_index}: {exc}") from exc
        total = 0 if data is None else len(data)
        names = [c["name"] for c in columns]
        kinds: dict[str, str] = {}
        if data is not None:
            for name in names:
                kinds[name] = _column_kind(data[name])
        for column in columns:
            column["kind"] = kinds.get(column["name"], "text")
        if sort:
            if sort not in names:
                raise InspectError(400, f"unknown column {sort!r}")
            if kinds.get(sort) == "array":
                raise InspectError(400, f"cannot sort by the array column {sort!r}")
        if data is None or offset >= total:
            rows_index = np.zeros(0, dtype=np.int64)
        elif sort:
            rows_index = _sort_order(data[sort], desc)[offset:offset + limit]
        else:
            rows_index = np.arange(offset, min(offset + limit, total), dtype=np.int64)
        by_column = [_cells(data[name][rows_index], kinds[name]) for name in names] if len(rows_index) else []
    rows = [list(r) for r in zip(*by_column, strict=True)] if by_column else []
    return {
        "hdu": hdu_index, "total": total, "offset": offset, "limit": limit,
        "sort": sort or None, "desc": bool(desc), "columns": columns,
        "rows": rows, "row_index": [int(i) for i in rows_index],
    }


def _column_stats(column: np.ndarray, kind: str, total: int) -> dict[str, Any]:
    arr = np.asarray(column)
    out: dict[str, Any] = {"kind": kind, "n": int(total)}
    if kind == "numeric":
        stats = array_stats(arr, bins=32)
        out.update({k: stats[k] for k in ("n_finite", "n_nan", "min", "max", "mean", "std",
                                          "median", "mad_std", "percentiles", "histogram")})
        out["n_null"] = stats["n"] - stats["n_finite"]
        if arr.dtype.kind in "iu" and arr.size <= TABLE_UNIQUE_MAX_ROWS:
            out["n_unique"] = int(np.unique(arr).size)
    elif kind == "bool":
        out["n_true"] = int(np.count_nonzero(arr))
        out["n_false"] = int(arr.size - np.count_nonzero(arr))
    elif kind == "text":
        sample = arr[:TABLE_UNIQUE_MAX_ROWS]
        values, counts = np.unique(np.char.strip(sample.astype("U")), return_counts=True)
        order = np.argsort(-counts, kind="stable")[:8]
        out["n_unique"] = int(values.size)
        out["n_empty"] = int(counts[values == ""].sum()) if (values == "").any() else 0
        out["top"] = [[str(values[i]), int(counts[i])] for i in order]
        out["unique_sampled"] = bool(arr.size > TABLE_UNIQUE_MAX_ROWS)
    else:
        out["shape"] = list(arr.shape[1:]) if arr.dtype.kind != "O" else None
    return out


def table_stats(path: str, hdu_index: int) -> dict[str, Any]:
    """Per-column statistics (a strided sample above :data:`TABLE_STATS_MAX_ROWS`)."""
    with _open(path) as hdul:
        hdu = _table_hdu(hdul, hdu_index)
        columns = _table_columns(hdu)
        try:
            data = hdu.data
        except (OSError, ValueError, TypeError) as exc:
            raise InspectError(422, f"cannot read table HDU {hdu_index}: {exc}") from exc
        total = 0 if data is None else len(data)
        step = max(1, math.ceil(total / TABLE_STATS_MAX_ROWS)) if total else 1
        out = []
        for column in columns:
            name = column["name"]
            if data is None:
                out.append({"name": name, "kind": "text", "n": 0})
                continue
            values = data[name][::step]
            kind = _column_kind(values)
            out.append({"name": name, "unit": column["unit"], **_column_stats(values, kind, total)})
    return {"hdu": hdu_index, "total": total, "sampled": step if step > 1 else None, "columns": out}


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def _load_json(path: str) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as fh:
            value = json.load(fh)
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _same_file(record_path: Any, real: str, base_dirs: tuple[str, ...]) -> bool:
    if not isinstance(record_path, str) or not record_path:
        return False
    if os.path.isabs(record_path):
        return os.path.realpath(record_path) == real
    return any(os.path.realpath(os.path.join(base, record_path)) == real for base in base_dirs)


def _checkpoint_records() -> dict[str, dict[str, Any]]:
    """Model ids of the checkpoints (``<ckpt root>/[*/]*/provenance.json``)."""
    root = os.path.dirname(os.path.realpath(Config.DEFAULT_CHECKPOINT_DIR))
    out: dict[str, dict[str, Any]] = {}
    for pattern in ("*/provenance.json", "*/*/provenance.json"):
        for path in glob.glob(os.path.join(root, pattern)):
            record = _load_json(path)
            if record and isinstance(record.get("id"), str):
                out.setdefault(record["id"], {"file": _safe_relpath(path),
                                              "checkpoint": _safe_relpath(os.path.dirname(path)),
                                              "kind": "checkpoint", "record": record})
    return out


def _lookup(pid: str, search_dirs: tuple[str, ...],
            checkpoints: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if not _PROV_ID.match(pid):
        return None
    for directory in search_dirs:
        for path in sorted(glob.glob(os.path.join(glob.escape(directory), f"{pid}.*.json"))):
            match = _SIDECAR.match(os.path.basename(path))
            record = _load_json(path)
            if match and record is not None:
                return {"file": _safe_relpath(path), "kind": match.group(2), "record": record}
    return checkpoints.get(pid)


def provenance(path: str) -> dict[str, Any]:
    """The file's stamp, the sidecars that describe it and its related records."""
    real = os.path.realpath(path)
    directory = os.path.dirname(real)
    stamp = None
    with contextlib.suppress(InspectError):
        if _large_gzip(real):
            stamp = _stamp(_gzip_primary_header(real))
        else:
            with _open(real) as hdul:
                stamp = _stamp(_hdu_at(hdul, 0).header)
    base_dirs = (os.getcwd(), os.fspath(os.path.dirname(os.path.realpath(Config.DATA_DIR))))
    sidecars = []
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        names = []
    for name in names:
        match = _SIDECAR.match(name)
        if not match:
            continue
        record = _load_json(os.path.join(directory, name))
        if record is None:
            continue
        is_self = stamp is not None and record.get("id") == stamp.get("id")
        if not is_self and not _same_file(record.get("path"), real, base_dirs):
            continue
        sidecars.append({
            "file": _safe_relpath(os.path.join(directory, name)), "id": match.group(1),
            "kind": match.group(2), "current": bool(is_self), "record": record,
        })
    # The current record first, then earlier runs newest first.
    sidecars.sort(key=lambda s: str(s["record"].get("created_at", "")), reverse=True)
    sidecars.sort(key=lambda s: not s["current"])
    current = next((s for s in sidecars if s["current"]), None)
    wanted: list[tuple[str, str]] = []
    if stamp is not None:
        if stamp.get("produced_by"):
            wanted.append(("produced_by", stamp["produced_by"]))
        wanted.extend(("parent", p) for p in stamp.get("parents") or [])
    if current is not None:
        rec = current["record"]
        if rec.get("produced_by"):
            wanted.append(("produced_by", str(rec["produced_by"])))
        wanted.extend(("parent", str(p)) for p in rec.get("parents") or [])
    related = []
    seen: set[str] = set()
    search_dirs = (os.path.realpath(Config.PROV_DIR), directory)
    checkpoints: dict[str, dict[str, Any]] | None = None
    for role, pid in wanted:
        if pid in seen:
            continue
        seen.add(pid)
        if checkpoints is None:
            checkpoints = _checkpoint_records()
        found = _lookup(pid, search_dirs, checkpoints)
        related.append({"role": role, "id": pid, **(found or {"file": None, "kind": None, "record": None})})
    return {"stamp": stamp, "sidecars": sidecars, "related": related,
            "stale_sidecars": sum(1 for s in sidecars if not s["current"]) if stamp else 0}
