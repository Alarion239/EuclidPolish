"""The star catalogue explorer of Data › Catalog (``GET /api/catalog/stars``).

Serves the synchronised FASRC-mirror ``stars.csv`` (the brightest-N query's
43k-star catalogue on netscratch, pulled into ``data/_fasrc_cache``) as
compact columnar rows — **never** the stale 200-row ``data/euclid_stars``
copy. Cache-only: no SSH, no rsync, works offline; the explicit refresh is
``POST /api/status/refresh-catalog``.

Each row: ``[id, ra, dec, mag, flux_uJy, fluxerr_uJy, field, b_VIS, b_Y_E,
b_J_E, b_H_E, nav]``. A band code packs the cutout status of that band:
bit 0 valid (a cutout at some size passed validation), bit 1 corrupted
(downloaded but rejected: NaN/Inf, all-zero or constant pixels, unopenable),
bit 2 download failed (no mosaic tile matched, or bad coordinates); bits
``3+i``: valid at ``sizes[i]``. No bit = pending (never attempted).
``band_stats`` and the ``summary`` states are exclusive (valid > corrupted >
failed > pending): every band row, and the summary, sum to the star count;
the summary state of a star is its best band.
``nav`` = 1 when the star is in the cutouts navigator (valid in all four
bands at the navigator's common size).
"""
from __future__ import annotations

import math
import os
import threading
from collections.abc import Iterable
from typing import Any

from euclid_polish.catalog.catalog_object import CatalogObject
from euclid_polish.config import Config
from euclid_polish.sky.observation.q1_fields import q1_field_for
from euclid_polish.web.helpers.status import (
    _cached_fasrc_catalog_dir,
    _valid_4band_select,
    catalog_cache_info,
    read_catalog_objects,
)

BIT_VALID, BIT_CORRUPTED, BIT_FAILED = 1, 2, 4
SIZE_SHIFT = 3
COLUMNS = ("id", "ra", "dec", "mag", "flux_uJy", "fluxerr_uJy", "field",
           "b_VIS", "b_Y_E", "b_J_E", "b_H_E", "nav")

_LOCK = threading.Lock()
_MEMO: dict[str, tuple[tuple[int, int], dict[str, Any]]] = {}


def _band_names() -> list[str]:
    return [band.name for band in Config.BANDS]


def _sizes(objects: Iterable[CatalogObject]) -> list[int]:
    sizes: set[int] = set()
    for obj in objects:
        for kind in ("valid", "corrupted", "download_failed"):
            for per_size in (obj.flags.get(kind) or {}).values():
                if isinstance(per_size, dict):
                    sizes.update(int(s) for s, ok in per_size.items() if ok)
    return sorted(sizes)


def band_code(obj: CatalogObject, band: str, sizes: list[int]) -> int:
    code = 0
    valid = (obj.flags.get("valid") or {}).get(band) or {}
    if any(valid.values()):
        code |= BIT_VALID
    if any(((obj.flags.get("corrupted") or {}).get(band) or {}).values()):
        code |= BIT_CORRUPTED
    if any(((obj.flags.get("download_failed") or {}).get(band) or {}).values()):
        code |= BIT_FAILED
    for i, size in enumerate(sizes):
        if valid.get(str(size)):
            code |= 1 << (SIZE_SHIFT + i)
    return code


_STATE_ORDER = ("valid", "corrupted", "failed", "pending")


def code_state(code: int) -> str:
    """The one exclusive state of a band code: valid > corrupted > failed > pending.

    A band can carry several bits at once (a rejected 255 px cutout and a
    valid 511 px one); it counts once, under its best outcome — the same rule
    as the explorer's ``bandState`` so the band table matches the filters.
    """
    if code & BIT_VALID:
        return "valid"
    if code & BIT_CORRUPTED:
        return "corrupted"
    if code & BIT_FAILED:
        return "failed"
    return "pending"


def star_state(codes: Iterable[int]) -> str:
    """A star's overall state: its best band (valid in any band wins)."""
    states = {code_state(code) for code in codes}
    return next((st for st in _STATE_ORDER if st in states), "pending")


def _round(value: float | None, digits: int) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), digits)


def build_payload(objects: list[CatalogObject]) -> dict[str, Any]:
    """The explorer payload for ``objects`` (pure; the route adds freshness)."""
    bands = _band_names()
    sizes = _sizes(objects)
    nav_size, nav_objects = _valid_4band_select(objects)
    nav_ids = {int(o.id) for o in nav_objects if o.id is not None}
    rows: list[list[Any]] = []
    per_band = {b: {"valid": 0, "corrupted": 0, "failed": 0, "pending": 0,
                    "by_size": {str(s): 0 for s in sizes}} for b in bands}
    status = {"valid": 0, "corrupted": 0, "failed": 0, "pending": 0}
    valid_all4 = 0
    mags: list[float] = []
    for obj in objects:
        codes = [band_code(obj, band, sizes) for band in bands]
        for band, code in zip(bands, codes, strict=True):
            stats = per_band[band]
            stats[code_state(code)] += 1
            for i, size in enumerate(sizes):
                if code & (1 << (SIZE_SHIFT + i)):
                    stats["by_size"][str(size)] += 1
        status[star_state(codes)] += 1
        valid_all4 += int(all(code & BIT_VALID for code in codes))
        ra, dec = _round(obj.ra, 6), _round(obj.dec, 6)
        mag = _round(obj.magnitude, 4)
        if mag is not None:
            mags.append(mag)
        field = q1_field_for(ra, dec) if ra is not None and dec is not None else None
        rows.append([
            int(obj.id) if obj.id is not None else None, ra, dec, mag,
            _round(obj.flux_psf_uJy, 3), _round(obj.fluxerr_psf_uJy, 4), field or "",
            *codes, int(obj.id is not None and int(obj.id) in nav_ids),
        ])
    return {
        "columns": list(COLUMNS),
        "rows": rows,
        "bands": bands,
        "sizes": sizes,
        "bits": {"valid": BIT_VALID, "corrupted": BIT_CORRUPTED, "failed": BIT_FAILED,
                 "size_shift": SIZE_SHIFT},
        "summary": {
            "total": len(objects), **status, "valid_all4": valid_all4,
            "navigator": {"size": nav_size, "count": len(nav_ids)},
            "mag_min": min(mags) if mags else None, "mag_max": max(mags) if mags else None,
        },
        "band_stats": [{"band": band, **per_band[band]} for band in bands],
    }


def stars_payload() -> dict[str, Any]:
    """``GET /api/catalog/stars``: the mirror's rows + freshness, or
    ``present: false`` (never a fallback to the stale local copy)."""
    info = catalog_cache_info()
    directory = _cached_fasrc_catalog_dir()
    if directory is None:
        return {"present": False, "source": "fasrc-mirror", **info, "columns": list(COLUMNS),
                "rows": [], "bands": _band_names(), "sizes": [], "summary": None,
                "band_stats": []}
    path = os.path.join(directory, Config.CATALOG_FILE)
    st = os.stat(path)
    key = (int(st.st_size), int(st.st_mtime_ns))
    real = os.path.realpath(path)
    with _LOCK:
        hit = _MEMO.get(real)
    if hit is not None and hit[0] == key:
        payload = hit[1]
    else:
        payload = build_payload(read_catalog_objects(path))
        with _LOCK:
            _MEMO.clear()
            _MEMO[real] = (key, payload)
    return {"present": True, "source": "fasrc-mirror", **info, **payload}


def stars_by_id() -> dict[int, dict[str, Any]]:
    """``id → {ra, dec, mag}`` of the mirror (for the cutouts gallery)."""
    directory = _cached_fasrc_catalog_dir()
    if directory is None:
        return {}
    out: dict[int, dict[str, Any]] = {}
    for obj in read_catalog_objects(os.path.join(directory, Config.CATALOG_FILE)):
        if obj.id is None:
            continue
        out[int(obj.id)] = {"ra": _round(obj.ra, 6), "dec": _round(obj.dec, 6),
                            "mag": _round(obj.magnitude, 4)}
    return out


__all__ = ["BIT_CORRUPTED", "BIT_FAILED", "BIT_VALID", "COLUMNS", "SIZE_SHIFT", "band_code",
           "build_payload", "code_state", "star_state", "stars_by_id", "stars_payload"]
