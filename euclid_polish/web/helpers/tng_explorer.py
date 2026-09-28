"""The TNG50 property explorer of Synthetic › Galaxies (templates) (``GET /api/tng/properties``).

Reads the local calibration CSVs only (no SSH, no TNG API, no writes):

* ``data/_tng_infographics/tng_properties.csv`` — per galaxy: SFR [M☉/yr],
  stellar mass [M☉], total bound ("halo") mass [M☉], group-catalogue stellar
  half-mass radius [kpc] (``tng.properties``, from the TNG API);
* ``tng_atlas_parameters.csv`` (+ ``.meta.json``) — per galaxy × viewpoint:
  the measured VIS effective radius of the SKIRT frame (``native_re_px`` /
  ``native_re_kpc``) that population generation matches to COSMOS.

Rows: ``[id, sfr, mass_stars, m_halo, reff, re_kpc, re_kpc_min, re_kpc_max,
n_orient, local]`` with ``re_kpc`` the mean measured radius over the
viewpoints; ``orientations[id] = [[orientation, native_re_px,
native_re_kpc], …]``; ``local`` = SKIRT frames of that galaxy on this
machine (``data/tng_skirt/<id>/``).
"""
from __future__ import annotations

import csv
import json
import math
import os
import threading
from typing import Any

from euclid_polish.config import Config

CALIBRATION_SUBDIR = "_tng_infographics"
PROPERTIES_CSV = "tng_properties.csv"
ATLAS_CSV = "tng_atlas_parameters.csv"
COLUMNS = ("id", "sfr", "mass_stars", "m_halo", "reff", "re_kpc", "re_kpc_min", "re_kpc_max",
           "n_orient", "local")

_LOCK = threading.Lock()
_MEMO: dict[str, Any] = {}


def calibration_dir() -> str:
    return os.path.join(Config.DATA_DIR, CALIBRATION_SUBDIR)


def _num(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _stat(path: str) -> tuple[int, int] | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return int(st.st_size), int(st.st_mtime_ns)


def _file_info(path: str, rows: int | None) -> dict[str, Any]:
    try:
        st = os.stat(path)
    except OSError:
        return {"present": False, "name": os.path.basename(path), "rows": 0, "mtime": None}
    return {"present": True, "name": os.path.basename(path), "rows": rows,
            "mtime": float(st.st_mtime), "size_bytes": int(st.st_size)}


def _read_rows(path: str) -> list[dict[str, str]]:
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def local_galaxies(tng_dir: str | None = None) -> dict[str, int]:
    """``subhalo id → number of local SKIRT FITS frames``."""
    root = tng_dir or Config.TNG_SKIRT_DIR
    out: dict[str, int] = {}
    try:
        names = os.listdir(root)
    except OSError:
        return out
    for name in names:
        folder = os.path.join(root, name)
        if not name.isdigit() or not os.path.isdir(folder):
            continue
        try:
            frames = [f for f in os.listdir(folder) if f.lower().endswith(".fits")]
        except OSError:
            continue
        if frames:
            out[name] = len(frames)
    return out


def build_payload(properties: list[dict[str, str]], atlas: list[dict[str, str]],
                  local: dict[str, int]) -> dict[str, Any]:
    """The explorer rows from the two CSVs (pure)."""
    orientations: dict[str, list[list[float | int | None]]] = {}
    for row in atlas:
        gid = str(row.get("subhalo_id") or "").strip()
        if not gid:
            continue
        orient = _num(row.get("orientation"))
        orientations.setdefault(gid, []).append([
            int(orient) if orient is not None else None,
            _num(row.get("native_re_px")), _num(row.get("native_re_kpc")),
        ])
    for values in orientations.values():
        values.sort(key=lambda v: (v[0] is None, v[0] or 0))
    by_id: dict[str, dict[str, Any]] = {}
    for row in properties:
        gid = str(row.get("id") or "").strip()
        if gid:
            by_id[gid] = row
    # Galaxies measured in the atlas but missing from the property cache keep
    # their atlas copies of the TNG properties.
    atlas_props: dict[str, dict[str, Any]] = {}
    for row in atlas:
        gid = str(row.get("subhalo_id") or "").strip()
        if gid and gid not in atlas_props:
            atlas_props[gid] = {"sfr": row.get("sfr_msun_yr"), "mass_stars": row.get("mass_stars_msun"),
                                "m_halo": row.get("m_halo_msun"), "reff": row.get("groupcat_reff_kpc")}
    ids = sorted(set(by_id) | set(orientations),
                 key=lambda g: (not g.isdigit(), int(g) if g.isdigit() else 0, g))
    rows = []
    n_quenched = n_missing = 0
    for gid in ids:
        props = by_id.get(gid) or atlas_props.get(gid) or {}
        radii = [v[2] for v in orientations.get(gid, []) if v[2] is not None]
        sfr = _num(props.get("sfr"))
        if sfr is None:
            n_missing += 1
        elif sfr == 0:
            n_quenched += 1
        rows.append([
            int(gid) if gid.isdigit() else gid, sfr, _num(props.get("mass_stars")),
            _num(props.get("m_halo")), _num(props.get("reff")),
            sum(radii) / len(radii) if radii else None,
            min(radii) if radii else None, max(radii) if radii else None,
            len(orientations.get(gid, [])), int(local.get(gid, 0)),
        ])
    return {
        "columns": list(COLUMNS), "rows": rows, "orientations": orientations,
        "summary": {"n": len(rows), "n_quenched": n_quenched, "n_missing_sfr": n_missing,
                    "n_in_atlas": sum(1 for gid in ids if gid in orientations),
                    "n_local": sum(1 for gid in ids if local.get(gid))},
    }


def properties_payload() -> dict[str, Any]:
    """``GET /api/tng/properties`` (memoised per CSV state)."""
    root = calibration_dir()
    props_path = os.path.join(root, PROPERTIES_CSV)
    atlas_path = os.path.join(root, ATLAS_CSV)
    local = local_galaxies()
    key = (_stat(props_path), _stat(atlas_path), tuple(sorted(local.items())))
    with _LOCK:
        if _MEMO.get("key") == key:
            return _MEMO["payload"]
    properties, atlas = _read_rows(props_path), _read_rows(atlas_path)
    meta = None
    try:
        with open(atlas_path + ".meta.json") as handle:
            meta = json.load(handle)
    except (OSError, ValueError):
        meta = None
    payload = {
        "present": bool(properties or atlas),
        "files": {"properties": _file_info(props_path, len(properties)),
                  "atlas": _file_info(atlas_path, len(atlas))},
        "atlas_meta": meta if isinstance(meta, dict) else None,
        **build_payload(properties, atlas, local),
    }
    with _LOCK:
        _MEMO["key"], _MEMO["payload"] = key, payload
    return payload


__all__ = ["ATLAS_CSV", "COLUMNS", "PROPERTIES_CSV", "build_payload", "calibration_dir",
           "local_galaxies", "properties_payload"]
