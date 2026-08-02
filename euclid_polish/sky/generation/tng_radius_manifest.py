"""Validated native TNG VIS half-light-radius manifests.

The SKIRT atlas is remote and expensive to scan during every field draw.  This
module turns the scan into an explicit, fingerprinted prerequisite: generation
may use only entries from a manifest whose atlas inventory and property cache
still match the files that were measured.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from euclid_polish.config import Config
from euclid_polish.sky.generation.tng_galaxy import (
    N_ORIENTATIONS,
    list_tng_galaxies,
    load_tng_frame,
    tng_fits_path,
)
from euclid_polish.skirt.image import measure_halflight_radius_px
from euclid_polish.sky.generation.redshift_model import load_tng_properties

MANIFEST_VERSION = 1
ALGORITHM_VERSION = "centered-vis-cog-v1"
DEFAULT_MANIFEST_NAME = "tng_radius_manifest.json"


def manifest_path(tng_dir: str, path: str | None = None) -> str:
    return str(path or os.path.join(
        Config.DATA_DIR, "_tng_infographics", DEFAULT_MANIFEST_NAME
    ))


def _file_identity(path: str) -> dict[str, int | str]:
    stat = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _property_identity(path: str | None) -> dict[str, int | str | None]:
    if not path or not os.path.isfile(path):
        return {"path": os.path.abspath(path) if path else None,
                "size": 0, "mtime_ns": 0}
    return _file_identity(path)


def _fingerprint(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _inventory(
    tng_dir: str,
    galaxies: list[tuple[str, str]],
    properties_path: str | None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for gdir, gid in galaxies:
        for orientation in range(1, N_ORIENTATIONS + 1):
            for band in ("VIS", "Y", "J", "H"):
                path = tng_fits_path(gdir, gid, orientation, band)
                if os.path.isfile(path):
                    entries.append(_file_identity(path))
                else:
                    entries.append({"path": os.path.abspath(path),
                                    "size": 0, "mtime_ns": 0})
    entries.sort(key=lambda row: str(row["path"]))
    return entries


def build_manifest(
    tng_dir: str,
    *,
    properties_path: str | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Measure every completed atlas orientation and return a report.

    A report is written atomically when ``output_path`` is supplied, including
    invalid entries, so the UI can show exactly why a job is not submit-ready.
    """
    galaxies = list_tng_galaxies(tng_dir)
    properties_path = properties_path or os.path.join(
        Config.DATA_DIR, "_tng_infographics", "tng_properties.csv"
    )
    properties = load_tng_properties(properties_path)
    entries: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for gdir, gid in galaxies:
        props = properties.get(str(gid), {})
        mass = float(props.get("mass_stars", float("nan")))
        has_properties = bool(np.isfinite(mass) and mass > 0.0)
        for orientation in range(1, N_ORIENTATIONS + 1):
            vis_path = tng_fits_path(gdir, gid, orientation, "VIS")
            row: dict[str, Any] = {
                "subhalo_id": str(gid),
                "orientation": int(orientation),
                "vis_path": os.path.abspath(vis_path),
                "valid": False,
            }
            try:
                identity = _file_identity(vis_path)
                frame = load_tng_frame(vis_path)
                re_px = float(measure_halflight_radius_px(frame))
                if not has_properties:
                    raise ValueError("missing finite positive mass_stars property")
                if not np.isfinite(re_px) or re_px <= 0.0:
                    raise ValueError("VIS curve-of-growth returned no positive R_e")
                row.update({
                    "valid": True,
                    "native_re_px": re_px,
                    "shape": [int(frame.shape[0]), int(frame.shape[1])],
                    "vis_file": identity,
                })
            except (OSError, ValueError, TypeError) as exc:
                row["error"] = str(exc)
                failures.append({
                    "subhalo_id": str(gid), "orientation": str(orientation),
                    "error": str(exc),
                })
            entries.append(row)

    inventory = _inventory(tng_dir, galaxies, properties_path)
    inventory_fingerprint = _fingerprint({
        "algorithm_version": ALGORITHM_VERSION,
        "inventory": inventory,
        "properties": _property_identity(properties_path),
    })
    valid_count = sum(bool(row.get("valid")) for row in entries)
    expected_count = len(galaxies) * N_ORIENTATIONS
    report: dict[str, Any] = {
        "version": MANIFEST_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "tng_dir": os.path.abspath(tng_dir),
        "properties_file": _property_identity(properties_path),
        "atlas_inventory_fingerprint": inventory_fingerprint,
        "expected_count": int(expected_count),
        "valid_count": int(valid_count),
        "failed_count": int(len(failures)),
        "valid": bool(expected_count > 0 and valid_count == expected_count),
        "failures": failures,
        "entries": entries,
    }
    report["manifest_fingerprint"] = _fingerprint({
        key: report[key] for key in (
            "version", "algorithm_version", "properties_file",
            "atlas_inventory_fingerprint", "expected_count", "valid_count",
            "entries",
        )
    })
    if output_path:
        write_manifest(output_path, report)
    return report


def write_manifest(path: str, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    os.replace(tmp, target)


def load_manifest(path: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def validate_manifest(
    tng_dir: str,
    *,
    properties_path: str | None = None,
    manifest: dict[str, Any] | None = None,
    manifest_path_value: str | None = None,
) -> dict[str, Any]:
    """Cheap submit-time validation against file inventory and manifest rows."""
    properties_path = properties_path or os.path.join(
        Config.DATA_DIR, "_tng_infographics", "tng_properties.csv"
    )
    path = manifest_path(tng_dir, manifest_path_value)
    payload = manifest if manifest is not None else load_manifest(path)
    if payload is None:
        return {"valid": False, "reason": f"missing radius manifest: {path}",
                "path": path}
    galaxies = list_tng_galaxies(tng_dir)
    inventory = _inventory(tng_dir, galaxies, properties_path)
    current_fp = _fingerprint({
        "algorithm_version": ALGORITHM_VERSION,
        "inventory": inventory,
        "properties": _property_identity(properties_path),
    })
    reasons: list[str] = []
    if payload.get("version") != MANIFEST_VERSION:
        reasons.append("unsupported radius-manifest version")
    if payload.get("algorithm_version") != ALGORITHM_VERSION:
        reasons.append("radius algorithm version changed")
    if not payload.get("valid"):
        reasons.append("manifest contains invalid or incomplete measurements")
    if payload.get("atlas_inventory_fingerprint") != current_fp:
        reasons.append("atlas files or TNG properties changed since measurement")
    expected = len(galaxies) * N_ORIENTATIONS
    try:
        manifest_expected = int(payload.get("expected_count", -1))
    except (TypeError, ValueError):
        manifest_expected = -1
    try:
        manifest_valid = int(payload.get("valid_count", -1))
    except (TypeError, ValueError):
        manifest_valid = -1
    if manifest_expected != expected:
        reasons.append(f"expected {expected} radius rows, manifest has "
                       f"{payload.get('expected_count')}")
    if manifest_valid != expected:
        reasons.append("not every completed atlas orientation has a valid R_e")
    return {
        "valid": not reasons,
        "path": path,
        "manifest_fingerprint": payload.get("manifest_fingerprint", ""),
        "atlas_inventory_fingerprint": current_fp,
        "expected_count": expected,
        "valid_count": payload.get("valid_count", 0),
        "failed_count": payload.get("failed_count", 0),
        "reasons": reasons,
    }


def radius_lookup(manifest: dict[str, Any]) -> dict[tuple[str, int], float]:
    """Return validated ``(subhalo_id, orientation) -> native R_e`` rows."""
    if not manifest.get("valid"):
        raise ValueError("cannot use an invalid TNG radius manifest")
    lookup: dict[tuple[str, int], float] = {}
    for row in manifest.get("entries", []):
        if not row.get("valid"):
            raise ValueError("radius manifest contains an invalid entry")
        value = float(row["native_re_px"])
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("radius manifest contains a non-positive R_e")
        lookup[(str(row["subhalo_id"]), int(row["orientation"]))] = value
    return lookup
