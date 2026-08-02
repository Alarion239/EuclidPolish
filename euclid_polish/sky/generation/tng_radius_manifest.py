"""Validated native TNG VIS half-light-radius manifests.

The SKIRT atlas is remote and expensive to scan during every field draw.  This
module turns the scan into an explicit, fingerprinted prerequisite: generation
may use only entries from a manifest whose atlas inventory and property cache
still match the files that were measured.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
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
from euclid_polish.sky.generation.redshift_model import (
    TNG_NATIVE_PC_PER_PIXEL,
    load_tng_properties,
)

MANIFEST_VERSION = 1
ALGORITHM_VERSION = "centered-vis-cog-v1"
DEFAULT_MANIFEST_NAME = "tng_radius_manifest.json"
PARAMETER_SUMMARY_VERSION = 1
DEFAULT_PARAMETER_SUMMARY_NAME = "tng_atlas_parameters.csv"
PARAMETER_SUMMARY_FIELDS = (
    "subhalo_id", "orientation", "native_re_px", "native_re_kpc",
    "frame_height_px", "frame_width_px", "mass_stars_msun",
    "logmass_stars", "sfr_msun_yr", "m_halo_msun",
    "groupcat_reff_kpc",
)


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


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    workers: int = 1,
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
    tasks: list[tuple[str, str, int, bool]] = []
    for gdir, gid in galaxies:
        props = properties.get(str(gid), {})
        mass = float(props.get("mass_stars", float("nan")))
        has_properties = bool(np.isfinite(mass) and mass > 0.0)
        for orientation in range(1, N_ORIENTATIONS + 1):
            tasks.append((gdir, str(gid), orientation, has_properties))

    def measure(task: tuple[str, str, int, bool]) -> tuple[dict[str, Any], dict[str, str] | None]:
        gdir, gid, orientation, has_properties = task
        vis_path = tng_fits_path(gdir, gid, orientation, "VIS")
        row: dict[str, Any] = {
            "subhalo_id": gid,
            "orientation": int(orientation),
            "vis_path": os.path.abspath(vis_path),
            "valid": False,
        }
        failure = None
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
            failure = {
                "subhalo_id": gid,
                "orientation": str(orientation),
                "error": str(exc),
            }
        return row, failure

    worker_count = max(1, int(workers))
    if worker_count == 1:
        measured = map(measure, tasks)
        measured_rows = list(measured)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            measured_rows = list(pool.map(measure, tasks))
    entries = [row for row, _ in measured_rows]
    failures = [failure for _, failure in measured_rows if failure is not None]

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


def parameter_summary_meta_path(path: str | Path) -> Path:
    target = Path(path)
    return target.with_suffix(target.suffix + ".meta.json")


def write_parameter_summary(
    path: str | Path,
    manifest: dict[str, Any],
    *,
    properties_path: str,
) -> dict[str, Any]:
    """Flatten a complete radius manifest and TNG properties into one CSV."""
    if not manifest.get("valid"):
        raise ValueError("cannot summarize an invalid TNG radius manifest")
    properties = load_tng_properties(properties_path)
    rows: list[dict[str, int | float | str]] = []
    galaxy_ids: set[str] = set()
    for entry in manifest.get("entries", []):
        if not entry.get("valid"):
            raise ValueError("radius manifest contains an invalid entry")
        gid = str(entry["subhalo_id"])
        props = properties.get(gid) or {}
        mass = float(props.get("mass_stars", float("nan")))
        native_re_px = float(entry["native_re_px"])
        shape = entry.get("shape") or []
        if (
            not np.isfinite(mass) or mass <= 0.0
            or not np.isfinite(native_re_px) or native_re_px <= 0.0
            or len(shape) != 2
        ):
            raise ValueError(f"TNG{gid} has incomplete summary parameters")
        row = {
            "subhalo_id": gid,
            "orientation": int(entry["orientation"]),
            "native_re_px": native_re_px,
            "native_re_kpc": (
                native_re_px * TNG_NATIVE_PC_PER_PIXEL / 1000.0
            ),
            "frame_height_px": int(shape[0]),
            "frame_width_px": int(shape[1]),
            "mass_stars_msun": mass,
            "logmass_stars": float(np.log10(mass)),
            "sfr_msun_yr": float(props.get("sfr", float("nan"))),
            "m_halo_msun": float(props.get("m_halo", float("nan"))),
            "groupcat_reff_kpc": float(props.get("reff", float("nan"))),
        }
        rows.append(row)
        galaxy_ids.add(gid)
    rows.sort(key=lambda row: (int(str(row["subhalo_id"])), int(row["orientation"])))

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PARAMETER_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, target)
    meta = {
        "version": PARAMETER_SUMMARY_VERSION,
        "kind": "tng_atlas_parameters",
        "valid": True,
        "algorithm_version": manifest.get("algorithm_version"),
        "manifest_fingerprint": manifest.get("manifest_fingerprint"),
        "atlas_inventory_fingerprint": manifest.get(
            "atlas_inventory_fingerprint"
        ),
        "properties_sha256": _file_sha256(properties_path),
        "csv_sha256": _file_sha256(target),
        "row_count": len(rows),
        "galaxy_count": len(galaxy_ids),
        "orientations_per_galaxy": N_ORIENTATIONS,
    }
    meta["summary_fingerprint"] = _fingerprint(meta)
    meta_target = parameter_summary_meta_path(target)
    meta_tmp = meta_target.with_suffix(meta_target.suffix + ".tmp")
    meta_tmp.write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    os.replace(meta_tmp, meta_target)
    return meta


def load_parameter_summary(path: str | Path) -> dict[str, Any]:
    """Load and fingerprint-check a compact atlas parameter summary."""
    target = Path(path)
    meta_path = parameter_summary_meta_path(target)
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"missing atlas parameter metadata: {meta_path}") from exc
    if (
        meta.get("version") != PARAMETER_SUMMARY_VERSION
        or meta.get("kind") != "tng_atlas_parameters"
        or not meta.get("valid")
    ):
        raise ValueError("atlas parameter summary metadata is invalid")
    summary_identity = {
        key: value for key, value in meta.items()
        if key != "summary_fingerprint"
    }
    if meta.get("summary_fingerprint") != _fingerprint(summary_identity):
        raise ValueError("atlas parameter summary metadata fingerprint changed")
    try:
        csv_sha256 = _file_sha256(target)
    except OSError as exc:
        raise ValueError(f"missing atlas parameter summary: {target}") from exc
    if meta.get("csv_sha256") != csv_sha256:
        raise ValueError("atlas parameter summary CSV fingerprint changed")
    rows: list[dict[str, Any]] = []
    with target.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            try:
                row = {
                    "subhalo_id": str(raw["subhalo_id"]),
                    "orientation": int(raw["orientation"]),
                    **{
                        key: float(raw[key])
                        for key in PARAMETER_SUMMARY_FIELDS
                        if key not in {"subhalo_id", "orientation"}
                    },
                }
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("atlas parameter summary has malformed rows") from exc
            if (
                not np.isfinite(row["native_re_px"])
                or row["native_re_px"] <= 0.0
                or not np.isfinite(row["mass_stars_msun"])
                or row["mass_stars_msun"] <= 0.0
            ):
                raise ValueError("atlas parameter summary has invalid core values")
            rows.append(row)
    if len(rows) != int(meta.get("row_count", -1)):
        raise ValueError("atlas parameter summary row count changed")
    keys = {(row["subhalo_id"], row["orientation"]) for row in rows}
    if len(keys) != len(rows):
        raise ValueError("atlas parameter summary has duplicate orientations")
    galaxy_ids = {row["subhalo_id"] for row in rows}
    expected = len(galaxy_ids) * int(meta.get("orientations_per_galaxy", 0))
    if len(rows) != expected or len(galaxy_ids) != int(meta.get("galaxy_count", -1)):
        raise ValueError("atlas parameter summary is incomplete")
    required_orientations = set(range(1, N_ORIENTATIONS + 1))
    for gid in galaxy_ids:
        orientations = {
            row["orientation"] for row in rows if row["subhalo_id"] == gid
        }
        if orientations != required_orientations:
            raise ValueError(f"TNG{gid} has incomplete orientation parameters")
    return {"meta": meta, "rows": rows}


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
