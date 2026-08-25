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
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from euclid_polish.config import Config
from euclid_polish.tng._image import (
    _load_tng_plane,
    _measure_halflight_radius_px,
)
from euclid_polish.tng.atlas import TNGGalaxy, _scan_complete_galaxies
from euclid_polish.tng.catalog import TNGPropertyCatalog
from euclid_polish.tng.types import N_ORIENTATIONS, TNG_FITS_BANDS, TNG_NATIVE_PC_PER_PIXEL

MANIFEST_VERSION = 1
ALGORITHM_VERSION = "centered-vis-cog-v1"
DEFAULT_MANIFEST_NAME = "tng_radius_manifest.json"
PARAMETER_SUMMARY_VERSION = 1
PARAMETER_SUMMARY_FIELDS = (
    "subhalo_id", "orientation", "native_re_px", "native_re_kpc",
    "frame_height_px", "frame_width_px", "mass_stars_msun",
    "logmass_stars", "sfr_msun_yr", "m_halo_msun",
    "groupcat_reff_kpc",
)


@dataclass(frozen=True, slots=True)
class TNGRadiusManifest(Mapping[tuple[str, int], float]):
    """Immutable native VIS half-light radii indexed by atlas orientation."""

    _radii: Mapping[tuple[str, int], float] = field(repr=False)
    fingerprint: str
    _max_radii: Mapping[str, float] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        radii: dict[tuple[str, int], float] = {}
        for raw_key, raw_radius in self._radii.items():
            try:
                raw_subhalo_id, raw_orientation = raw_key
                subhalo_id = str(raw_subhalo_id).strip()
                orientation = int(raw_orientation)
                radius = float(raw_radius)
            except (TypeError, ValueError) as exc:
                raise ValueError("malformed TNG radius-manifest entry") from exc
            if not subhalo_id:
                raise ValueError("TNG radius manifest contains an empty subhalo id")
            if orientation not in range(1, N_ORIENTATIONS + 1):
                raise ValueError(
                    f"TNG radius orientation must be in 1..{N_ORIENTATIONS}, "
                    f"got {orientation!r}"
                )
            if not np.isfinite(radius) or radius <= 0.0:
                raise ValueError("TNG radius manifest contains a non-positive R_e")
            key = (subhalo_id, orientation)
            if key in radii:
                raise ValueError(f"duplicate TNG radius-manifest entry {key!r}")
            radii[key] = radius
        fingerprint = str(self.fingerprint).strip()
        if not fingerprint:
            raise ValueError("TNG radius manifest has no fingerprint")
        object.__setattr__(self, "_radii", MappingProxyType(radii))
        object.__setattr__(self, "fingerprint", fingerprint)
        max_radii: dict[str, float] = {}
        for (subhalo_id, _), radius in radii.items():
            max_radii[subhalo_id] = max(max_radii.get(subhalo_id, 0.0), radius)
        object.__setattr__(self, "_max_radii", MappingProxyType(max_radii))

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> TNGRadiusManifest:
        """Validate and materialize the live radius-manifest JSON schema."""
        if payload.get("version") != MANIFEST_VERSION:
            raise ValueError("unsupported TNG radius-manifest version")
        if payload.get("algorithm_version") != ALGORITHM_VERSION:
            raise ValueError("unsupported TNG radius-measurement algorithm")
        if not payload.get("valid"):
            raise ValueError("cannot load an invalid TNG radius manifest")
        claimed_fingerprint = str(payload.get("manifest_fingerprint", ""))
        if claimed_fingerprint != _manifest_fingerprint(payload):
            raise ValueError("TNG radius-manifest fingerprint does not match its rows")
        entries = payload.get("entries")
        if not isinstance(entries, list):
            raise ValueError("TNG radius manifest entries must be a list")

        radii: dict[tuple[str, int], float] = {}
        for entry in entries:
            if not isinstance(entry, Mapping) or not entry.get("valid"):
                raise ValueError("TNG radius manifest contains an invalid entry")
            try:
                key = (str(entry["subhalo_id"]), int(entry["orientation"]))
                radius = float(entry["native_re_px"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("malformed TNG radius-manifest entry") from exc
            if key in radii:
                raise ValueError(f"duplicate TNG radius-manifest entry {key!r}")
            radii[key] = radius

        try:
            expected_count = int(payload.get("expected_count", len(radii)))
            valid_count = int(payload.get("valid_count", len(radii)))
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid TNG radius-manifest row counts") from exc
        if expected_count != len(radii) or valid_count != len(radii):
            raise ValueError("TNG radius manifest is incomplete")
        return cls(radii, claimed_fingerprint)

    @classmethod
    def read(cls, path: str | Path) -> TNGRadiusManifest:
        """Read a valid existing manifest without measuring or repairing it."""
        payload = load_manifest(str(path))
        if payload is None:
            raise ValueError(f"missing or malformed TNG radius manifest: {path}")
        return cls.from_payload(payload)

    def __getitem__(self, key: tuple[str, int]) -> float:
        subhalo_id, orientation = key
        return self._radii[(str(subhalo_id), int(orientation))]

    def __iter__(self) -> Iterator[tuple[str, int]]:
        return iter(self._radii)

    def __len__(self) -> int:
        return len(self._radii)

    def radius(self, subhalo_id: str, orientation: int) -> float:
        """Return the native half-light radius for one atlas view."""
        return self[(str(subhalo_id), int(orientation))]

    def max_radius(self, subhalo_id: str) -> float:
        """Return the largest native radius among one galaxy's orientations."""
        return self._max_radii[str(subhalo_id)]

    def __repr__(self) -> str:
        return (
            f"TNGRadiusManifest(entries={len(self)}, "
            f"fingerprint={self.fingerprint!r})"
        )


def manifest_path(path: str | None = None) -> str:
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
    return cast(dict[str, int | str | None], _file_identity(path))


def _fingerprint(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_MANIFEST_FINGERPRINT_FIELDS = (
    "version",
    "algorithm_version",
    "properties_file",
    "atlas_inventory_fingerprint",
    "expected_count",
    "valid_count",
    "entries",
)


def _manifest_fingerprint(payload: Mapping[str, Any]) -> str:
    """Recompute the canonical digest covering every rendered-radius row."""
    try:
        covered = {key: payload[key] for key in _MANIFEST_FINGERPRINT_FIELDS}
        return _fingerprint(covered)
    except (KeyError, TypeError, ValueError):
        return ""


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(
    galaxies: Sequence[TNGGalaxy],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for galaxy in galaxies:
        for orientation in range(1, N_ORIENTATIONS + 1):
            for band in TNG_FITS_BANDS:
                path = galaxy.fits_path(orientation, band)
                if path.is_file():
                    entries.append(_file_identity(str(path)))
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
    reuse_existing: bool = True,
) -> dict[str, Any]:
    """Measure every completed atlas orientation and return a report.

    A report is written atomically when ``output_path`` is supplied, including
    invalid entries, so the UI can show exactly why a job is not submit-ready.
    """
    galaxies = _scan_complete_galaxies(Path(tng_dir))
    properties_path = properties_path or os.path.join(
        Config.DATA_DIR, "_tng_infographics", "tng_properties.csv"
    )
    properties = TNGPropertyCatalog.read(properties_path)
    reusable: dict[tuple[str, int], dict[str, Any]] = {}
    if reuse_existing and output_path:
        previous = load_manifest(output_path)
        if (
            previous
            and previous.get("version") == MANIFEST_VERSION
            and previous.get("algorithm_version") == ALGORITHM_VERSION
            and previous.get("manifest_fingerprint")
            == _manifest_fingerprint(previous)
        ):
            for entry in previous.get("entries", []):
                if not isinstance(entry, dict) or not entry.get("valid"):
                    continue
                try:
                    key = (
                        str(entry["subhalo_id"]),
                        int(entry["orientation"]),
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                reusable[key] = dict(entry)
    tasks: list[tuple[TNGGalaxy, int, bool]] = []
    for galaxy in galaxies:
        props = properties.get(galaxy.subhalo_id)
        mass = (
            props.stellar_mass_msun if props is not None else float("nan")
        )
        has_properties = bool(np.isfinite(mass) and mass > 0.0)
        for orientation in range(1, N_ORIENTATIONS + 1):
            tasks.append((galaxy, orientation, has_properties))

    def measure(
        task: tuple[TNGGalaxy, int, bool],
    ) -> tuple[dict[str, Any], dict[str, str] | None, bool]:
        galaxy, orientation, has_properties = task
        gid = galaxy.subhalo_id
        vis_path = galaxy.fits_path(orientation, "VIS")
        row: dict[str, Any] = {
            "subhalo_id": gid,
            "orientation": int(orientation),
            "vis_path": os.path.abspath(vis_path),
            "valid": False,
        }
        failure = None
        try:
            identity = _file_identity(str(vis_path))
            previous = reusable.get((gid, orientation))
            if (
                has_properties
                and previous is not None
                and previous.get("vis_file") == identity
            ):
                previous["vis_path"] = os.path.abspath(vis_path)
                return previous, None, True
            frame = _load_tng_plane(vis_path, "VIS")
            re_px = float(_measure_halflight_radius_px(frame, band="VIS"))
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
        return row, failure, False

    worker_count = max(1, int(workers))
    if worker_count == 1:
        measured = map(measure, tasks)
        measured_rows = list(measured)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            measured_rows = list(pool.map(measure, tasks))
    entries = [row for row, _, _ in measured_rows]
    failures = [
        failure for _, failure, _ in measured_rows if failure is not None
    ]
    reused_count = sum(reused for _, _, reused in measured_rows)

    inventory = _inventory(galaxies)
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
        "reused_count": int(reused_count),
        "measured_count": int(len(entries) - reused_count),
        "valid": bool(expected_count > 0 and valid_count == expected_count),
        "failures": failures,
        "entries": entries,
    }
    report["manifest_fingerprint"] = _manifest_fingerprint(report)
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
    properties = TNGPropertyCatalog.read(properties_path)
    rows: list[dict[str, int | float | str]] = []
    galaxy_ids: set[str] = set()
    for entry in manifest.get("entries", []):
        if not entry.get("valid"):
            raise ValueError("radius manifest contains an invalid entry")
        gid = str(entry["subhalo_id"])
        props = properties.get(gid)
        mass = (
            props.stellar_mass_msun if props is not None else float("nan")
        )
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
            "sfr_msun_yr": (
                props.sfr_msun_yr if props is not None else float("nan")
            ),
            "m_halo_msun": (
                props.halo_mass_msun if props is not None else float("nan")
            ),
            "groupcat_reff_kpc": (
                props.stellar_halfmass_radius_kpc
                if props is not None else float("nan")
            ),
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
        writer.writerows(cast(Any, rows))
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
    path = manifest_path(manifest_path_value)
    payload = manifest if manifest is not None else load_manifest(path)
    if payload is None:
        return {"valid": False, "reason": f"missing radius manifest: {path}",
                "path": path}
    galaxies = _scan_complete_galaxies(Path(tng_dir))
    inventory = _inventory(galaxies)
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
    if payload.get("manifest_fingerprint") != _manifest_fingerprint(payload):
        reasons.append("radius-manifest fingerprint does not match its rows")
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


def ensure_manifest(
    tng_dir: str,
    *,
    properties_path: str | None = None,
    manifest_path_value: str | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Return a valid manifest, incrementally repairing it when necessary.

    Existing rows are reused only when their VIS file identity still matches.
    Atlas additions therefore measure just the new orientations, while a
    replaced VIS frame is remeasured before any generation worker starts.
    """
    path = manifest_path(manifest_path_value)
    initial = validate_manifest(
        tng_dir,
        properties_path=properties_path,
        manifest_path_value=path,
    )
    if initial.get("valid"):
        return dict(initial, repaired=False, reused_count=0, measured_count=0)

    report = build_manifest(
        tng_dir,
        properties_path=properties_path,
        output_path=path,
        workers=workers,
        reuse_existing=True,
    )
    result = validate_manifest(
        tng_dir,
        properties_path=properties_path,
        manifest_path_value=path,
    )
    result.update({
        "repaired": True,
        "repair_reasons": list(initial.get("reasons") or []),
        "reused_count": int(report.get("reused_count", 0)),
        "measured_count": int(report.get("measured_count", 0)),
    })
    if not result.get("valid"):
        failures = [
            str(item.get("error") or "unknown radius measurement failure")
            for item in report.get("failures", [])[:3]
        ]
        detail = list(result.get("reasons") or []) + failures
        raise ValueError(
            "TNG radius manifest repair failed: " + "; ".join(detail)
        )
    return result
