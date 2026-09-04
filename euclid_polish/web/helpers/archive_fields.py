"""Validated access to the shared multipoint Euclid archive-field collection.

The collection is generated on FASRC from the independent VIS-noise support
pointings, then synchronized as one manifest and one four-band FITS bundle per
sample.  This module is the single local read boundary: callers never glob the
cutout directory or trust paths embedded by the remote filesystem.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.io.fits.verify import VerifyError

from euclid_polish.config import Config
from euclid_polish.photometry import adu_per_s_to_electrons_factor

ARCHIVE_FIELDS_SUBDIR = Config.EuclidSky.ARCHIVE_FIELDS_SUBDIR
ARCHIVE_FIELDS_MANIFEST = Config.EuclidSky.ARCHIVE_FIELDS_MANIFEST_FILENAME
SOURCE_SAMPLING_SUBDIR = "vis_noise_samples"
SOURCE_SAMPLING_MANIFEST = "vis_noise_sampling_manifest.json"
MANIFEST_KIND = "euclid_archive_fields"
MANIFEST_VERSION = 1
TILE_SIZE = 256
SOURCE_SAMPLE_COUNT = 44
POSITIONS_PER_PARENT = 5
SAMPLE_COUNT = SOURCE_SAMPLE_COUNT * POSITIONS_PER_PARENT
BAND_NAMES: tuple[str, ...] = tuple(Config.LR_INPUT_BAND_NAMES)
_Q1_FIELDS = frozenset({"EDF-N", "EDF-S", "EDF-F"})
_COMPLETE_STATUSES = frozenset({"written", "cached", "complete", "completed"})
_ALLOWED_STATUSES = _COMPLETE_STATUSES | frozenset({"planned", "failed"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FINGERPRINT_OMIT = frozenset({
    "created_at", "output_path", "status", "error", "collection_fingerprint",
})
_TOTAL_ELECTRON_UNITS = frozenset({
    "e-", "e-/pixel", "electron", "electron/pixel", "electrons",
    "electrons/pixel",
})
_ARCHIVE_RATE_UNITS = frozenset({
    "", "adu/s", "adu/sec", "count/s", "count/sec", "counts/s",
    "counts/sec", "electron/s", "electron/sec", "electrons/s",
    "electrons/sec", "e-/s", "e-/sec",
})


class ArchiveFieldError(ValueError):
    """The archive-field manifest or one of its FITS bundles is invalid."""


@dataclass(frozen=True)
class ArchiveField:
    """One independently positioned four-band archive sample."""

    sample_id: int
    source_sample_id: int
    parent_id: str
    field: str
    ra: float
    dec: float
    source_release: str
    source_plan_fingerprint: str
    position_index: int
    position_name: str
    path: Path
    bundle_sha256: str
    bands: Mapping[str, Any]
    record: Mapping[str, Any]


def collection_root() -> Path:
    """Return the relocatable local collection root."""
    return Path(Config.EUCLID_SKY_DIR) / ARCHIVE_FIELDS_SUBDIR


def manifest_path() -> Path:
    return collection_root() / ARCHIVE_FIELDS_MANIFEST


def source_manifest_path() -> Path:
    return Path(Config.EUCLID_SKY_DIR) / SOURCE_SAMPLING_SUBDIR / SOURCE_SAMPLING_MANIFEST


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ArchiveFieldError(f"cannot read {path}: {exc}") from exc
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compute_plan_fingerprint(plan: Mapping[str, Any]) -> str:
    """Canonical acquisition-plan fingerprint shared with the generator."""
    return _canonical_sha256(dict(plan))


def _fingerprint_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _fingerprint_value(item)
            for key, item in value.items()
            if key not in _FINGERPRINT_OMIT
        }
    if isinstance(value, (list, tuple)):
        return [_fingerprint_value(item) for item in value]
    return value


def compute_collection_fingerprint(manifest: Mapping[str, Any]) -> str:
    """Fingerprint successful sample identity/provenance, not file locations.

    Retry timestamps, status wording, errors, and remote/local paths are
    intentionally excluded.  A synchronized collection therefore retains the
    same identity after relocation, while any science-bearing metadata change
    invalidates it.
    """
    raw_samples = manifest.get("samples")
    samples = raw_samples if isinstance(raw_samples, list) else []
    completed = [
        sample for sample in samples
        if isinstance(sample, Mapping)
        and str(sample.get("status") or "").lower() in _COMPLETE_STATUSES
    ]
    try:
        completed.sort(key=lambda sample: int(sample["sample_id"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ArchiveFieldError("archive samples have invalid sample_id values") from exc
    core = {
        key: manifest[key]
        for key in (
            "version", "kind", "source_release", "source", "plan",
            "plan_fingerprint",
        )
        if key in manifest
    }
    core["samples"] = completed
    try:
        return _canonical_sha256(_fingerprint_value(core))
    except (TypeError, ValueError) as exc:
        raise ArchiveFieldError("archive manifest is not canonically serializable") from exc


def manifest_fingerprint(path: Path | str | None = None) -> str:
    """Return the SHA256 of the exact synchronized manifest bytes."""
    return _sha256(Path(path) if path is not None else manifest_path())


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArchiveFieldError(f"{label} must be an object")
    return value


def _require_sha(value: Any, label: str) -> str:
    fingerprint = str(value or "").strip().lower()
    if not _SHA256.fullmatch(fingerprint):
        raise ArchiveFieldError(f"{label} must be a lowercase SHA256 fingerprint")
    return fingerprint


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int:
        raise ArchiveFieldError(f"{label} must be an integer")
    result = int(value)
    if result < minimum:
        raise ArchiveFieldError(f"{label} must be an integer >= {minimum}")
    return result


def _require_coordinate(value: Any, label: str, *, dec: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ArchiveFieldError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise ArchiveFieldError(f"{label} must be finite")
    if dec and not -90.0 <= result <= 90.0:
        raise ArchiveFieldError(f"{label} must be within [-90, 90]")
    if not dec and not 0.0 <= result < 360.0:
        raise ArchiveFieldError(f"{label} must be within [0, 360)")
    return result


def _completed_samples(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    samples = manifest["samples"]
    assert isinstance(samples, list)
    return [
        sample for sample in samples
        if isinstance(sample, Mapping)
        and str(sample.get("status") or "").lower() in _COMPLETE_STATUSES
    ]


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("kind") != MANIFEST_KIND:
        raise ArchiveFieldError(f"manifest kind must be {MANIFEST_KIND!r}")
    if type(manifest.get("version")) is not int or manifest["version"] != MANIFEST_VERSION:
        raise ArchiveFieldError(f"archive manifest version must be {MANIFEST_VERSION}")

    source_release = str(manifest.get("source_release") or "").strip()
    if not source_release:
        raise ArchiveFieldError("manifest has no source_release")
    source = _require_mapping(manifest.get("source"), "source")
    plan = _require_mapping(manifest.get("plan"), "plan")
    if source.get("manifest_kind") != "euclid_vis_noise_sampling":
        raise ArchiveFieldError("source manifest kind must be euclid_vis_noise_sampling")
    if type(source.get("manifest_version")) is not int or source["manifest_version"] != 1:
        raise ArchiveFieldError("source manifest version must be 1")
    source_manifest_sha = _require_sha(
        source.get("manifest_sha256"), "source.manifest_sha256",
    )
    source_plan = _require_sha(
        source.get("plan_fingerprint"), "source.plan_fingerprint",
    )
    if str(plan.get("source_release") or "") != source_release:
        raise ArchiveFieldError("plan source_release disagrees with manifest")
    if _require_sha(
        plan.get("source_manifest_sha256"), "plan.source_manifest_sha256",
    ) != source_manifest_sha:
        raise ArchiveFieldError("source manifest fingerprint disagrees between source and plan")
    if _require_sha(
        plan.get("source_plan_fingerprint"), "plan.source_plan_fingerprint",
    ) != source_plan:
        raise ArchiveFieldError("source plan fingerprint disagrees between source and plan")
    if list(plan.get("bands") or []) != list(BAND_NAMES):
        raise ArchiveFieldError(f"plan bands must be {list(BAND_NAMES)!r}")
    if _require_int(
        plan.get("cutout_size_vis_pixels"), "plan.cutout_size_vis_pixels", minimum=1,
    ) != TILE_SIZE:
        raise ArchiveFieldError(f"archive cutout size must be {TILE_SIZE}")
    source_count = _require_int(
        plan.get("source_sample_count"), "plan.source_sample_count", minimum=1,
    )
    positions = _require_int(
        plan.get("positions_per_parent"), "plan.positions_per_parent", minimum=1,
    )
    if source_count != SOURCE_SAMPLE_COUNT or positions != POSITIONS_PER_PARENT:
        raise ArchiveFieldError(
            f"archive plan must contain {SOURCE_SAMPLE_COUNT} source pointings "
            f"and {POSITIONS_PER_PARENT} positions per pointing"
        )
    recorded_plan_fingerprint = _require_sha(
        manifest.get("plan_fingerprint"), "plan_fingerprint",
    )
    if recorded_plan_fingerprint != compute_plan_fingerprint(plan):
        raise ArchiveFieldError("plan_fingerprint does not match the acquisition plan")

    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise ArchiveFieldError("manifest samples must be a list")
    if len(samples) != SAMPLE_COUNT:
        raise ArchiveFieldError(
            "sample count does not match source_sample_count * positions_per_parent"
        )
    seen_ids: set[int] = set()
    source_parents: dict[int, str] = {}
    source_positions: Counter[int] = Counter()
    for offset, raw_sample in enumerate(samples):
        sample = _require_mapping(raw_sample, f"samples[{offset}]")
        sample_id = _require_int(sample.get("sample_id"), f"samples[{offset}].sample_id")
        if sample_id in seen_ids:
            raise ArchiveFieldError(f"duplicate sample_id {sample_id}")
        seen_ids.add(sample_id)
        source_sample_id = _require_int(
            sample.get("source_sample_id"),
            f"sample {sample_id} source_sample_id",
        )
        if source_sample_id >= SOURCE_SAMPLE_COUNT:
            raise ArchiveFieldError(f"sample {sample_id} source_sample_id is out of range")
        position_index = _require_int(
            sample.get("position_index"), f"sample {sample_id} position_index",
        )
        if position_index >= POSITIONS_PER_PARENT:
            raise ArchiveFieldError(f"sample {sample_id} position_index is out of range")
        if not str(sample.get("position_name") or "").strip():
            raise ArchiveFieldError(f"sample {sample_id} has no position_name")
        if sample_id != source_sample_id * POSITIONS_PER_PARENT + position_index:
            raise ArchiveFieldError(f"sample {sample_id} identity does not match its position")
        parent_id = str(sample.get("parent_id") or "").strip()
        if not parent_id:
            raise ArchiveFieldError(f"sample {sample_id} has no parent_id")
        previous_parent = source_parents.setdefault(source_sample_id, parent_id)
        if previous_parent != parent_id:
            raise ArchiveFieldError(
                f"source pointing {source_sample_id} maps to multiple parents"
            )
        source_positions[source_sample_id] += 1
        field = str(sample.get("field") or "")
        if field not in _Q1_FIELDS:
            raise ArchiveFieldError(f"sample {sample_id} has unknown field {field!r}")
        _require_coordinate(sample.get("ra"), f"sample {sample_id} ra")
        _require_coordinate(sample.get("dec"), f"sample {sample_id} dec", dec=True)
        if str(sample.get("source_release") or "") != source_release:
            raise ArchiveFieldError(f"sample {sample_id} source_release disagrees")
        if _require_sha(
            sample.get("source_plan_fingerprint"),
            f"sample {sample_id} source_plan_fingerprint",
        ) != source_plan:
            raise ArchiveFieldError(f"sample {sample_id} source plan disagrees")
        if _require_sha(
            sample.get("plan_fingerprint"),
            f"sample {sample_id} plan_fingerprint",
        ) != recorded_plan_fingerprint:
            raise ArchiveFieldError(f"sample {sample_id} archive plan disagrees")
        status = str(sample.get("status") or "").lower()
        if status not in _ALLOWED_STATUSES:
            raise ArchiveFieldError(f"sample {sample_id} has invalid status {status!r}")
        if status not in _COMPLETE_STATUSES:
            continue
        expected_name = f"field_{sample_id:04d}.fits"
        if Path(str(sample.get("output_path") or "")).name != expected_name:
            raise ArchiveFieldError(
                f"sample {sample_id} output_path must end in {expected_name}"
            )
        _require_sha(sample.get("bundle_sha256"), f"sample {sample_id} bundle_sha256")
        bands = _require_mapping(sample.get("bands"), f"sample {sample_id} bands")
        if set(bands) != set(BAND_NAMES):
            raise ArchiveFieldError(
                f"sample {sample_id} must describe exactly {list(BAND_NAMES)!r}"
            )
        for band_name in BAND_NAMES:
            band = _require_mapping(bands[band_name], f"sample {sample_id} {band_name}")
            try:
                shape = tuple(int(side) for side in band["shape"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ArchiveFieldError(
                    f"sample {sample_id} {band_name} has no valid shape"
                ) from exc
            if shape != (TILE_SIZE, TILE_SIZE):
                raise ArchiveFieldError(
                    f"sample {sample_id} {band_name} shape must be "
                    f"{TILE_SIZE}x{TILE_SIZE}"
                )
    if seen_ids != set(range(len(samples))):
        raise ArchiveFieldError("sample_id values must be contiguous from zero")
    if set(source_parents) != set(range(SOURCE_SAMPLE_COUNT)) or any(
        source_positions[index] != POSITIONS_PER_PARENT
        for index in range(SOURCE_SAMPLE_COUNT)
    ):
        raise ArchiveFieldError(
            "each source pointing must contribute exactly "
            f"{POSITIONS_PER_PARENT} positions"
        )
    if len(set(source_parents.values())) != SOURCE_SAMPLE_COUNT:
        raise ArchiveFieldError("source pointings must use independent parent mosaics")

    recorded_collection = _require_sha(
        manifest.get("collection_fingerprint"), "collection_fingerprint",
    )
    if recorded_collection != compute_collection_fingerprint(manifest):
        raise ArchiveFieldError("collection_fingerprint does not match the manifest")


def load_manifest(path: Path | str | None = None) -> dict[str, Any]:
    """Load and strictly validate one archive collection manifest."""
    target = Path(path) if path is not None else manifest_path()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ArchiveFieldError(f"archive manifest is unavailable: {target}") from exc
    except json.JSONDecodeError as exc:
        raise ArchiveFieldError(f"archive manifest is invalid JSON: {target}") from exc
    if not isinstance(payload, dict):
        raise ArchiveFieldError("archive manifest must be a JSON object")
    _validate_manifest(payload)
    return payload


def _bundle_path(sample: Mapping[str, Any], root: Path) -> Path:
    sample_id = int(sample["sample_id"])
    # Always rebase the stable basename beneath the synchronized collection.
    # Remote absolute paths are provenance only and must never escape the data
    # root when the local web process opens a bundle.
    return root / "cutouts" / f"field_{sample_id:04d}.fits"


def iter_fields(
    manifest: Mapping[str, Any] | None = None,
    *,
    manifest_file: Path | str | None = None,
) -> Iterator[ArchiveField]:
    """Yield completed fields in stable ``sample_id`` order."""
    target = Path(manifest_file) if manifest_file is not None else manifest_path()
    payload = load_manifest(target) if manifest is None else dict(manifest)
    _validate_manifest(payload)
    root = target.parent
    samples = sorted(_completed_samples(payload), key=lambda item: int(item["sample_id"]))
    for sample in samples:
        yield ArchiveField(
            sample_id=int(sample["sample_id"]),
            source_sample_id=int(sample["source_sample_id"]),
            parent_id=str(sample["parent_id"]),
            field=str(sample["field"]),
            ra=float(sample["ra"]),
            dec=float(sample["dec"]),
            source_release=str(sample["source_release"]),
            source_plan_fingerprint=str(sample["source_plan_fingerprint"]),
            position_index=int(sample.get("position_index", 0)),
            position_name=str(sample.get("position_name") or "sample"),
            path=_bundle_path(sample, root),
            bundle_sha256=str(sample["bundle_sha256"]),
            bands=_require_mapping(sample["bands"], f"sample {sample['sample_id']} bands"),
            record=sample,
        )


def _normalised_unit(header: fits.Header) -> str:
    return "".join(str(header.get("BUNIT") or "").strip().lower().split())


def _band_electrons(data: np.ndarray, header: fits.Header, band_name: str, path: Path) -> np.ndarray:
    unit = _normalised_unit(header)
    if unit in _TOTAL_ELECTRON_UNITS:
        converted = np.asarray(data, dtype=np.float64)
    else:
        if unit not in _ARCHIVE_RATE_UNITS:
            raise ArchiveFieldError(f"{path}: unsupported {band_name} BUNIT {unit!r}")
        try:
            zeropoint = float(header["MAGZERO"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ArchiveFieldError(
                f"{path}: {band_name} archive-rate image needs finite MAGZERO"
            ) from exc
        if not math.isfinite(zeropoint):
            raise ArchiveFieldError(
                f"{path}: {band_name} archive-rate image needs finite MAGZERO"
            )
        converted = np.asarray(data, dtype=np.float64) * adu_per_s_to_electrons_factor(
            zeropoint, Config.get_band(band_name),
        )
    if not np.all(np.isfinite(converted)):
        raise ArchiveFieldError(f"{path}: {band_name} contains non-finite pixels")
    return np.asarray(converted, dtype=np.float32)


def _validate_primary(
    header: fits.Header,
    field: ArchiveField,
    manifest: Mapping[str, Any],
) -> None:
    expected = {
        "SAMPLEID": field.sample_id,
        "SRC_ID": field.source_sample_id,
        "PARENT": field.parent_id,
        "Q1FIELD": field.field,
        "RELEASE": field.source_release,
        "SRCPLAN": field.source_plan_fingerprint,
        "PLANHASH": str(manifest["plan_fingerprint"]),
    }
    for key, wanted in expected.items():
        if str(header.get(key)) != str(wanted):
            raise ArchiveFieldError(
                f"{field.path}: FITS {key}={header.get(key)!r} disagrees with {wanted!r}"
            )
    for key, wanted in (("RA", field.ra), ("DEC", field.dec)):
        try:
            actual = float(header[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ArchiveFieldError(f"{field.path}: FITS {key} is invalid") from exc
        if not math.isfinite(actual) or not np.isclose(actual, wanted, rtol=0.0, atol=1e-8):
            raise ArchiveFieldError(
                f"{field.path}: FITS {key}={actual!r} disagrees with {wanted!r}"
            )


def load_field(
    field: ArchiveField | int,
    manifest: Mapping[str, Any] | None = None,
    *,
    manifest_file: Path | str | None = None,
) -> np.ndarray:
    """Load one strict four-band bundle as ``(256, 256, 4)`` electrons."""
    target = Path(manifest_file) if manifest_file is not None else manifest_path()
    payload = load_manifest(target) if manifest is None else dict(manifest)
    _validate_manifest(payload)
    if isinstance(field, int):
        record = next((item for item in iter_fields(payload, manifest_file=target)
                       if item.sample_id == field), None)
        if record is None:
            raise ArchiveFieldError(f"archive sample {field} is unavailable")
        field = record
    if not field.path.is_file():
        raise ArchiveFieldError(f"archive field bundle is unavailable: {field.path}")
    expected_bundle_sha = str(field.record["bundle_sha256"])
    if _sha256(field.path) != expected_bundle_sha:
        raise ArchiveFieldError(f"{field.path}: bundle SHA256 does not match the manifest")
    try:
        with fits.open(field.path, memmap=False) as hdul:
            hdul.verify("exception")
            if len(hdul) != len(BAND_NAMES) + 1:
                raise ArchiveFieldError(
                    f"{field.path}: expected one primary and four image HDUs"
                )
            if not isinstance(hdul[0], fits.PrimaryHDU) or hdul[0].data is not None:
                raise ArchiveFieldError(f"{field.path}: primary HDU must be dataless")
            _validate_primary(hdul[0].header, field, payload)
            extensions = [str(hdu.name) for hdu in hdul[1:]]
            if extensions != list(BAND_NAMES):
                raise ArchiveFieldError(
                    f"{field.path}: image HDUs must be ordered as {list(BAND_NAMES)!r}"
                )
            bands: list[np.ndarray] = []
            for band_name, hdu in zip(BAND_NAMES, hdul[1:], strict=True):
                if not isinstance(hdu, (fits.ImageHDU, fits.CompImageHDU)) or hdu.data is None:
                    raise ArchiveFieldError(f"{field.path}: {band_name} is not an image HDU")
                data = np.asarray(hdu.data)
                if data.shape != (TILE_SIZE, TILE_SIZE):
                    raise ArchiveFieldError(
                        f"{field.path}: {band_name} shape is {data.shape}, expected "
                        f"{TILE_SIZE}x{TILE_SIZE}"
                    )
                bands.append(_band_electrons(data, hdu.header, band_name, field.path))
    except ArchiveFieldError:
        raise
    except (OSError, ValueError, VerifyError) as exc:
        raise ArchiveFieldError(f"archive field bundle is unreadable: {field.path}") from exc
    return np.stack(bands, axis=-1).astype(np.float32, copy=False)


def currentness(
    manifest: Mapping[str, Any] | None = None,
    *,
    manifest_file: Path | str | None = None,
    source_file: Path | str | None = None,
) -> dict[str, Any]:
    """Compare the archive collection with the currently synchronized source plan."""
    target = Path(manifest_file) if manifest_file is not None else manifest_path()
    payload = load_manifest(target) if manifest is None else dict(manifest)
    _validate_manifest(payload)
    source_target = Path(source_file) if source_file is not None else source_manifest_path()
    reasons: list[str] = []
    try:
        source_payload = json.loads(source_target.read_text(encoding="utf-8"))
        local_source_sha = _sha256(source_target)
    except (OSError, json.JSONDecodeError, ArchiveFieldError):
        source_payload = None
        local_source_sha = None
        reasons.append("source VIS-pointing manifest is unavailable")
    source_sha = local_source_sha
    if source_payload is not None and not isinstance(source_payload, Mapping):
        reasons.append("source VIS-pointing manifest is not a JSON object")
    if isinstance(source_payload, Mapping):
        sync = source_payload.get("sync")
        if isinstance(sync, Mapping):
            remote_sha = str(sync.get("remote_manifest_sha256") or "").lower()
            # The sync route deliberately rewrites local paths/status and adds
            # this block, so its local byte hash differs from the immutable
            # FASRC source.  The recorded, validated remote digest is the
            # science identity; raw bytes are only a fallback for unsynced
            # manifests.
            if _SHA256.fullmatch(remote_sha):
                source_sha = remote_sha
    expected_sha = str(payload["source"]["manifest_sha256"])
    expected_plan = str(payload["source"]["plan_fingerprint"])
    if source_sha is not None and source_sha != expected_sha:
        reasons.append("source VIS-pointing manifest has changed")
    if isinstance(source_payload, Mapping):
        if source_payload.get("kind") != "euclid_vis_noise_sampling":
            reasons.append("source VIS-pointing manifest kind is invalid")
        if source_payload.get("version") != 1:
            reasons.append("source VIS-pointing manifest version is invalid")
        if str(source_payload.get("source_release") or "") != str(payload["source_release"]):
            reasons.append("source archive release has changed")
        if str(source_payload.get("plan_fingerprint") or "") != expected_plan:
            reasons.append("source VIS-pointing plan has changed")
    return {
        "current": not reasons,
        "reasons": reasons,
        "source_manifest_path": str(source_target),
        "source_manifest_sha256": source_sha,
        "local_source_manifest_sha256": local_source_sha,
        "expected_source_manifest_sha256": expected_sha,
        "source_plan_fingerprint": expected_plan,
    }


def is_current(
    manifest: Mapping[str, Any] | None = None,
    *,
    manifest_file: Path | str | None = None,
    source_file: Path | str | None = None,
) -> bool:
    return bool(currentness(
        manifest, manifest_file=manifest_file, source_file=source_file,
    )["current"])


def availability(
    *,
    manifest_file: Path | str | None = None,
    source_file: Path | str | None = None,
) -> dict[str, Any]:
    """Return non-raising readiness, provenance, and independent-pointing counts."""
    target = Path(manifest_file) if manifest_file is not None else manifest_path()
    base: dict[str, Any] = {
        "available": target.is_file(),
        "valid": False,
        "ready": False,
        "complete": False,
        "current": False,
        "reasons": [],
        "sample_count": 0,
        "planned_sample_count": 0,
        "parent_count": 0,
        "fields": {},
        "bands": list(BAND_NAMES),
        "tile_size": TILE_SIZE,
        "manifest_path": str(target),
        "manifest_fingerprint": None,
        "collection_fingerprint": None,
        "source_release": None,
        "source_plan_fingerprint": None,
        "source_manifest_sha256": None,
    }
    if not target.is_file():
        base["reasons"] = ["multipoint archive manifest is unavailable"]
        return base
    try:
        manifest = load_manifest(target)
        fields = list(iter_fields(manifest, manifest_file=target))
        missing = [field.path for field in fields if not field.path.is_file()]
        provenance = currentness(
            manifest, manifest_file=target, source_file=source_file,
        )
        reasons = list(provenance["reasons"])
        if missing:
            reasons.append(f"{len(missing)} synchronized FITS bundles are missing")
        planned = len(manifest["samples"])
        complete = len(fields) == planned and not missing
        if len(fields) != planned:
            reasons.append(f"only {len(fields)} of {planned} planned samples are complete")
        base.update({
            "valid": True,
            # Consumer-safe readiness is intentionally stronger than simple
            # presence.  The viewer and population comparison must never
            # silently fall back to a partial or source-stale baseline.
            "ready": complete and bool(provenance["current"]),
            "complete": complete,
            "current": bool(provenance["current"]),
            "reasons": reasons,
            "sample_count": len(fields),
            "planned_sample_count": planned,
            "parent_count": len({field.parent_id for field in fields}),
            "fields": dict(sorted(Counter(field.field for field in fields).items())),
            "manifest_fingerprint": manifest_fingerprint(target),
            "collection_fingerprint": str(manifest["collection_fingerprint"]),
            "source_release": str(manifest["source_release"]),
            "source_plan_fingerprint": str(manifest["source"]["plan_fingerprint"]),
            "source_manifest_sha256": str(manifest["source"]["manifest_sha256"]),
        })
    except ArchiveFieldError as exc:
        base["reasons"] = [str(exc)]
    return base


__all__ = [
    "ARCHIVE_FIELDS_MANIFEST",
    "ARCHIVE_FIELDS_SUBDIR",
    "ArchiveField",
    "ArchiveFieldError",
    "BAND_NAMES",
    "MANIFEST_KIND",
    "MANIFEST_VERSION",
    "POSITIONS_PER_PARENT",
    "SAMPLE_COUNT",
    "SOURCE_SAMPLE_COUNT",
    "TILE_SIZE",
    "availability",
    "collection_root",
    "compute_collection_fingerprint",
    "compute_plan_fingerprint",
    "currentness",
    "is_current",
    "iter_fields",
    "load_field",
    "load_manifest",
    "manifest_fingerprint",
    "manifest_path",
    "source_manifest_path",
]
