"""The synthetic training records behind Data › Records (spec §8.4).

* **Random access.** A TFRecord file is a sequence of ``uint64 length |
  uint32 masked-crc32c(length) | data | uint32 masked-crc32c(data)`` frames.
  :func:`tfrecord_offsets` scans only the 12-byte headers (seeking over the
  payloads, verifying each length CRC) and caches the frame offsets per file
  state, so :func:`read_record` deserialises exactly one record and
  :func:`record_count` never parses an image — the viewer used to deserialise
  every record before the one it showed (≈ 400 MB to reach index 99).
* **Inventory.** :func:`records_inventory` lists, per split, the dirty (LR),
  hr, clean and ``sources_<split>.csv`` files with sizes and record counts.
* **Truth sources.** :func:`record_sources` returns one record's sources from
  the generator's sidecar CSV (HR pixel coordinates, pixel centres at
  integers), :func:`record_source_detail` one source with every column,
  :func:`sources_summary` a per-record census for the split.
* **SR tier.** The SR cubes (``<vis>/sky_sr/sr_<split>_NNNN.npy``) carry a
  per-split manifest recording the model identity (STARFULL members + the
  production combiner fingerprint) and the input records' file state, so
  :func:`sr_state` tells current / stale / partial / missing / unknown.
"""
from __future__ import annotations

import contextlib
import csv
import glob
import hashlib
import json
import math
import os
import struct
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

from euclid_polish import ensemble_registry
from euclid_polish.config import Config
from euclid_polish.image import Image
from euclid_polish.image.tfio import deserialize_image, tfrecord_path
from euclid_polish.provenance.records import Artifact, Format

#: Subsets we generate SR for, in priority order. ``test`` first: it's the
#: held-out set evals (e.g. the power-spectrum summary) prefer.
SUBSETS = ("test", "validate", "train")

#: Record kinds of a split: the model input (dirty = LR), the starfull target
#: (hr) and the starless clean scene.
RECORD_KINDS = ("dirty", "hr", "clean")


# ---------------------------------------------------------------------------
# TFRecord framing: masked CRC32C and the header scan
# ---------------------------------------------------------------------------

def _crc32c_table() -> tuple[int, ...]:
    poly = 0x82F63B78                       # Castagnoli, reflected
    table = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = (c >> 1) ^ poly if c & 1 else c >> 1
        table.append(c)
    return tuple(table)


_CRC_TABLE = _crc32c_table()
_HEADER = struct.Struct("<QI")


def crc32c(data: bytes) -> int:
    c = 0xFFFFFFFF
    for byte in data:
        c = _CRC_TABLE[(c ^ byte) & 0xFF] ^ (c >> 8)
    return c ^ 0xFFFFFFFF


def masked_crc32c(data: bytes) -> int:
    """TFRecord's masked CRC32C (``((crc >> 15) | (crc << 17)) + 0xa282ead8``)."""
    c = crc32c(data)
    return (((c >> 15) | (c << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def _scan_offsets(path: str) -> tuple[list[int], bool]:
    """Frame start offsets of ``path`` and whether the file ended cleanly
    (``False``: a bad header CRC, or a frame running past the end — a
    truncated rsync or not a TFRecord at all)."""
    offsets: list[int] = []
    size = os.path.getsize(path)
    with open(path, "rb") as handle:
        pos = 0
        while pos < size:
            head = handle.read(_HEADER.size)
            if len(head) < _HEADER.size:
                return offsets, False
            length, crc = _HEADER.unpack(head)
            if masked_crc32c(head[:8]) != crc:
                return offsets, False
            end = pos + _HEADER.size + length + 4
            if end > size:
                return offsets, False
            offsets.append(pos)
            pos = end
            handle.seek(pos)
    return offsets, True


_OFFSETS: OrderedDict[str, tuple[tuple[int, int], list[int], bool]] = OrderedDict()
_OFFSETS_MAX = 64
_LOCK = threading.Lock()


def _file_key(path: str) -> tuple[int, int]:
    st = os.stat(path)
    return int(st.st_size), int(st.st_mtime_ns)


def tfrecord_offsets(path: str) -> tuple[list[int], bool]:
    """``(offsets, complete)`` of a TFRecord, cached by file size + mtime.
    Raises ``OSError`` when the file is absent."""
    real = os.path.realpath(path)
    key = _file_key(real)
    with _LOCK:
        hit = _OFFSETS.get(real)
        if hit is not None and hit[0] == key:
            _OFFSETS.move_to_end(real)
            return hit[1], hit[2]
    offsets, complete = _scan_offsets(real)
    with _LOCK:
        _OFFSETS[real] = (key, offsets, complete)
        _OFFSETS.move_to_end(real)
        while len(_OFFSETS) > _OFFSETS_MAX:
            _OFFSETS.popitem(last=False)
    return offsets, complete


def record_count(path: str) -> int | None:
    """Records in one TFRecord: ``0`` when absent, ``None`` when truncated or
    corrupt (render as "—"), else the count. Headers only."""
    if not os.path.exists(path):
        return 0
    try:
        offsets, complete = tfrecord_offsets(path)
    except OSError:
        return None
    return len(offsets) if complete else None


def read_record(path: str, index: int) -> Image:
    """Record ``index`` (0-based position) of a TFRecord, deserialised alone.
    ``IndexError`` past the readable records."""
    offsets, _complete = tfrecord_offsets(path)
    if index < 0 or index >= len(offsets):
        raise IndexError(f"record {index} out of range ({len(offsets)} readable)")
    with open(path, "rb") as handle:
        handle.seek(offsets[index])
        length, _crc = _HEADER.unpack(handle.read(_HEADER.size))
        payload = handle.read(length)
    return deserialize_image(payload)


# ---------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------

def _file_entry(path: str, *, records: bool) -> dict[str, Any] | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    entry: dict[str, Any] = {"name": os.path.basename(path), "size_bytes": int(st.st_size),
                             "mtime": float(st.st_mtime)}
    if records:
        entry["count"] = record_count(path)
    return entry


def sources_csv_path(records_dir: str, subset: str) -> str:
    return os.path.join(records_dir, f"sources_{subset}.csv")


def records_inventory(records_dir: str) -> dict[str, dict[str, Any]]:
    """Per split: ``{files: {dirty|hr|clean: {name, size_bytes, mtime, count}
    | None, sources: {…} | None}, count, present}`` (``count`` = the largest
    readable record count; ``present`` = any TFRecord on disk)."""
    out: dict[str, dict[str, Any]] = {}
    for subset in SUBSETS:
        files: dict[str, Any] = {
            kind: _file_entry(tfrecord_path(records_dir, f"{kind}_{subset}"), records=True)
            for kind in RECORD_KINDS
        }
        files["sources"] = _file_entry(sources_csv_path(records_dir, subset), records=False)
        counts = [f["count"] for k, f in files.items() if k != "sources" and f and f.get("count")]
        out[subset] = {
            "files": files,
            "count": max(counts) if counts else 0,
            "present": any(files[kind] for kind in RECORD_KINDS),
        }
    return out


_GEOMETRY: dict[tuple[str, tuple[int, int]], dict[str, Any]] = {}


def record_geometry(path: str) -> dict[str, Any] | None:
    """``{height, width, pixscale}`` of a TFRecord's first record (cached)."""
    try:
        key = (os.path.realpath(path), _file_key(path))
    except OSError:
        return None
    hit = _GEOMETRY.get(key)
    if hit is not None:
        return hit
    try:
        rec = read_record(path, 0)
    except (OSError, IndexError, ValueError):
        return None
    data = rec.data
    geometry = {"height": int(data.shape[0]), "width": int(data.shape[1]),
                "pixscale": float(rec.pixel_scale_arcsec or 0.0)}
    _GEOMETRY[key] = geometry
    return geometry


def split_geometry(records_dir: str, subset: str) -> dict[str, Any]:
    """The HR (hr, else clean) and LR (dirty) grids of a split."""
    hr = None
    for kind in ("hr", "clean"):
        hr = record_geometry(tfrecord_path(records_dir, f"{kind}_{subset}"))
        if hr is not None:
            break
    lr = record_geometry(tfrecord_path(records_dir, f"dirty_{subset}"))
    return {"hr": hr, "lr": lr}


# ---------------------------------------------------------------------------
# truth sources (sources_<subset>.csv)
# ---------------------------------------------------------------------------

#: Numeric columns sent per source in the compact (table / map) rows.
_SOURCE_NUMBERS = (
    "x_pix", "y_pix", "flux_vis_e", "flux_y_e", "flux_j_e", "flux_h_e", "z",
    "re_arcsec", "theta_E_arcsec", "orientation", "temperature_k",
    "mag_y_e", "mag_j_e", "mag_h_e", "target_vis_mag",
)
_SOURCE_TEXT = ("render", "subhalo_id", "source_subhalo_id", "sfr_class")
_JSON_COLUMNS = ("tng_render_trace", "lens_tng_trace", "source_tng_trace")
_TYPES = ("galaxy", "star", "lens")


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "1.0", "true", "yes", "on"}


def _compact(row: Mapping[str, str], position: int) -> dict[str, Any]:
    kind = (row.get("type") or "").strip() or "other"
    out: dict[str, Any] = {"row": position, "type": kind}
    for key in _SOURCE_NUMBERS:
        out[key] = _number(row.get(key))
    for key in _SOURCE_TEXT:
        out[key] = (row.get(key) or "").strip() or None
    out["off_field"] = _truthy(row.get("off_field"))
    # One VIS magnitude per source: a star's sampled magnitude, a galaxy's
    # achieved 2FWHM aperture magnitude (else its target), none for a lens.
    out["mag_vis"] = (_number(row.get("mag_vis")) if kind == "star"
                      else _number(row.get("achieved_vis_2fwhm_mag"))
                      or _number(row.get("target_vis_mag")))
    return out


_SOURCES: OrderedDict[str, tuple[tuple[int, int], dict[int, list[dict[str, Any]]]]] = OrderedDict()
_SOURCES_MAX = 6


def _parsed_sources(path: str) -> dict[int, list[dict[str, Any]]] | None:
    """``field_index → compact rows`` (CSV order), cached per file state;
    ``None`` when the CSV is absent."""
    try:
        key = _file_key(path)
    except OSError:
        return None
    real = os.path.realpath(path)
    with _LOCK:
        hit = _SOURCES.get(real)
        if hit is not None and hit[0] == key:
            return hit[1]
    by_field: dict[int, list[dict[str, Any]]] = {}
    with open(real, newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                field = int(float(row.get("field_index") or ""))
            except ValueError:
                continue
            rows = by_field.setdefault(field, [])
            rows.append(_compact(row, len(rows)))
    with _LOCK:
        _SOURCES[real] = (key, by_field)
        while len(_SOURCES) > _SOURCES_MAX:
            _SOURCES.popitem(last=False)
    return by_field


def _counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = dict.fromkeys(_TYPES, 0) | {"other": 0, "off_field": 0}
    for row in rows:
        counts[row["type"] if row["type"] in _TYPES else "other"] += 1
        counts["off_field"] += int(bool(row["off_field"]))
    return counts


def record_sources(records_dir: str, subset: str, field_index: int) -> dict[str, Any]:
    """One record's truth sources: ``{subset, field_index, present, sources,
    counts}``; positions are HR pixels (pixel centres at integers)."""
    parsed = _parsed_sources(sources_csv_path(records_dir, subset))
    rows = (parsed or {}).get(int(field_index), [])
    return {"subset": subset, "field_index": int(field_index), "present": parsed is not None,
            "sources": rows, "counts": _counts(rows)}


def record_source_detail(records_dir: str, subset: str, field_index: int,
                         row: int) -> dict[str, Any]:
    """Every column of one source (the ``row``-th of record ``field_index``):
    numbers parsed, JSON trace columns decoded, empty cells ``None``.
    ``KeyError`` when there is no such source."""
    path = sources_csv_path(records_dir, subset)
    if not os.path.isfile(path):
        raise KeyError("no sources file")
    position = 0
    with open(path, newline="") as handle:
        for raw in csv.DictReader(handle):
            try:
                field = int(float(raw.get("field_index") or ""))
            except ValueError:
                continue
            if field != int(field_index):
                continue
            if position == int(row):
                values: dict[str, Any] = {}
                for key, value in raw.items():
                    if key is None:
                        continue
                    if key in _JSON_COLUMNS:
                        try:
                            values[key] = json.loads(value) if value else None
                        except ValueError:
                            values[key] = value
                    elif value in ("", None):
                        values[key] = None
                    else:
                        number = _number(value)
                        values[key] = number if number is not None else value
                return {"subset": subset, "field_index": int(field_index), "row": int(row),
                        "source": _compact(raw, int(row)), "values": values}
            position += 1
    raise KeyError(f"record {field_index} has no source {row}")


def sources_summary(records_dir: str, subset: str) -> dict[str, Any]:
    """Per-record census of a split: ``{present, fields: [{field_index,
    galaxy, star, lens, other, off_field, n, brightest_star_mag,
    brightest_galaxy_mag, total_vis_e}]}``."""
    parsed = _parsed_sources(sources_csv_path(records_dir, subset))
    fields = []
    for field, rows in sorted((parsed or {}).items()):
        stars = [r["mag_vis"] for r in rows if r["type"] == "star" and r["mag_vis"] is not None]
        galaxies = [r["mag_vis"] for r in rows if r["type"] == "galaxy" and r["mag_vis"] is not None]
        total = sum(r["flux_vis_e"] or 0.0 for r in rows if not r["off_field"])
        fields.append({"field_index": field, **_counts(rows), "n": len(rows),
                       "brightest_star_mag": min(stars) if stars else None,
                       "brightest_galaxy_mag": min(galaxies) if galaxies else None,
                       "total_vis_e": total})
    return {"subset": subset, "present": parsed is not None, "fields": fields}


# ---------------------------------------------------------------------------
# SR cubes + their model identity
# ---------------------------------------------------------------------------

def sky_sr_dir() -> str:
    """Local directory holding generated sky SR cubes (one ``.npy`` each)."""
    return os.path.join(Config.VIS_DIR, "sky_sr")


def sr_path(subset: str, idx: int) -> str:
    return os.path.join(sky_sr_dir(), f"sr_{subset}_{int(idx):04d}.npy")


def sr_count(subset: str) -> int:
    """How many SR cubes have been generated for ``subset``."""
    return len(glob.glob(os.path.join(sky_sr_dir(), f"sr_{subset}_*.npy")))


def sr_manifest_path(subset: str) -> str:
    return os.path.join(sky_sr_dir(), f"sr_{subset}.json")


_FINGERPRINTS: OrderedDict[str, tuple[tuple[int, int], str]] = OrderedDict()


def records_fingerprint(path: str) -> str:
    """A content fingerprint of a TFRecord: SHA-1 over every frame's header and
    payload CRC (the writer's own checksum of each record's bytes), so it changes
    when any record changes but **not** when only the mtime does (every FASRC
    pull stamps ``os.utime``). Cached by size + mtime; raises ``OSError``."""
    real = os.path.realpath(path)
    key = _file_key(real)
    with _LOCK:
        hit = _FINGERPRINTS.get(real)
        if hit is not None and hit[0] == key:
            return hit[1]
    offsets, complete = tfrecord_offsets(real)
    digest = hashlib.sha1(struct.pack("<QQ?", key[0], len(offsets), complete))
    with open(real, "rb") as handle:
        for pos in offsets:
            handle.seek(pos)
            head = handle.read(_HEADER.size)
            length, _crc = _HEADER.unpack(head)
            handle.seek(pos + _HEADER.size + length)
            digest.update(head)
            digest.update(handle.read(4))
    value = digest.hexdigest()
    with _LOCK:
        _FINGERPRINTS[real] = (key, value)
        _FINGERPRINTS.move_to_end(real)
        while len(_FINGERPRINTS) > _OFFSETS_MAX:
            _FINGERPRINTS.popitem(last=False)
    return value


def _records_stamp(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    try:
        st = os.stat(path)
        fingerprint = records_fingerprint(path)
    except OSError:
        return None
    return {"name": os.path.basename(path), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns),
            "fingerprint": fingerprint}


def _records_changed(recorded: Mapping[str, Any], now: Mapping[str, Any]) -> bool:
    """Whether the input records differ from the ones an SR was made from: by
    content fingerprint, or by size alone for a manifest that predates it
    (never by mtime — a re-sync of unchanged records touches it)."""
    if recorded.get("size") != now.get("size"):
        return True
    was = recorded.get("fingerprint")
    return bool(was) and was != now.get("fingerprint")


def write_sr_manifest(subset: str, identity: Mapping[str, Any] | None, records_path: str | None,
                      *, count: int, model_label: str | None = None) -> dict[str, Any]:
    """Record which model made ``subset``'s SR cubes from which records."""
    manifest = {
        "subset": subset, "count": int(count), "model_label": model_label,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "identity": dict(identity) if identity else None,
        "records": _records_stamp(records_path),
    }
    os.makedirs(sky_sr_dir(), exist_ok=True)
    path = sr_manifest_path(subset)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as handle:
        json.dump(manifest, handle, indent=1)
    os.replace(tmp, path)
    return manifest


def read_sr_manifest(subset: str) -> dict[str, Any] | None:
    try:
        with open(sr_manifest_path(subset)) as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def clear_sr(subset: str) -> int:
    """Delete ``subset``'s SR cubes and manifest; the number of cubes removed."""
    removed = 0
    for path in glob.glob(os.path.join(sky_sr_dir(), f"sr_{subset}_*.npy")):
        try:
            os.remove(path)
            removed += 1
        except OSError:
            pass
    with contextlib.suppress(OSError):
        os.remove(sr_manifest_path(subset))
    return removed


def sr_state(subset: str, current: Mapping[str, Any] | None,
             records_path: str | None) -> dict[str, Any]:
    """The SR tier of one split against the model an SR run would load now
    (``current`` identity, ``None`` = unknown) and the input records:
    ``{state: current|stale|partial|missing|unknown, reasons, count,
    records_count, manifest}``."""
    count = sr_count(subset)
    records_count = record_count(records_path) if records_path and os.path.exists(records_path) else None
    manifest = read_sr_manifest(subset)
    out: dict[str, Any] = {"count": count, "records_count": records_count, "manifest": manifest,
                           "reasons": []}
    if count == 0:
        return {**out, "state": "missing"}
    if manifest is None:
        return {**out, "state": "unknown",
                "reasons": ["the SR predates model tracking — regenerate to verify it"]}
    reasons: list[str] = []
    recorded = manifest.get("identity") or {}
    if current and recorded:
        was, now = list(recorded.get("member_labels") or []), list(current.get("member_labels") or [])
        if was != now:
            reasons.append(f"members changed ({len(was)} → {len(now)} STARFULL members)")
        if (recorded.get("combiner_kind") or None) != (current.get("combiner_kind") or None):
            reasons.append(f"combiner changed ({recorded.get('combiner_kind') or 'mean'} → "
                           f"{current.get('combiner_kind') or 'mean'})")
        elif (recorded.get("combiner_fingerprint") or None) != (current.get("combiner_fingerprint") or None):
            reasons.append("the production combiner was refitted")
    stamp, now_stamp = manifest.get("records"), _records_stamp(records_path)
    if stamp and now_stamp and _records_changed(stamp, now_stamp):
        reasons.append("the records were regenerated since the SR")
    if reasons:
        return {**out, "state": "stale", "reasons": reasons}
    if records_count is not None and count < records_count:
        return {**out, "state": "partial",
                "reasons": [f"{count} of {records_count} records have an SR"]}
    return {**out, "state": "current"}


# ---------------------------------------------------------------------------
# presence helpers
# ---------------------------------------------------------------------------

def checkpoint_present(checkpoint: str | None = None) -> bool:
    """True when a usable model is on disk (cheap; no TensorFlow import).

    With no argument this asks the ensemble registry — THE model is the
    ensemble, so "a checkpoint exists" means "at least one active member".
    An explicit ``checkpoint`` dir keeps the old single-dir probe (a TF
    checkpoint dir carries a ``checkpoint`` pointer file plus per-step
    ``*.index`` shards).
    """
    if checkpoint is None:
        base = ensemble_registry.default_ensemble_dir()
        return bool(ensemble_registry.load_registry(base)["active"])
    if not os.path.isdir(checkpoint):
        return False
    return (os.path.isfile(os.path.join(checkpoint, "checkpoint"))
            or bool(glob.glob(os.path.join(checkpoint, "*.index"))))


def records_present(records_dir: str, subset: str = "validate") -> bool:
    """True when the dirty (LR) TFRecord for ``subset`` is in the local cache."""
    return os.path.exists(tfrecord_path(records_dir, f"dirty_{subset}"))


def present_subsets(records_dir: str) -> list[str]:
    return [s for s in SUBSETS if records_present(records_dir, s)]


def record_sr_cube(store, npy_path: str, subset: str, idx: int, *,
                   model_id=None, input_id=None, produced_by=None,
                   git=None, sidecar_dir: str | None = None) -> Artifact:
    """Persist an SR-cutout :class:`Artifact` for one SR cube, next to the data.

    Parents are ``(model_id, input_id)`` (whichever are known) so the cube can
    later be told apart from a stale one. The sidecar is named by the artifact's
    id; the ``.npy`` keeps its viewer-visible name.
    """
    parents = tuple(p for p in (model_id, input_id) if p is not None)
    art = Artifact.sr_cutout(
        id=store.mint(), git=git, produced_by=produced_by,
        format=Format.NPY, path=npy_path, parents=parents,
        descriptors={"subset": subset, "index": int(idx)},
    )
    store.put(art, sidecar_dir=sidecar_dir or sky_sr_dir())
    return art
