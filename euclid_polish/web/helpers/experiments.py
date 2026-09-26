"""Model comparison on REAL tiles (Sky → Experiments, spec §7.3 / §9.2).

An experiment = a set of real tiles (``source/id``) × a set of model specs
(:mod:`model_catalog`). The job (:func:`run_experiment`):

1. resolves the specs once (unavailable ones are *skipped* with the reason)
   and refuses (:class:`DiskSpaceError`) when the new outputs would leave
   less than ``MIN_FREE_BYTES`` free;
2. per tile, computes the union of the members the specs read — each member
   SR once, cached on disk under ``experiments/cache/<source>/<id>/`` keyed by
   the member's checkpoint fingerprint and the LR content hash (a retrained
   member or a re-cached tile invalidates it). The cache is bounded
   (:class:`MemberCacheBudget`): at most ``MEMBER_CACHE_BUDGET_BYTES`` over all
   tiles, least-recently-used entries evicted, and no cache write while the
   disk keeps less than ``MIN_FREE_BYTES + MEMBER_CACHE_BUDGET_BYTES`` free
   (the SR is then used from memory only);
3. applies every spec (reusing a current output — a store output whose spec
   fingerprint and LR hash still match, or a current legacy SR of the NEXUS /
   pair pipelines, which is scored and copied into the store) into the
   output store (``experiments/outputs/<source>/<id>/<slug>.fits``);
4. computes the real-data metrics (:mod:`real_metrics`) per (tile, spec) —
   plus, for spatial-gate specs, which members carry the weight in the
   brightest LR pixels — and pools them per spec;
5. writes the experiment record ``experiments/records/<id>.json``
   progressively (``running`` → ``done`` | ``failed`` | ``cancelled``).

Only one experiment computes at a time (TensorFlow members are memory
heavy); a second job waits, cancellable, for the first.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from euclid_polish.web.helpers import model_catalog, real_metrics, real_tiles
from euclid_polish.web.jobs import JobCancelled

RECORD_VERSION = 1
_EXPERIMENT_ID = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$")
_RUN_LOCK = threading.Lock()
#: Members that carry the most gate weight over the brightest LR pixels.
CORE_WEIGHT_TOP = 3
#: Upper bound on the member-SR cache over every tile (``.npy`` + sidecars).
MEMBER_CACHE_BUDGET_BYTES = 4 * 1024 ** 3
#: Free space an experiment must leave on the data disk after its outputs.
MIN_FREE_BYTES = 5 * 1024 ** 3
#: Assumed LR side when a tile's shape is unknown (disk estimates only).
_DEFAULT_SIDE = 256

Progress = Callable[[int, int, str], None]


class DiskSpaceError(ValueError):
    """The experiment's outputs would not fit on the data disk."""

    def __init__(self, message: str, *, needed: int, free: int) -> None:
        super().__init__(message)
        self.needed = int(needed)
        self.free = int(free)


def records_root() -> Path:
    return model_catalog.experiments_root() / "records"


def cache_root() -> Path:
    return model_catalog.member_cache_root()


def free_bytes(path: Path) -> int:
    """Free bytes on the file system holding ``path`` (its nearest existing
    ancestor when it does not exist yet)."""
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return int(shutil.disk_usage(probe).free)


def _gib(value: int) -> str:
    return f"{value / 1024 ** 3:.2f} GiB"


def new_experiment_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def check_experiment_id(identifier: str) -> str:
    if not _EXPERIMENT_ID.fullmatch(str(identifier or "")):
        raise KeyError(f"unknown experiment {identifier!r}")
    return str(identifier)


def _write_record(record: Mapping[str, Any]) -> None:
    path = records_root() / f"{record['id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def get_experiment(identifier: str) -> dict[str, Any]:
    """One experiment record (:class:`KeyError` when unknown)."""
    path = records_root() / f"{check_experiment_id(identifier)}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise KeyError(f"unknown experiment {identifier!r}") from exc
    return payload


def list_experiments(limit: int = 200) -> list[dict[str, Any]]:
    """Experiment summaries, newest first (no per-tile results)."""
    root = records_root()
    if not root.is_dir():
        return []
    out = []
    for path in sorted(root.glob("*.json"), reverse=True)[:limit]:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out.append({key: record.get(key) for key in (
            "id", "label", "created", "finished", "status", "job_id", "tiles", "models",
            "skipped", "summary", "errors", "counts")})
    return out


def experiments_for_tile(source: str, identifier: str) -> list[str]:
    ref = f"{source}/{identifier}"
    return [item["id"] for item in list_experiments() if ref in (item.get("tiles") or [])]


# ---------------------------------------------------------------------------
# the per-tile member cache
# ---------------------------------------------------------------------------

def member_cache_dir(source: str, identifier: str) -> Path:
    return model_catalog.member_cache_dir(source, identifier)


def _label_slug(label: str) -> str:
    return model_catalog.member_name(label)


class MemberCacheBudget:
    """Byte budget over the whole member-SR cache, least recently used first.

    Scans ``cache_root()`` once (``*.npy`` + their ``.json`` sidecars, oldest
    mtime first); a cache hit touches the file (LRU), a write evicts the
    least recently used entries until the cache fits
    ``MEMBER_CACHE_BUDGET_BYTES`` again. :meth:`admits` refuses a write that
    is larger than the budget or would leave less than
    ``MIN_FREE_BYTES + MEMBER_CACHE_BUDGET_BYTES`` free — the cache is an
    optimisation and must stop well before the experiment outputs do.
    """

    def __init__(self, root: Path | None = None, *, budget: int | None = None,
                 min_free: int | None = None) -> None:
        self.root = Path(root) if root is not None else cache_root()
        self.budget = int(MEMBER_CACHE_BUDGET_BYTES if budget is None else budget)
        self.min_free = int((MIN_FREE_BYTES + self.budget) if min_free is None else min_free)
        self._files: dict[Path, int] = {}
        found = []
        if self.root.is_dir():
            for path in self.root.rglob("*.npy"):
                if path.name.startswith("."):
                    continue
                with contextlib.suppress(OSError):
                    found.append((path.stat().st_mtime_ns, str(path), path,
                                  self._size(path)))
        for _mtime, _name, path, size in sorted(found):
            self._files[path] = size
        self.total = sum(self._files.values())
        self.evicted = 0
        self.not_cached = 0

    @staticmethod
    def _size(path: Path) -> int:
        size = 0
        for item in (path, path.with_suffix(".json")):
            with contextlib.suppress(OSError):
                size += item.stat().st_size
        return size

    def admits(self, nbytes: int) -> bool:
        if nbytes > self.budget:
            return False
        return free_bytes(self.root) - int(nbytes) >= self.min_free

    def used(self, path: Path) -> None:
        """A cache hit: mark ``path`` most recently used."""
        with contextlib.suppress(OSError):
            os.utime(path)
        if path in self._files:
            self._files[path] = self._files.pop(path)

    def added(self, path: Path) -> None:
        """A new cache entry: account for it, then evict down to the budget."""
        self.total -= self._files.pop(path, 0)
        size = self._size(path)
        self._files[path] = size
        self.total += size
        while self.total > self.budget:
            victim = next((item for item in self._files if item != path), None)
            if victim is None:
                break
            self.total -= self._files.pop(victim)
            for item in (victim, victim.with_suffix(".json")):
                with contextlib.suppress(OSError):
                    item.unlink()
            self.evicted += 1


class CachedTileMembers:
    """:class:`model_catalog.MemberSource` for one tile backed by the disk
    cache: a member's SR is reused while its checkpoint fingerprint and the
    tile's LR hash match, else computed with ``runner.predict`` and stored
    (when ``budget`` admits it; the SR stays in memory for the tile either
    way)."""

    def __init__(self, source: str, identifier: str, lr_e: np.ndarray, *, lr_sha: str,
                 runner: Any, fingerprints: Mapping[str, str | None],
                 on_compute: Callable[[str], None] | None = None,
                 budget: MemberCacheBudget | None = None) -> None:
        self.directory = member_cache_dir(source, identifier)
        self.lr = np.asarray(lr_e, np.float32)
        self.lr_sha = lr_sha
        self.runner = runner
        self.fingerprints = dict(fingerprints)
        self.on_compute = on_compute
        self.budget = budget
        self.memory: dict[str, np.ndarray] = {}
        self.computed: list[str] = []
        self.reused: list[str] = []
        self.not_cached: list[str] = []

    def _paths(self, label: str) -> tuple[Path, Path]:
        slug = _label_slug(label)
        return self.directory / f"{slug}.npy", self.directory / f"{slug}.json"

    def cached(self, label: str) -> bool:
        array_path, meta_path = self._paths(label)
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return (array_path.is_file() and meta.get("lr_sha") == self.lr_sha
                and meta.get("member_fingerprint") == self.fingerprints.get(label)
                and self.fingerprints.get(label) is not None)

    def _store(self, label: str, value: np.ndarray) -> None:
        array_path, meta_path = self._paths(label)
        if self.budget is not None and not self.budget.admits(value.nbytes):
            self.not_cached.append(label)
            self.budget.not_cached += 1
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = array_path.with_name(f".{array_path.stem}.{os.getpid()}.tmp.npy")
        np.save(temporary, value)
        os.replace(temporary, array_path)
        meta_path.write_text(json.dumps({
            "label": label, "member_fingerprint": self.fingerprints.get(label),
            "lr_sha": self.lr_sha, "shape": [int(v) for v in value.shape],
            "created": datetime.now(UTC).isoformat(),
        }, indent=2) + "\n", encoding="utf-8")
        if self.budget is not None:
            self.budget.added(array_path)

    def get(self, label: str) -> np.ndarray:
        if label in self.memory:
            return self.memory[label]
        array_path, _meta_path = self._paths(label)
        if self.cached(label):
            value = np.load(array_path).astype(np.float32)
            if self.budget is not None:
                self.budget.used(array_path)
            self.reused.append(label)
        else:
            if self.on_compute is not None:
                self.on_compute(label)
            value = np.asarray(self.runner.predict(self.lr, label), np.float32)
            self._store(label, value)
            self.computed.append(label)
        self.memory[label] = value
        return value


def delete_tile_outputs(source: str, identifier: str) -> dict[str, Any]:
    """Remove a tile's model outputs and its member-SR cache."""
    removed = model_catalog.delete_outputs(source, identifier)
    freed = 0
    cache = member_cache_dir(source, identifier)
    if cache.is_dir():
        for path in cache.rglob("*"):
            if path.is_file():
                freed += path.stat().st_size
                removed.append(str(path))
        shutil.rmtree(cache)
    return {"removed": removed, "removed_count": len(removed), "cache_bytes_freed": freed}


# ---------------------------------------------------------------------------
# metrics helpers
# ---------------------------------------------------------------------------

def gate_core_weights(weights: np.ndarray, lr_e: np.ndarray, labels: Sequence[str],
                      band_names: Sequence[str]) -> dict[str, list[list[Any]]]:
    """Per band, the members with the largest mean gate weight over the SR
    pixels of the brightest 1 % of LR pixels: ``{band: [[label, w], …]}``."""
    out: dict[str, list[list[Any]]] = {}
    lr = np.asarray(lr_e, np.float64)
    factor = weights.shape[0] // lr.shape[0]
    for band_index, band in enumerate(band_names):
        plane = lr[..., band_index]
        finite = np.isfinite(plane)
        if not finite.any():
            continue
        bright = finite & (plane >= np.percentile(plane[finite], 99.0)) & (plane > 0)
        if not bright.any():
            continue
        mask = np.kron(bright, np.ones((factor, factor))) > 0
        mask = mask[:weights.shape[0], :weights.shape[1]]
        mean = weights[mask][:, :, band_index].mean(axis=0)
        order = np.argsort(-mean)[:CORE_WEIGHT_TOP]
        out[band] = [[str(labels[i]), round(float(mean[i]), 4)] for i in order]
    return out


def _compact(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Per-band metrics without the per-peak rows (records stay small)."""
    return {"per_band": metrics.get("per_band"), "summary": metrics.get("summary"),
            **({"gate_core_weights": metrics["gate_core_weights"]}
               if "gate_core_weights" in metrics else {})}


# ---------------------------------------------------------------------------
# the job
# ---------------------------------------------------------------------------

def _acquire(check: Callable[[], None], log: Callable[[str], None]) -> None:
    waited = False
    while not _RUN_LOCK.acquire(timeout=1.0):
        if not waited:
            log("waiting for the running experiment to finish")
            waited = True
        check()


def output_bytes(entry: real_tiles.TileEntry) -> int:
    """Bytes of one stored SR output of a tile (``(4, 2H, 2W)`` float32 +
    header and sidecar)."""
    height, width = entry.shape or (_DEFAULT_SIDE, _DEFAULT_SIDE)
    factor = model_catalog.SR_FACTOR
    return int(height * factor * width * factor * len(entry.bands) * 4 + 64 * 1024)


def check_disk(entries: Sequence[real_tiles.TileEntry],
               specs: Sequence[model_catalog.ModelSpec]) -> int:
    """The bytes the new (tile, spec) outputs need; :class:`DiskSpaceError`
    when writing them would leave less than ``MIN_FREE_BYTES`` free. Outputs
    already in the store are replaced in place and not counted."""
    needed = 0
    for entry in entries:
        stored = model_catalog.list_outputs(entry.source, entry.id)
        needed += sum(output_bytes(entry) for item in specs if item.spec not in stored)
    free = free_bytes(model_catalog.experiments_root())
    if free - needed < MIN_FREE_BYTES:
        raise DiskSpaceError(
            f"not enough disk space: the outputs need {_gib(needed)}, {_gib(free)} is "
            f"free and {_gib(MIN_FREE_BYTES)} must stay free — delete cached outputs "
            "(POST /api/real/<source>/<id>/delete-outputs) or pick fewer tiles/models",
            needed=needed, free=free)
    return needed


def plan(tile_refs: Iterable[tuple[str, str]], specs: Iterable[str]
         ) -> tuple[list[real_tiles.TileEntry], list[model_catalog.ModelSpec],
                    dict[str, str]]:
    """Validate an experiment request: ``(entries, runnable specs, skipped)``.

    Raises :class:`real_tiles.RealTileError` (unknown / not four-band tile),
    :class:`DiskSpaceError` (the outputs would not fit) or
    :class:`ValueError` (bad or unknown spec, nothing runnable).
    """
    entries = []
    for source, identifier in tile_refs:
        entry = real_tiles.get_entry(source, identifier)
        if not entry.model_ready:
            raise real_tiles.RealTileError(
                409, f"{entry.ref} has no four-band LR yet ({', '.join(entry.bands)})")
        entries.append(entry)
    if not entries:
        raise ValueError("select at least one real tile (tiles=source/id,…)")
    catalog = model_catalog.list_specs()
    runnable: list[model_catalog.ModelSpec] = []
    skipped: dict[str, str] = {}
    for spec in model_catalog.parse_specs(list(specs)):
        try:
            item = model_catalog.resolve_spec(spec, catalog)
        except KeyError as exc:
            raise ValueError(str(exc.args[0]) if exc.args else str(exc)) from exc
        if item.available:
            runnable.append(item)
        else:
            skipped[item.spec] = str(item.reason)
    if not runnable:
        detail = "; ".join(f"{k}: {v}" for k, v in skipped.items()) or "no models given"
        raise ValueError(f"no runnable model in the request ({detail})")
    check_disk(entries, runnable)
    return entries, runnable, skipped


def run_experiment(
    tile_refs: Sequence[tuple[str, str]],
    specs: Sequence[str],
    *,
    experiment_id: str | None = None,
    label: str | None = None,
    job_id: str | None = None,
    progress: Progress | None = None,
    check_cancelled: Callable[[], None] | None = None,
    runner: Any | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run an experiment (see the module docstring); returns the record."""
    entries, runnable, skipped = plan(tile_refs, specs)
    identifier = experiment_id or new_experiment_id()
    check = check_cancelled or (lambda: None)
    tick = progress or (lambda *_a: None)
    record: dict[str, Any] = {
        "version": RECORD_VERSION, "id": identifier, "label": label or "",
        "created": datetime.now(UTC).isoformat(), "finished": None,
        "status": "running", "job_id": job_id,
        "tiles": [entry.ref for entry in entries],
        "models": [item.spec for item in runnable] + list(skipped),
        "skipped": skipped,
        "fingerprints": {item.spec: item.fingerprint for item in runnable},
        "model_labels": {item.spec: item.label for item in runnable},
        "definitions": real_metrics.definitions(),
        "results": {}, "summary": {}, "errors": {},
        "counts": {"members_computed": 0, "members_reused": 0,
                   "members_not_cached": 0, "members_evicted": 0,
                   "outputs_computed": 0, "outputs_reused": 0},
    }
    _write_record(record)
    needed = model_catalog.needed_members(runnable)
    current_fps = {item.spec: item.fingerprint for item in runnable}
    if runner is None:
        # Restore only the members of specs that will actually run: a spec
        # whose output already carries its fingerprint on every tile is
        # reused (the runner widens itself if that guess proves wrong).
        to_run = [item for item in runnable if any(
            (real_tiles.tile_outputs(entry, current_fps).get(item.spec) or {}).get(
                "fingerprint") != item.fingerprint for entry in entries)]
        runner = model_catalog.EnsembleMemberRunner(
            labels=model_catalog.needed_members(to_run))
    total = len(entries) * (len(needed) + len(runnable))
    step = 0
    per_spec: dict[str, list[dict[str, Any]]] = {item.spec: [] for item in runnable}
    locked = False
    started = time.monotonic()
    try:
        _acquire(check, log)
        locked = True
        fingerprints = model_catalog.member_fingerprints(needed)
        budget = MemberCacheBudget()
        for tile_number, entry in enumerate(entries, start=1):
            check()
            tile = real_tiles.get_tile(entry.source, entry.id, entry=entry)
            lr = tile.lr_e
            finite = np.isfinite(lr)
            lr_input = np.where(finite, lr, 0.0).astype(np.float32)
            lr_sha = model_catalog.array_sha(lr_input)
            outputs = real_tiles.tile_outputs(entry, current_fps)
            tile_results: dict[str, Any] = {}

            def on_compute(member: str, _n=tile_number) -> None:
                nonlocal step
                step += 1
                tick(step, total, f"tile {_n}/{len(entries)} · member {member}")

            members = CachedTileMembers(
                entry.source, entry.id, lr_input, lr_sha=lr_sha, runner=runner,
                fingerprints=fingerprints, on_compute=on_compute, budget=budget)
            for item in runnable:
                check()
                meta = outputs.get(item.spec)
                legacy = bool(meta and meta.get("legacy"))
                # A legacy SR records no LR hash: it was made from this tile's
                # own LR file, so its identity alone decides.
                current = (meta is not None and meta.get("fingerprint") == item.fingerprint
                           and (meta.get("lr_sha") == lr_sha
                                or (legacy and meta.get("lr_sha") is None)))
                try:
                    if current and not legacy and (meta.get("metrics") or {}).get(
                            "version") == real_metrics.METRICS_VERSION:
                        metrics = meta["metrics"]
                        record["counts"]["outputs_reused"] += 1
                        state = "reused"
                    elif current:
                        # A current SR without metrics (a legacy NEXUS / pair
                        # inference SR, or an older store output): score it,
                        # no model run; a legacy SR is copied into the store.
                        sr, _header, _meta = real_tiles.load_output(
                            entry, item.spec, current=item.fingerprint)
                        metrics = real_metrics.tile_metrics(lr, sr)
                        if legacy:
                            meta = model_catalog.save_output(
                                entry.source, entry.id, item, sr,
                                lr_header=tile.wcs_header, lr_sha=lr_sha,
                                extra={"experiment_id": identifier, "metrics": metrics,
                                       "from_legacy": meta.get("path"),
                                       "nonfinite_lr_fraction": float(1.0 - finite.mean())})
                        else:
                            meta = model_catalog.update_output_meta(
                                entry.source, entry.id, item.spec, {"metrics": metrics})
                        record["counts"]["outputs_reused"] += 1
                        state = "reused"
                    else:
                        sr = model_catalog.predict(item, lr_input, members)
                        metrics = real_metrics.tile_metrics(lr, sr)
                        weights = (model_catalog.gate_weights(item, lr_input, members)
                                   if item.combiner_kind == model_catalog.PRODUCTION_KIND else None)
                        if weights is not None:
                            metrics["gate_core_weights"] = gate_core_weights(
                                weights, lr, item.member_labels, metrics["bands"])
                        meta = model_catalog.save_output(
                            entry.source, entry.id, item, sr,
                            lr_header=tile.wcs_header, lr_sha=lr_sha,
                            extra={"experiment_id": identifier, "metrics": metrics,
                                   "nonfinite_lr_fraction": float(1.0 - finite.mean())})
                        record["counts"]["outputs_computed"] += 1
                        state = "computed"
                    per_spec[item.spec].append(metrics)
                    tile_results[item.spec] = {
                        "state": state, "fingerprint": item.fingerprint,
                        "file": meta.get("file") if meta else None,
                        "metrics": _compact(metrics),
                    }
                except (JobCancelled, KeyboardInterrupt):
                    raise
                except Exception as exc:  # noqa: BLE001 - keep the other (tile, spec) results
                    record["errors"][f"{entry.ref}|{item.spec}"] = f"{type(exc).__name__}: {exc}"
                step += 1
                tick(min(step, total), total,
                     f"tile {tile_number}/{len(entries)} · {item.spec}")
            record["counts"]["members_computed"] += len(members.computed)
            record["counts"]["members_reused"] += len(members.reused)
            record["counts"]["members_not_cached"] = budget.not_cached
            record["counts"]["members_evicted"] = budget.evicted
            record["results"][entry.ref] = tile_results
            record["summary"] = {spec: real_metrics.aggregate(items)
                                 for spec, items in per_spec.items() if items}
            _write_record(record)
        record["status"] = ("failed" if record["errors"] and not any(per_spec.values())
                            else "done")
    except (JobCancelled, KeyboardInterrupt):
        record["status"] = "cancelled"
        raise
    except Exception as exc:
        record["status"] = "failed"
        record["errors"]["experiment"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if locked:
            _RUN_LOCK.release()
        record["finished"] = datetime.now(UTC).isoformat()
        record["duration_s"] = round(time.monotonic() - started, 3)
        record["summary"] = {spec: real_metrics.aggregate(items)
                             for spec, items in per_spec.items() if items}
        with contextlib.suppress(OSError, ValueError, TypeError):
            _write_record(record)
    return record


def job_result(record: Mapping[str, Any]) -> dict[str, Any]:
    """Small JSON result for the jobs API (the record itself is fetched by id)."""
    return {"experiment_id": record.get("id"), "status": record.get("status"),
            "tiles": len(record.get("tiles") or []), "models": record.get("models"),
            "errors": len(record.get("errors") or {}), "counts": record.get("counts")}


__all__ = [
    "CachedTileMembers",
    "DiskSpaceError",
    "MEMBER_CACHE_BUDGET_BYTES",
    "MIN_FREE_BYTES",
    "MemberCacheBudget",
    "check_disk",
    "check_experiment_id",
    "delete_tile_outputs",
    "free_bytes",
    "experiments_for_tile",
    "gate_core_weights",
    "get_experiment",
    "job_result",
    "list_experiments",
    "member_cache_dir",
    "new_experiment_id",
    "output_bytes",
    "plan",
    "records_root",
    "run_experiment",
]
