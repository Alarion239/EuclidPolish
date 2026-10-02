"""System facts and the Home health checks (System › Code and Storage, Home).

``GET /api/system``
    The runtime (Python, platform, key package versions, Node when it is on
    the server's ``PATH``), FREE SPACE on the data disk with a warning level,
    the last measured disk usage of every data root and the experiments'
    member-cache budget. Cheap: nothing is walked on a GET.
``POST /api/system/disk-usage/refresh``
    Measure every data root in a local job (one at a time, kind
    ``system-disk-usage``); the result is kept in memory and in a small JSON
    file so it survives a restart.
``GET /api/system/production``
    Home's production numbers without the heavy ensemble status: the scalar
    keys of the STARFULL ``eval_summary.json`` (the production gate's
    ``spatial_gate_*`` block included), when it was written, whether it is
    stale (the evaluation check: members or test records changed) and the
    active STARFULL / starless member counts.
``GET /api/system/alerts``
    The Home health checks, each ``ok | warn | bad | unknown``: free disk
    space, real SR products vs the production model, the production gate vs
    the STARFULL membership, the evaluation vs members and test records, the
    PSNR-vs-knee curves, the training records' noise model vs
    ``Config.NOISE_MODEL`` and results newer than the tracking log's last
    entry. Memoised for :data:`ALERTS_TTL_S` (``?fresh=1`` recomputes); a
    check that raises reads ``unknown`` with its error.

Everything is local: nothing here needs FASRC.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flask import jsonify, request

from euclid_polish import ensemble_registry
from euclid_polish.config import Config
from euclid_polish.image.collection import ImageSet
from euclid_polish.image.tfio import tfrecord_path
from euclid_polish.provenance.defaults import default_store as provenance_store
from euclid_polish.tracking.store import TrackingStore
from euclid_polish.web import errors
from euclid_polish.web.helpers import experiments, model_catalog, sky_atlas, system_alerts
from euclid_polish.web.helpers.ensemble_viz import knee_psnr_status
from euclid_polish.web.helpers.paths import _sky_records_local_dir
from euclid_polish.web.jobs import REGISTRY

GIB = 1024 ** 3
#: Free space below which the data disk reads ``warn`` (also at ≥ 95 % used).
DISK_WARN_FREE_BYTES = 25 * GIB
#: Free space below which it reads ``bad``: under this the experiments'
#: member-SR cache stops writing (5 GiB floor + 4 GiB budget) and a new
#: experiment is close to being refused (507).
DISK_BAD_FREE_BYTES = 10 * GIB
DISK_WARN_USED_FRACTION = 0.95
#: A measured disk usage older than this reads ``stale`` (the UI offers a refresh).
DISK_USAGE_TTL_S = 6 * 3600.0
DISK_USAGE_JOB_KIND = "system-disk-usage"
DISK_USAGE_CACHE_PATH = os.path.expanduser("~/.euclid_polish/system_disk_usage.json")
ALERTS_TTL_S = 30.0
#: A result newer than the last tracking entry by more than this is "unlogged".
TRACKING_LAG_S = 3600.0
#: The repository checkout (poster/, output/ live here).
REPO_ROOT = str(Path(__file__).resolve().parents[3])

_PACKAGES = ("flask", "werkzeug", "numpy", "scipy", "astropy", "tensorflow", "photutils")
_LOCK = threading.Lock()
_SPAWN_LOCK = threading.Lock()
_DISK_CACHE: dict[str, Any] | None = None
_RUNTIME: dict[str, Any] | None = None
_ALERTS: tuple[float, dict[str, Any]] | None = None
_NOISE_CACHE: dict[tuple[str, int, int], str | None] = {}


def reset_caches() -> None:
    """Forget every memoised value (tests; a restart does the same)."""
    global _DISK_CACHE, _RUNTIME, _ALERTS
    with _LOCK:
        _DISK_CACHE = None
        _RUNTIME = None
        _ALERTS = None
        _NOISE_CACHE.clear()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _gib(value: float) -> str:
    return f"{value / GIB:.1f} GiB"


# ---------------------------------------------------------------------------
# runtime
# ---------------------------------------------------------------------------

def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def node_version() -> str | None:
    """``node --version`` when Node is on the server's PATH, else ``None``."""
    node = shutil.which("node")
    if not node:
        return None
    try:
        out = subprocess.run([node, "--version"], capture_output=True, text=True,
                             timeout=3, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    text = (out.stdout or "").strip()
    return text or None


def runtime_info() -> dict[str, Any]:
    """Python / platform / packages / Node (computed once per process)."""
    global _RUNTIME
    with _LOCK:
        if _RUNTIME is not None:
            return _RUNTIME
    info = {
        "python": {"version": platform.python_version(),
                   "implementation": platform.python_implementation(),
                   "executable": sys.executable},
        "platform": {"system": platform.system(), "release": platform.release(),
                     "machine": platform.machine(), "platform": platform.platform(terse=True)},
        "packages": {name: _package_version(name) for name in _PACKAGES},
        "node": node_version(),
    }
    with _LOCK:
        _RUNTIME = info
    return info


# ---------------------------------------------------------------------------
# free space
# ---------------------------------------------------------------------------

def disk_level(free: int, total: int) -> str:
    """``ok | warn | bad | unknown`` for the data disk (see the thresholds)."""
    if total <= 0:
        return "unknown"
    if free < DISK_BAD_FREE_BYTES:
        return "bad"
    if free < DISK_WARN_FREE_BYTES or (total - free) / total >= DISK_WARN_USED_FRACTION:
        return "warn"
    return "ok"


def _existing(path: str) -> str:
    probe = Path(os.path.abspath(path))
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return str(probe)


def disk_payload(path: str | None = None) -> dict[str, Any]:
    """Free space of the file system holding ``path`` (default: the data dir)."""
    where = _existing(path or Config.DATA_DIR)
    try:
        usage = shutil.disk_usage(where)
        total, free = int(usage.total), int(usage.free)
    except OSError:
        total = free = 0
    return {
        "path": where, "total_bytes": total, "free_bytes": free,
        "used_bytes": max(0, total - free),
        "used_fraction": (total - free) / total if total > 0 else None,
        "level": disk_level(free, total),
        "warn_below_bytes": DISK_WARN_FREE_BYTES, "bad_below_bytes": DISK_BAD_FREE_BYTES,
        "warn_used_fraction": DISK_WARN_USED_FRACTION,
    }


# ---------------------------------------------------------------------------
# disk usage per data root (measured by a job)
# ---------------------------------------------------------------------------

def data_roots() -> list[dict[str, str]]:
    """Every root worth measuring: each directory under ``Config.DATA_DIR``,
    the checkpoints, the tracking store and the repo's ``poster/``/``output/``."""
    roots: list[dict[str, str]] = []
    data = os.path.abspath(Config.DATA_DIR)
    with contextlib.suppress(OSError):
        for entry in sorted(os.scandir(data), key=lambda e: e.name):
            if entry.is_dir(follow_symlinks=False):
                roots.append({"id": f"data/{entry.name}", "label": entry.name,
                              "path": entry.path, "group": "data"})
    ckpt = os.path.dirname(os.path.abspath(Config.DEFAULT_CHECKPOINT_DIR.rstrip("/"))) or "."
    for rid, label, path in (
        ("ckpt", "Checkpoints", ckpt),
        ("tracking", "Tracking", os.path.abspath(Config.TRACKING_DIR)),
        ("poster", "Poster", os.path.join(REPO_ROOT, "poster")),
        ("output", "Output", os.path.join(REPO_ROOT, "output")),
    ):
        roots.append({"id": rid, "label": label, "path": path, "group": "repo"})
    seen: set[str] = set()
    unique = []
    for root in roots:
        real = os.path.realpath(root["path"])
        if real in seen:
            continue
        seen.add(real)
        unique.append(root)
    return unique


def measure_tree(path: str, *, tick: Callable[[], None] | None = None) -> dict[str, Any]:
    """Apparent size and file count under ``path`` (symlinks not followed)."""
    if not os.path.isdir(path):
        return {"bytes": 0, "files": 0, "exists": False}
    total = files = seen = 0
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            seen += 1
            if tick is not None and seen % 2000 == 0:
                tick()
            try:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    stack.append(entry.path)
                elif entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
                    files += 1
            except OSError:
                continue
    return {"bytes": int(total), "files": int(files), "exists": True}


def _write_cache(payload: dict[str, Any]) -> None:
    directory = os.path.dirname(DISK_USAGE_CACHE_PATH)
    os.makedirs(directory, exist_ok=True)
    tmp = DISK_USAGE_CACHE_PATH + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(payload, handle)
    os.replace(tmp, DISK_USAGE_CACHE_PATH)


def compute_disk_usage(progress: Callable[[int, int, str], None] | None = None) -> dict[str, Any]:
    """Measure every root (+ the experiments' outputs and member cache), keep
    the result in memory and in :data:`DISK_USAGE_CACHE_PATH`."""
    global _DISK_CACHE
    roots = data_roots()
    extra = [
        {"id": "experiments/cache", "label": "Experiments · member-SR cache",
         "path": str(model_catalog.member_cache_root())},
        {"id": "experiments/outputs", "label": "Experiments · model outputs",
         "path": str(model_catalog.outputs_root())},
    ]
    items = []
    todo = [*roots, *extra]
    for index, root in enumerate(todo):
        if progress is not None:
            progress(index, len(todo), root["label"])
        tick = (lambda i=index, r=root: progress(i, len(todo), r["label"])) if progress else None
        items.append({**root, **measure_tree(root["path"], tick=tick)})
    measured = [item for item in items if not item["id"].startswith("experiments/")]
    measured.sort(key=lambda item: item["bytes"], reverse=True)
    by_id = {item["id"]: item for item in items}
    payload = {
        "computed_at": _now_iso(), "computed_ts": time.time(),
        "items": measured,
        "total_bytes": sum(item["bytes"] for item in measured),
        "experiments": {"cache_bytes": by_id["experiments/cache"]["bytes"],
                        "outputs_bytes": by_id["experiments/outputs"]["bytes"]},
    }
    if progress is not None:
        progress(len(todo), len(todo), "done")
    with _LOCK:
        _DISK_CACHE = payload
    _write_cache(payload)
    return payload


def _read_disk_cache() -> dict[str, Any] | None:
    global _DISK_CACHE
    with _LOCK:
        if _DISK_CACHE is not None:
            return _DISK_CACHE
    try:
        with open(DISK_USAGE_CACHE_PATH) as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    with _LOCK:
        _DISK_CACHE = payload
    return payload


def _running_refresh() -> str | None:
    for job in REGISTRY.list(summary=True):
        if job.get("kind") == DISK_USAGE_JOB_KIND and job.get("status") == "running":
            return str(job["job_id"])
    return None


def spawn_disk_usage_refresh() -> str:
    """Start the measuring job unless one already runs (its id either way)."""
    with _SPAWN_LOCK:
        running = _running_refresh()
        if running is not None:
            return running
        return REGISTRY.spawn("System: measure disk usage per data root",
                              lambda cap: _summary(compute_disk_usage(progress=cap.tick)),
                              kind=DISK_USAGE_JOB_KIND)


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {"computed_at": payload.get("computed_at"), "roots": len(payload.get("items") or []),
            "total_bytes": payload.get("total_bytes")}


def disk_usage_status() -> dict[str, Any]:
    """The last measurement (never measures): items, age, ``stale``, job."""
    cache = _read_disk_cache() or {}
    computed_ts = cache.get("computed_ts")
    stale = not isinstance(computed_ts, (int, float)) or time.time() - computed_ts > DISK_USAGE_TTL_S
    return {
        "items": list(cache.get("items") or []),
        "computed_at": cache.get("computed_at"),
        "total_bytes": cache.get("total_bytes"),
        "stale": bool(stale),
        "ttl_s": DISK_USAGE_TTL_S,
        "refresh_job": _running_refresh(),
        "experiments": cache.get("experiments"),
    }


def system_payload() -> dict[str, Any]:
    roots = disk_usage_status()
    measured = roots.get("experiments") or {}
    return {
        **runtime_info(),
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "data_dir": os.path.abspath(Config.DATA_DIR),
        "noise_model": Config.NOISE_MODEL,
        "disk": disk_payload(),
        "roots": roots,
        "experiments": {
            "cache_budget_bytes": experiments.MEMBER_CACHE_BUDGET_BYTES,
            "min_free_bytes": experiments.MIN_FREE_BYTES,
            "cache_bytes": measured.get("cache_bytes"),
            "outputs_bytes": measured.get("outputs_bytes"),
            "measured_at": roots.get("computed_at"),
        },
    }


# ---------------------------------------------------------------------------
# Home health checks
# ---------------------------------------------------------------------------

def records_dir() -> str:
    """The local mirror of the synthetic TFRecords (the Synthetic › Records source)."""
    return _sky_records_local_dir()


def experiment_records_root() -> Path:
    return experiments.records_root()


def check_disk() -> dict[str, Any]:
    disk = disk_payload()
    used = disk["used_fraction"]
    pct = f" ({used * 100:.0f} % used)" if used is not None else ""
    title = f"{_gib(disk['free_bytes'])} free on the data disk{pct}"
    detail = (f"Experiments refuse to start when fewer than {_gib(experiments.MIN_FREE_BYTES)} "
              f"would stay free; the member-SR cache stops writing below "
              f"{_gib(experiments.MIN_FREE_BYTES + experiments.MEMBER_CACHE_BUDGET_BYTES)}.")
    return {"state": disk["level"], "title": title, "detail": detail, "to": "/system/storage",
            "facts": {"free_bytes": disk["free_bytes"], "total_bytes": disk["total_bytes"]}}


#: (sky layer id, real source, label) of the sources that carry production SRs.
_SR_LAYERS = (
    ("nexus-tiles", "nexus", "NEXUS tiles"),
    ("real-tiles", "tile", "Cached 25.6″ tiles"),
    ("real-fields", "field", "Legacy real-field tiles"),
    ("poster", "poster", "Poster galaxy"),
    ("pairs", "pair", "JWST × Euclid pairs"),
)


def check_real_sr() -> dict[str, Any]:
    """Production SR state of every real tile (the sky layers' ``state``)."""
    sources = []
    totals = {"current": 0, "stale": 0, "missing": 0}
    for layer_id, source, label in _SR_LAYERS:
        features = sky_atlas.layer_features(layer_id).get("features") or []
        if not features:
            continue
        counts = {"current": 0, "stale": 0, "missing": 0}
        for feature in features:
            state = (feature.get("props") or {}).get("state")
            if state in counts:
                counts[state] += 1
        for key, value in counts.items():
            totals[key] += value
        sources.append({"source": source, "label": label, **counts})
    facts = {**totals, "sources": sources}
    if totals["stale"]:
        worst = ", ".join(f"{s['label']} {s['stale']}" for s in sources if s["stale"])
        return {"state": "warn",
                "title": f"{totals['stale']} real SR products are stale",
                "detail": (f"{worst}: made by an older model than the production gate. "
                           "Rerun the production model on them from Sky › Targets."),
                "to": "/sky/targets", "facts": facts}
    if totals["current"]:
        return {"state": "ok", "title": f"All {totals['current']} real production SRs are current",
                "detail": None, "to": "/sky/targets", "facts": facts}
    return {"state": "ok", "title": "No real production SRs yet",
            "detail": "Run the production model on real tiles from Sky › Targets.",
            "to": "/sky/targets", "facts": facts}


def check_combiner() -> dict[str, Any]:
    """Is the production gate fitted for the current STARFULL members?"""
    production = next((spec for spec in model_catalog.list_specs()
                       if spec.spec == model_catalog.SPEC_PRODUCTION), None)
    if production is None:
        return {"state": "unknown", "title": "No production model in the catalogue",
                "detail": None, "to": "/models/combiner"}
    if production.available:
        n = len(production.member_labels)
        return {"state": "ok", "title": f"Production gate fitted for the current {n} members",
                "detail": None, "to": "/models/combiner", "facts": {"members": n}}
    return {"state": "warn", "title": "The production gate does not match the members",
            "detail": production.reason, "to": "/models/combiner"}


_EVALUATE_ACTION = {
    "label": "Evaluate", "method": "POST", "url": "/ensemble/evaluate",
    "params": {"mode": "starfull"},
    "confirm": ("Run the STARFULL evaluation on the local test records? It loads every "
                "active member (TensorFlow) and takes several minutes."),
}


def _records_match(records_fp: str, rdir: str, subset: str) -> bool | None:
    """Whether the recorded ``kind:size:mtime_ns|…`` still matches the files
    (``None`` when it cannot be compared)."""
    parts = [part.split(":") for part in str(records_fp).split("|") if part]
    if not parts or not rdir or not os.path.isdir(rdir):
        return None
    for fields in parts:
        if len(fields) != 3:
            return None
        kind, size, mtime = fields
        try:
            st = os.stat(tfrecord_path(rdir, f"{kind}_{subset}"))
        except OSError:
            return False
        if str(st.st_size) != size or str(st.st_mtime_ns) != mtime:
            return False
    return True


def check_evaluation() -> dict[str, Any]:
    """The STARFULL eval summary vs the active members and the test records."""
    path = model_catalog.regime_dir() / "eval_summary.json"
    try:
        summary = json.loads(path.read_text())
    except (OSError, ValueError):
        summary = None
    if not isinstance(summary, dict):
        return {"state": "warn", "title": "The STARFULL ensemble is not evaluated yet",
                "detail": "Evaluate it on the local test records to get the production numbers.",
                "to": "/models/leaderboard", "action": _EVALUATE_ACTION}
    recorded = [str(x) for x in (summary.get("member_labels")
                                 or summary.get("per_member_labels") or [])]
    active = model_catalog.active_member_labels()
    evaluated_at = datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(timespec="seconds")
    facts = {"evaluated_members": len(recorded), "active_members": len(active),
             "evaluated_at": evaluated_at, "n_scored": summary.get("n_scored")}
    if set(recorded) != set(active):   # the same members in another order are still current
        added = [label for label in active if label not in recorded]
        removed = [label for label in recorded if label not in active]
        change = ", ".join(filter(None, [
            f"+{len(added)} new" if added else "", f"−{len(removed)} retired" if removed else ""]))
        return {"state": "warn", "title": "The evaluation predates the current members",
                "detail": (f"Evaluated {len(recorded)} members; {len(active)} are active now"
                           f"{f' ({change})' if change else ''}."),
                "to": "/models/leaderboard", "action": _EVALUATE_ACTION, "facts": facts}
    identity = summary.get("eval_identity") or {}
    subset = str(identity.get("subset") or "test")
    match = (_records_match(identity["records_fp"], records_dir(), subset)
             if identity.get("records_fp") else None)
    if match is False:
        return {"state": "warn", "title": "The evaluation predates the current test records",
                "detail": (f"The {subset} records changed since the evaluation "
                           f"({evaluated_at[:10]}); its numbers describe the old fields."),
                "to": "/models/leaderboard", "action": _EVALUATE_ACTION, "facts": facts}
    return {"state": "ok",
            "title": f"Evaluation current ({len(active)} members, {subset} records)",
            "detail": None if match else "The test records could not be compared.",
            "to": "/models/leaderboard", "facts": facts}


def starless_member_labels() -> list[str]:
    """``NN·psnr`` labels of the registry-active starless (opt-in) members."""
    return list(ensemble_registry.regime_labels(model_catalog.ensemble_dir(), True))


def production_payload() -> dict[str, Any]:
    """``GET /api/system/production`` (see the module docstring)."""
    path = model_catalog.regime_dir() / "eval_summary.json"
    try:
        summary = json.loads(path.read_text())
    except (OSError, ValueError):
        summary = None
    scalars = None
    evaluated_at = None
    stale, reason = False, None
    if isinstance(summary, dict):
        scalars = {key: value for key, value in summary.items()
                   if value is None or isinstance(value, (bool, int, float, str))}
        with contextlib.suppress(OSError):
            evaluated_at = datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(timespec="seconds")
        check = check_evaluation()
        stale = check["state"] != "ok"
        reason = check["title"] if stale else None
    return {
        "eval_summary": scalars, "evaluated_at": evaluated_at, "stale": stale, "stale_reason": reason,
        "members": len(model_catalog.active_member_labels()),
        "starless_members": len(starless_member_labels()),
    }


def check_knee() -> dict[str, Any]:
    status = knee_psnr_status(False) or {}
    action = {"label": "Compute", "method": "POST", "url": "/ensemble/knee-psnr",
              "params": {"mode": "starfull"},
              "confirm": "Recompute the PSNR-vs-knee curves from the cached test cubes?"}
    if not status.get("available"):
        return {"state": "warn", "title": "PSNR-vs-knee curves not computed",
                "detail": status.get("reason"), "to": "/models/leaderboard", "action": action}
    if status.get("stale"):
        return {"state": "warn", "title": "PSNR-vs-knee curves are stale",
                "detail": "The cubes, members or combiners changed since they were computed.",
                "to": "/models/leaderboard", "action": action}
    return {"state": "ok", "title": "PSNR-vs-knee curves current", "detail": None,
            "to": "/models/leaderboard"}


def _find_noise_model(value: Any) -> str | None:
    """The ``noise_model`` identity anywhere in a config snapshot."""
    if isinstance(value, dict):
        direct = value.get("noise_model")
        if isinstance(direct, str) and direct:
            return direct
        values: Iterable[Any] = value.values()
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        return None
    for nested in values:
        found = _find_noise_model(nested)
        if found:
            return found
    return None


def record_noise_model(path: str) -> str | None:
    """The noise model that generated a TFRecord (its first record's
    provenance stamp → the generation run in the local provenance store), or
    ``None`` when that run is not recorded locally. Cached per file state."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (path, st.st_size, st.st_mtime_ns)
    if key in _NOISE_CACHE:
        return _NOISE_CACHE[key]
    model = None
    with contextlib.suppress(Exception):
        image = next(iter(ImageSet.read(path, num_images=1)))
        stamp = image.prov_stamp()
        if stamp is not None and stamp.produced_by is not None:
            process = provenance_store().get_or_none(stamp.produced_by)
            model = _find_noise_model(getattr(getattr(process, "config", None), "fields", None))
    _NOISE_CACHE[key] = model
    return model


def check_records_noise() -> dict[str, Any]:
    """Were the local dirty records generated with today's noise model?"""
    rdir = records_dir()
    files = sorted(Path(rdir).glob("dirty_*.tfrecord")) if rdir and os.path.isdir(rdir) else []
    if not files:
        return {"state": "unknown", "title": "No local training records",
                "detail": "Sync the records from FASRC in Synthetic › Records.", "to": "/synthetic/records"}
    seen = {path.stem: record_noise_model(str(path)) for path in files}
    wrong = {name: model for name, model in seen.items() if model and model != Config.NOISE_MODEL}
    facts = {"records": seen, "noise_model": Config.NOISE_MODEL}
    if wrong:
        listed = "; ".join(f"{name}: {model}" for name, model in wrong.items())
        return {"state": "bad", "title": "Training records use an older noise model",
                "detail": (f"{listed} — the code uses {Config.NOISE_MODEL}. Regenerate these "
                           "splits (Synthetic › Records › synthetic_generate)."),
                "to": "/synthetic/records", "facts": facts}
    if all(model is None for model in seen.values()):
        return {"state": "unknown", "title": "Noise model of the local records unverified",
                "detail": ("The generation run's provenance is not in the local store (the "
                           "records were generated on FASRC). Only the local splits are "
                           "examined — FASRC-only splits such as dirty_train/hr_train are "
                           "not checked here."),
                "to": "/synthetic/records", "facts": {**facts, "local_only": True}}
    return {"state": "ok", "title": f"Records match {Config.NOISE_MODEL}",
            "detail": None, "to": "/synthetic/records", "facts": facts}


_HEADING = re.compile(
    r"^##\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)\s*$",
    re.MULTILINE)


def last_log_entry(text: str) -> datetime | None:
    """The newest ``## <ISO time>`` heading of the tracking notebook."""
    newest = None
    for match in _HEADING.finditer(text or ""):
        raw = match.group(1).replace("Z", "+00:00")
        try:
            stamp = datetime.fromisoformat(raw)
        except ValueError:
            continue
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=UTC)
        if newest is None or stamp > newest:
            newest = stamp
    return newest


def _tracked_results() -> list[tuple[str, float]]:
    """(label, mtime) of the results the notebook should mention."""
    regime = model_catalog.regime_dir()
    out = []
    for label, path in (
        ("evaluation", regime / "eval_summary.json"),
        ("production gate fit", regime / model_catalog.PRODUCTION_ARTIFACT_DIR / "combiner.npz"),
        ("PSNR vs knee", regime / "ensemble_knee_psnr.json"),
    ):
        with contextlib.suppress(OSError):
            out.append((label, path.stat().st_mtime))
    root = experiment_records_root()
    with contextlib.suppress(OSError):
        newest = max((p.stat().st_mtime for p in Path(root).glob("*.json")), default=None)
        if newest is not None:
            out.append(("experiment", newest))
    return out


def check_tracking() -> dict[str, Any]:
    text = TrackingStore(Config.TRACKING_DIR).read_log()
    last = last_log_entry(text)
    if last is None:
        return {"state": "unknown", "title": "No tracking log entries",
                "detail": "Start a campaign and log results in Notebook › Log.", "to": "/notebook/log"}
    cutoff = last.timestamp() + TRACKING_LAG_S
    newer = sorted(((label, mtime) for label, mtime in _tracked_results() if mtime > cutoff),
                   key=lambda item: item[1], reverse=True)
    facts = {"last_entry": last.isoformat(timespec="seconds"),
             "unlogged": [{"label": label, "at": datetime.fromtimestamp(mtime, UTC).isoformat(
                 timespec="seconds")} for label, mtime in newer]}
    day = last.date().isoformat()
    if newer:
        listed = ", ".join(f"{label} {datetime.fromtimestamp(mtime, UTC).date().isoformat()}"
                           for label, mtime in newer)
        return {"state": "warn", "title": f"Results since the last tracking entry ({day})",
                "detail": f"{listed} — log them in Notebook › Log.", "to": "/notebook/log",
                "facts": facts}
    return {"state": "ok", "title": f"Tracking log up to date (last entry {day})", "detail": None,
            "to": "/notebook/log", "facts": facts}


CHECK_LABELS = {
    "disk": "Disk space", "real-sr": "Real SR products", "combiner": "Production gate",
    "evaluation": "Evaluation", "knee": "Knee PSNR", "records-noise": "Records noise model",
    "tracking": "Tracking log",
}

CHECKS: tuple[tuple[str, Callable[[], dict[str, Any]]], ...] = (
    ("disk", lambda: check_disk()),
    ("real-sr", lambda: check_real_sr()),
    ("combiner", lambda: check_combiner()),
    ("evaluation", lambda: check_evaluation()),
    ("knee", lambda: check_knee()),
    ("records-noise", lambda: check_records_noise()),
    ("tracking", lambda: check_tracking()),
)

_ORDER = {"bad": 0, "warn": 1}


def run_checks() -> list[dict[str, Any]]:
    out = []
    for check_id, fn in CHECKS:
        label = CHECK_LABELS.get(check_id, check_id)
        try:
            result = dict(fn() or {})
        except Exception as exc:  # noqa: BLE001 - one broken check must not hide the rest
            result = {"state": "unknown", "title": f"{label}: check failed",
                      "detail": f"{type(exc).__name__}: {exc}"}
        state = result.get("state")
        out.append({
            "id": check_id, "label": label,
            "state": state if state in {"ok", "warn", "bad", "unknown"} else "unknown",
            "title": result.get("title") or label,
            "detail": result.get("detail"),
            "to": result.get("to"),
            **({"action": result["action"]} if result.get("action") else {}),
            **({"facts": result["facts"]} if result.get("facts") is not None else {}),
        })
    return out


def alerts_payload(*, fresh: bool = False) -> dict[str, Any]:
    global _ALERTS
    with _LOCK:
        cached = _ALERTS
    if not fresh and cached is not None and time.monotonic() - cached[0] < ALERTS_TTL_S:
        return cached[1]
    checks = run_checks()
    counts = {"bad": 0, "warn": 0, "ok": 0, "unknown": 0}
    for check in checks:
        counts[check["state"]] += 1
    alerts = sorted((c for c in checks if c["state"] in _ORDER), key=lambda c: _ORDER[c["state"]])
    payload = {"computed_at": _now_iso(), "ttl_s": ALERTS_TTL_S, "checks": checks,
               "alerts": alerts, "counts": counts}
    with _LOCK:
        _ALERTS = (time.monotonic(), payload)
    return payload


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def register(app):
    errors.json_errors_for(app, "/api/system")

    @app.get("/api/system")
    def api_system():
        """Runtime, free space, last disk usage per root (never measures)."""
        return jsonify(system_payload())

    @app.post("/api/system/disk-usage/refresh")
    def api_system_disk_usage_refresh():
        """Measure every data root in a local job (one at a time)."""
        return jsonify({"ok": True, "job_id": spawn_disk_usage_refresh()})

    @app.get("/api/system/production")
    def api_system_production():
        """Home's production numbers (the STARFULL eval summary headline)."""
        return jsonify(production_payload())

    @app.get("/api/system/alerts")
    def api_system_alerts():
        """The Home health checks (memoised; ``?fresh=1`` recomputes)."""
        return jsonify(alerts_payload(fresh=_truthy(request.args.get("fresh"))))

    @app.get("/api/system/loop")
    def api_system_loop():
        """The staleness service: one verdict per Loop stage (Home's Loop
        strip and System › Lineage). Memoised; ``?fresh=1`` recomputes it
        and the alerts it reads. Read-only: never starts a job."""
        fresh = _truthy(request.args.get("fresh"))
        return jsonify(system_alerts.loop_payload(
            alerts=lambda: alerts_payload(fresh=fresh), check_records_noise=check_records_noise, fresh=fresh))
