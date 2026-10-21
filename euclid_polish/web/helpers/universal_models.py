"""Universal models (Models › Universal): single-image multi-knee members
compared as models in their own right — their training time and hardware,
their score on the Leaderboard's test fields, and the combiners' training time
weighted by the gate's shares.

A universal model (:func:`~euclid_polish.ensemble_registry.is_universal_origin`)
is either an active ensemble member whose recipe is universal (class
``member``, e.g. 195/196 — it stays in the ensemble) or a STANDALONE
registered universal model (class ``universal``: pulled by
:func:`job_universal_pull`, never in the mean, the gates, Evaluate or any
cache keyed on the active members; it loads as an ensemble of one through
``EnsembleModel(labels=[…], include_universal=True)``).

* **Training time** of a model = Σ ``duration_s`` over every non-baseline row
  of its ``training_log.csv`` — rolled-back windows included (real compute),
  never deduplicated — in hours (:func:`train_time`); the hardware comes from
  the FASRC job ledger (:func:`hardware`).
* **Combiner training time** = Σ over the members a gate READS of
  ``hours × weight``: the member's PEAK gate share
  (:func:`~euclid_polish.eval.gate_members.member_peak_weights` — the max over
  bands of its all-pixel, source-pixel and brightness-bin mean weights, the
  "used by the gate" rule; the default), its mean share (mean over bands of
  the all-pixel share) or 1 (every member). The production gate's shares come
  from its payload (``spatial_gate_combiner_evals.json``); a variant's from
  ``<regime>/<variant dir>_evals.json``, which the score job computes from the
  validate cubes.
* **Scores**: the Leaderboard's exact metric — knee-integrated PSNR against
  ``hr_test`` blurred to the target PSF, on the Leaderboard's test fields —
  stored per model in ``<vis>/ensemble/starfull/universal/`` (``<name>.npz``
  per-field curves + ``index.json``; the read side lives in
  :mod:`~euclid_polish.web.helpers.ensemble_viz`). Never in a cube bucket.
* **Spectra** (Models › Spectra): made by the same pass over the same test
  fields — the output's power spectrum against the truth's, their coherence
  and transfer, and the direction structure of the error
  (:mod:`~euclid_polish.eval.model_spectra`) — stored as
  ``<name>.spectra.json`` next to the curves; served by
  :mod:`~euclid_polish.web.helpers.universal_spectra`.

:func:`universal_payload` (``GET /ensemble/universal.json``) only reads: no
SSH, no TensorFlow, no writes. :func:`job_universal_score` is a local job;
:func:`job_universal_pull` needs FASRC.
"""

from __future__ import annotations

import contextlib
import csv
import glob
import io
import json
import os
import shlex
import shutil
import threading
import time
import zipfile
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime

import numpy as np

from euclid_polish import ensemble_registry
from euclid_polish.config import Config
from euclid_polish.ensemble import EnsembleModel, member_fingerprint
from euclid_polish.eval.combiner import BAND_NAMES, COMBINER_MODELS, combiner_artifact_fingerprint
from euclid_polish.eval.gate_members import member_peak_weights
from euclid_polish.eval.knee_psnr import KNEE_GRID_E, integrated_psnr, knee_psnr, stretched_truth
from euclid_polish.eval.model_spectra import SpectraAccumulator, field_sums
from euclid_polish.eval.spatial_gate import SPATIAL_GATE_KIND, load_spatial_gate
from euclid_polish.image.collection import ImageSet
from euclid_polish.image.tfio import tfrecord_path
from euclid_polish.model import _checkpoint_exists
from euclid_polish.training.inference import checkpoint_step, infer_checkpoint_learned_output_knee
from euclid_polish.training.target_blur import blur_target_array
from euclid_polish.training.trainer import TRAINING_LOG_FILENAME, prune_orphaned_checkpoints
from euclid_polish.web.helpers import ensemble_viz as ev
from euclid_polish.web.helpers.purge_requests import request_stale_purge
from euclid_polish.web.remote import STATE

#: Model spec prefixes (contract C9) of a standalone universal model and of
#: an ensemble member.
UNIVERSAL_SPEC_PREFIX = "universal:"
MEMBER_SPEC_PREFIX = "member:"

#: Hardware label per SLURM partition. ``gpu`` = one full A100-SXM4-80GB;
#: ``gpu_test`` books 4 MIG 3g.20gb slices (for the CPUs) but trains on one.
_HARDWARE_LABELS = {"gpu": "A100", "gpu_test": "A100 MIG slice"}

_TRAIN_TIME_LOCK = threading.Lock()
_TRAIN_TIME_CACHE: dict[tuple[str, int, int], dict] = {}
_TRAIN_TIME_CACHE_MAX = 512


# --------------------------------------------------------------------------- #
# Training time and hardware
# --------------------------------------------------------------------------- #

def _wall_iso(value: float | None) -> str | None:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(float(value), UTC).isoformat(timespec="seconds")
    except (OverflowError, OSError, ValueError):
        return None


def _train_time_from_rows(rows: Iterable[dict]) -> dict:
    """:func:`train_time` of parsed ``training_log.csv`` rows."""
    seconds = 0.0
    n_rows = rollback = 0
    top: int | None = None
    walls: list[float] = []
    for row in rows:
        if ev._truthy(row.get("is_baseline")):
            continue                     # a resume marker: no training in it
        step = ev._as_int(row.get("step"))
        if step is None:
            continue
        n_rows += 1
        if top is not None and step <= top:
            rollback += 1                # re-trains a window after a rollback
        top = step if top is None else max(top, step)
        duration = ev._finite(row.get("duration_s"))
        if duration is not None and duration >= 0:
            seconds += duration
        wall = ev._finite(row.get("wall_time"))
        if wall is not None:
            walls.append(wall)
    return {"hours": round(seconds / 3600.0, 4) if n_rows else None,
            "steps": top, "rows": n_rows, "rollback_rows": rollback,
            "first_wall": _wall_iso(min(walls)) if walls else None,
            "last_wall": _wall_iso(max(walls)) if walls else None}


def _empty_train_time() -> dict:
    return {"hours": None, "steps": None, "rows": 0, "rollback_rows": 0,
            "first_wall": None, "last_wall": None}


def _read_log_text(path: str) -> str | None:
    """The ``training_log.csv`` text of a member dir or of an archived
    member's tracking zip (stored at the zip's root); ``None`` without one."""
    if path.endswith(".zip") and os.path.isfile(path):
        try:
            with zipfile.ZipFile(path) as zf:
                return zf.read(TRAINING_LOG_FILENAME).decode("utf-8", errors="replace")
        except (OSError, KeyError, zipfile.BadZipFile):
            return None
    try:
        with open(os.path.join(path, TRAINING_LOG_FILENAME), encoding="utf-8",
                  errors="replace") as handle:
            return handle.read()
    except OSError:
        return None


def train_time(path: str) -> dict:
    """A model's training time from its ``training_log.csv`` — in the member
    dir ``path``, or inside an archived member's tracking zip when ``path``
    is that ``.zip``: ``{hours, steps, rows, rollback_rows, first_wall,
    last_wall}``.

    ``hours`` = Σ ``duration_s`` (the trainer's wall time per logged window:
    training + validation, not checkpoint saves or job start-up) over every
    non-baseline row, rolled-back windows included — never deduplicated —
    ``None`` without a log or rows. ``steps`` = the furthest logged step,
    ``rows`` = non-baseline rows, ``rollback_rows`` = rows re-training a step
    already logged, ``first_wall``/``last_wall`` = ISO UTC of the
    ``wall_time`` span. Cached per file state (size + mtime)."""
    log = path if path.endswith(".zip") else os.path.join(path, TRAINING_LOG_FILENAME)
    try:
        st = os.stat(log)
    except OSError:
        return _empty_train_time()
    key = (os.path.abspath(log), st.st_size, st.st_mtime_ns)
    with _TRAIN_TIME_LOCK:
        hit = _TRAIN_TIME_CACHE.get(key)
    if hit is not None:
        return dict(hit)
    text = _read_log_text(path)
    if text is None:
        return _empty_train_time()
    try:
        result = _train_time_from_rows(csv.DictReader(io.StringIO(text)))
    except csv.Error:
        return _empty_train_time()
    with _TRAIN_TIME_LOCK:
        if len(_TRAIN_TIME_CACHE) >= _TRAIN_TIME_CACHE_MAX:
            _TRAIN_TIME_CACHE.clear()
        _TRAIN_TIME_CACHE[key] = result
    return dict(result)


def _member_names_of(job: dict) -> list[str]:
    out = []
    for raw in job.get("member_names") or []:
        try:
            out.append(ensemble_registry.member_name(raw))
        except ValueError:
            continue
    return out


def ledger_jobs(name: str, jobs: list[dict] | None = None) -> list[dict]:
    """Every ``ensemble_train`` ledger row naming the member — the add/fork
    job that created it (``member_names``) and every continue job that
    resumed it (``members``) — newest first, as
    :func:`~euclid_polish.web.helpers.ensemble_viz.training_jobs` rows plus
    ``task_index`` (the member's array task in that job)."""
    name = ensemble_registry.member_name(name)
    out = []
    for job in ev.training_jobs() if jobs is None else jobs:
        names = _member_names_of(job)
        if name in names:
            out.append({**job, "task_index": names.index(name)})
    return out


def hardware(name: str, jobs: list[dict] | None = None) -> dict:
    """``{partition, label, jobs}`` a member trained on, from the ledger:
    ``partition`` = the one SLURM partition of its jobs (``"mixed"`` when
    they differ, ``None`` without a ledger row); ``label`` = ``"A100"``
    (``gpu``), ``"A100 MIG slice"`` (``gpu_test``), ``"mixed"`` or ``None``;
    ``jobs`` = the job ids, newest first."""
    found = ledger_jobs(name, jobs)
    partitions = list(dict.fromkeys(str(j["partition"]) for j in found if j.get("partition")))
    labels = {_HARDWARE_LABELS.get(p) for p in partitions}
    if not partitions:
        partition = label = None
    else:
        partition = partitions[0] if len(partitions) == 1 else "mixed"
        label = next(iter(labels)) if len(labels) == 1 else "mixed"
    return {"partition": partition, "label": label,
            "jobs": [str(j.get("jobid") or "") for j in found]}


#: SLURM states after which a member's dir holds a checkpoint to pull (the
#: Members tab's ``TERMINAL_WITH_CHECKPOINT``).
_PULLABLE_STATES = {"COMPLETED", "TIMEOUT"}


def _state_head(state: str | None) -> str:
    return str(state or "").split(" ", 1)[0].strip().upper()


def _finished(state: str | None) -> bool:
    """A SLURM state that will not change any more (``CANCELLED by …`` too)."""
    head = _state_head(state)
    return bool(head) and head not in ev._LIVE_SLURM_STATES


def _pullable(state: str | None) -> bool:
    """The newest job ended in a state that leaves a checkpoint behind
    (COMPLETED, or TIMEOUT: stopped by the time limit after saving)."""
    return _state_head(state) in _PULLABLE_STATES


def fasrc_universal_rows(jobs: list[dict] | None = None,
                         registry: dict | None = None) -> list[dict]:
    """Universal models the ledger's add/fork jobs created (a fork of a
    universal model is one) that are not here: neither active, tombstoned,
    read by a local gate or held by a cube manifest
    (:func:`~euclid_polish.web.helpers.ensemble_viz.known_member_names`),
    nor a registered universal model WITH a local checkpoint — a registered
    one without (a pull that failed, or a dir the mirror removed) is listed
    with ``registered: true`` so it can be pulled again. ``{name, jobids,
    state, num_res_blocks, steps, knee, submitted_at, ended_at, partition,
    finished, pullable, registered}`` sorted by name. ``state``,
    ``ended_at`` and ``partition`` are the newest job's; ``finished`` = that
    job reached a final state; ``pullable`` = it ended COMPLETED or TIMEOUT
    (a checkpoint to pull); ``steps`` = the add job's budget, raised to a
    continue job's absolute target."""
    jobs = ev.training_jobs() if jobs is None else jobs
    base = ev.ensemble_dir()
    reg = registry if registry is not None else ensemble_registry.load_registry(base)
    on_disk = {os.path.basename(d) for d in ensemble_registry.universal_member_dirs(base)}
    registered = set(reg["universal"])
    known = ev.known_member_names({**reg, "universal": sorted(registered & on_disk)})
    rows = []
    for name, recipe in sorted(ev.ledger_member_recipes(jobs, base=base).items()):
        if name in known or not ensemble_registry.is_universal_origin(recipe):
            continue
        found = ledger_jobs(name, jobs)
        newest = found[0]
        steps = recipe.get("steps")
        for job in found:
            if (job.get("mode") == "continue" and job.get("continue_basis") == "target"
                    and job.get("target_steps")):
                steps = max(int(steps or 0), int(job["target_steps"]))
        rows.append({
            "name": name, "jobids": [str(j.get("jobid") or "") for j in found],
            "state": newest.get("state"),
            "num_res_blocks": recipe.get("num_res_blocks"), "steps": steps,
            "knee": "learned" if recipe.get("learn_output_knee") else "fixed",
            "submitted_at": found[-1].get("submitted_at"),
            "ended_at": newest.get("ended_at"), "partition": newest.get("partition"),
            "finished": _finished(newest.get("state")),
            "pullable": _pullable(newest.get("state")),
            "registered": name in registered,
        })
    return rows


# --------------------------------------------------------------------------- #
# Which models are universal
# --------------------------------------------------------------------------- #

def universal_targets(base: str | None = None) -> dict[str, tuple[str, str]]:
    """Every local universal model: ``{name: (class, dir)}`` in name order —
    class ``member`` for an active STARFULL member whose recipe is universal,
    ``universal`` for a registered universal model with a checkpoint."""
    base = base or ev.ensemble_dir()
    reg = ensemble_registry.load_registry(base)
    out: dict[str, tuple[str, str]] = {}
    for name in reg["active"]:
        d = os.path.join(base, name)
        if (_checkpoint_exists(d) and not ensemble_registry.member_is_starless(d)
                and ensemble_registry.is_universal_origin(ev._member_origin(d))):
            out[name] = (ensemble_registry.MEMBER_CLASS, d)
    for d in ensemble_registry.universal_member_dirs(base):
        if not ensemble_registry.member_is_starless(d):
            out[os.path.basename(d)] = (ensemble_registry.UNIVERSAL_CLASS, d)
    return dict(sorted(out.items(), key=lambda kv: _number(kv[0])))


def _name_of(label: str) -> str | None:
    """``"196·psnr"`` → ``"member_196"``; ``None`` for a label of no member
    dir (e.g. a ``·loss`` track)."""
    try:
        return ensemble_registry.member_name(label)
    except ValueError:
        return None


def _number(name: str) -> int:
    tail = str(name).removeprefix("member_")
    return int(tail) if tail.isdigit() else -1


# --------------------------------------------------------------------------- #
# Gate weights per gate
# --------------------------------------------------------------------------- #

def _variant_evals_path(regime_dir: str, name: str) -> str:
    """``<regime>/<variant dir>_evals.json`` — the production gate's is its
    combiner payload; a variant's holds ``{artifact_fp, gate_diagnostics}``."""
    return os.path.join(regime_dir, f"{name}_evals.json")


def _validate_cube_labels(regime_dir: str) -> set[str] | None:
    manifest = ev._read_json_file(os.path.join(regime_dir, "cubes_validate", "viz_index.json"))
    if manifest is None:
        return None
    return {str(v) for v in manifest.get("member_labels") or []}


def _weights_from_diagnostic(diag: dict, labels: list[str]
                             ) -> tuple[list[float] | None, list[float] | None, str | None]:
    """``(peak shares, mean shares, reason)`` per fitted member of a gate."""
    try:
        peaks = member_peak_weights(diag, len(labels))
    except ValueError as exc:
        return None, None, str(exc)
    usage = diag.get("usage") or {}
    means = []
    for i in range(len(labels)):
        vals = [ev._finite(v[i]) for v in usage.values() if isinstance(v, list) and i < len(v)]
        vals = [v for v in vals if v is not None]
        means.append(float(np.mean(vals)) if vals else 0.0)
    return peaks, means, None


def gate_weights(variant: dict, regime_dir: str, *,
                 validate_labels: set[str] | None = None) -> dict:
    """One gate's per-member shares, from cached diagnostics only:
    ``{source: "production"|"variant"|None, reason, measurable, max: {label:
    peak share}, mean: {label: mean share}}``. ``variant`` is a
    :func:`~euclid_polish.web.helpers.ensemble_viz.combiner_variants` row; the
    diagnostic must belong to its artifact on disk now (``artifact_fp``).
    ``measurable``: a variant without current shares whose read members all
    have validate cubes — the score job can measure it now."""
    labels = [str(v) for v in variant.get("member_labels") or []]
    name = str(variant.get("name"))
    none = {"source": None, "reason": None, "measurable": False, "max": {}, "mean": {}}
    if variant.get("production"):
        payload = ev._read_json_file(os.path.join(
            regime_dir, COMBINER_MODELS[SPATIAL_GATE_KIND].payload_name))
        diag = (payload or {}).get("gate_diagnostics") or {}
        if not payload or not diag:
            return {**none, "reason": "no production gate payload yet — open Models › "
                                      "Combiner once to compute it"}
        if [str(v) for v in payload.get("member_labels") or []] != labels:
            return {**none, "reason": "the production gate payload is for another fit — "
                                      "open Models › Combiner once to refresh it"}
        if diag.get("artifact_fp") not in (None, variant.get("fingerprint")):
            return {**none, "reason": "the production gate's weight diagnostic predates the "
                                      "gate on disk — open Models › Combiner once to refresh it"}
        source = "production"
    else:
        reads = [str(v) for v in variant.get("reads") or labels]
        missing = ([lb for lb in reads if lb not in validate_labels]
                   if validate_labels is not None else None)
        measurable = missing == []
        cached = ev._read_json_file(_variant_evals_path(regime_dir, name))
        diag = (cached or {}).get("gate_diagnostics") or {}
        if cached and cached.get("artifact_fp") != variant.get("fingerprint"):
            return {**none, "measurable": measurable,
                    "reason": "computed for an earlier fit of this gate — “Measure gate "
                              "weights” (the score job) refreshes it"}
        if not cached or not diag.get("available"):
            if missing is None:
                reason = "no validate cube cache to measure its weights on"
            elif missing:
                reason = (f"the validate cubes lack {len(missing)} member(s) it reads ("
                          + ", ".join(missing[:4]) + ("…" if len(missing) > 4 else "") + ")")
            else:
                reason = ("not measured yet — “Measure gate weights” (the score job) "
                          "measures it from the validate cubes")
            return {**none, "reason": reason, "measurable": measurable}
        source = "variant"
    peaks, means, reason = _weights_from_diagnostic(diag, labels)
    if peaks is None or means is None:
        return {**none, "reason": reason}
    return {"source": source, "reason": None, "measurable": False,
            "max": dict(zip(labels, peaks, strict=True)),
            "mean": dict(zip(labels, means, strict=True))}


def _gate_weight_diagnostics(cap) -> dict:
    """Measure, from the validate cubes, the weight diagnostic of every gate
    variant that lacks a current one (backups and the production gate
    excluded — production keeps its own payload), into
    ``<regime>/<variant dir>_evals.json``. A variant whose read members have
    no validate cubes is reported unavailable and nothing is written."""
    regime = ev._ensemble_regime_dir(False)
    production = COMBINER_MODELS[SPATIAL_GATE_KIND].artifact_dir
    names = [os.path.basename(d) for d in sorted(glob.glob(os.path.join(regime, "spatial_gate_*")))
             if os.path.isdir(d)]
    names = [n for n in names if n != production and not n.startswith(ev.GATE_BACKUP_PREFIX)
             and ev._VARIANT_NAME.fullmatch(n)]
    computed: list[str] = []
    reused: list[str] = []
    unavailable: dict[str, str] = {}
    for i, name in enumerate(names):
        cap.tick(i, len(names), f"gate weights: {name}")
        fp = combiner_artifact_fingerprint(regime, name)
        if fp is None:
            continue
        path = _variant_evals_path(regime, name)
        cached = ev._read_json_file(path) or {}
        diag = cached.get("gate_diagnostics") or {}
        if (cached.get("artifact_fp") == fp and diag.get("available")
                and diag.get("schema") == ev._GATE_DIAGNOSTIC_SCHEMA):
            reused.append(name)
            continue
        try:
            comb = load_spatial_gate(os.path.join(regime, name))
            if comb is None:
                unavailable[name] = "the gate artifact does not load (older schema)"
                continue
            diag = ev._spatial_gate_weight_diagnostic(comb, starless=False)
        except Exception as exc:  # noqa: BLE001 — one gate never stops the others
            unavailable[name] = f"{type(exc).__name__}: {exc}"
            continue
        if not diag.get("available"):
            unavailable[name] = str(diag.get("reason") or "unavailable")
            continue
        diag["artifact_fp"] = fp
        ev._atomic_json(path, {"artifact_fp": fp, "member_labels": list(comb.member_labels),
                               "computed_at": datetime.now(UTC).isoformat(timespec="seconds"),
                               "gate_diagnostics": diag})
        computed.append(name)
    if names:
        cap.tick(len(names), len(names), "gate weights measured")
    return {"computed": computed, "reused": reused, "unavailable": unavailable}


# --------------------------------------------------------------------------- #
# The payload
# --------------------------------------------------------------------------- #

def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


def _mean(values: Iterable) -> float | None:
    vals = [v for v in (ev._finite(x) for x in values) if v is not None]
    return float(np.mean(vals)) if vals else None


def _knee_model(knee: dict, model_id: str) -> dict | None:
    """``{integrated[4], mean, stale}`` of one row of the knee payload."""
    if not knee.get("available"):
        return None
    row = next((m for m in knee.get("models", []) or [] if m.get("id") == model_id), None)
    if row is None:
        return None
    integrated = [ev._finite(v) for v in row.get("integrated") or []]
    return {"integrated": integrated, "mean": _round(_mean(integrated)),
            "stale": bool(knee.get("stale"))}


def _universal_reference(name: str, scores: dict, identity_ok: bool) -> dict | None:
    """``{integrated[4], mean, stale, model}`` of the universal model that is
    production, from the universal scores (``None`` until it is scored)."""
    entry = scores["models"].get(name)
    if not entry:
        return None
    integrated = [ev._finite(v) for v in entry.get("integrated") or []]
    fresh = entry.get("fingerprint") == member_fingerprint(os.path.join(ev.ensemble_dir(), name))
    return {"integrated": integrated, "mean": _round(ev._finite(entry.get("mean"))),
            "stale": not (fresh and identity_ok), "model": name}


def _comparable(knee: dict, identity: dict | None) -> bool:
    """Whether the knee payload's production/mean numbers were scored on the
    same fields, records and target as the universal scores."""
    return ev.universal_knee_comparable(knee, identity)


def _parse_time(value) -> datetime | None:
    try:
        stamp = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=UTC)


def _compare_knee_rows(regime_dir: str) -> dict[str, dict]:
    """``{method: {report, created, integrated, identity, member_fps}}`` —
    each gate method's knee row from the NEWEST saved compare report that
    scored it (the latest report first, then every saved one, newest first);
    ``identity``/``member_fps`` are ``None`` for a report written before
    compare reports recorded them."""
    paths = [os.path.join(regime_dir, ev._COMPARE_LATEST)]
    folder = os.path.join(regime_dir, ev._COMPARE_DIR)
    ids = sorted((os.path.basename(p)[:-5] for p in glob.glob(os.path.join(folder, "*.json"))),
                 reverse=True)
    paths += [os.path.join(folder, f"{rid}.json") for rid in ids if ev._COMPARE_ID.fullmatch(rid)]
    out: dict[str, dict] = {}
    for path in paths:
        report = ev._read_json_file(path)
        if not report:
            continue
        identity = report.get("identity") if isinstance(report.get("identity"), dict) else None
        fps = report.get("member_fps") if isinstance(report.get("member_fps"), dict) else None
        for method, row in ((report.get("knee") or {}).get("methods") or {}).items():
            if method not in out and isinstance(row, dict) and row.get("integrated"):
                out[method] = {"report": report.get("id"), "created": report.get("created"),
                               "integrated": [ev._finite(v) for v in row["integrated"]],
                               "identity": identity, "member_fps": fps}
    return out


def _fingerprint_resolver(base: str) -> Callable[[str], str | None]:
    """label → the member's current checkpoint fingerprint (memoised)."""
    memo: dict[str, str | None] = {}

    def fingerprint_of(label: str) -> str | None:
        if label not in memo:
            name = _name_of(label)
            memo[label] = member_fingerprint(os.path.join(base, name)) if name else None
        return memo[label]

    return fingerprint_of


def _fingerprint_mtime(fingerprint: str | None) -> datetime | None:
    """The checkpoint index mtime a :func:`member_fingerprint` ends with."""
    try:
        return datetime.fromtimestamp(int(str(fingerprint).rsplit(":", 1)[1]), UTC)
    except (IndexError, ValueError, OverflowError, OSError):
        return None


def _compare_stale(hit: dict, variant: dict, fingerprint_of: Callable[[str], str | None]) -> bool:
    """A compare knee row is stale when the report predates the gate's fit
    (or its promotion), a member it reads is no longer active, or a member
    it reads has another checkpoint now than the one the report scored —
    the report's recorded ``member_fps``; a legacy report without them:
    a read member's checkpoint index is newer than the report."""
    created = _parse_time(hit.get("created"))
    made = [t for t in (_parse_time(variant.get("fitted_at")),
                        _parse_time((variant.get("fit") or {}).get("promoted_at"))) if t]
    if created is None or (made and created < max(made)):
        return True
    if not bool((variant.get("membership") or {}).get("current")):
        return True
    reads = [str(v) for v in variant.get("reads") or []]
    recorded = hit.get("member_fps")
    if isinstance(recorded, dict):
        return any(recorded.get(label) != fingerprint_of(label) for label in reads)
    for label in reads:
        saved = _fingerprint_mtime(fingerprint_of(label))
        if saved is not None and saved > created:
            return True
    return False


def _hours_resolver(base: str, registry: dict) -> Callable[[str], float | None]:
    """label → the member's training hours: its dir in the ensemble dir, else
    its archived tracking zip (tombstone), else ``None``."""
    tombstones = {str(t.get("name")): t for t in registry["archived"]}
    memo: dict[str, float | None] = {}

    def hours_of(label: str) -> float | None:
        if label in memo:
            return memo[label]
        name = _name_of(label)
        if name is None:
            memo[label] = None
            return None
        hours = train_time(os.path.join(base, name))["hours"]
        if hours is None and name in tombstones:
            found = ensemble_registry.find_archived_zip(tombstones[name], Config.TRACKING_DIR)
            if found is not None:
                hours = train_time(found[1])["hours"]
        memo[label] = hours
        return hours

    return hours_of


def _combiner_row(variant: dict, *, hours_of: Callable[[str], float | None],
                  weights: dict, compare: dict[str, dict],
                  fingerprint_of: Callable[[str], str | None],
                  identity: dict | None, knee_comparable: bool) -> dict:
    production = bool(variant.get("production"))
    name = str(variant.get("name"))
    short = name.removeprefix("spatial_gate_")
    reads = [str(v) for v in variant.get("reads") or []]
    knee_row = variant.get("knee") or {}
    integrated = None
    source = None
    stale = False
    comparable = False
    if production and knee_row.get("source") == "knee":
        integrated = [ev._finite(v) for v in knee_row.get("integrated") or []]
        source, stale = "knee-psnr", bool(knee_row.get("stale"))
        comparable = knee_comparable
    elif (hit := compare.get(f"gate:{name}")) is not None:
        integrated = hit["integrated"]
        source, stale = "compare", _compare_stale(hit, variant, fingerprint_of)
        comparable = ev.report_identity_matches(hit.get("identity"), identity)
    members = []
    for label in reads:
        members.append({"label": label, "name": _name_of(label),
                        "hours": hours_of(label),
                        "max_weight": _round(weights["max"].get(label), 6),
                        "mean_weight": _round(weights["mean"].get(label), 6)})
    known = [m for m in members if m["hours"] is not None]
    has_weights = weights["source"] is not None
    hours: dict[str, float | None] = {"max": None, "mean": None, "sum": None}
    if known:
        hours["sum"] = _round(sum(m["hours"] for m in known))
        if has_weights:
            hours["max"] = _round(sum(m["hours"] * (m["max_weight"] or 0.0) for m in known))
            hours["mean"] = _round(sum(m["hours"] * (m["mean_weight"] or 0.0) for m in known))
    fit_seconds = ev._finite((variant.get("fit") or {}).get("fit_seconds"))
    return {
        "id": "production" if production else f"gate:{short}",
        "name": name, "label": "production gate" if production else f"gate {short}",
        "spec": variant.get("spec") or ("production" if production else f"gate:{short}"),
        "production": production, "reads": reads, "n_reads": len(reads),
        "integrated": integrated, "mean": _round(_mean(integrated or [])) if integrated else None,
        "score_source": source, "stale": bool(stale), "comparable": bool(comparable),
        "weights_source": weights["source"], "weights_reason": weights["reason"],
        "members": members, "hours": hours,
        "missing_hours": [m["label"] for m in members if m["hours"] is None],
        "fit_hours": _round(fit_seconds / 3600.0) if fit_seconds is not None else None,
    }


def _combiner_table(*, base: str, registry: dict, identity: dict | None,
                    knee_comparable: bool) -> tuple[list[dict], list[str]]:
    """:func:`combiner_rows` plus the ids of the gates whose shares are
    missing but measurable now (``weights_pending``)."""
    regime = ev._regime_dir_ro(False)
    variants = [v for v in ev.combiner_variants(False).get("variants") or []
                if v.get("kind") == "gate" and not v.get("backup")]
    variants.sort(key=lambda v: (not v.get("production"), str(v.get("name"))))
    validate_labels = _validate_cube_labels(regime)
    compare = _compare_knee_rows(regime)
    hours_of = _hours_resolver(base, registry)
    fingerprint_of = _fingerprint_resolver(base)
    rows: list[dict] = []
    pending: list[str] = []
    for v in variants:
        weights = gate_weights(v, regime, validate_labels=validate_labels)
        row = _combiner_row(v, hours_of=hours_of, compare=compare, weights=weights,
                            fingerprint_of=fingerprint_of, identity=identity,
                            knee_comparable=knee_comparable)
        rows.append(row)
        if weights["measurable"]:
            pending.append(row["id"])
    return rows, pending


def combiner_rows(*, base: str, registry: dict, identity: dict | None = None,
                  knee_comparable: bool = False) -> list[dict]:
    """One row per gate (the production gate first, then every variant by
    name; promotion backups left out): its knee-integrated PSNR (production:
    the knee payload; a variant: the newest compare report that scored it),
    whether that score is ``comparable`` with the universal scores under
    ``identity`` (production: ``knee_comparable``; a compare report: its
    recorded identity), the members it reads with their training hours and
    shares, and its training time weighted by the peak share (``hours.max``),
    the mean share (``hours.mean``) and unweighted (``hours.sum``)."""
    return _combiner_table(base=base, registry=registry, identity=identity,
                           knee_comparable=knee_comparable)[0]


def _model_row(name: str, cls: str, d: str, *, jobs: list[dict], by_member: dict,
               scores: dict, identity_ok: bool, production: dict | None,
               comparable: bool) -> dict:
    origin = ev._member_origin(d) or {}
    tt = train_time(d)
    step = tt["steps"] if tt["steps"] is not None else ev._member_last_step(d)
    fingerprint = member_fingerprint(d)
    entry = scores["models"].get(name)
    stored = entry or {}
    fresh = bool(entry) and stored.get("fingerprint") == fingerprint
    score = None
    if entry:
        integrated = [ev._finite(v) for v in entry.get("integrated") or []]
        score = {"integrated": integrated, "mean": ev._finite(entry.get("mean")),
                 "n_fields": ev._as_int(entry.get("n_fields")),
                 "stale": not (fresh and identity_ok), "scored_at": entry.get("scored_at")}
    vs = None
    if score and score["mean"] is not None and production and production["mean"] is not None \
            and comparable:
        vs = _round(score["mean"] - production["mean"])
    learned = stored.get("learned_knees") if fresh else None
    progress = ev.member_progress(step, origin, by_member.get(name))
    return {
        "name": name, "label": ensemble_registry.member_label(name),
        "spec": (UNIVERSAL_SPEC_PREFIX if cls == ensemble_registry.UNIVERSAL_CLASS
                 else MEMBER_SPEC_PREFIX) + name,
        "class": cls,
        "num_res_blocks": ev._as_int(origin.get("num_res_blocks")),
        "steps": step,
        # The scored checkpoint's step and the training time/steps logged when
        # it was scored (null for entries scored before they were recorded) —
        # kept for a stale score, so the page can place it where it was made.
        "served_step": ev._as_int(stored.get("served_step")),
        "scored_hours": ev._finite(stored.get("scored_hours")),
        "scored_steps": ev._as_int(stored.get("scored_steps")),
        "target_steps": progress["target_steps"], "status": progress["status"],
        "batch_size": ev._as_int(origin.get("batch_size")),
        "crops_per_field": ev._as_int(origin.get("crops_per_field")),
        "knee": "learned" if origin.get("learned_output_knee") else "fixed",
        "output_knee": ev._finite(origin.get("output_knee")),
        "learned_knees": ([ev._finite(v) for v in learned] if isinstance(learned, list)
                          else None),
        "jit_compile": bool(origin.get("jit_compile")),
        "mixed_precision": bool(origin.get("mixed_precision")),
        "bootstrap": origin.get("bootstrap"), "seed": origin.get("seed"),
        "train_hours": tt["hours"], "train_rows": tt["rows"],
        "hardware": hardware(name, jobs),
        "fingerprint": fingerprint,
        "score": score, "vs_production_db": vs,
        "created_at": origin.get("created_at"), "commit": origin.get("commit"),
    }


def _empty_payload() -> dict:
    return {"available": False, "knees": list(KNEE_GRID_E), "bands": list(BAND_NAMES),
            "integration": {"from_e": KNEE_GRID_E[0], "to_e": KNEE_GRID_E[-1]},
            "identity": None, "production": None, "mean_model": None, "comparable": False,
            "models": [], "combiners": [], "fasrc": [], "weights_pending": [], "notes": []}


def universal_payload(starless: bool = False) -> dict:
    """Models › Universal (``GET /ensemble/universal.json``): every local
    universal model with its recipe, training time, hardware and test score;
    every gate with its score and weighted training time; the universal
    models still only on FASRC; plus the production gate's and plain mean's
    knee-integrated PSNR for reference. Reads cached files only (no SSH, no
    TensorFlow, nothing written); answers on an empty ensemble."""
    out = _empty_payload()
    if starless:
        out["notes"].append("universal models are STARFULL only")
        return out
    base = ev.ensemble_dir()
    registry = ensemble_registry.load_registry(base)
    jobs = ev.training_jobs()
    by_member = ev._jobs_by_member(jobs)
    knee = ev.knee_psnr_status(False)
    scores = ev.read_universal_scores()
    identity = scores.get("identity") if isinstance(scores.get("identity"), dict) else None
    current = ev.universal_score_identity()
    identity_ok = ev.universal_identity_matches(identity, current)
    universal_production = ensemble_registry.production_name(base)
    if universal_production is not None:
        # Production is one universal model: its own score, on the same store.
        production = _universal_reference(universal_production, scores, identity_ok)
        comparable = production is not None
    else:
        production = _knee_model(knee, SPATIAL_GATE_KIND)
        comparable = _comparable(knee, identity)
    models = [_model_row(name, cls, d, jobs=jobs, by_member=by_member, scores=scores,
                         identity_ok=identity_ok, production=production, comparable=comparable)
              for name, (cls, d) in universal_targets(base).items()]
    fasrc = fasrc_universal_rows(jobs, registry)
    combiners, pending = _combiner_table(base=base, registry=registry, identity=identity,
                                         knee_comparable=comparable)
    out.update({
        "available": bool(models or fasrc),
        "identity": identity,
        "production": production,
        # the universal model that is production (null: production is the gate)
        "production_model": universal_production,
        "mean_model": _knee_model(knee, "ensemble_mean"),
        # production/mean_model scored on the same fields, records and target
        # as the universal scores (the rows' vs_production_db needs it).
        "comparable": comparable,
        "models": models,
        "combiners": combiners,
        "fasrc": fasrc,
        # gates whose shares are missing but measurable from the validate
        # cubes now (the score job measures them first)
        "weights_pending": pending,
    })
    notes = out["notes"]
    missing = [n for n in registry["universal"] if n not in {m["name"] for m in models}]
    if missing:
        notes.append(f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'} registered "
                     "but not on disk with a checkpoint (pull in progress, or removed by the "
                     "mirror) — pull again")
    if pending:
        notes.append(f"{len(pending)} gate(s) without measured weights ("
                     + ", ".join(pending) + ") — “Measure gate weights” measures them from "
                     "the validate cubes")
    unscored = [m["name"] for m in models if m["score"] is None]
    if unscored:
        notes.append(f"{len(unscored)} universal model(s) not scored yet: "
                     + ", ".join(unscored) + " — run “Score”")
    stale = [m["name"] for m in models if m["score"] and m["score"]["stale"]]
    if stale:
        notes.append(f"{len(stale)} score(s) stale (checkpoint, test fields or records changed): "
                     + ", ".join(stale))
    if current.get("records_fp") is None:
        notes.append("the test records are not local — scores cannot be checked against "
                     "them (sync them on Synthetic › Records)")
    if universal_production is not None:
        pass                      # production is a universal model: no gate to evaluate
    elif not knee.get("available"):
        notes.append("no knee-PSNR payload — the production gate's ∫PSNR is unknown "
                     "(Models › Leaderboard: compute the PSNR-vs-knee curves)")
    elif knee.get("stale"):
        notes.append("the knee-PSNR payload is stale — production and plain-mean ∫PSNR refer "
                     "to older cubes; re-run Evaluate for a like-for-like comparison")
    if universal_production is not None:
        notes.append(f"production is universal model {universal_production.removeprefix('member_')}"
                     " — Δ vs production compares with its score")
    elif production and identity and not comparable:
        notes.append("the knee-PSNR payload was scored on other fields or records than the "
                     "universal models — Δ vs production is left out")
    notes.append("training time = Σ duration_s of every training_log.csv row (rolled-back "
                 "windows included); A100 MIG-slice hours are wall hours, not "
                 "A100-equivalent")
    return out


# --------------------------------------------------------------------------- #
# The score job
# --------------------------------------------------------------------------- #

def _test_pairs(records_dir: str, wanted: list[int]) -> Iterator[tuple[int, np.ndarray, object]]:
    """``(index, LR array, HR record)`` of every wanted test field present in
    both ``dirty_test`` and ``hr_test`` (records are in index order)."""
    if not wanted:
        return
    last = max(wanted) + 1
    targets = iter(ImageSet.read(tfrecord_path(records_dir, "hr_test"), num_images=last))
    lrs = iter(ImageSet.read(tfrecord_path(records_dir, "dirty_test"), num_images=last))
    target, lr = next(targets, None), next(lrs, None)
    for index in sorted(set(wanted)):
        while target is not None and (target.index or 0) < index:
            target = next(targets, None)
        while lr is not None and (lr.index or 0) < index:
            lr = next(lrs, None)
        if target is None or lr is None:
            return
        if target.index == index and lr.index == index:
            yield index, np.asarray(lr.data, np.float32), target


def _field_truth(target_record, fwhm: float) -> np.ndarray:
    """The Leaderboard's truth of one field: the HR record blurred to the
    target PSF."""
    return np.asarray(blur_target_array(
        np.asarray(target_record.data, np.float32), fwhm,
        pixel_scale_arcsec=target_record.pixel_scale_arcsec), np.float32)


def _field_curve(sr: np.ndarray, target_record, fwhm: float) -> np.ndarray:
    """PSNR at every knee and band of one field — the Leaderboard's metric,
    the stretched truth reused."""
    truth = _field_truth(target_record, fwhm)
    return knee_psnr(sr, truth, truth_asinh=stretched_truth(truth))


def _field_measures(sr: np.ndarray, target_record, fwhm: float) -> tuple[np.ndarray, dict]:
    """One field's PSNR curve (:func:`_field_curve`) and spectra partial sums
    (:func:`~euclid_polish.eval.model_spectra.field_sums`), one truth for
    both."""
    truth = _field_truth(target_record, fwhm)
    return knee_psnr(sr, truth, truth_asinh=stretched_truth(truth)), field_sums(sr, truth)


def _score_model(cap, name: str, d: str, identity: dict, records_dir: str, *,
                 done: int, total: int) -> tuple[dict, np.ndarray, list[int], dict]:
    """Score one universal model on the identity's test fields: ``(index
    entry, curves (F, K, C), fields, spectra record)`` — the spectra come
    from the same outputs and truths. The model loads as an ensemble of one
    (the ensemble-only rule)."""
    label = ensemble_registry.member_label(name)
    started = time.monotonic()
    fingerprint = member_fingerprint(d)           # before loading: a pull mid-score → stale
    logged = train_time(d)                        # the training this checkpoint had
    logged_steps = logged["steps"] if logged["steps"] is not None else ev._member_last_step(d)
    model = EnsembleModel(ev.ensemble_dir(), labels=[label], include_universal=True,
                          starless=False)
    fwhm = float(identity["target_psf_fwhm_arcsec"])
    n_wanted = len(identity["indices"])
    fields: list[int] = []
    curves: list[np.ndarray] = []
    spectra = SpectraAccumulator()
    pending: list[tuple[int, Future]] = []

    def drain(limit: int) -> None:
        while len(pending) > limit:
            index, future = pending.pop(0)
            curve, sums = future.result()
            curves.append(curve)
            spectra.merge(sums)
            fields.append(index)

    with ThreadPoolExecutor(max_workers=ev._KNEE_PSNR_WORKERS) as pool:
        for n, (index, lr, target) in enumerate(_test_pairs(records_dir, identity["indices"])):
            sr = np.asarray(model.member_arrays(lr)[0], np.float32)
            pending.append((index, pool.submit(_field_measures, sr, target, fwhm)))
            drain(2 * ev._KNEE_PSNR_WORKERS)
            cap.tick(done + n + 1, total, f"{name}: field {n + 1}/{n_wanted}")
        drain(0)
    if not curves:
        raise RuntimeError("none of the test fields is in both dirty_test and hr_test")
    stack = np.stack(curves)
    integrated = integrated_psnr(stack.mean(axis=0))
    learned = infer_checkpoint_learned_output_knee(d)
    entry = {"fingerprint": fingerprint,
             "integrated": np.round(integrated, 4).tolist(),
             "mean": round(float(np.mean(integrated)), 4),
             "n_fields": len(fields),
             "scored_at": datetime.now(UTC).isoformat(timespec="seconds"),
             "seconds": round(time.monotonic() - started, 1),
             "learned_knees": ([round(float(q), 4) for q in learned]
                               if learned is not None else None),
             "served_step": checkpoint_step(d),
             "scored_hours": logged["hours"], "scored_steps": logged_steps}
    record = {**spectra.result(list(BAND_NAMES)), "name": name, "fingerprint": fingerprint,
              "identity": identity, "served_step": entry["served_step"],
              "computed_at": entry["scored_at"]}
    return entry, stack, fields, record


def _save_curves(store: str, name: str, fields: list[int], curves: np.ndarray) -> None:
    path = os.path.join(store, f"{name}.npz")
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as handle:
        np.savez(handle, fields=np.asarray(fields, np.int64),
                 knees=np.asarray(KNEE_GRID_E, np.float64),
                 curves=np.asarray(curves, np.float64))
    os.replace(tmp, path)


def _needs_spectra(name: str, d: str, identity: dict, *, force: bool) -> bool:
    """Whether a model's spectra record is missing or stale (as a score:
    another checkpoint, or other fields/target/records)."""
    record = None if force else ev.read_universal_spectra(name)
    if not record or record.get("fingerprint") != member_fingerprint(d):
        return True
    return not ev.universal_identity_matches(record.get("identity"), identity)


def _needs_score(entry: dict | None, d: str, stored_identity: dict | None,
                 identity: dict, *, force: bool) -> bool:
    """Whether a model's stored score is missing or stale (another checkpoint,
    or fields/target/records that changed — records that are not local
    cannot be checked)."""
    if force or not entry or entry.get("fingerprint") != member_fingerprint(d):
        return True
    return not ev.universal_identity_matches(stored_identity, identity)


def job_universal_score(cap, *, members: list[str] | None = None, force: bool = False,
                        spectra: bool = False) -> dict:
    """Measure the gate weights, then score the universal models (every local
    one, or ``members``) on the Leaderboard's test fields with its exact
    knee-integrated PSNR, into the universal scores store; every model scored
    also gets its spectra record (Models › Spectra). Local job
    (``kind="universal-score"``).

    The weight diagnostic of every gate variant lacking a current one is
    measured FIRST, from the validate cubes only — so it runs even when
    nothing needs scoring or the test records are not local. A model whose
    entry is current (same checkpoint fingerprint, same identity) is reused
    unless ``force`` — or, with ``spectra``, unless its spectra record is
    missing or stale. Without local test records the job still succeeds when
    nothing needs scoring (scores are left as they are) and fails, after the
    gate weights, when something does.

    Result ``{scored, reused, failed: {name: error}, gates: {computed,
    reused, unavailable: {variant: reason}}, identity, spectra: [names whose
    spectra record was written]}``."""
    base = ev.ensemble_dir()
    targets = universal_targets(base)
    if members:
        wanted = list(dict.fromkeys(ensemble_registry.member_name(m) for m in members))
        unknown = [n for n in wanted if n not in targets]
        if unknown:
            raise RuntimeError("not a local universal model with a checkpoint: "
                               + ", ".join(unknown))
        targets = {n: targets[n] for n in wanted}
    gates = _gate_weight_diagnostics(cap)          # validate cubes only — first
    if gates["computed"] or gates["unavailable"]:
        print(f"  • gate weights: {len(gates['computed'])} measured, "
              f"{len(gates['reused'])} current, {len(gates['unavailable'])} unavailable")
    index = ev.read_universal_scores()
    records_dir = ev._sky_records_local_dir()
    identity = ev.universal_score_identity()

    def needs_work(name: str, d: str) -> bool:
        return (_needs_score(index["models"].get(name), d, index.get("identity"), identity,
                             force=force)
                or (spectra and _needs_spectra(name, d, identity, force=force)))

    if not records_dir or identity["records_fp"] is None:
        why = (ev._NO_SKY_RECORDS if not records_dir else
               "the test records (dirty_test + hr_test) are not local — sync them on "
               "Synthetic › Records.")
        need = [n for n, (_cls, d) in targets.items() if needs_work(n, d)]
        if need:
            raise RuntimeError(f"{why} ({len(need)} model(s) need a score: "
                               f"{', '.join(need)}; the gate weights were measured first)")
        print("  • every universal model's score is current (test records not local)")
        return {"scored": [], "reused": list(targets), "failed": {}, "gates": gates,
                "identity": index.get("identity"), "spectra": []}
    store = ev.universal_scores_dir()
    os.makedirs(store, exist_ok=True)
    index_path = os.path.join(store, ev.UNIVERSAL_SCORES_INDEX)
    if index.get("identity") != identity:
        # Other fields, records or target: every stored score is void.
        for name in index["models"]:
            with contextlib.suppress(OSError):
                os.remove(os.path.join(store, f"{name}.npz"))
        index = {"schema": ev.UNIVERSAL_SCORES_SCHEMA, "identity": identity, "models": {}}
        ev._atomic_json(index_path, index)
    todo = [n for n, (_cls, d) in targets.items() if needs_work(n, d)]
    reused = [n for n in targets if n not in todo]
    per_model = len(identity["indices"])
    total = max(1, len(todo) * per_model)
    scored: list[str] = []
    failed: dict[str, str] = {}
    for k, name in enumerate(todo):
        d = targets[name][1]
        print(f"  • scoring {name} on {per_model} test fields")
        try:
            entry, curves, fields, record = _score_model(cap, name, d, identity, records_dir,
                                                         done=k * per_model, total=total)
        except Exception as exc:  # noqa: BLE001 — one model never voids the others
            failed[name] = f"{type(exc).__name__}: {exc}"
            print(f"  ! {name} not scored: {failed[name]}")
            continue
        _save_curves(store, name, fields, curves)
        index["models"][name] = entry
        ev._atomic_json(index_path, index)
        ev._atomic_json(ev.universal_spectra_path(name), record)
        scored.append(name)
        print(f"  ✓ {name}: ∫PSNR {entry['mean']:.3f} dB "
              f"({' / '.join(f'{v:.2f}' for v in entry['integrated'])}) on "
              f"{entry['n_fields']} fields, {entry['seconds']:.0f} s"
              + ("" if record["available"] else f" (no spectra: {record['reason']})"))
    if failed and not scored and not reused:
        raise RuntimeError("no universal model could be scored: "
                           + "; ".join(f"{n}: {e}" for n, e in failed.items()))
    return {"scored": scored, "reused": reused, "failed": failed, "gates": gates,
            "identity": identity, "spectra": list(scored)}


# --------------------------------------------------------------------------- #
# The pull job
# --------------------------------------------------------------------------- #

#: First output line of :func:`_remote_probe`'s command: whether the remote
#: member dir holds a checkpoint.
_CKPT_YES, _CKPT_NO = "checkpoint=1", "checkpoint=0"


def _remote_probe_command(member_dir: str) -> str:
    """One shell line: ``checkpoint=1|0`` (a ``checkpoint`` manifest or an
    ``*.index`` file in the dir), then the dir's ``origin.json``."""
    q = shlex.quote(member_dir)
    return (f"if [ -f {q}/checkpoint ] || ls {q}/*.index >/dev/null 2>&1; "
            f"then echo {_CKPT_YES}; else echo {_CKPT_NO}; fi; cat {q}/origin.json")


def _remote_probe(ssh, remote_dir: str, name: str) -> tuple[dict | None, str]:
    """A FASRC member's ``origin.json`` when its dir also holds a checkpoint
    (one ssh call), else ``(None, reason)``."""
    member_dir = f"{remote_dir}/{name}"
    rc, out, err = ssh.run(_remote_probe_command(member_dir), timeout=60)
    head, _sep, body = str(out or "").partition("\n")
    if head.strip() == _CKPT_NO:
        return None, ("no checkpoint on FASRC (the job ended before its first save, or "
                      "the dir is gone)")
    if rc != 0 or head.strip() != _CKPT_YES:
        return None, f"no origin.json on FASRC ({(err or '').strip()[:160] or member_dir})"
    try:
        origin = json.loads(body)
    except (TypeError, ValueError):
        return None, "its origin.json on FASRC is unreadable"
    if not isinstance(origin, dict):
        return None, "its origin.json on FASRC is unreadable"
    return origin, ""


def _pulled_checkpoint(local: str) -> bool:
    """The rsync delivered a loadable checkpoint: its manifest AND the index
    it names (rsync copies ``checkpoint`` before the ``ckpt-*`` files)."""
    return _checkpoint_exists(local) and member_fingerprint(local) is not None


def _undo_failed_pull(base: str, name: str, local: str, *, created_dir: bool,
                      was_registered: bool) -> str:
    """Clean up after a failed rsync; returns what became of the name. The
    partial dir this pull created is removed FIRST (a ``checkpoint``
    manifest without its weights would read as a pulled model, and — once
    unregistered — the bootstrap would activate it); then the registration
    this pull added is removed. A registration that predates the pull, or a
    dir that was already there with a checkpoint, stays."""
    if created_dir:
        shutil.rmtree(local, ignore_errors=True)
    if was_registered:
        return "it stays registered; pull again"
    if os.path.isdir(local) and _checkpoint_exists(local):
        return "it stays registered (a checkpoint is on disk); pull again"
    try:
        ensemble_registry.remove_universal_entry(base, name)
    except ValueError:
        return "pull again"
    return "not registered; pull again"


def _pull_one(ssh, base: str, remote: str, name: str, ensemble_names: set[str]) -> str | None:
    """Pull one universal model; ``None`` on success, else why it was not."""
    reg = ensemble_registry.load_registry(base)
    if name in reg["active"]:
        return "an active ensemble member — the ensemble pull refreshes it"
    if name in {str(t.get("name")) for t in reg["archived"]}:
        return "archived (tombstoned) — restore it from its zip instead"
    if name in ensemble_names and name not in reg["universal"]:
        return ("an ensemble member (a local gate or cube manifest reads it) — the ensemble "
                "pull restores it")
    origin, why = _remote_probe(ssh, remote, name)
    if origin is None:
        return why
    if not ensemble_registry.is_universal_origin(origin):
        return ("not a universal model (its origin.json has no asinh_knees with a fixed or "
                "learned output knee)")
    was_registered = name in reg["universal"]
    local = os.path.join(base, name)
    created_dir = not os.path.exists(local)
    try:
        ensemble_registry.add_universal_entry(base, name)          # before the rsync
    except ValueError as exc:
        return str(exc)
    try:
        rc, _out, err = ssh.rsync_pull(f"{remote}/{name}/", local, timeout=3600)
        failure = (None if _pulled_checkpoint(local) else
                   f"the rsync left no loadable checkpoint (rc={rc}: {(err or '').strip()[:200]})")
    except Exception as exc:  # noqa: BLE001 — a dead link fails this name only
        failure = f"the rsync failed ({type(exc).__name__}: {exc})"
    if failure is None:
        return None
    outcome = _undo_failed_pull(base, name, local, created_dir=created_dir,
                                was_registered=was_registered)
    return f"{failure} — {outcome}"


def job_universal_pull(cap, *, members: list[str]) -> dict:
    """Pull universal models from FASRC as STANDALONE models (never ensemble
    members). Per name: refuse a malformed, active or tombstoned name, or one
    a local gate or cube manifest reads (an ensemble member whose dir
    vanished); in ONE ssh call check that the remote dir holds a checkpoint
    and read its ``origin.json``, refusing a dir without a checkpoint ("no
    checkpoint on FASRC") or a model that is not universal — nothing is
    registered then; register it
    (:func:`~euclid_polish.ensemble_registry.add_universal_entry`) BEFORE the
    rsync, so the registry bootstrap never activates the arriving dir; rsync
    ``<remote ensemble>/<name>/`` → ``<ensemble>/<name>/`` (as the ensemble
    pull does) and prune orphaned checkpoint files. A failed rsync (an
    exception, or no loadable checkpoint after it) removes the registration
    this call added (and the partial dir it created); a name's failure never
    stops the others. Local job (``kind="universal-pull"``) that needs FASRC.

    Result ``{pulled: [names], refused: {name: reason}}``."""
    if not members:
        raise RuntimeError("name the universal models to pull")
    ssh = STATE.ssh
    if ssh is None or not ssh.is_connected():
        raise RuntimeError("not connected to FASRC — connect on the FASRC tab first.")
    base = ev.ensemble_dir()
    os.makedirs(base, exist_ok=True)
    remote = ev.remote_ensemble_dir().rstrip("/")
    refused: dict[str, str] = {}
    names: list[str] = []
    for raw in members:
        try:
            names.append(ensemble_registry.member_name(raw))
        except ValueError:
            refused[str(raw)] = "not a member name"
    names = list(dict.fromkeys(names))
    ensemble_names = ev.gate_and_cube_member_names()
    pulled: list[str] = []
    for i, name in enumerate(names):
        cap.tick(i, len(names), f"pull {name}")
        local = os.path.join(base, name)
        try:
            why = _pull_one(ssh, base, remote, name, ensemble_names)
            pruned = 0 if why is not None else sum(
                prune_orphaned_checkpoints(track)
                for track in (local, os.path.join(local, "loss_best")) if os.path.isdir(track))
        except Exception as exc:  # noqa: BLE001 — one name never stops the others
            why = f"{type(exc).__name__}: {exc}"
        if why is not None:
            refused[name] = why
            continue
        print(f"  ✓ {name} pulled as a universal model"
              + (f" ({pruned} stale checkpoint file(s) pruned)" if pruned else ""))
        pulled.append(name)
    cap.tick(len(names), len(names), "done")
    for name, reason in refused.items():
        print(f"  ! {name} refused: {reason}")
    if pulled:
        # Real-tile member-SR cache entries of an older checkpoint go stale.
        request_stale_purge(f"pulled universal {', '.join(pulled)}")
    return {"pulled": pulled, "refused": refused}
