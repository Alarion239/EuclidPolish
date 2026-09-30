"""Past-run resource usage and next-run SLURM resource recommendations.

Pure: takes job-ledger rows (``list[dict[str, str]]``, the
:class:`~euclid_polish.observability.job_log.JobLog` CSV), never touches
Flask, the file system or FASRC. ``web/routes/resources.py`` serves it.

Three layers:

* :func:`normalize_row` — one ledger row → :class:`RunUsage` (numbers parsed
  defensively: blank / garbage → ``None``, never raises), with the step
  profile's amount of work (``units``) and similarity ``key``.
* :func:`recommend` — CPUs / GPUs / memory / time for a planned run from the
  most similar past runs (match levels ``exact`` → ``similar`` → ``step``).
* :func:`summarize_step` — per-step dashboard numbers (state counts, median
  efficiencies, allocated-vs-used CPU/GPU/memory hours).

Array jobs are ONE ledger row whose allocation fields are per task and whose
elapsed is the max over tasks (``web/sacct.py``), so per task is the unit the
advisor recommends in. Which runs count as evidence: ``COMPLETED`` (full
evidence), ``OUT_OF_MEMORY`` (a memory lower bound: it needed more than it
asked for) and ``TIMEOUT`` (a time lower bound: it needed more than it ran);
``FAILED`` / ``CANCELLED`` are errors, not resource problems — counted in the
summaries only.
"""

from __future__ import annotations

import json
import math
import re
import shlex
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Tunables (the design spec's numbers)
# ---------------------------------------------------------------------------

#: A match level needs this many usable runs to win outright.
MIN_LEVEL_RUNS = 3
#: ``high`` confidence needs this many runs at ``exact`` / ``similar``.
HIGH_CONFIDENCE_RUNS = 5
#: Only the newest runs of the chosen level are used.
MAX_BASIS_RUNS = 20
#: Headroom over the p90 of memory / time.
HEADROOM = 1.2
#: Time never gets less than this over the p90 estimate.
MIN_TIME_MARGIN_S = 5 * 60
#: An out-of-memory run's request times this is a floor for the next one.
OOM_FLOOR_FACTOR = 1.25
#: A timed-out run that would time out again is given 1.5x its elapsed.
TIMEOUT_BUMP_FACTOR = 1.5
#: CPU steps with a median CPU efficiency below this get fewer CPUs.
CPU_REDUCE_BELOW_EFFICIENCY = 0.35
#: Headroom over the p75 of busy cores when reducing CPUs.
CPU_REDUCE_HEADROOM = 1.3
#: GPU steps are "starved" below this median GPU utilisation (percent) ...
GPU_STARVED_BELOW_UTIL = 60.0
#: ... while their CPUs are this busy (median CPU efficiency).
GPU_STARVED_ABOVE_CPU_EFFICIENCY = 0.75
#: Starved GPU steps get this many times the CPUs their runs had.
GPU_STARVED_CPU_FACTOR = 1.5
#: Per-unit rates come from runs of comparable size: within this factor of
#: the planned work either way (a smoke run's rate is mostly its startup, a
#: big run amortises it).
COMPARABLE_WORK_FACTOR = 4.0
#: At most this many rounds of "change the CPUs, re-match the runs at the
#: new count" (each round only moves the count one way, so it ends sooner).
_MAX_CPU_ROUNDS = 8
#: Memory is rounded up to a multiple of this (and never below it), GB.
MEMORY_STEP_GB = 4
#: Time is rounded up to a multiple of this (and never below it), seconds.
TIME_STEP_S = 15 * 60
#: Time is never recommended above this (the partition limit), seconds.
TIME_CAP_S = 3 * 86400

COMPLETED = "COMPLETED"
OUT_OF_MEMORY = "OUT_OF_MEMORY"
TIMEOUT = "TIMEOUT"
#: States whose runs are evidence for a recommendation.
COUNTED_STATES = frozenset({COMPLETED, OUT_OF_MEMORY, TIMEOUT})
#: Of those, the runs whose peak memory / utilisation ran to the end (an
#: out-of-memory run was killed partway: only its request is evidence).
_FULL_RUN_STATES = frozenset({COMPLETED, TIMEOUT})

#: Summary state bucket of each (normalised) SLURM state.
_STATE_BUCKETS: dict[str, str] = {
    COMPLETED: "completed",
    OUT_OF_MEMORY: "oom",
    TIMEOUT: "timeout", "DEADLINE": "timeout",
    "FAILED": "failed", "NODE_FAIL": "failed", "BOOT_FAIL": "failed",
    "CANCELLED": "cancelled", "PREEMPTED": "cancelled", "REVOKED": "cancelled",
    "RUNNING": "running", "PENDING": "running", "REQUEUED": "running",
    "CONFIGURING": "running", "COMPLETING": "running", "RESIZING": "running",
    "SUSPENDED": "running",
}
STATE_KEYS = ("completed", "oom", "timeout", "failed", "cancelled", "running")
_FINISHED_BUCKETS = frozenset({"completed", "oom", "timeout", "failed", "cancelled"})

#: The resource fields a recommendation covers (the form's names).
RESOURCE_FIELDS = ("n_cpus", "n_gpus", "memory", "time_limit")

#: Dashboard step order: these first, then by the newest submission.
PINNED_STEPS = ("ensemble_train", "synthetic_generate")

GPU_MEMORY_NOTE = (
    "GPU memory is not a signal: TensorFlow preallocates the whole card, so "
    "used GPU memory always reads ~100%."
)

_TRUE_WORDS = frozenset({"1", "true", "yes", "on"})


# ---------------------------------------------------------------------------
# Defensive parsing
# ---------------------------------------------------------------------------

def _float(value: Any) -> float | None:
    """A finite float from a CSV cell / JSON value, else ``None``."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _int(value: Any) -> int | None:
    """An integer (``"4"``, ``4``, ``"4.0"``), else ``None``."""
    out = _float(value)
    if out is None or out != int(out):
        return None
    return int(out)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value if value is not None else "").strip().lower() in _TRUE_WORDS


_MEMORY_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*([KMGT]?)(?:I?B)?$", re.IGNORECASE)
_MEMORY_FACTOR_MB = {"K": 1 / 1024, "M": 1.0, "G": 1024.0, "T": 1024.0 ** 2}


def parse_memory_mb(value: Any, *, bare_unit: str = "M") -> float | None:
    """SLURM memory text → MB: ``"32G"``, ``"17GB"``, ``"4000M"``, ``"1T"``.

    A plain number means ``bare_unit`` — MB by default (SLURM's ``--mem``
    semantics, what an old ledger row meant); pass ``"G"`` for a console form
    value (the step form appends ``G`` to a bare number before submitting).
    Blank / garbage → ``None``.
    """
    match = _MEMORY_RE.match(str(value if value is not None else "").strip())
    if not match:
        return None
    number, unit = match.groups()
    mb = float(number) * _MEMORY_FACTOR_MB[(unit or bare_unit).upper()]
    return mb if mb > 0 else None


_TIME_RE = re.compile(r"^(?:(\d+)-)?(\d+(?::\d+){0,2})$")


def parse_time_s(value: Any) -> float | None:
    """SLURM time limit → seconds.

    ``MM``, ``MM:SS``, ``H:MM:SS``, ``D-HH``, ``D-HH:MM``, ``D-HH:MM:SS``.
    Blank, ``UNLIMITED`` or garbage → ``None``.
    """
    match = _TIME_RE.match(str(value if value is not None else "").strip())
    if not match:
        return None
    days, rest = match.groups()
    parts = [int(p) for p in rest.split(":")]
    if days is not None:
        hours, minutes, seconds = (parts + [0, 0])[:3]
        total = int(days) * 86400 + hours * 3600 + minutes * 60 + seconds
    elif len(parts) == 1:
        total = parts[0] * 60
    elif len(parts) == 2:
        total = parts[0] * 60 + parts[1]
    else:
        total = parts[0] * 3600 + parts[1] * 60 + parts[2]
    return float(total) if total > 0 else None


def normalize_state(value: Any) -> str:
    """``"cancelled by 1234"`` → ``"CANCELLED"``; ``"OOM"`` → ``"OUT_OF_MEMORY"``."""
    words = str(value or "").split()
    state = words[0].upper() if words else ""
    return OUT_OF_MEMORY if state == "OOM" else state


def _parse_params(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    try:
        parsed = json.loads(str(value or "") or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def round_memory_gb(mb: float) -> int:
    """Round ``mb`` UP to a multiple of :data:`MEMORY_STEP_GB` GB (≥ one step)."""
    steps = math.ceil(mb / 1024.0 / MEMORY_STEP_GB - 1e-9)
    return max(1, steps) * MEMORY_STEP_GB


def format_memory(gb: int) -> str:
    """``36`` → ``"36G"`` (the form's memory text)."""
    return f"{int(gb)}G"


def round_time_s(seconds: float) -> int:
    """Round UP to :data:`TIME_STEP_S`, at least one step, at most :data:`TIME_CAP_S`."""
    steps = math.ceil(seconds / TIME_STEP_S - 1e-9)
    return min(TIME_CAP_S, max(1, steps) * TIME_STEP_S)


def format_time(seconds: float) -> str:
    """Seconds → SLURM ``H:MM:SS`` (``D-HH:MM:SS`` from 24 h)."""
    total = int(round(seconds))
    days, rest = divmod(total, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    if days:
        return f"{days}-{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours}:{minutes:02d}:{secs:02d}"


def _human_duration(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.1f} h"
    if seconds >= 60:
        return f"{seconds / 60:.0f} min"
    return f"{seconds:.0f} s"


def _gb(mb: float) -> str:
    gb = mb / 1024.0
    return f"{gb:.1f} GB" if gb < 100 else f"{gb:.0f} GB"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def percentile(values: Iterable[float | None], q: float) -> float | None:
    """numpy-style linear-interpolation percentile of the non-``None`` values
    (``q`` in 0–100); one value is that value; none → ``None``."""
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def median(values: Iterable[float | None]) -> float | None:
    return percentile(values, 50)


# ---------------------------------------------------------------------------
# Step profiles (workload scaling)
# ---------------------------------------------------------------------------

Key = tuple[Any, ...]


@dataclass(frozen=True)
class StepProfile:
    """How a step's task params translate into work and similarity.

    ``units(params)`` is the amount of work (scales time; ``None`` = unknown
    for this plan), ``key(params)`` a categorical similarity key plus its
    human label. ``memory_scales_with_cpus``: the step runs one worker per
    CPU, so memory is compared per CPU. ``time_scales_with_cpus``: those
    workers share the work, so a run measured at more CPUs than planned is
    stretched by the ratio (never shrunk: fewer CPUs is slower, more is not
    reliably faster). ``gpu``: a GPU step (historical steps no longer in the
    registry fall back to this).
    """

    units: Callable[[Mapping[str, Any]], float | None] | None = None
    units_label: str = ""
    key: Callable[[Mapping[str, Any]], tuple[Key, str]] | None = None
    memory_scales_with_cpus: bool = False
    time_scales_with_cpus: bool = False
    gpu: bool = False

    @property
    def has_units(self) -> bool:
        return self.units is not None

    def work(self, params: Mapping[str, Any]) -> tuple[float | None, str]:
        """``(units, units_label)`` of a run with ``params``."""
        if self.units is None:
            return None, self.units_label
        try:
            units = self.units(params)
        except (TypeError, ValueError, AttributeError):
            units = None
        return (units if units is not None and units > 0 else None), self.units_label

    def similarity(self, params: Mapping[str, Any]) -> tuple[Key, str]:
        """``(key, key_label)``; ``((), "")`` for a step without a key."""
        if self.key is None:
            return (), ""
        try:
            return self.key(params)
        except (TypeError, ValueError, AttributeError):
            return ("?",), "unreadable params"


_SPLIT_COUNT_PARAM = {"train": "n_train", "validate": "n_valid", "test": "n_test"}
_SPLIT_ALIASES = {"train": "train", "validate": "validate", "valid": "validate",
                  "val": "validate", "test": "test"}
_SPLITS = ("train", "validate", "test")


def _split_list(value: Any) -> list[str]:
    items = value if isinstance(value, list) else str(value or "").split(",")
    return [s for s in dict.fromkeys(_SPLIT_ALIASES.get(str(i).strip().lower(), "")
                                     for i in items) if s]


def _synthetic_plan(params: Mapping[str, Any]) -> tuple[str, list[str]]:
    """``("all" | "splits" | "resume", splits)`` of a generation run.

    ``regenerate_splits`` and ``force`` are the structured knobs; an older
    submission may carry them only in ``extra_flags`` (``--force``,
    ``--regenerate-splits[=]a,b``), which ``run_pipeline.py`` honours too.
    """
    force = _truthy(params.get("force"))
    splits = _split_list(params.get("regenerate_splits"))
    if not force and not splits:
        try:
            tokens = shlex.split(str(params.get("extra_flags") or ""))
        except ValueError:
            tokens = []
        for i, token in enumerate(tokens):
            if token == "--force":
                force = True
            elif token.startswith("--regenerate-splits="):
                splits = _split_list(token.split("=", 1)[1])
            elif token == "--regenerate-splits" and i + 1 < len(tokens):
                splits = _split_list(tokens[i + 1])
    if force or set(splits) == set(_SPLITS):
        return "all", list(_SPLITS)
    if splits:
        return "splits", [s for s in _SPLITS if s in splits]
    return "resume", []


def _synthetic_units(params: Mapping[str, Any]) -> float | None:
    """Images actually generated: the regenerated splits' scene counts
    (``None`` for a cache-first resume, whose work depends on the cache)."""
    mode, splits = _synthetic_plan(params)
    if mode == "resume":
        return None
    counts = [_int(params.get(_SPLIT_COUNT_PARAM[s])) for s in splits]
    if any(c is None for c in counts):
        return None
    return float(sum(c for c in counts if c is not None))


def _synthetic_key(params: Mapping[str, Any]) -> tuple[Key, str]:
    mode, splits = _synthetic_plan(params)
    what = mode if mode != "splits" else ",".join(sorted(splits))
    image_size = _int(params.get("image_size"))
    onthefly = _truthy(params.get("onthefly_train"))
    label = {"all": "all splits", "resume": "cache-first resume"}.get(mode, "+".join(splits))
    parts = [label]
    if image_size is not None:
        parts.append(f"{image_size} px")
    if onthefly:
        parts.append("on-the-fly train")
    return (what, image_size, onthefly), " · ".join(parts)


def _ensemble_mode(params: Mapping[str, Any]) -> str:
    return str(params.get("mode") or "add").strip().lower() or "add"


def _ensemble_units(params: Mapping[str, Any]) -> float | None:
    """Training steps per member (= per array task)."""
    if _ensemble_mode(params) == "continue":
        basis = str(params.get("continue_basis") or "extra").strip().lower()
        if basis != "extra":
            return None  # up to an absolute target: depends on each member's step
        return _float(params.get("extra_steps"))
    return _float(params.get("steps"))


def _member_spec(params: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = params.get("member_spec")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw) if raw.strip() else []
        except ValueError:
            raw = []
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _ensemble_key(params: Mapping[str, Any]) -> tuple[Key, str]:
    kind = "continue" if _ensemble_mode(params) == "continue" else "new"
    batch_size = _int(params.get("batch_size"))
    hr_crop = _int(params.get("hr_crop_size"))
    spec = _member_spec(params)
    run_blocks = _int(params.get("num_res_blocks"))
    blocks = {_int(item.get("num_res_blocks")) or run_blocks for item in spec} or {run_blocks}
    depth = tuple(sorted(blocks, key=lambda b: (b is None, b or 0)))
    multi_knee = bool(str(params.get("asinh_knees") or "").strip()) or any(
        item.get("asinh_knees") not in (None, "", []) for item in spec)
    parts = ["new members" if kind == "new" else "continue",
             f"batch {batch_size}" if batch_size is not None else "default batch",
             "/".join(str(b) if b is not None else "default" for b in depth) + " blocks",
             "multi-knee" if multi_knee else "single knee"]
    if hr_crop is not None:
        parts.append(f"crop {hr_crop}")
    return (kind, batch_size, depth, multi_knee, hr_crop), " · ".join(parts)


def _steps_units(params: Mapping[str, Any]) -> float | None:
    return _float(params.get("steps"))


PROFILES: dict[str, StepProfile] = {
    "synthetic_generate": StepProfile(
        units=_synthetic_units, units_label="images", key=_synthetic_key,
        # One generation worker per CPU: ≈1.5 GB each (15 GB @ 10, 29–31 @ 20),
        # and validate+test takes ~2x longer at 10 CPUs than at 32.
        memory_scales_with_cpus=True, time_scales_with_cpus=True),
    "ensemble_train": StepProfile(
        units=_ensemble_units, units_label="steps", key=_ensemble_key, gpu=True),
    "train": StepProfile(units=_steps_units, units_label="steps", gpu=True),
    "lensfinder_train": StepProfile(units=_steps_units, units_label="steps", gpu=True),
    "lens_isolation_train": StepProfile(units=_steps_units, units_label="steps", gpu=True),
}
_GENERIC_PROFILE = StepProfile()


def profile_for(step_id: str) -> StepProfile:
    """The step's profile (every other step: no units, empty key)."""
    return PROFILES.get(step_id, _GENERIC_PROFILE)


# ---------------------------------------------------------------------------
# Row normalisation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunUsage:
    """One ledger row, parsed: what the run asked for and what it used."""

    jobid: str
    step_id: str
    submitted_at: str
    state: str
    partition: str
    label: str
    cpus: int | None
    gpus: int | None
    req_memory: str
    req_memory_mb: float | None
    req_time_limit: str
    req_time_s: float | None
    elapsed_s: float | None
    cpu_efficiency: float | None
    cores_used: float | None
    peak_mem_mb: float | None
    mem_ratio: float | None
    time_ratio: float | None
    gpu_util: float | None
    gpu_mem_used_mb: float | None
    units: float | None
    units_label: str
    key: Key
    key_label: str
    params: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def counted(self) -> bool:
        """Evidence for a recommendation (finished for a resource reason)."""
        return self.state in COUNTED_STATES and (self.elapsed_s or 0) > 0

    @property
    def bucket(self) -> str:
        return _STATE_BUCKETS.get(self.state, "")

    def to_dict(self) -> dict[str, Any]:
        """The API's RunUsage JSON (no params / key: they are large / internal)."""
        return {
            "jobid": self.jobid,
            "submitted_at": self.submitted_at,
            "state": self.state,
            "partition": self.partition,
            "cpus": self.cpus,
            "gpus": self.gpus,
            "req_memory": self.req_memory,
            "req_memory_mb": _round(self.req_memory_mb, 1),
            "req_time_limit": self.req_time_limit,
            "req_time_s": self.req_time_s,
            "elapsed_s": self.elapsed_s,
            "cpu_efficiency": _round(self.cpu_efficiency, 4),
            "cores_used": _round(self.cores_used, 2),
            "peak_mem_mb": _round(self.peak_mem_mb, 1),
            "mem_ratio": _round(self.mem_ratio, 4),
            "time_ratio": _round(self.time_ratio, 4),
            "gpu_util": _round(self.gpu_util, 1),
            "gpu_mem_used_mb": _round(self.gpu_mem_used_mb, 1),
            "units": _clean_units(self.units),
            "units_label": self.units_label,
            "key_label": self.key_label,
            "label": self.label,
        }


def _round(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def _clean_units(value: float | None) -> float | int | None:
    if value is None:
        return None
    return int(value) if value == int(value) else value


def _max_or_none(*values: float | None) -> float | None:
    present = [v for v in values if v is not None]
    return max(present) if present else None


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den <= 0:
        return None
    return num / den


def normalize_row(row: Mapping[str, Any]) -> RunUsage:
    """One ledger row → :class:`RunUsage`. Never raises on bad cells."""
    step_id = str(row.get("step_id") or "")
    profile = profile_for(step_id)
    params = _parse_params(row.get("params_json"))
    cpus = _int(row.get("alloc_cpus")) or _int(row.get("req_cpus"))
    gpus = _int(row.get("alloc_gpus")) or _int(row.get("req_gpus"))
    req_memory = str(row.get("req_memory") or "")
    req_memory_mb = parse_memory_mb(req_memory)
    if req_memory_mb is None:
        alloc_mb = _float(row.get("alloc_memory_mb"))
        req_memory_mb = alloc_mb if alloc_mb and alloc_mb > 0 else None
    req_time_limit = str(row.get("req_time_limit") or "")
    req_time_s = parse_time_s(req_time_limit)
    elapsed_s = _float(row.get("elapsed_seconds"))
    if elapsed_s is not None and elapsed_s < 0:
        elapsed_s = None
    cpu_efficiency = _float(row.get("cpu_efficiency"))
    if cpu_efficiency is not None and cpu_efficiency < 0:
        cpu_efficiency = None
    cores_used = (cpu_efficiency * cpus
                  if cpu_efficiency is not None and cpus is not None else None)
    peak_mem_mb = _max_or_none(_float(row.get("max_rss_mb")),
                               _float(row.get("jobstats_cpu_memory_used_mb")))
    gpu_util = _float(row.get("jobstats_gpu_util"))
    if gpu_util is None:
        gpu_util = _float(row.get("gpu_util_mean"))
    gpu_mem_used_mb = _float(row.get("jobstats_gpu_memory_used_mb"))
    if gpu_mem_used_mb is None:
        gpu_mem_used_mb = _float(row.get("gpu_mem_peak_mb"))
    units, units_label = profile.work(params)
    key, key_label = profile.similarity(params)
    return RunUsage(
        jobid=str(row.get("jobid") or ""),
        step_id=step_id,
        submitted_at=str(row.get("submitted_at") or ""),
        state=normalize_state(row.get("state")),
        partition=str(row.get("partition") or ""),
        label=str(row.get("label") or ""),
        cpus=cpus,
        gpus=gpus,
        req_memory=req_memory,
        req_memory_mb=req_memory_mb,
        req_time_limit=req_time_limit,
        req_time_s=req_time_s,
        elapsed_s=elapsed_s,
        cpu_efficiency=cpu_efficiency,
        cores_used=cores_used,
        peak_mem_mb=peak_mem_mb,
        mem_ratio=_ratio(peak_mem_mb, req_memory_mb),
        time_ratio=_ratio(elapsed_s, req_time_s),
        gpu_util=gpu_util,
        gpu_mem_used_mb=gpu_mem_used_mb,
        units=units,
        units_label=units_label,
        key=key,
        key_label=key_label,
        params=params,
    )


def normalize_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[RunUsage, ...]:
    return tuple(normalize_row(row) for row in rows)


def _as_runs(items: Iterable[RunUsage | Mapping[str, Any]]) -> list[RunUsage]:
    return [item if isinstance(item, RunUsage) else normalize_row(item) for item in items]


def newest_first(runs: Iterable[RunUsage]) -> list[RunUsage]:
    """By ``submitted_at`` (ISO-8601 UTC: string order is time order)."""
    return sorted(runs, key=lambda r: r.submitted_at, reverse=True)


def latest_counted(runs: Iterable[RunUsage]) -> RunUsage | None:
    """The newest run that is evidence (its params seed the dashboard's
    "next run like the last one")."""
    return next((r for r in newest_first(runs) if r.counted), None)


# ---------------------------------------------------------------------------
# Match levels
# ---------------------------------------------------------------------------

LEVEL_EXACT = "exact"
LEVEL_SIMILAR = "similar"
LEVEL_STEP = "step"


@dataclass(frozen=True)
class Basis:
    level: str | None
    level_label: str
    runs: tuple[RunUsage, ...]
    fell_back: bool           # the first attempted level had < MIN_LEVEL_RUNS runs


def select_basis(runs: Sequence[RunUsage], profile: StepProfile, key: Key,
                 key_label: str, cpus: int | None) -> Basis:
    """The runs to learn from: the first match level with ≥ 3 usable runs
    (``exact`` = same key and CPU count, only when memory scales with CPUs;
    ``similar`` = same key, skipped for a step without a key; ``step``),
    else the most specific non-empty level. At most the 20 newest."""
    usable = newest_first(r for r in runs if r.counted)
    levels: list[tuple[str, str, list[RunUsage]]] = []
    settings = f"same settings ({key_label})" if key_label else "same settings"
    if profile.memory_scales_with_cpus and cpus is not None and key:
        levels.append((LEVEL_EXACT, f"{settings} at {cpus} CPUs",
                       [r for r in usable if r.key == key and r.cpus == cpus]))
    if key:
        levels.append((LEVEL_SIMILAR, settings, [r for r in usable if r.key == key]))
    levels.append((LEVEL_STEP, "every run of this step", usable))
    chosen = next((lv for lv in levels if len(lv[2]) >= MIN_LEVEL_RUNS), None)
    if chosen is None:
        chosen = next((lv for lv in levels if lv[2]), None)
    if chosen is None:
        return Basis(None, "no past runs", (), fell_back=True)
    level, label, selected = chosen
    return Basis(level, label, tuple(selected[:MAX_BASIS_RUNS]),
                 fell_back=chosen is not levels[0])


# ---------------------------------------------------------------------------
# The recommendation
# ---------------------------------------------------------------------------

def _clean_current(current: Mapping[str, Any] | None) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in RESOURCE_FIELDS:
        value = (current or {}).get(name)
        text = "" if value is None else str(value).strip()
        out[name] = text or None
    return out


def _fill(current: dict[str, str | None],
          defaults: Mapping[str, Any] | None) -> dict[str, str | None]:
    """``current`` with blanks taken from ``defaults`` (the baseline a
    "keep" rule keeps)."""
    filled = dict(current)
    for name, value in _clean_current(defaults).items():
        if filled.get(name) is None:
            filled[name] = value
    return filled


def _same_value(name: str, current: str | None, recommended: str | None) -> bool:
    """Compare parsed quantities (``"32G"`` == ``"32768M"``), not strings."""
    if current is None or recommended is None:
        return current == recommended
    if name in ("n_cpus", "n_gpus"):
        return _int(current) == _int(recommended)
    if name == "memory":
        a, b = parse_memory_mb(current, bare_unit="G"), parse_memory_mb(recommended, bare_unit="G")
        return a is not None and b is not None and abs(a - b) < 1.0
    a_s, b_s = parse_time_s(current), parse_time_s(recommended)
    return a_s is not None and b_s is not None and abs(a_s - b_s) < 1.0


def recommend(
    step_id: str,
    runs: Iterable[RunUsage | Mapping[str, Any]],
    *,
    params: Mapping[str, Any] | None = None,
    current: Mapping[str, Any] | None = None,
    needs_gpu: bool | None = None,
    fixed_cpus: int | None = None,
    fixed_gpus: int | None = None,
    defaults: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resources for the next run of ``step_id`` with task ``params``.

    ``runs`` are ledger rows or :class:`RunUsage` (other steps' are ignored);
    ``current`` the form's ``{n_cpus, n_gpus, memory, time_limit}`` (any may
    be missing: ``defaults`` fill the blanks the "keep" rules keep). Returns
    the API's Recommendation (without ``ok``): ``available`` is false — and
    the current values are echoed — when the step has no usable past run.
    The recommendation is its own recommendation: asking again with its
    resources (after Apply) changes nothing.
    """
    profile = profile_for(step_id)
    params = dict(params or {})
    step_runs = [r for r in _as_runs(runs) if r.step_id == step_id]
    gpu_step = bool(needs_gpu) if needs_gpu is not None else (
        profile.gpu or any((r.gpus or 0) > 0 for r in step_runs))
    form = _clean_current(current)
    base = _fill(form, defaults)
    if fixed_cpus is not None:
        base["n_cpus"] = str(int(fixed_cpus))
    if fixed_gpus is not None:
        base["n_gpus"] = str(int(fixed_gpus))

    planned_units, units_label = profile.work(params)
    key, key_label = profile.similarity(params)
    current_cpus = _int(base.get("n_cpus"))
    basis = select_basis(step_runs, profile, key, key_label, current_cpus)
    notes: list[str] = []
    warnings: list[str] = []
    reasons: dict[str, str] = {}
    if gpu_step:
        notes.append(GPU_MEMORY_NOTE)

    # ---- CPUs --------------------------------------------------------------
    # A new count re-matches the runs at that count (a step whose memory
    # scales with its CPUs has a per-count exact level) until the rule holds
    # there, so memory and time come from runs like the recommended one and
    # asking again after Apply changes nothing.
    rec_cpus = current_cpus
    cpu_note = ""
    if fixed_cpus is None and current_cpus is not None and basis.runs:
        for _ in range(_MAX_CPU_ROUNDS):
            cpus, reason, note = _cpu_rule(basis.runs, gpu_step, rec_cpus)
            if cpus == rec_cpus:
                break
            rec_cpus, reasons["n_cpus"], cpu_note = cpus, reason, note
            basis = select_basis(step_runs, profile, key, key_label, rec_cpus)

    result: dict[str, Any] = {
        "step_id": step_id,
        "available": bool(basis.runs),
        "confidence": "low",
        "resources": dict(base),
        "current": form,
        "changes": [],
        "basis": {
            "level": basis.level,
            "level_label": basis.level_label,
            "n_runs": len(basis.runs),
            "jobids": [r.jobid for r in basis.runs],
            "units": _clean_units(planned_units),
            "units_label": units_label,
            "rate_s_per_unit": None,
        },
        "notes": notes,
        "warnings": warnings,
    }
    if not basis.runs:
        notes.append("No completed, out-of-memory or timed-out run of this step yet: "
                     "nothing to learn from.")
        return result

    n = len(basis.runs)
    if basis.fell_back and basis.level is not None:
        notes.append(f"Fewer than {MIN_LEVEL_RUNS} runs at the closest match: "
                     f"using {basis.level_label}.")
    if cpu_note:
        notes.append(cpu_note)
    if fixed_cpus is not None:
        reasons["n_cpus"] = "fixed for this step"
    if fixed_gpus is not None:
        reasons["n_gpus"] = "fixed for this step"
    if rec_cpus is not None:
        result["resources"]["n_cpus"] = str(rec_cpus)

    # ---- memory ------------------------------------------------------------
    memory_mb, memory_reason = _recommend_memory(basis.runs, profile, rec_cpus)
    if memory_mb is not None:
        result["resources"]["memory"] = format_memory(round_memory_gb(memory_mb))
        reasons["memory"] = memory_reason

    # ---- time --------------------------------------------------------------
    time_s, time_reason, rate = _recommend_time(
        basis.runs, planned_units, units_label, profile, rec_cpus, notes, warnings)
    if time_s is not None:
        result["resources"]["time_limit"] = format_time(time_s)
        reasons["time_limit"] = time_reason
        result["basis"]["rate_s_per_unit"] = _round(rate, 6)

    # ---- confidence ----------------------------------------------------------
    # A step without a key has one kind of run: its step level is "similar".
    alike = basis.level in (LEVEL_EXACT, LEVEL_SIMILAR) or not key
    units_known = rate is not None or not profile.has_units
    if n >= HIGH_CONFIDENCE_RUNS and alike and units_known:
        result["confidence"] = "high"
    elif n >= MIN_LEVEL_RUNS:
        result["confidence"] = "medium"
    if n < MIN_LEVEL_RUNS:
        warnings.append(f"Only {_plural(n, 'past run')} to learn from: treat this as a rough guide.")

    for name in RESOURCE_FIELDS:
        recommended = result["resources"].get(name)
        if recommended is None or _same_value(name, form.get(name), recommended):
            continue
        result["changes"].append({
            "field": name,
            "current": form.get(name),
            "recommended": recommended,
            "reason": reasons.get(name) or "the current field is blank",
        })
    return result


def _cpu_rule(runs: Sequence[RunUsage], gpu_step: bool, cpus: int) -> tuple[int, str, str]:
    """``(cpus, reason, note)`` the basis ``runs`` suggest for a form asking
    ``cpus``; the count unchanged (and blank texts) when the rule keeps it.

    A CPU step whose median efficiency is below 35% gets the p75 of its busy
    cores + 30%. A GPU step never loses CPUs; a starved one (GPU mostly idle
    while its CPUs are busy) gets 1.5x the CPUs its runs had, so asking again
    with that count keeps it.
    """
    full = [r for r in runs if r.state in _FULL_RUN_STATES] or list(runs)
    eff = median(r.cpu_efficiency for r in full)
    if eff is None:
        return cpus, "", ""
    if not gpu_step:
        busy = percentile((r.cores_used for r in full), 75)
        if busy is None or eff >= CPU_REDUCE_BELOW_EFFICIENCY:
            return cpus, "", ""
        fewer = max(1, math.ceil(busy * CPU_REDUCE_HEADROOM - 1e-9))
        if fewer >= cpus:
            return cpus, "", ""
        return fewer, (f"median CPU efficiency {eff:.0%} over {_plural(len(full), 'run')}: "
                       f"p75 {busy:.1f} cores busy (+30%)"), "Fewer CPUs may lengthen the run."
    util = median(r.gpu_util for r in full)
    had = median(r.cpus for r in full)
    if (util is None or had is None or util >= GPU_STARVED_BELOW_UTIL
            or eff <= GPU_STARVED_ABOVE_CPU_EFFICIENCY):
        return cpus, "", ""
    more = math.ceil(had * GPU_STARVED_CPU_FACTOR - 1e-9)
    if more <= cpus:
        return cpus, "", ""
    return more, (f"median GPU utilisation {util:.0f}% while CPU efficiency is {eff:.0%} "
                  f"over {_plural(len(full), 'run')} at {had:g} CPUs (×1.5)"), (
        "GPU waits on the input pipeline (starved): add CPUs.")


def _recommend_memory(runs: Sequence[RunUsage], profile: StepProfile,
                      cpus: int | None) -> tuple[float | None, str]:
    """``(MB before rounding, reason)``; ``(None, "")`` without evidence.

    p90 of the completed / timed-out runs' peaks × 1.2; every out-of-memory
    run of the basis sets a floor of its request × 1.25. Both are per CPU
    (× ``cpus``) when the step's memory scales with its CPUs.
    """
    per_cpu = profile.memory_scales_with_cpus and cpus is not None
    full = [r for r in runs if r.state in _FULL_RUN_STATES and r.peak_mem_mb is not None]
    ooms = [r for r in runs if r.state == OUT_OF_MEMORY and r.req_memory_mb is not None]
    if per_cpu:
        full = [r for r in full if r.cpus]
        ooms = [r for r in ooms if r.cpus]

    def scaled(r: RunUsage, mb: float) -> float:
        return mb / r.cpus * cpus if per_cpu and r.cpus and cpus is not None else mb

    peak = percentile((scaled(r, r.peak_mem_mb) for r in full if r.peak_mem_mb is not None), 90)
    floor = max((scaled(r, r.req_memory_mb) * OOM_FLOOR_FACTOR
                 for r in ooms if r.req_memory_mb is not None), default=None)
    if peak is None and floor is None:
        return None, ""
    parts: list[str] = []
    if peak is not None:
        asked = median(r.req_memory_mb for r in full)
        if per_cpu:
            per = percentile((r.peak_mem_mb / r.cpus for r in full
                              if r.peak_mem_mb is not None and r.cpus), 90) or 0.0
            parts.append(f"p90 peak {per / 1024:.2f} GB/CPU × {cpus} CPUs = {_gb(peak)} "
                         f"over {_plural(len(full), 'run')} (+20%)")
        else:
            of = f" of {_gb(asked)}" if asked else ""
            parts.append(f"p90 peak {_gb(peak)}{of} over {_plural(len(full), 'run')} (+20%)")
    if ooms:
        worst = max(ooms, key=lambda r: r.req_memory_mb or 0.0)
        parts.append(f"{_plural(len(ooms), 'OOM')} at up to {_gb(worst.req_memory_mb or 0.0)}"
                     + (f" ({worst.cpus} CPUs)" if per_cpu else ""))
    mb = max(v for v in (peak * HEADROOM if peak is not None else None, floor) if v is not None)
    return mb, "; ".join(parts)


def _comparable(runs: Sequence[RunUsage], planned_units: float) -> list[RunUsage]:
    """The runs whose work is within :data:`COMPARABLE_WORK_FACTOR` of the plan."""
    lo, hi = planned_units / COMPARABLE_WORK_FACTOR, planned_units * COMPARABLE_WORK_FACTOR
    return [r for r in runs if r.units and lo <= r.units <= hi]


def _recommend_time(runs: Sequence[RunUsage], planned_units: float | None, units_label: str,
                    profile: StepProfile, cpus: int | None, notes: list[str],
                    warnings: list[str]) -> tuple[float | None, str, float | None]:
    """``(seconds, reason, p90 rate or None)``; ``(None, "", None)`` without
    evidence. Appends the units / timeout / cap notes and warnings.

    The per-unit rate is over the runs of comparable size (all sizes when
    none is). On a step whose time scales with its CPUs, a run measured at
    more than the planned ``cpus`` counts as ``elapsed × its CPUs / cpus``.
    """
    timed = [r for r in runs if r.state in _FULL_RUN_STATES and r.elapsed_s]
    if not timed:
        return None, "", None

    def stretch(r: RunUsage) -> float:
        if not profile.time_scales_with_cpus or not r.cpus or not cpus:
            return 1.0
        return max(1.0, r.cpus / cpus)

    def elapsed(r: RunUsage) -> float:
        return (r.elapsed_s or 0.0) * stretch(r)

    with_units = [r for r in timed if r.units]
    per_unit = planned_units is not None and bool(with_units)
    rate: float | None = None
    if per_unit and planned_units is not None:
        used = _comparable(with_units, planned_units)
        if not used:
            used = with_units
            notes.append(f"No past run within {COMPARABLE_WORK_FACTOR:g}× of the planned work: "
                         "the time rate mixes runs of every size.")

        def needed(r: RunUsage) -> float:
            """What ``r`` would take for the planned work (at the planned CPUs)."""
            return elapsed(r) / (r.units or 1.0) * planned_units

        rate = percentile((elapsed(r) / (r.units or 1.0) for r in used), 90) or 0.0
        estimate = rate * planned_units
        unit = units_label[:-1] if units_label.endswith("s") else units_label
        of = " of comparable size" if len(used) < len(with_units) else ""
        reason = (f"p90 {rate:.3g} s/{unit} × {_clean_units(planned_units):,} {units_label} = "
                  f"{_human_duration(estimate)} over {_plural(len(used), 'run')}{of}")
    else:
        if profile.has_units:
            notes.append(
                "Units unknown for this plan (cache-first resume / continue to target): "
                "time from whole-run elapsed." if planned_units is None else
                "No past run with a known amount of work: time from whole-run elapsed.")
        used = timed
        needed = elapsed
        estimate = percentile((elapsed(r) for r in used), 90) or 0.0
        reason = f"p90 elapsed {_human_duration(estimate)} over {_plural(len(used), 'run')}"
    if any(stretch(r) > 1.0 for r in used):
        reason += f", runs at more CPUs stretched to {cpus}"
    seconds = max(estimate * HEADROOM, estimate + MIN_TIME_MARGIN_S)
    # A timed-out run is a lower bound: if it would time out again under the
    # (rounded) recommendation, give it 1.5x what it already ran.
    timeouts = [needed(r) for r in used if r.state == TIMEOUT]
    reason += " (+20%, at least +5 min)"
    if timeouts:
        reason += f"; {_plural(len(timeouts), 'timeout')}"
        longest = max(timeouts)
        if longest >= round_time_s(seconds):
            seconds = longest * TIMEOUT_BUMP_FACTOR
            warnings.append(
                f"A past run timed out after the equivalent of {_human_duration(longest)}: "
                "bumped to 1.5× that.")
    if seconds > TIME_CAP_S:
        warnings.append(f"Capped at {format_time(TIME_CAP_S)} (the partition limit); "
                        "the run may need more.")
    return round_time_s(seconds), reason, rate


# ---------------------------------------------------------------------------
# Summaries (dashboard)
# ---------------------------------------------------------------------------

def summarize_step(step_id: str, runs: Iterable[RunUsage | Mapping[str, Any]], *,
                   label: str | None = None, needs_gpu: bool | None = None) -> dict[str, Any]:
    """The API's StepSummary over every row of ``step_id``.

    State counts and ``success_rate`` (completed / finished) cover every row;
    medians and the p90 peak are over the runs that are evidence (completed,
    out-of-memory, timed out); the allocated-vs-used hours are over every
    finished run with an elapsed time (failed and cancelled runs wasted their
    allocation too). An array job counts ONE task's allocation.
    """
    profile = profile_for(step_id)
    step_runs = [r for r in _as_runs(runs) if r.step_id == step_id]
    states: dict[str, int] = dict.fromkeys(STATE_KEYS, 0)
    for r in step_runs:
        if r.bucket:
            states[r.bucket] += 1
    finished = sum(states[k] for k in _FINISHED_BUCKETS)
    counted = [r for r in step_runs if r.counted]
    spent = [r for r in step_runs if r.bucket in _FINISHED_BUCKETS and (r.elapsed_s or 0) > 0]
    gpu_step = bool(needs_gpu) if needs_gpu is not None else (
        profile.gpu or any((r.gpus or 0) > 0 for r in step_runs))

    def hours(pairs: Iterable[tuple[float | None, float | None]], scale: float = 1.0) -> float:
        total = sum(a * (b or 0.0) for a, b in pairs if a is not None)
        return round(total * scale / 3600.0, 3)

    return {
        "step_id": step_id,
        "label": label or step_id,
        "needs_gpu": gpu_step,
        "runs": len(step_runs),
        "states": states,
        "success_rate": _round(states["completed"] / finished, 4) if finished else None,
        "last_submitted_at": max((r.submitted_at for r in step_runs if r.submitted_at),
                                 default=None),
        "cpu_efficiency": _round(median(r.cpu_efficiency for r in counted), 4),
        "gpu_util": _round(median(r.gpu_util for r in counted), 1),
        "mem_ratio": _round(median(r.mem_ratio for r in counted), 4),
        "time_ratio": _round(median(r.time_ratio for r in counted), 4),
        "peak_mem_p90_mb": _round(percentile((r.peak_mem_mb for r in counted), 90), 1),
        "cpu_hours_alloc": hours((r.cpus, r.elapsed_s) for r in spent),
        "cpu_hours_used": hours((r.cores_used, r.elapsed_s) for r in spent),
        "gpu_hours_alloc": hours((r.gpus, r.elapsed_s) for r in spent),
        "gpu_hours_used": hours(
            ((r.gpus or 0) * r.gpu_util / 100.0 if r.gpu_util is not None else None, r.elapsed_s)
            for r in spent),
        "mem_gb_hours_alloc": hours(((r.req_memory_mb, r.elapsed_s) for r in spent), 1 / 1024),
        "mem_gb_hours_used": hours(((r.peak_mem_mb, r.elapsed_s) for r in spent), 1 / 1024),
    }


def order_summaries(summaries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """:data:`PINNED_STEPS` first, then the newest ``last_submitted_at`` first."""
    items = list(summaries)
    pinned = [s for p in PINNED_STEPS for s in items if s["step_id"] == p]
    rest = sorted((s for s in items if s["step_id"] not in PINNED_STEPS),
                  key=lambda s: (s.get("last_submitted_at") or "", s["step_id"]), reverse=True)
    return pinned + rest


__all__ = [
    "COUNTED_STATES",
    "PROFILES",
    "RESOURCE_FIELDS",
    "Basis",
    "RunUsage",
    "StepProfile",
    "format_memory",
    "format_time",
    "latest_counted",
    "median",
    "newest_first",
    "normalize_row",
    "normalize_rows",
    "normalize_state",
    "order_summaries",
    "parse_memory_mb",
    "parse_time_s",
    "percentile",
    "profile_for",
    "recommend",
    "round_memory_gb",
    "round_time_s",
    "select_basis",
    "summarize_step",
]
