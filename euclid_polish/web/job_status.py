"""Structured job-status aggregator.

Pulls the JSONL events stream that :class:`~euclid_polish.observability
.reporter.Reporter` writes on the FASRC side and folds it into a
:class:`JobStatus` value the web UI can render directly.

Design
------
* **One round-trip per fetch.** A single ``cat <events_path>`` over the
  existing ControlMaster SSH session pulls the entire file. Events
  files stay small (a few hundred KB even after a long job), so we
  don't bother with incremental tailing.

* **One pass per fetch.** Folding events into a status is O(N) over
  the events list — last ``stage`` event wins, last ``step`` wins,
  all ``warn`` / ``error`` accumulate. Trivially extensible: add a
  new ``kind`` and a new handler.

* **Stateless.** Each ``fetch`` call returns a fresh :class:`JobStatus`
  derived purely from what's on disk. No long-lived background loop,
  no per-job caches in process memory.
"""

from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from euclid_polish.config import Config

# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Event:
    """One structured log entry (warning or error)."""

    ts:   float
    msg:  str

    def to_dict(self) -> dict[str, Any]:
        return {"ts": float(self.ts), "msg": str(self.msg)}


@dataclass(frozen=True)
class StepProgress:
    """A ``(current, total, label)`` triple as emitted by ``set_step``."""

    current: int
    total:   int
    label:   str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "current": int(self.current),
            "total":   int(self.total),
            "label":   str(self.label),
        }

    @property
    def fraction(self) -> float:
        return self.current / self.total if self.total > 0 else 0.0


@dataclass(frozen=True)
class ParallelProgress:
    """Aggregate progress of a parallel phase (many worker processes).

    ``current``/``total`` is the cumulative count summed across all worker
    units; ``active_workers`` is how many distinct processes emitted a
    progress event within the recent window (i.e. are crunching right now);
    ``n_workers`` is how many were requested.
    """

    current:        int
    total:          int
    active_workers: int
    n_workers:      int

    def to_dict(self) -> dict[str, Any]:
        return {
            "current":        int(self.current),
            "total":          int(self.total),
            "active_workers": int(self.active_workers),
            "n_workers":      int(self.n_workers),
        }


#: A worker is counted "active" if its most recent event is within this many
#: seconds of the newest event in the stream (using event timestamps, not
#: the server clock — worker/server clocks may differ). Generous enough to
#: cover a slow single image (a 512² generate+forward can take several
#: seconds, during which a worker emits nothing).
_PARALLEL_ACTIVE_WINDOW_S = 60.0

#: How many of the most recent samples the "live" gauge averages over —
#: the smoothing the user sees so a single spiky reading doesn't make the
#: number jump. 6 samples ≈ the last minute at the sampler's 10 s cadence.
_RESOURCE_SMOOTH_WINDOW = 6
#: Cap on the per-metric series handed to the UI sparkline; the newest
#: points are kept. Bounds the status payload on a multi-hour job.
_RESOURCE_SERIES_CAP = 60

#: Cap on the training-metrics history handed to the UI. One sample per
#: evaluate (every ``evaluate_every`` steps), so 500 covers a very long
#: run; the newest are kept. The full curve still lives in the training-log
#: artifact — this is just the live readout.
_METRICS_CAP = 500


@dataclass(frozen=True)
class ResourceUsage:
    """CPU + GPU utilisation folded from the ``resource`` sample stream.

    ``*_percent`` are the *smoothed live* values (mean of the last
    :data:`_RESOURCE_SMOOTH_WINDOW` samples) the card shows ticking; the
    ``*_mean`` / ``*_peak`` aggregates span the whole run and feed the
    post-mortem. ``cpu_series`` / ``gpu_series`` are the recent raw
    samples for a sparkline. Any metric absent from the stream (e.g. GPU
    on a CPU-only job) stays ``None`` / empty.
    """

    cpu_percent:     float | None   = None
    gpu_percent:     float | None   = None
    gpu_mem_percent: float | None   = None
    cpu_mean:        float | None   = None
    cpu_peak:        float | None   = None
    gpu_mean:        float | None   = None
    gpu_peak:        float | None   = None
    gpu_mem_peak:    float | None   = None
    n_samples:       int               = 0
    cpu_series:      tuple[float, ...] = ()
    gpu_series:      tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_percent":     self.cpu_percent,
            "gpu_percent":     self.gpu_percent,
            "gpu_mem_percent": self.gpu_mem_percent,
            "cpu_mean":        self.cpu_mean,
            "cpu_peak":        self.cpu_peak,
            "gpu_mean":        self.gpu_mean,
            "gpu_peak":        self.gpu_peak,
            "gpu_mem_peak":    self.gpu_mem_peak,
            "n_samples":       int(self.n_samples),
            "cpu_series":      list(self.cpu_series),
            "gpu_series":      list(self.gpu_series),
        }


@dataclass(frozen=True)
class StageEvent:
    """One stage entry: when the script entered it + what it called it."""

    ts:   float
    name: str

    def to_dict(self) -> dict[str, Any]:
        return {"ts": float(self.ts), "name": str(self.name)}


@dataclass(frozen=True)
class JobStatus:
    """Structured status of one job, derived from its events stream."""

    stage:        str | None            = None
    #: Full ordered history of stage transitions; the UI renders this
    #: as a checklist with timestamps so the user can see what stages
    #: ran before the current one. ``stage`` is the last entry's name
    #: (or None when no stage event has been seen yet).
    stages:       tuple[StageEvent, ...]   = ()
    step:         StepProgress | None   = None
    #: Steps per wall-clock second within the *current* stage, computed
    #: from the (current, ts) pairs the script has emitted since the
    #: latest ``set_stage``. ``None`` when fewer than two step events
    #: have arrived in this stage (rate is undefined).
    step_rate_per_s: float | None       = None
    #: Estimated seconds until the current stage's ``step.total`` is
    #: reached, at the current rate. ``None`` whenever
    #: ``step_rate_per_s`` is ``None`` or ``total`` is missing/zero.
    step_eta_s:    float | None         = None
    #: Present only during a parallel phase: cumulative count + how many
    #: worker processes are active right now. ``step`` mirrors the same
    #: cumulative (current/total) so the progress bar + rate/ETA work
    #: unchanged; ``parallel`` adds the live worker count.
    parallel:     ParallelProgress | None = None
    #: Smoothed live CPU/GPU utilisation + run aggregates, folded from the
    #: ``resource`` sample stream. ``None`` when the job emitted no samples
    #: (older scripts, or one that didn't start the ResourceSampler).
    resources:    ResourceUsage | None  = None
    #: Latest training-metrics sample folded from the ``metric`` event
    #: stream (loss, PSNR per lane, gradient norms, …) — the live training
    #: progress the WebUI used to scrape from the ``.out`` log. ``None``
    #: until the first evaluate emits a metric event.
    latest_metrics: dict[str, Any] | None = None
    #: Full (capped) history of metric samples in arrival order — one per
    #: evaluate. Drives the live loss/PSNR readout and the in-card
    #: validation list without reading the training-log file.
    metrics:        tuple[dict[str, Any], ...] = ()
    #: The latest metric sample that wrote a checkpoint (``saved`` truthy),
    #: as a short ``"step N"`` marker — the events-native replacement for
    #: the ``Checkpoint saved`` log line that triggers the ckpt mirror.
    last_checkpoint: str | None          = None
    warnings:     tuple[Event, ...]        = ()
    errors:       tuple[Event, ...]        = ()
    #: When the events file was last fetched (server clock). Lets the
    #: UI dim cards whose backing file hasn't been touched in a while.
    last_fetched: float                    = field(default_factory=time.time)
    #: True iff the events file exists and produced at least one valid
    #: event. False for jobs that haven't started writing yet, or for
    #: scripts that don't use :class:`Reporter`.
    has_events:   bool                     = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage":           self.stage,
            "stages":          [s.to_dict() for s in self.stages],
            "step":            self.step.to_dict() if self.step else None,
            "step_rate_per_s": self.step_rate_per_s,
            "step_eta_s":      self.step_eta_s,
            "parallel":        self.parallel.to_dict() if self.parallel else None,
            "resources":       self.resources.to_dict() if self.resources else None,
            "latest_metrics":  self.latest_metrics,
            "metrics":         list(self.metrics),
            "last_checkpoint": self.last_checkpoint,
            "warnings":        [w.to_dict() for w in self.warnings],
            "errors":          [e.to_dict() for e in self.errors],
            "last_fetched":    float(self.last_fetched),
            "has_events":      bool(self.has_events),
        }


# ---------------------------------------------------------------------------
# Folding logic — events → status
# ---------------------------------------------------------------------------

def fold_events(text: str) -> JobStatus:
    """Parse the events file's text and fold it into a :class:`JobStatus`.

    Lines that fail to parse as JSON are skipped silently — a job
    crashing mid-write can leave a partial last line. Lines with an
    unknown ``kind`` are also skipped; we'd rather lose forward
    compatibility than 500 the status endpoint.
    """
    stage:     str | None          = None
    stages:    list[StageEvent]        = []
    step:      StepProgress | None = None
    warnings:  list[Event]             = []
    errors:    list[Event]             = []
    saw_any    = False
    #: Per-stage step history: list of (ts, current, total) tuples for
    #: every step event since the latest ``stage`` event. Cleared on
    #: every stage transition so rate/ETA only ever average inside the
    #: current stage — a fast download stage's rate doesn't pollute the
    #: ETA of a slow training stage that follows.
    stage_step_history: list[tuple[float, int, int]] = []

    # Parallel-phase aggregation. ``by_worker`` keeps each worker unit's
    # latest (ts, current, total, pid); ``parallel_history`` records the
    # cumulative-over-time so the bar's rate/ETA work on the aggregate.
    parallel_total:   int = 0
    parallel_workers: int = 0
    by_worker: dict[str, tuple[float, int, int, Any]] = {}
    parallel_history: list[tuple[float, int, int]] = []

    # Resource-utilisation samples (CPU/GPU). Kept as separate per-metric
    # series because a CPU-only job emits ``cpu`` without ``gpu``.
    res_cpu:     list[float] = []
    res_gpu:     list[float] = []
    res_gpu_mem: list[float] = []

    # Training-metrics samples (one per evaluate) folded from the ``metric``
    # event stream. ``last_checkpoint`` is the latest sample that wrote a
    # checkpoint — the events-native replacement for the ".out" log line.
    metrics:         list[dict[str, Any]] = []
    last_checkpoint: str | None        = None

    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            ev = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict):
            continue
        kind  = ev.get("kind")
        value = ev.get("value")
        ts    = ev.get("ts")
        try:
            ts_f = float(ts)
        except (TypeError, ValueError):
            ts_f = 0.0
        saw_any = True

        if kind == "stage" and isinstance(value, str):
            stage = value
            stages.append(StageEvent(ts=ts_f, name=value))
            # Reset per-stage step history — a new stage means we start
            # measuring rate from its first step event.
            stage_step_history = []
        elif kind == "step" and isinstance(value, dict):
            try:
                cur = int(value.get("current", 0))
                tot = int(value.get("total", 0))
                step = StepProgress(
                    current=cur,
                    total=tot,
                    label=str(value.get("label", "")),
                )
            except (TypeError, ValueError):
                continue
            stage_step_history.append((ts_f, cur, tot))
        elif kind == "parallel" and isinstance(value, dict):
            try:
                parallel_total   = int(value.get("total", 0))
                parallel_workers = int(value.get("workers", 0))
            except (TypeError, ValueError):
                pass
            # A new parallel phase (e.g. train → validate) starts fresh:
            # reset the per-worker aggregation so its bar tracks only this
            # phase, the same way a ``stage`` event resets the step history.
            by_worker = {}
            parallel_history = []
        elif kind == "worker" and isinstance(value, dict):
            try:
                wid = str(value.get("worker_id", ""))
                cur = int(value.get("current", 0))
                tot = int(value.get("total", 0))
            except (TypeError, ValueError):
                continue
            by_worker[wid] = (ts_f, cur, tot, value.get("pid"))
            # Cumulative across every worker unit seen so far.
            cumulative = sum(v[1] for v in by_worker.values())
            grand = parallel_total or sum(v[2] for v in by_worker.values())
            parallel_history.append((ts_f, cumulative, grand))
        elif kind == "resource" and isinstance(value, dict):
            for key, bucket in (("cpu", res_cpu), ("gpu", res_gpu),
                                ("gpu_mem", res_gpu_mem)):
                if key in value:
                    with contextlib.suppress(TypeError, ValueError):
                        bucket.append(float(value[key]))
        elif kind == "metric" and isinstance(value, dict):
            # One evaluate's metrics. Stamp the event ts so a consumer can
            # plot against wall time even if the producer didn't include it.
            sample = dict(value)
            sample.setdefault("wall_time", ts_f)
            metrics.append(sample)
            # ``saved`` truthy ⇒ this eval wrote a checkpoint; remember it as
            # a compact marker the mirror trigger keys on (cheaper + more
            # reliable than grepping ".out" for "Checkpoint saved").
            if value.get("saved"):
                step_no = value.get("step")
                last_checkpoint = (f"step {int(step_no)}"
                                   if step_no is not None else "saved")
        elif kind == "warn" and isinstance(value, str):
            warnings.append(Event(ts=ts_f, msg=value))
        elif kind == "error" and isinstance(value, str):
            errors.append(Event(ts=ts_f, msg=value))
        # Unknown kinds are silently dropped — forward compatibility.

    # Parallel phase: fold the per-worker stream into one cumulative bar
    # (so the existing progress/rate/ETA path works unchanged) plus a live
    # count of how many worker processes are crunching right now.
    parallel: ParallelProgress | None = None
    if parallel_history:
        _, cum, grand = parallel_history[-1]
        step = StepProgress(current=cum, total=grand, label="")
        newest_ts = max(v[0] for v in by_worker.values())
        active_pids = {
            v[3] for v in by_worker.values()
            if v[3] is not None and v[1] < v[2]
            and (newest_ts - v[0]) <= _PARALLEL_ACTIVE_WINDOW_S
        }
        parallel = ParallelProgress(
            current=cum, total=grand,
            active_workers=len(active_pids),
            n_workers=parallel_workers,
        )

    # Rate + ETA from steps within the *current* stage, via an EMA of
    # per-step duration so recent intervals dominate. A plain
    # total/total average lets a slow first interval (graph compilation,
    # cold disk cache) pin the ETA pessimistic for the whole run; the
    # EMA decays it away. We average *seconds per step* (not steps/s)
    # so events that report uneven step strides — one tile vs 50 train
    # steps — fold in on a common per-step basis. Intervals with
    # non-positive Δt or Δsteps (clock skew, a rewound counter, a resume
    # that re-emits an earlier step) are skipped, not folded.
    # During a parallel phase the cumulative-over-time history drives the
    # rate/ETA; otherwise the per-stage serial step history does.
    rate_history = parallel_history if parallel_history else stage_step_history
    step_rate_per_s: float | None = None
    step_eta_s:      float | None = None
    ema_spp: float | None = None        # EMA of seconds-per-step
    for (t0, c0, _), (t1, c1, _tot1) in zip(
        rate_history, rate_history[1:], strict=False
    ):
        d_t = t1 - t0
        d_n = c1 - c0
        if d_t <= 0 or d_n <= 0:
            continue
        spp = d_t / d_n
        ema_spp = (spp if ema_spp is None
                   else (1.0 - Config.WebFetch.STEP_RATE_EMA_ALPHA) * ema_spp
                        + Config.WebFetch.STEP_RATE_EMA_ALPHA * spp)
    if ema_spp is not None and ema_spp > 0:
        step_rate_per_s = 1.0 / ema_spp
        _, last_cur, last_tot = rate_history[-1]
        if last_tot > 0:
            remaining = max(0, last_tot - last_cur)
            step_eta_s = remaining * ema_spp

    # Resource utilisation: smoothed live value (last-N mean) + run
    # aggregates (mean/peak) + a capped recent series for the sparkline.
    def _smooth(xs: list[float]) -> float | None:
        if not xs:
            return None
        tail = xs[-_RESOURCE_SMOOTH_WINDOW:]
        return sum(tail) / len(tail)

    def _mean(xs: list[float]) -> float | None:
        return (sum(xs) / len(xs)) if xs else None

    resources: ResourceUsage | None = None
    if res_cpu or res_gpu:
        resources = ResourceUsage(
            cpu_percent=_smooth(res_cpu),
            gpu_percent=_smooth(res_gpu),
            gpu_mem_percent=_smooth(res_gpu_mem),
            cpu_mean=_mean(res_cpu),
            cpu_peak=max(res_cpu) if res_cpu else None,
            gpu_mean=_mean(res_gpu),
            gpu_peak=max(res_gpu) if res_gpu else None,
            gpu_mem_peak=max(res_gpu_mem) if res_gpu_mem else None,
            n_samples=max(len(res_cpu), len(res_gpu)),
            cpu_series=tuple(res_cpu[-_RESOURCE_SERIES_CAP:]),
            gpu_series=tuple(res_gpu[-_RESOURCE_SERIES_CAP:]),
        )

    return JobStatus(
        stage=stage,
        stages=tuple(stages),
        step=step,
        step_rate_per_s=step_rate_per_s,
        step_eta_s=step_eta_s,
        parallel=parallel,
        resources=resources,
        latest_metrics=(metrics[-1] if metrics else None),
        metrics=tuple(metrics[-_METRICS_CAP:]),
        last_checkpoint=last_checkpoint,
        warnings=tuple(warnings),
        errors=tuple(errors),
        last_fetched=time.time(),
        has_events=saw_any,
    )


# ---------------------------------------------------------------------------
# SSH-backed fetcher
# ---------------------------------------------------------------------------

class _SSHRunner(Protocol):
    """Minimal SSH interface :class:`JobStatusFetcher` needs.

    Matches the shape of the SSH connection wrapper used elsewhere in
    :mod:`euclid_polish.web` — declared as a Protocol so tests can pass
    a stub.
    """

    def run(self, cmd: str, *, timeout: float = ...) -> tuple[int, str, str]: ...

    def is_connected(self) -> bool: ...


class JobStatusFetcher:
    """Read a remote ``.events`` file and fold it into :class:`JobStatus`.

    One SSH ``cat`` per fetch — no local caching, no background loop.
    Status responses are short-lived (the UI polls every ~1.5 s and
    rebuilds the card on every response), so a stale cache would only
    create staleness bugs without buying anything.
    """

    def __init__(self, ssh: _SSHRunner | None) -> None:
        self.ssh = ssh

    def fetch(self, events_path: str | None) -> JobStatus:
        """Return the current status for the job at ``events_path``.

        Empty :class:`JobStatus` (``has_events=False``) is returned
        whenever:

        * ``events_path`` is ``None`` (job was submitted by an older
          script that doesn't write a events file),
        * the SSH session is missing or disconnected,
        * the file doesn't exist remotely or is empty.
        """
        if not events_path or self.ssh is None or not self.ssh.is_connected():
            return JobStatus()
        # ``cat`` exits non-zero if the file doesn't exist yet (the job
        # is still queued). Treat that as "no events" rather than an
        # error so the UI can show a "waiting for first event…" state.
        rc, out, _err = self.ssh.run(
            f"cat {events_path} 2>/dev/null || true", timeout=10,
        )
        if rc != 0 or not out:
            return JobStatus()
        return fold_events(out)
