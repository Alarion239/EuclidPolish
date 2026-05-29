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

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

from euclid_polish.config import Config


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Event:
    """One structured log entry (warning or error)."""

    ts:   float
    msg:  str

    def to_dict(self) -> Dict[str, Any]:
        return {"ts": float(self.ts), "msg": str(self.msg)}


@dataclass(frozen=True)
class StepProgress:
    """A ``(current, total, label)`` triple as emitted by ``set_step``."""

    current: int
    total:   int
    label:   str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current": int(self.current),
            "total":   int(self.total),
            "label":   str(self.label),
        }

    @property
    def fraction(self) -> float:
        return self.current / self.total if self.total > 0 else 0.0


@dataclass(frozen=True)
class StageEvent:
    """One stage entry: when the script entered it + what it called it."""

    ts:   float
    name: str

    def to_dict(self) -> Dict[str, Any]:
        return {"ts": float(self.ts), "name": str(self.name)}


@dataclass(frozen=True)
class JobStatus:
    """Structured status of one job, derived from its events stream."""

    stage:        Optional[str]            = None
    #: Full ordered history of stage transitions; the UI renders this
    #: as a checklist with timestamps so the user can see what stages
    #: ran before the current one. ``stage`` is the last entry's name
    #: (or None when no stage event has been seen yet).
    stages:       Tuple[StageEvent, ...]   = ()
    step:         Optional[StepProgress]   = None
    #: Steps per wall-clock second within the *current* stage, computed
    #: from the (current, ts) pairs the script has emitted since the
    #: latest ``set_stage``. ``None`` when fewer than two step events
    #: have arrived in this stage (rate is undefined).
    step_rate_per_s: Optional[float]       = None
    #: Estimated seconds until the current stage's ``step.total`` is
    #: reached, at the current rate. ``None`` whenever
    #: ``step_rate_per_s`` is ``None`` or ``total`` is missing/zero.
    step_eta_s:    Optional[float]         = None
    warnings:     Tuple[Event, ...]        = ()
    errors:       Tuple[Event, ...]        = ()
    #: When the events file was last fetched (server clock). Lets the
    #: UI dim cards whose backing file hasn't been touched in a while.
    last_fetched: float                    = field(default_factory=time.time)
    #: True iff the events file exists and produced at least one valid
    #: event. False for jobs that haven't started writing yet, or for
    #: scripts that don't use :class:`Reporter`.
    has_events:   bool                     = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage":           self.stage,
            "stages":          [s.to_dict() for s in self.stages],
            "step":            self.step.to_dict() if self.step else None,
            "step_rate_per_s": self.step_rate_per_s,
            "step_eta_s":      self.step_eta_s,
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
    stage:     Optional[str]          = None
    stages:    List[StageEvent]        = []
    step:      Optional[StepProgress] = None
    warnings:  List[Event]             = []
    errors:    List[Event]             = []
    saw_any    = False
    #: Per-stage step history: list of (ts, current, total) tuples for
    #: every step event since the latest ``stage`` event. Cleared on
    #: every stage transition so rate/ETA only ever average inside the
    #: current stage — a fast download stage's rate doesn't pollute the
    #: ETA of a slow training stage that follows.
    stage_step_history: List[Tuple[float, int, int]] = []

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
        elif kind == "warn" and isinstance(value, str):
            warnings.append(Event(ts=ts_f, msg=value))
        elif kind == "error" and isinstance(value, str):
            errors.append(Event(ts=ts_f, msg=value))
        # Unknown kinds are silently dropped — forward compatibility.

    # Rate + ETA from steps within the *current* stage, via an EMA of
    # per-step duration so recent intervals dominate. A plain
    # total/total average lets a slow first interval (graph compilation,
    # cold disk cache) pin the ETA pessimistic for the whole run; the
    # EMA decays it away. We average *seconds per step* (not steps/s)
    # so events that report uneven step strides — one tile vs 50 train
    # steps — fold in on a common per-step basis. Intervals with
    # non-positive Δt or Δsteps (clock skew, a rewound counter, a resume
    # that re-emits an earlier step) are skipped, not folded.
    step_rate_per_s: Optional[float] = None
    step_eta_s:      Optional[float] = None
    ema_spp: Optional[float] = None        # EMA of seconds-per-step
    for (t0, c0, _), (t1, c1, tot1) in zip(
        stage_step_history, stage_step_history[1:]
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
        _, last_cur, last_tot = stage_step_history[-1]
        if last_tot > 0:
            remaining = max(0, last_tot - last_cur)
            step_eta_s = remaining * ema_spp

    return JobStatus(
        stage=stage,
        stages=tuple(stages),
        step=step,
        step_rate_per_s=step_rate_per_s,
        step_eta_s=step_eta_s,
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

    def run(self, cmd: str, *, timeout: float = ...) -> Tuple[int, str, str]: ...

    def is_connected(self) -> bool: ...


class JobStatusFetcher:
    """Read a remote ``.events`` file and fold it into :class:`JobStatus`.

    One SSH ``cat`` per fetch — no local caching, no background loop.
    Status responses are short-lived (the UI polls every ~1.5 s and
    rebuilds the card on every response), so a stale cache would only
    create staleness bugs without buying anything.
    """

    def __init__(self, ssh: Optional[_SSHRunner]) -> None:
        self.ssh = ssh

    def fetch(self, events_path: Optional[str]) -> JobStatus:
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
