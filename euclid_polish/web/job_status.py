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
class JobStatus:
    """Structured status of one job, derived from its events stream."""

    stage:        Optional[str]            = None
    step:         Optional[StepProgress]   = None
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
            "stage":        self.stage,
            "step":         self.step.to_dict() if self.step else None,
            "warnings":     [w.to_dict() for w in self.warnings],
            "errors":       [e.to_dict() for e in self.errors],
            "last_fetched": float(self.last_fetched),
            "has_events":   bool(self.has_events),
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
    step:      Optional[StepProgress] = None
    warnings:  List[Event]             = []
    errors:    List[Event]             = []
    saw_any    = False

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
        elif kind == "step" and isinstance(value, dict):
            try:
                step = StepProgress(
                    current=int(value.get("current", 0)),
                    total=int(value.get("total", 0)),
                    label=str(value.get("label", "")),
                )
            except (TypeError, ValueError):
                continue
        elif kind == "warn" and isinstance(value, str):
            warnings.append(Event(ts=ts_f, msg=value))
        elif kind == "error" and isinstance(value, str):
            errors.append(Event(ts=ts_f, msg=value))
        # Unknown kinds are silently dropped — forward compatibility.

    return JobStatus(
        stage=stage,
        step=step,
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
