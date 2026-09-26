"""
Simple in-memory background-job tracker for the web UI.

A "job" is one long-running pipeline step (generate a clean field,
run the forward model, extract a PSF). The tracker:

  * gives each job a short UUID id (and an optional ``kind`` tag)
  * runs the callable in a background thread
  * captures stdout/stderr into a string buffer that the UI can poll
  * records ``"running"`` / ``"done"`` / ``"failed"`` / ``"cancelled"``
    status + return value (small JSON-safe results are exposed to the UI)
  * supports cooperative cancel: :meth:`Job.cancel` sets a flag and the
    job raises :class:`JobCancelled` at its next ``cap.tick(...)`` (or tqdm
    update inside ``cap.tqdm_hook``)
  * keeps at most ``max_finished`` finished jobs (oldest evicted)

Not durable: jobs are lost when the Flask process exits. That is fine
for an interactive single-user localhost UI; if multi-process durability
is ever needed, swap the dict for a redis-backed queue.

HTTP contract (C2): ``GET /api/jobs[?summary=1]``, ``GET /api/jobs/<id>``,
``POST /api/jobs/<id>/cancel`` — see ``euclid_polish/web/API.md``.
"""

from __future__ import annotations

import builtins
import contextlib
import io
import json
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import tqdm as _tqdm_module
from tqdm import auto as _tqdm_auto

# Results larger than this (as compact JSON) are not echoed to the UI.
MAX_RESULT_BYTES = 64 * 1024
# Finished jobs kept by a registry before the oldest are evicted.
MAX_FINISHED_JOBS = 200


class JobCancelled(BaseException):  # noqa: N818 - public contract name (C2)
    """Raised inside a job's thread at its next ``tick`` after a cancel.

    Derives from :class:`BaseException` (like ``KeyboardInterrupt``) so the
    ``except Exception`` blocks common in job targets cannot swallow it.
    """


def _json_safe(value: Any) -> Any:
    """``value`` when it is strict JSON of at most 64 KB, else ``None``."""
    if value is None:
        return None
    try:
        encoded = json.dumps(value, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError, RecursionError):
        return None
    if len(encoded.encode("utf-8")) > MAX_RESULT_BYTES:
        return None
    return json.loads(encoded)


# ---------------------------------------------------------------------------
# Job record
# ---------------------------------------------------------------------------

@dataclass
class Job:
    """A single background task with progress reporting."""

    job_id:    str
    label:     str
    status:    str                 = "running"   # running | done | failed | cancelled
    started:   float               = field(default_factory=time.time)
    finished:  float | None     = None
    result:    Any                 = None
    error:     str | None       = None
    log_buf:   io.StringIO         = field(default_factory=io.StringIO)
    kind:      str | None       = None
    cancel_requested: bool = False
    # Progress fields — set by jobs via ``_LogCapture.tick(...)``. Optional;
    # ``progress_total = 0`` means "indeterminate".
    progress_current: int = 0
    progress_total:   int = 0
    progress_label:   str = ""
    _progress_stage_started: float | None = field(
        default=None, repr=False)
    _progress_last_updated: float | None = field(default=None, repr=False)
    _progress_rate: float | None = field(default=None, repr=False)
    _result_json: Any = field(default=None, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def append_log(self, msg: str) -> None:
        with self._lock:
            self.log_buf.write(msg)

    def set_progress(self, current: int, total: int, label: str = "") -> None:
        with self._lock:
            now = time.time()
            current = int(current)
            total = int(total)
            stage_changed = (
                self._progress_stage_started is None
                or total != self.progress_total
                or current < self.progress_current
            )
            if stage_changed:
                self._progress_stage_started = now
                self._progress_rate = None
            elif (self._progress_last_updated is not None
                  and current > self.progress_current):
                elapsed = now - self._progress_last_updated
                if elapsed > 0:
                    instantaneous = (
                        (current - self.progress_current) / elapsed)
                    self._progress_rate = (
                        instantaneous
                        if self._progress_rate is None
                        else 0.25 * instantaneous + 0.75 * self._progress_rate
                    )
            self.progress_current = current
            self.progress_total = total
            self._progress_last_updated = now
            if label:
                self.progress_label = label

    @property
    def log(self) -> str:
        with self._lock:
            return self.log_buf.getvalue()

    @property
    def progress_pct(self) -> float:
        with self._lock:
            if self.progress_total <= 0:
                return 0.0
            return 100.0 * self.progress_current / self.progress_total

    def cancel(self) -> bool:
        """Request a cooperative cancel; False when the job already ended."""
        with self._lock:
            if self.status != "running":
                return False
            self.cancel_requested = True
            return True

    def raise_if_cancelled(self) -> None:
        """Raise :class:`JobCancelled` when a cancel has been requested."""
        if self.cancel_requested:
            raise JobCancelled(self.job_id)

    @property
    def cancellable(self) -> bool:
        with self._lock:
            return self.status == "running" and not self.cancel_requested

    def complete(self, result: Any) -> None:
        with self._lock:
            self.result = result
            self._result_json = _json_safe(result)
            self.finished = time.time()
            self.status = "done"

    def fail(self, error: str) -> None:
        with self._lock:
            self.error = error
            self.log_buf.write(f"\nERROR: {error}\n")
            self.finished = time.time()
            self.status = "failed"

    def mark_cancelled(self) -> None:
        with self._lock:
            self.log_buf.write("\nCancelled by user.\n")
            self.finished = time.time()
            self.status = "cancelled"

    def to_dict(self, *, summary: bool = False) -> dict[str, Any]:
        """JSON view (contract C2). ``summary=True`` drops the log text."""
        with self._lock:
            now = time.time()
            # Keep the log payload small — the UI only renders the last ~4 KB.
            log = self.log_buf.getvalue()
            log_tail = log[-4000:] if len(log) > 4000 else log
            if self.progress_total <= 0:
                progress_pct = 0.0
            else:
                progress_pct = 100.0 * self.progress_current / self.progress_total
            stage_elapsed = (
                max(0.0, now - self._progress_stage_started)
                if self._progress_stage_started is not None else None)
            eta_seconds = None
            if (self.status == "running" and self.progress_total > 0
                    and self.progress_current < self.progress_total
                    and self._progress_rate is not None
                    and self._progress_rate > 0):
                eta_seconds = (
                    (self.progress_total - self.progress_current)
                    / self._progress_rate)
            elif self.status == "done" and self.progress_total > 0:
                eta_seconds = 0.0
            return {
                "job_id":   self.job_id,
                "label":    self.label,
                "kind":     self.kind,
                "status":   self.status,
                "started":  self.started,
                "finished": self.finished,
                "duration": (self.finished or time.time()) - self.started,
                "error":    self.error,
                "cancellable": (self.status == "running"
                                and not self.cancel_requested),
                "cancel_requested": self.cancel_requested,
                "result":   self._result_json if self.status == "done" else None,
                "log":      None if summary else log_tail,
                "log_truncated": len(log) > len(log_tail),
                "progress": {
                    "current": self.progress_current,
                    "total":   self.progress_total,
                    "pct":     round(progress_pct, 1),
                    "label":   self.progress_label,
                    "stage_elapsed": stage_elapsed,
                    "rate_per_second": self._progress_rate,
                    "eta_seconds": eta_seconds,
                    "updated_ago_seconds": (
                        max(0.0, now - self._progress_last_updated)
                        if self._progress_last_updated is not None else None),
                },
            }


# ---------------------------------------------------------------------------
# Tracker (process-global)
# ---------------------------------------------------------------------------

class JobRegistry:
    """Thread-safe job dict + spawn helper (keeps ``max_finished`` done jobs)."""

    def __init__(self, max_finished: int = MAX_FINISHED_JOBS) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self.max_finished = int(max_finished)

    def list(self, *, summary: bool = False) -> builtins.list[dict[str, Any]]:
        """Newest first; ``summary=True`` omits every job's log text."""
        with self._lock:
            jobs = builtins.list(self._jobs.values())
        return sorted(
            (j.to_dict(summary=summary) for j in jobs),
            key=lambda d: d["started"], reverse=True,
        )

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool | None:
        """Request a cancel: True if flagged, False if already finished,
        None if unknown."""
        job = self.get(job_id)
        if job is None:
            return None
        return job.cancel()

    def _evict_finished(self) -> None:
        """Drop the oldest finished jobs beyond ``max_finished``."""
        with self._lock:
            finished = sorted(
                (j for j in self._jobs.values() if j.status != "running"),
                key=lambda j: (j.finished or j.started, j.started),
            )
            excess = len(finished) - self.max_finished
            for job in finished[:max(0, excess)]:
                del self._jobs[job.job_id]

    def spawn(
        self,
        label: str,
        target: Callable[[_LogCapture], Any],
        kind: str | None = None,
    ) -> str:
        """Run ``target(log_capture)`` in a daemon thread; return the job id.

        ``target`` receives a small helper that lets it write to the
        job's log buffer (and that monkey-patches print() to redirect
        stdout into the same buffer while it's running). ``kind`` is a
        free-form tag (e.g. ``"fasrc-env-update"``) the UI can group by.
        """
        job = Job(job_id=uuid.uuid4().hex[:8], label=label, kind=kind)
        with self._lock:
            self._jobs[job.job_id] = job

        def _runner() -> None:
            try:
                cap = _LogCapture(job)
                with cap:
                    job.raise_if_cancelled()
                    result = target(cap)
                job.complete(result)
            except JobCancelled:
                job.mark_cancelled()
            except Exception as e:
                error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                job.fail(error)
            finally:
                self._evict_finished()

        threading.Thread(target=_runner, daemon=True, name=f"job-{job.job_id}").start()
        return job.job_id


class _ThreadLocalStream:
    """Route each bound thread to its job and all other writes downstream."""

    def __init__(self, fallback) -> None:
        self._fallback = fallback
        self._local = threading.local()

    @contextlib.contextmanager
    def bind(self, job: Job):
        previous = getattr(self._local, "job", None)
        self._local.job = job
        try:
            yield
        finally:
            if previous is None:
                del self._local.job
            else:
                self._local.job = previous

    def write(self, message: str):
        job = getattr(self._local, "job", None)
        if job is not None:
            job.append_log(message)
            return len(message)
        return self._fallback.write(message)

    def flush(self) -> None:
        if getattr(self._local, "job", None) is None:
            self._fallback.flush()

    def __getattr__(self, name: str):
        return getattr(self._fallback, name)


_STREAM_INSTALL_LOCK = threading.Lock()


def _stream_proxies() -> tuple[_ThreadLocalStream, _ThreadLocalStream]:
    """Install stream routers around the currently active process streams."""
    with _STREAM_INSTALL_LOCK:
        if not isinstance(sys.stdout, _ThreadLocalStream):
            sys.stdout = _ThreadLocalStream(sys.stdout)
        if not isinstance(sys.stderr, _ThreadLocalStream):
            sys.stderr = _ThreadLocalStream(sys.stderr)
        return sys.stdout, sys.stderr


class _LogCapture:
    """Context manager that redirects stdout/stderr into a job's log buffer
    AND lets the job report structured progress.

    Background threads inherit the process stdout, so naive ``print``
    would interleave with the Flask server log. This redirect keeps the
    job's output isolated and pollable via the UI.

    Also exposes:

      * :meth:`tick(current, total, label=None)`  — direct progress update;
        raises :class:`JobCancelled` once a cancel has been requested
      * :meth:`check_cancelled()` — the same check without a progress update
      * :meth:`tqdm_hook()` context — replaces ``tqdm.tqdm`` for the
        duration of a block so any code using ``tqdm`` (the downloader,
        trainer, EPSFBuilder progress bar) drives the job's progress
        bar automatically (and honours cancel on every update).
    """

    def __init__(self, job: Job) -> None:
        self.job = job
        self._stack: contextlib.ExitStack | None = None

    def __enter__(self) -> _LogCapture:
        stdout, stderr = _stream_proxies()
        self._stack = contextlib.ExitStack()
        self._stack.enter_context(stdout.bind(self.job))
        self._stack.enter_context(stderr.bind(self.job))
        return self

    def __exit__(self, *exc):
        assert self._stack is not None
        self._stack.close()
        return False

    def write(self, msg: str) -> None:
        """Direct write, in case the user has a non-stdout logger."""
        self.job.append_log(msg)

    def tick(self, current: int, total: int, label: str = "") -> None:
        """Update the job's progress fields. ``total=0`` means indeterminate.

        Raises :class:`JobCancelled` when the job has been asked to stop."""
        self.job.raise_if_cancelled()
        self.job.set_progress(current, total, label)

    def check_cancelled(self) -> None:
        """Raise :class:`JobCancelled` when the job has been asked to stop."""
        self.job.raise_if_cancelled()

    @contextlib.contextmanager
    def tqdm_hook(self, label: str = ""):
        """Patch ``tqdm.tqdm`` so any code using it inside this block updates
        the job's progress bar.

        Usage::

            with cap.tqdm_hook("downloading cutouts"):
                downloader.download(show_progress=True)
        """

        job = self.job

        class _JobTqdm(_tqdm_module.tqdm):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                desc = self.desc or label or "working"
                job.set_progress(0, self.total or 0, desc)

            def update(self, n=1):
                job.raise_if_cancelled()
                super().update(n)
                desc = self.desc or label or "working"
                job.set_progress(int(self.n), int(self.total or 0), desc)

            def close(self):
                super().close()
                # Force the bar to 100% when the iteration ends naturally.
                if self.total:
                    job.set_progress(int(self.total), int(self.total), self.desc or label)

        original_module = _tqdm_module.tqdm
        original_auto   = _tqdm_auto.tqdm
        _tqdm_module.tqdm = _JobTqdm
        _tqdm_auto.tqdm   = _JobTqdm
        try:
            yield self
        finally:
            _tqdm_module.tqdm = original_module
            _tqdm_auto.tqdm   = original_auto
            # Reset progress when the block ends to avoid stale state.
            job.set_progress(0, 0, "")


# Module-singleton tracker; one per process is plenty.
REGISTRY = JobRegistry()
