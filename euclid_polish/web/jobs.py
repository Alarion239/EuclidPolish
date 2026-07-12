"""
Simple in-memory background-job tracker for the web UI.

A "job" is one long-running pipeline step (generate a clean field,
run the forward model, extract a PSF). The tracker:

  * gives each job a short UUID id
  * runs the callable in a background thread
  * captures stdout/stderr into a string buffer that the UI can poll
  * records ``"running"`` / ``"done"`` / ``"failed"`` status + return value

Not durable: jobs are lost when the Flask process exits. That is fine
for an interactive single-user localhost UI; if multi-process durability
is ever needed, swap the dict for a redis-backed queue.
"""

from __future__ import annotations

import builtins
import contextlib
import io
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

# ---------------------------------------------------------------------------
# Job record
# ---------------------------------------------------------------------------

@dataclass
class Job:
    """A single background task with progress reporting."""

    job_id:    str
    label:     str
    status:    str                 = "running"   # running | done | failed
    started:   float               = field(default_factory=time.time)
    finished:  float | None     = None
    result:    Any                 = None
    error:     str | None       = None
    log_buf:   io.StringIO         = field(default_factory=io.StringIO)
    # Progress fields — set by jobs via ``_LogCapture.tick(...)``. Optional;
    # ``progress_total = 0`` means "indeterminate".
    progress_current: int = 0
    progress_total:   int = 0
    progress_label:   str = ""
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def append_log(self, msg: str) -> None:
        with self._lock:
            self.log_buf.write(msg)

    def set_progress(self, current: int, total: int, label: str = "") -> None:
        with self._lock:
            self.progress_current = int(current)
            self.progress_total   = int(total)
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

    def complete(self, result: Any) -> None:
        with self._lock:
            self.result = result
            self.finished = time.time()
            self.status = "done"

    def fail(self, error: str) -> None:
        with self._lock:
            self.error = error
            self.log_buf.write(f"\nERROR: {error}\n")
            self.finished = time.time()
            self.status = "failed"

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            # Keep the log payload small — the UI only renders the last ~4 KB.
            log = self.log_buf.getvalue()
            log_tail = log[-4000:] if len(log) > 4000 else log
            if self.progress_total <= 0:
                progress_pct = 0.0
            else:
                progress_pct = 100.0 * self.progress_current / self.progress_total
            return {
                "job_id":   self.job_id,
                "label":    self.label,
                "status":   self.status,
                "started":  self.started,
                "finished": self.finished,
                "duration": (self.finished or time.time()) - self.started,
                "error":    self.error,
                "log":      log_tail,
                "log_truncated": len(log) > len(log_tail),
                "progress": {
                    "current": self.progress_current,
                    "total":   self.progress_total,
                    "pct":     round(progress_pct, 1),
                    "label":   self.progress_label,
                },
            }


# ---------------------------------------------------------------------------
# Tracker (process-global)
# ---------------------------------------------------------------------------

class JobRegistry:
    """Thread-safe job dict + spawn helper."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def list(self) -> builtins.list[dict[str, Any]]:
        """Newest first."""
        with self._lock:
            return sorted(
                (j.to_dict() for j in self._jobs.values()),
                key=lambda d: d["started"], reverse=True,
            )

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def spawn(self, label: str, target: Callable[[_LogCapture], Any]) -> str:
        """Run ``target(log_capture)`` in a daemon thread; return the job id.

        ``target`` receives a small helper that lets it write to the
        job's log buffer (and that monkey-patches print() to redirect
        stdout into the same buffer while it's running).
        """
        job = Job(job_id=uuid.uuid4().hex[:8], label=label)
        with self._lock:
            self._jobs[job.job_id] = job

        def _runner() -> None:
            try:
                cap = _LogCapture(job)
                with cap:
                    result = target(cap)
                job.complete(result)
            except Exception as e:
                error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                job.fail(error)

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

      * :meth:`tick(current, total, label=None)`  — direct progress update
      * :meth:`tqdm_hook()` context — replaces ``tqdm.tqdm`` for the
        duration of a block so any code using ``tqdm`` (the downloader,
        trainer, EPSFBuilder progress bar) drives the job's progress
        bar automatically.
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
        """Update the job's progress fields. ``total=0`` means indeterminate."""
        self.job.set_progress(current, total, label)

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
