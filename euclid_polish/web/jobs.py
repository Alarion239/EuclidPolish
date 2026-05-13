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

import contextlib
import io
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


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
    finished:  Optional[float]     = None
    result:    Any                 = None
    error:     Optional[str]       = None
    log_buf:   io.StringIO         = field(default_factory=io.StringIO)
    # Progress fields — set by jobs via ``_LogCapture.tick(...)``. Optional;
    # ``progress_total = 0`` means "indeterminate".
    progress_current: int = 0
    progress_total:   int = 0
    progress_label:   str = ""

    def append_log(self, msg: str) -> None:
        self.log_buf.write(msg)

    def set_progress(self, current: int, total: int, label: str = "") -> None:
        self.progress_current = int(current)
        self.progress_total   = int(total)
        if label:
            self.progress_label = label

    @property
    def log(self) -> str:
        return self.log_buf.getvalue()

    @property
    def progress_pct(self) -> float:
        if self.progress_total <= 0:
            return 0.0
        return 100.0 * self.progress_current / self.progress_total

    def to_dict(self) -> Dict[str, Any]:
        # Keep the log payload small — the UI only renders the last ~4 KB.
        log = self.log
        log_tail = log[-4000:] if len(log) > 4000 else log
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
                "pct":     round(self.progress_pct, 1),
                "label":   self.progress_label,
            },
        }


# ---------------------------------------------------------------------------
# Tracker (process-global)
# ---------------------------------------------------------------------------

class JobRegistry:
    """Thread-safe job dict + spawn helper."""

    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def list(self) -> List[Dict[str, Any]]:
        """Newest first."""
        with self._lock:
            return sorted(
                (j.to_dict() for j in self._jobs.values()),
                key=lambda d: d["started"], reverse=True,
            )

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def spawn(self, label: str, target: Callable[["_LogCapture"], Any]) -> str:
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
                    job.result = target(cap)
                job.finished = time.time()
                job.status   = "done"        # set status LAST so pollers
                                              # always see populated fields
            except Exception as e:
                job.error    = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                job.append_log(f"\nERROR: {job.error}\n")
                job.finished = time.time()
                job.status   = "failed"      # status flip after error+finished

        threading.Thread(target=_runner, daemon=True, name=f"job-{job.job_id}").start()
        return job.job_id


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
        self._stack: Optional[contextlib.ExitStack] = None

    def __enter__(self) -> "_LogCapture":
        self._stack = contextlib.ExitStack()
        self._stack.enter_context(contextlib.redirect_stdout(self.job.log_buf))
        self._stack.enter_context(contextlib.redirect_stderr(self.job.log_buf))
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
        import tqdm as _tqdm_module
        from tqdm import auto as _tqdm_auto

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
