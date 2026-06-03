"""Structured progress reporting for FASRC jobs.

A :class:`Reporter` emits one JSON object per line into the job's
``.events`` file. The web UI fetches that file and folds it into a
structured status card — see :mod:`euclid_polish.web.job_status`.

Events
------
Each line in the events file is a single JSON object::

    {"ts": 1731234567.89, "kind": "stage", "value": "Downloading HLSP tiles"}
    {"ts": 1731234568.12, "kind": "step",  "value": {"current": 3, "total": 25, "label": "tile 3"}}
    {"ts": 1731234568.15, "kind": "warn",  "value": "tile 12 checksum mismatch"}
    {"ts": 1731234568.20, "kind": "error", "value": "tile 14 missing from MAST"}

Adding a new event kind (``"metric"``, ``"checkpoint"``, ``"image"``,
…) is a single-line producer change plus a new consumer handler in
:mod:`euclid_polish.web.job_status` — no new files on disk.

Atomicity
---------
The writer opens with ``"a"`` mode (``O_APPEND``) and emits one
``json.dumps(...) + "\n"`` per event. POSIX guarantees that writes
smaller than ``PIPE_BUF`` (4 KB on Linux) appended to a single file
are atomic at the kernel — lines from concurrent writers never
interleave. Our event lines are ~100 bytes, so multi-process scripts
(``ProcessPoolExecutor`` children, etc.) can all share a single events
file without explicit locking. A per-instance ``threading.Lock``
guards the Python-side buffered write so threads in the same process
also don't interleave.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Any, Optional


#: Name of the env var the sbatch template sets to the per-job events
#: file path. Scripts read it via :meth:`Reporter.from_env`; tests can
#: instantiate :class:`Reporter` directly with an explicit path.
ENV_EVENTS_PATH = "EUCLID_POLISH_EVENTS_PATH"


class Reporter:
    """Append-only JSONL event stream for a single FASRC job.

    Scripts call :meth:`set_stage`, :meth:`set_step`, :meth:`warn`,
    and :meth:`error` to emit structured progress. When ``events_path``
    is ``None`` (e.g. running interactively outside SLURM) every emit
    becomes a no-op — the same script then runs in both contexts
    without conditional code paths.

    Calls also echo a one-liner to ``stderr`` so the raw ``.err`` log
    surfaces the same information for users who toggle the "show raw
    log" fallback. ``step`` events are deliberately silent on stderr —
    high-frequency progress would flood the log; scripts that want a
    raw progress bar should keep their existing ``tqdm`` alongside.
    """

    #: Minimum wall-clock gap (seconds) between two ``set_step`` stderr
    #: echoes. Every call still writes to the JSONL events file; the
    #: rate-limit is only on the human-readable echo so a 200k-step
    #: training loop doesn't flood ``.err``. 30 s gives roughly one
    #: line per minute on a typical job and is short enough to spot a
    #: stalled loop within a coffee.
    STEP_ECHO_INTERVAL_S: float = 30.0

    def __init__(self, events_path: Optional[str] = None) -> None:
        self.events_path = events_path
        # Guard concurrent threaded writers in the same process.
        # Multi-process writers don't need this; ``O_APPEND`` gives
        # kernel-level line atomicity for writes under PIPE_BUF.
        self._lock = threading.Lock()
        # Wall-clock of the last echoed step line; the first call always
        # echoes so a quick "did the loop start?" check works without a
        # 30 s pause.
        self._last_step_echo_ts: float = 0.0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> "Reporter":
        """Construct from ``EUCLID_POLISH_EVENTS_PATH``.

        Returns a no-op reporter when the env var is missing — so the
        same script works under SLURM (events file present) and during
        local development (no env var, no file, no errors).

        When the events file IS present, a Reporter is driving the WebUI
        progress bar, so any ``tqdm`` bar would just flood the job's
        ``.err`` log with redundant ASCII frames. Silence every tqdm bar
        in this process by exporting ``TQDM_DISABLE`` (tqdm reads it at
        each bar's creation). ``setdefault`` so an explicit override from
        the environment still wins. ``tqdm.write`` status lines (parsed
        from ``.out``) are unaffected — only the live bar is suppressed.
        """
        events_path = os.environ.get(ENV_EVENTS_PATH)
        if events_path:
            os.environ.setdefault("TQDM_DISABLE", "1")
        return cls(events_path=events_path)

    # ------------------------------------------------------------------
    # Public emit API
    # ------------------------------------------------------------------

    def set_stage(self, name: str) -> None:
        """Mark the current stage (e.g. ``"Downloading HLSP tiles"``).

        Stages are coarse-grained — one job typically has a handful of
        them. The UI renders the most recent stage as a chip above the
        progress bar.
        """
        self._emit("stage", str(name), echo=True)

    def set_step(self, current: int, total: int, label: str = "") -> None:
        """Record fine-grained progress within the current stage.

        ``current`` / ``total`` feed the UI progress bar; ``label`` is
        an optional short string for the current item. Every call
        appends to the JSONL events stream so the UI's poll sees the
        latest number; stderr echo is rate-limited to one line per
        :attr:`STEP_ECHO_INTERVAL_S` so a tight loop doesn't flood
        ``.err``. The first call inside a stage always echoes so a
        quick "did the loop actually start?" check works immediately.
        """
        cur_i = int(current)
        tot_i = int(total)
        lbl_s = str(label)
        self._emit("step", {
            "current": cur_i,
            "total":   tot_i,
            "label":   lbl_s,
        }, echo=False)

        # Rate-limited human-readable echo to ``.err``. The first echo
        # fires unconditionally (``_last_step_echo_ts`` starts at 0); the
        # next one waits ``STEP_ECHO_INTERVAL_S`` seconds.
        now = time.time()
        if (now - self._last_step_echo_ts) >= self.STEP_ECHO_INTERVAL_S:
            pct = (100.0 * cur_i / tot_i) if tot_i > 0 else 0.0
            tag = f" {lbl_s}" if lbl_s else ""
            print(
                f"STEP: {cur_i:,}/{tot_i:,} ({pct:.1f}%){tag}",
                file=sys.stderr, flush=True,
            )
            self._last_step_echo_ts = now

    def set_parallel(self, total: int, workers: int, label: str = "") -> None:
        """Announce a parallel phase: ``total`` items split across ``workers``
        worker processes.

        Emitted once by the *parent* before fanning out. The consumer uses
        it to know the job-level total (for the cumulative bar) and how many
        workers were requested. Workers then report their own progress via
        :meth:`set_worker_step`.
        """
        self._emit("parallel", {
            "total":   int(total),
            "workers": int(workers),
            "label":   str(label),
        }, echo=True)

    def set_worker_step(
        self, worker_id: Any, current: int, total: int, label: str = "",
    ) -> None:
        """Report one worker's progress within a parallel phase.

        Each worker process calls this with a STABLE ``worker_id`` for the
        unit of work it owns (e.g. a shard id) plus that unit's
        ``current``/``total``. The os PID is attached automatically so the
        consumer can count how many distinct processes are active. The
        consumer sums ``current`` across worker_ids for the cumulative
        count. ``echo=False`` — high-frequency, would flood the raw log;
        the parent's stage line is enough there.
        """
        self._emit("worker", {
            "worker_id": str(worker_id),
            "pid":       os.getpid(),
            "current":   int(current),
            "total":     int(total),
            "label":     str(label),
        }, echo=False)

    def sample_resources(self, sample: dict) -> None:
        """Emit one resource-utilisation sample (CPU%% / GPU%% / GPU mem%%).

        ``sample`` is a flat dict with any of ``cpu``, ``gpu``,
        ``gpu_mem``, ``gpu_mem_used_mb``, ``gpu_mem_total_mb`` (a
        CPU-only node emits just ``{"cpu": ...}``). High-frequency and
        machine-only, so ``echo=False`` — it never touches ``.err``. The
        web consumer folds a smoothed live gauge; the post-mortem
        summarises mean/peak into the job log. Empty samples are dropped.
        Produced by
        :class:`euclid_polish.observability.resource_sampler.ResourceSampler`.
        """
        if not sample:
            return
        self._emit("resource", dict(sample), echo=False)

    def warn(self, msg: str) -> None:
        """Append a warning to the structured stream and to stderr."""
        self._emit("warn", str(msg), echo=True, stderr_prefix="WARN")

    def error(self, msg: str) -> None:
        """Append an error to the structured stream and to stderr."""
        self._emit("error", str(msg), echo=True, stderr_prefix="ERROR")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _emit(
        self,
        kind: str,
        value: Any,
        *,
        echo: bool,
        stderr_prefix: Optional[str] = None,
    ) -> None:
        if self.events_path:
            line = json.dumps(
                {"ts": time.time(), "kind": kind, "value": value},
                separators=(",", ":"),
                ensure_ascii=False,
            ) + "\n"
            with self._lock:
                with open(self.events_path, "a", encoding="utf-8") as f:
                    f.write(line)

        if echo:
            prefix = stderr_prefix or kind.upper()
            try:
                rendered = value if isinstance(value, str) else json.dumps(
                    value, ensure_ascii=False,
                )
            except (TypeError, ValueError):
                rendered = repr(value)
            print(f"{prefix}: {rendered}", file=sys.stderr, flush=True)
