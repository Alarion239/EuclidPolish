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

    def __init__(self, events_path: Optional[str] = None) -> None:
        self.events_path = events_path
        # Guard concurrent threaded writers in the same process.
        # Multi-process writers don't need this; ``O_APPEND`` gives
        # kernel-level line atomicity for writes under PIPE_BUF.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> "Reporter":
        """Construct from ``EUCLID_POLISH_EVENTS_PATH``.

        Returns a no-op reporter when the env var is missing — so the
        same script works under SLURM (events file present) and during
        local development (no env var, no file, no errors).
        """
        return cls(events_path=os.environ.get(ENV_EVENTS_PATH))

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
        an optional short string for the current item. Silent on
        stderr by design — call sites are typically inside a loop.
        """
        self._emit("step", {
            "current": int(current),
            "total":   int(total),
            "label":   str(label),
        }, echo=False)

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
