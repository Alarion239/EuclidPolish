"""Per-stage wall-clock timer that appends one CSV row per stage.

The pipeline runs in roughly four phases:

  * ``init``     — Python import + dataset loaders ready (no parameter
                   scaling: same on every run).
  * ``generate`` — clean-field sky simulation (scales with
                   ``n_train`` + ``n_valid`` + ``image_size²``).
  * ``convolve`` — multi-band PSF + noise forward model (same scaling).
  * ``train``    — WDSR optimisation (scales with ``steps`` × ``batch_size``).

Recording each one separately lets the ETA model fit
``runtime ~ f(params)`` per stage rather than collapsing every flavour
of work into one ``sec/step`` number. Each row also carries the params
that were active for the run so an offline fit can de-bias by stage.

Schema (one row per completed stage):

    jobid,stage,started_at,ended_at,duration_seconds,
    params_dependent,n_train,n_valid,image_size,batch_size,steps

``params_dependent`` is 1 for stages whose runtime scales with the
form parameters and 0 for fixed overhead — the ETA model can drop
parameter-dependent samples whenever the form values don't match.
"""

from __future__ import annotations

import csv
import os
import time
from contextlib import contextmanager
from typing import Any

_HEADER = [
    "jobid", "stage", "started_at", "ended_at", "duration_seconds",
    "params_dependent",
    "n_train", "n_valid", "image_size", "batch_size", "steps",
]


class StageTimer:
    """Append one CSV row per stage. Threadsafe within a single process
    only — concurrent jobs should write to separate CSVs (the default
    pattern: one file per ``SLURM_JOB_ID``)."""

    def __init__(self, csv_path: str, *,
                 jobid: str,
                 params: dict[str, Any] | None = None) -> None:
        self.csv_path = csv_path
        self.jobid    = jobid
        self.params   = dict(params or {})
        parent = os.path.dirname(csv_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as fh:
                csv.writer(fh).writerow(_HEADER)

    # ------------------------------ public API ----------------------------

    @contextmanager
    def stage(self, name: str, *, params_dependent: bool):
        """Context manager: time the block, append one row on exit.

        The row is written even if the block raises, so a crashed
        stage still shows up in the timings table.
        """
        t0 = time.time()
        try:
            yield
        finally:
            self._append(name, params_dependent, t0, time.time())

    def mark(self, name: str, *, params_dependent: bool,
             started_at: float, ended_at: float | None = None) -> None:
        """Append a row for a stage whose boundaries were captured
        outside the context manager (e.g. ``init`` measured from the
        script's first instant up to the first stage starting)."""
        self._append(name, params_dependent, started_at,
                     ended_at if ended_at is not None else time.time())

    # ------------------------------ internal ------------------------------

    def _append(self, name: str, params_dependent: bool,
                t0: float, t1: float) -> None:
        with open(self.csv_path, "a", newline="") as fh:
            csv.writer(fh).writerow([
                self.jobid, name,
                f"{t0:.3f}", f"{t1:.3f}",
                f"{t1 - t0:.3f}",
                int(bool(params_dependent)),
                self.params.get("n_train",    ""),
                self.params.get("n_valid",    ""),
                self.params.get("image_size", ""),
                self.params.get("batch_size", ""),
                self.params.get("steps",      ""),
            ])
