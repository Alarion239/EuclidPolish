"""Per-stage wall-clock timings for non-interactive pipeline jobs.

This module deliberately lives outside :mod:`euclid_polish.training`: record
generation and forward modelling use the same timer, and importing a utility
for those stages must not initialize the model, visualization, or TensorFlow
training stack.
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
    """Append one CSV timing row per stage.

    The timer is safe within one process. Concurrent jobs should use separate
    CSV paths, as the pipeline's SLURM job-id default already does.
    """

    def __init__(
        self,
        csv_path: str,
        *,
        jobid: str,
        params: dict[str, Any] | None = None,
    ) -> None:
        self.csv_path = csv_path
        self.jobid = jobid
        self.params = dict(params or {})
        parent = os.path.dirname(csv_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as fh:
                csv.writer(fh).writerow(_HEADER)

    @contextmanager
    def stage(self, name: str, *, params_dependent: bool):
        """Time a block and append its row even when the block raises."""
        started_at = time.time()
        try:
            yield
        finally:
            self._append(
                name,
                params_dependent,
                started_at,
                time.time(),
            )

    def mark(
        self,
        name: str,
        *,
        params_dependent: bool,
        started_at: float,
        ended_at: float | None = None,
    ) -> None:
        """Append a row for a stage whose bounds were captured elsewhere."""
        self._append(
            name,
            params_dependent,
            started_at,
            ended_at if ended_at is not None else time.time(),
        )

    def _append(
        self,
        name: str,
        params_dependent: bool,
        started_at: float,
        ended_at: float,
    ) -> None:
        with open(self.csv_path, "a", newline="") as fh:
            csv.writer(fh).writerow([
                self.jobid,
                name,
                f"{started_at:.3f}",
                f"{ended_at:.3f}",
                f"{ended_at - started_at:.3f}",
                int(bool(params_dependent)),
                self.params.get("n_train", ""),
                self.params.get("n_valid", ""),
                self.params.get("image_size", ""),
                self.params.get("batch_size", ""),
                self.params.get("steps", ""),
            ])
