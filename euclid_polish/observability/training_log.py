"""Append-only, resume-continuous CSV of per-evaluation training metrics.

Stored next to the checkpoint (``<ckpt_dir>/training_log.csv``) so a run's
whole history travels with its weights. One row is written per validation:

  * **per-lane training losses** — synthetic / HST / star-anchor, each the
    mean over the eval window (empty when that lane is off this run),
  * **per-mode validation PSNRs** — synthetic (str/raw), HST (str/raw),
    star-anchor (masked at the star pixel),
  * gradient norms, and the composite **save-best score**.

This is the durable, structured record of a run — readable with pandas or
the dashboard without parsing stdout. It survives resumes (append-only). If
the column set changed since the last run, the stale file is rotated to a
timestamped ``.bak`` so each file stays internally consistent (a reader can
trust one file's header), and the new run starts a fresh, consistent file.
"""

from __future__ import annotations

import csv
import os
import time
from typing import Optional, Sequence


class TrainingLog:
    """Owns one append-only metrics CSV with a fixed column schema."""

    def __init__(self, path: str, columns: Sequence[str]) -> None:
        self.path = path
        self.columns = tuple(columns)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Set to the ``.bak`` path when an existing file with a stale header
        # was rotated out of the way (caller may surface it to the user).
        self.rotated_backup: Optional[str] = self._rotate_if_stale_header()

    def _rotate_if_stale_header(self) -> Optional[str]:
        """If an existing log's header doesn't match the current columns,
        rename it to a timestamped backup so appends never misalign."""
        if not (os.path.exists(self.path) and os.path.getsize(self.path) > 0):
            return None
        with open(self.path, "r", newline="") as fh:
            first_line = fh.readline().rstrip("\r\n")
        if first_line == ",".join(self.columns):
            return None
        backup = os.path.join(
            os.path.dirname(self.path) or ".",
            f"training_log.{time.strftime('%Y%m%d-%H%M%S')}.bak",
        )
        os.rename(self.path, backup)
        return backup

    def append(self, row: dict) -> None:
        """Append one row, writing the header first if the file is new/empty.
        Missing columns are written blank; unknown keys are ignored (the
        ``DictWriter`` is pinned to ``self.columns``)."""
        write_header = (not os.path.exists(self.path)
                        or os.path.getsize(self.path) == 0)
        with open(self.path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=self.columns,
                               extrasaction="ignore", restval="")
            if write_header:
                w.writeheader()
            w.writerow(row)
