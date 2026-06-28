"""Local CSV log of every FASRC submission.

One row per job. Two write phases:

  * **Submission** (immediately after ``sbatch``) — append a row with
    the request-time fields (resources asked for, params, paths).
    Post-mortem columns are empty.
  * **Post-mortem** (once the reconcile loop sees a terminal state) —
    update the same row's post-mortem columns with what ``sacct``
    reports (state, elapsed time, peak memory, exit code, …).

Why CSV?
--------
The sqlite :class:`~euclid_polish.web.fasrc_jobs.JobDB` keeps live
operational state (PENDING/RUNNING reconciliation, ETA, in-flight
progress). The CSV log is a separate, append-once-update-many analytics
record — easy to ``grep``, easy to pull into pandas, easy to diff.
Sqlite stays the source of truth for the UI's live state; the CSV is
the historical resource-usage ledger.

The two stores write through different code paths but stay consistent
because both are triggered from the same submit and reconcile call
sites. If they diverge, the CSV is the canonical user-facing record.
"""

from __future__ import annotations

import csv
import json
import os
import threading
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from typing import Any


def _utc_now_iso() -> str:
    """ISO 8601 timestamp in UTC, second precision. Stable for CSV diffs."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class JobRecord:
    """One row of the submission log.

    Request fields are populated by :meth:`JobLog.record_submission`;
    post-mortem fields default to empty strings and are filled in by
    :meth:`JobLog.record_post_mortem` once ``sacct`` reports.
    """

    # ---- Identity & submission timestamps -----------------------------
    jobid:           str = ""
    submitted_at:    str = ""           # ISO 8601 UTC

    # ---- Request-time fields ------------------------------------------
    step_id:         str = ""
    label:           str = ""
    partition:       str = ""
    req_cpus:        int = 0
    req_gpus:        int = 0
    req_memory:      str = ""           # e.g. "8G"
    req_time_limit:  str = ""           # e.g. "2:00:00"
    params_json:     str = ""           # script-specific params, JSON
    script_path:     str = ""
    log_path:        str = ""
    err_path:        str = ""
    events_path:     str = ""

    # ---- Post-mortem fields (filled from sacct) -----------------------
    state:           str = ""           # "COMPLETED", "FAILED", "OOM", …
    exit_code:       str = ""           # "0:0" / "1:0" / "0:9"
    started_at:      str = ""           # ISO 8601 UTC
    ended_at:        str = ""           # ISO 8601 UTC
    elapsed_seconds: str = ""           # float as string for CSV neutrality
    cpu_seconds:     str = ""           # CPUTimeRAW, float
    cpu_efficiency:  str = ""           # cpu_seconds / (elapsed × alloc_cpus)
    max_rss_mb:      str = ""           # peak resident memory, MB
    alloc_cpus:      str = ""
    alloc_gpus:      str = ""
    alloc_memory_mb: str = ""

    # ---- Live resource-utilisation summary (from the events stream) ---
    # Folded from the job's ``resource`` samples by the post-mortem (see
    # ``fasrc_jobs.fetch_resource_summary``). Percent. ``cpu_util_*`` are
    # node-wide CPU% (distinct from ``cpu_efficiency``, which is sacct's
    # allocated-core efficiency); ``gpu_*`` come from nvidia-smi.
    gpu_util_mean:   str = ""
    gpu_util_peak:   str = ""
    gpu_mem_peak:    str = ""           # peak GPU memory utilisation, %
    cpu_util_mean:   str = ""
    cpu_util_peak:   str = ""


class JobLog:
    """Append-once-update-many CSV log of every submission.

    Concurrency model: a single :class:`threading.Lock` serialises all
    reads and writes, so submission appends and post-mortem updates from
    the reconcile loop don't race. Updates re-write the whole file
    atomically (write to ``.tmp``, then ``os.replace``) — fine at our
    scale (typically <1000 lifetime rows; the rewrite cost is
    microseconds for that size).
    """

    #: Column order in the CSV. Pinned so consumers (pandas / shell
    #: pipelines) see a stable schema even when :class:`JobRecord`
    #: grows new fields.
    COLUMNS: list[str] = [f.name for f in fields(JobRecord)]

    def __init__(self, csv_path: str) -> None:
        self.csv_path = csv_path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        if not os.path.exists(csv_path):
            self._write_header()
        else:
            self._migrate_header_if_needed()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_submission(self, record: JobRecord) -> None:
        """Append a freshly-submitted job's row to the CSV.

        ``submitted_at`` is set automatically if the caller left it blank.
        """
        if not record.submitted_at:
            record.submitted_at = _utc_now_iso()
        with self._lock, open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.COLUMNS,
                               extrasaction="ignore")
            w.writerow(self._stringify(asdict(record)))

    def record_post_mortem(self, jobid: str, stats: dict[str, Any]) -> bool:
        """Fill the post-mortem columns for ``jobid`` from ``stats``.

        ``stats`` keys map to :class:`JobRecord` post-mortem field names
        (``state``, ``exit_code``, ``elapsed_seconds``, …). Missing keys
        leave the existing cell unchanged. Returns ``True`` if the row
        existed and was updated, ``False`` if no row matched.
        """
        with self._lock:
            rows = self._read_all_locked()
            for r in rows:
                if r.get("jobid") == jobid:
                    for k, v in stats.items():
                        if k in self.COLUMNS and v is not None:
                            r[k] = self._stringify_value(v)
                    self._write_all_locked(rows)
                    return True
            return False

    def get(self, jobid: str) -> dict[str, str] | None:
        """Return the row for ``jobid`` (or ``None`` if not present)."""
        with self._lock:
            for r in self._read_all_locked():
                if r.get("jobid") == jobid:
                    return r
            return None

    def list_all(self) -> list[dict[str, str]]:
        """Return every row in submission order."""
        with self._lock:
            return self._read_all_locked()

    # ------------------------------------------------------------------
    # Per-step history queries
    # ------------------------------------------------------------------

    def history_for_step(self, step_id: str) -> list[dict[str, str]]:
        """Return every row whose ``step_id`` matches, newest first.

        Drives the "previous runs" panel under each step's form. The
        rows carry both request-time fields (resources asked for, task
        params) and post-mortem fields (actuals, exit code, state) —
        callers project to whatever subset they want.
        """
        with self._lock:
            rows = [r for r in self._read_all_locked()
                    if r.get("step_id") == step_id]
        # Newest first. ``submitted_at`` is ISO-8601 in UTC; string
        # ordering is correct for that format.
        rows.sort(key=lambda r: r.get("submitted_at", ""), reverse=True)
        return rows

    def latest_match(
        self,
        step_id:     str,
        params_json: str,
    ) -> dict[str, str] | None:
        """Find the run that should seed the resources form, or ``None``.

        Selection rule (sign-off in conversation): the *most recent
        run whose task params match exactly and whose final SLURM state
        was* ``COMPLETED``. If no such row exists, fall back to the
        most recent matching row in any state — that gives the user
        a re-submit-as-is hint instead of staring at blanks. Matches
        nothing when ``params_json`` is empty (some steps have no task
        params at all; we don't want to leak resources across steps).
        """
        if not params_json:
            return None
        matches = [r for r in self.history_for_step(step_id)
                   if r.get("params_json", "") == params_json]
        if not matches:
            return None
        completed = [r for r in matches if r.get("state") == "COMPLETED"]
        return (completed[0] if completed else matches[0])

    # ------------------------------------------------------------------
    # Internals (lock held by the caller)
    # ------------------------------------------------------------------

    def _write_header(self) -> None:
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(self.COLUMNS)

    def _migrate_header_if_needed(self) -> None:
        """If the on-disk header is missing any current column, rewrite
        the file with the full schema and blank values for new columns.
        Keeps old logs readable across :class:`JobRecord` evolutions."""
        with self._lock:
            if not os.path.exists(self.csv_path):
                return
            with open(self.csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                header = list(reader.fieldnames or [])
            if header == self.COLUMNS:
                return
            self._write_all_locked(rows)

    def _read_all_locked(self) -> list[dict[str, str]]:
        if not os.path.exists(self.csv_path):
            return []
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _write_all_locked(self, rows: list[dict[str, Any]]) -> None:
        tmp = self.csv_path + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.COLUMNS,
                               extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: self._stringify_value(r.get(k, ""))
                            for k in self.COLUMNS})
        os.replace(tmp, self.csv_path)

    # ------------------------------------------------------------------
    # Value normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _stringify_value(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (dict, list)):
            return json.dumps(v, ensure_ascii=False, separators=(",", ":"))
        return str(v)

    @classmethod
    def _stringify(cls, d: dict[str, Any]) -> dict[str, str]:
        return {k: cls._stringify_value(v) for k, v in d.items()}
