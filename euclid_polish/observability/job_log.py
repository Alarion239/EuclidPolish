"""Local CSV log of every FASRC submission.

One row per job. Two write phases:

  * **Submission** (immediately after ``sbatch``) — append a row with
    the request-time fields (resources asked for, params, paths).
    Post-mortem columns are empty.
  * **Post-mortem** (once the reconcile loop sees a terminal state) —
    update the same row's post-mortem columns with Jobstats utilization and
    ``sacct`` lifecycle/accounting data (state, elapsed time, peak memory,
    exit code, …).

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
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from typing import Any

# Embedded population calibrations can make ``params_json`` substantially
# larger than Python's conservative 128 KiB CSV default.  Every JobLog read
# must accept rows that JobLog itself can write; keep a bounded allowance
# rather than requiring each server entry point to change process-global CSV
# state before importing the web application.
_CSV_FIELD_SIZE_LIMIT = 16 * 1024 * 1024


def _ensure_csv_field_size_limit() -> None:
    if csv.field_size_limit() < _CSV_FIELD_SIZE_LIMIT:
        csv.field_size_limit(_CSV_FIELD_SIZE_LIMIT)


def _utc_now_iso() -> str:
    """ISO 8601 timestamp in UTC, second precision. Stable for CSV diffs."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _file_version(st: os.stat_result) -> tuple[int, int, int]:
    """Identify one version of the ledger file for the read cache.

    A JobLog rewrite replaces the file (new inode), an append grows it, and
    any other edit moves its modification time.
    """
    return (st.st_ino, st.st_mtime_ns, st.st_size)


def _newest_first(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Sort ``rows`` newest first, in place. ``submitted_at`` is ISO-8601 in
    UTC, so string order is time order; equal timestamps keep file order."""
    rows.sort(key=lambda r: r.get("submitted_at", ""), reverse=True)
    return rows


@dataclass
class JobRecord:
    """One row of the submission log.

    Request fields are populated by :meth:`JobLog.record_submission`;
    post-mortem fields default to empty strings and are filled in by
    :meth:`JobLog.record_post_mortem` once Jobstats or ``sacct`` reports.
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
    cpu_seconds:     str = ""           # consumed CPU seconds, float
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
    gpu_mem_peak:    str = ""           # legacy, mixed units: live-sampler GPU memory % or sacct MB
    cpu_util_mean:   str = ""
    cpu_util_peak:   str = ""

    # ---- Normalized Jobstats / accounting fields ---------------------
    # These are additive so old CSV rows remain readable.  Memory values
    # explicitly carry their unit; utilization values are percentages.
    accounting_source:             str = ""
    accounting_collected_at:       str = ""
    accounting_attempted_at:       str = ""
    gpu_mem_peak_mb:               str = ""
    gpu_mem_util_peak:             str = ""
    jobstats_cpu_util:             str = ""
    jobstats_cpu_memory_util:      str = ""
    jobstats_cpu_memory_used_mb:   str = ""
    jobstats_cpu_memory_alloc_mb:  str = ""
    jobstats_gpu_util:             str = ""
    jobstats_gpu_memory_util:      str = ""
    jobstats_gpu_memory_used_mb:   str = ""
    jobstats_gpu_memory_total_mb:  str = ""
    jobstats_cpu_nodes_json:       str = ""
    jobstats_cpu_memory_nodes_json: str = ""
    jobstats_gpu_nodes_json:       str = ""
    jobstats_gpu_memory_nodes_json: str = ""
    jobstats_notes_json:           str = ""


class JobLog:
    """Append-once-update-many CSV log of every submission.

    Concurrency model: a single :class:`threading.Lock` serialises all
    reads and writes, so submission appends and post-mortem updates from
    the reconcile loop don't race. Updates re-write the whole file
    atomically (write to ``.tmp``, then ``os.replace``) — fine at our
    scale (hundreds of rows, a few MB).

    Reads are memoised: the parsed rows are kept with the file version
    (inode, mtime, size) they came from and re-parsed only when that
    changes, so a write by another process (``scripts/``, a second console)
    is still picked up. JobLog's own rewrites refresh the cache with the rows
    they wrote. Every read returns copies, so callers may change the rows
    they get without touching the cache.
    """

    #: Column order in the CSV. Pinned so consumers (pandas / shell
    #: pipelines) see a stable schema even when :class:`JobRecord`
    #: grows new fields.
    COLUMNS: list[str] = [f.name for f in fields(JobRecord)]

    def __init__(self, csv_path: str) -> None:
        _ensure_csv_field_size_limit()
        self.csv_path = csv_path
        self._lock = threading.Lock()
        # Parsed rows of the file version ``_cache_key`` names; shared, so
        # only ever copied out (see ``_rows_locked``).
        self._cache_rows: list[dict[str, str]] = []
        self._cache_key: tuple[int, int, int] | None = None
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
            # The next read parses the grown file.
            self._cache_key = None

    def record_post_mortem(self, jobid: str, stats: dict[str, Any]) -> bool:
        """Fill the post-mortem columns for ``jobid`` from ``stats``.

        ``stats`` keys map to :class:`JobRecord` post-mortem field names
        (``state``, ``exit_code``, ``elapsed_seconds``, …). Missing keys
        leave the existing cell unchanged. Returns ``True`` if the row
        existed and was updated, ``False`` if no row matched.
        """
        with self._lock:
            rows = self._rows_locked()
            for i, r in enumerate(rows):
                if r.get("jobid") == jobid:
                    # Change a copy: ``rows`` is the shared cache.
                    row = dict(r)
                    for k, v in stats.items():
                        if k in self.COLUMNS and v is not None:
                            row[k] = self._stringify_value(v)
                    self._write_all_locked([*rows[:i], row, *rows[i + 1:]])
                    return True
            return False

    def mark_accounting_attempt(self, jobid: str, at: float | None = None) -> bool:
        """Record a failed or in-progress accounting lookup timestamp."""
        return self.record_post_mortem(
            jobid, {"accounting_attempted_at": at if at is not None else time.time()},
        )

    def accounting_attempt_due(self, jobid: str, *, min_interval: float = 60.0) -> bool:
        """Whether another best-effort accounting lookup should run now."""
        row = self.get(jobid)
        if not row:
            return True
        try:
            attempted = float(row.get("accounting_attempted_at") or 0.0)
        except (TypeError, ValueError):
            return True
        return (time.time() - attempted) >= float(min_interval)

    def get(self, jobid: str) -> dict[str, str] | None:
        """Return the row for ``jobid`` (or ``None`` if not present)."""
        with self._lock:
            for r in self._rows_locked():
                if r.get("jobid") == jobid:
                    return dict(r)
            return None

    def get_many(self, jobids: Iterable[str]) -> dict[str, dict[str, str]]:
        """``{jobid: row}`` for each of ``jobids`` in the log, from one read.

        The rows :meth:`get` returns (the first row of a repeated jobid);
        jobids without a row are left out.
        """
        wanted = set(jobids)
        found: dict[str, dict[str, str]] = {}
        with self._lock:
            for r in self._rows_locked():
                jobid = r.get("jobid")
                if jobid is not None and jobid in wanted and jobid not in found:
                    found[jobid] = dict(r)
        return found

    def list_all(self) -> list[dict[str, str]]:
        """Return every row in submission order."""
        with self._lock:
            return [dict(r) for r in self._rows_locked()]

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
            rows = [dict(r) for r in self._rows_locked()
                    if r.get("step_id") == step_id]
        return _newest_first(rows)

    def history_by_step(self) -> dict[str, list[dict[str, str]]]:
        """:meth:`history_for_step` for every step at once, from one read.

        ``{step_id: rows newest first}``; a step without runs is absent.
        """
        by_step: dict[str, list[dict[str, str]]] = {}
        with self._lock:
            for r in self._rows_locked():
                step_id = r.get("step_id")
                if step_id is not None:
                    by_step.setdefault(step_id, []).append(dict(r))
        for rows in by_step.values():
            _newest_first(rows)
        return by_step

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
                version = _file_version(os.fstat(f.fileno()))
                reader = csv.DictReader(f)
                rows = list(reader)
                header = list(reader.fieldnames or [])
            if header == self.COLUMNS:
                # Already current: the rows just parsed seed the read cache.
                self._cache_rows, self._cache_key = rows, version
                return
            self._write_all_locked(rows)

    def _rows_locked(self) -> list[dict[str, str]]:
        """Every row, parsed again only when the file changed on disk.

        The list and its dicts are the cache itself: copy a row before
        changing it or handing it out.
        """
        try:
            if _file_version(os.stat(self.csv_path)) == self._cache_key:
                return self._cache_rows
            with open(self.csv_path, newline="", encoding="utf-8") as f:
                # The version of the file actually opened, taken before
                # parsing: a write landing mid-parse changes the file's
                # version, so the next read parses again.
                version = _file_version(os.fstat(f.fileno()))
                rows = list(csv.DictReader(f))
        except FileNotFoundError:
            self._cache_rows, self._cache_key = [], None
            return []
        self._cache_rows, self._cache_key = rows, version
        return rows

    def _write_all_locked(self, rows: list[dict[str, Any]]) -> None:
        out = [{k: self._stringify_value(r.get(k, "")) for k in self.COLUMNS}
               for r in rows]
        tmp = self.csv_path + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.COLUMNS,
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(out)
        # A rename keeps the file's inode, mtime and size, so this is the
        # version the replaced ledger reads as; ``out`` is exactly what a
        # re-read would parse back.
        version = _file_version(os.stat(tmp))
        os.replace(tmp, self.csv_path)
        self._cache_rows, self._cache_key = out, version

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
