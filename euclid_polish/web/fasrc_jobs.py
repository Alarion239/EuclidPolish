"""FASRC job tracking, submission helper, ETA heuristics.

Two layers:

  * :class:`JobDB` — persistent record of every job we've submitted from
    this UI. Sqlite at ``~/.euclid_polish/fasrc_jobs.db``. Survives
    Flask restarts so the ETA model has history to draw from.

  * :func:`submit_sbatch_script` — single helper used by every Flask
    submit handler: write the script over SSH, ``sbatch`` it, parse the
    job id, record in :class:`JobDB`. Script *rendering* lives in
    :mod:`euclid_polish.web.fasrc_pipeline`; this module just orchestrates
    the SSH+DB side.

The ETA model is intentionally simple: median seconds-per-step across
the user's last N completed jobs, multiplied by their requested step
count. If the in-flight job's log emits ``step X/Y`` we refine the ETA
live.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import statistics
import time
from typing import Any, Dict, List, Optional, Tuple

from euclid_polish.observability import JobLog, JobRecord
from euclid_polish.web import fasrc_config
from euclid_polish.web.sacct import fetch_sacct_stats

DB_DIR  = fasrc_config.CONFIG_DIR
DB_PATH = os.path.join(DB_DIR, "fasrc_jobs.db")
#: Append-once-update-many submission log. Separate from sqlite so the
#: file can be ``cat``'d / loaded into pandas / diffed in git without
#: any database tooling.
JOB_LOG_PATH = os.path.join(DB_DIR, "fasrc_job_log.csv")

SCHEMA = """
CREATE TABLE IF NOT EXISTS fasrc_jobs (
    jobid           TEXT PRIMARY KEY,
    submitted_at    REAL,
    label           TEXT,
    params_json     TEXT,
    script_path     TEXT,
    log_path        TEXT,
    err_path        TEXT,
    events_path     TEXT,
    state           TEXT,
    started_at      REAL,
    ended_at        REAL,
    progress_step   INTEGER DEFAULT 0,
    progress_total  INTEGER DEFAULT 0,
    last_seen       REAL,
    runtime_seconds REAL,
    step_id         TEXT
);
"""


def _ensure_schema_columns(conn: sqlite3.Connection) -> None:
    """Add new columns to an existing DB without losing data.

    Runs every time :class:`JobDB` is constructed. Sqlite's ``ALTER
    TABLE ADD COLUMN`` is non-failing if we wrap it — pre-existing
    column raises ``OperationalError`` which we swallow.
    """
    for col_def in (
        "runtime_seconds REAL",
        "step_id TEXT",
        "events_path TEXT",
    ):
        try:
            conn.execute(f"ALTER TABLE fasrc_jobs ADD COLUMN {col_def}")
        except sqlite3.OperationalError as e:
            if "duplicate column" not in str(e).lower():
                raise


# ---------------------------------------------------------------------------
# Sqlite-backed job history
# ---------------------------------------------------------------------------

class JobDB:
    def __init__(self, path: str = DB_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        with self._conn() as c:
            c.executescript(SCHEMA)
            _ensure_schema_columns(c)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=10)
        c.row_factory = sqlite3.Row
        return c

    # ---------------- CRUD --------------------------------------------------

    def insert(
        self,
        jobid: str,
        *,
        label:       str,
        params:      Dict[str, Any],
        script_path: str,
        log_path:    str,
        err_path:    str,
        events_path: Optional[str] = None,
    ) -> None:
        with self._conn() as c:
            c.execute(
                """
                INSERT OR REPLACE INTO fasrc_jobs
                  (jobid, submitted_at, label, params_json, script_path,
                   log_path, err_path, events_path, state, last_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)
                """,
                (jobid, time.time(), label, json.dumps(params),
                 script_path, log_path, err_path, events_path, time.time()),
            )

    def update_state(self, jobid: str, *, state: str,
                     started_at: Optional[float] = None,
                     ended_at: Optional[float] = None) -> None:
        sets, args = ["state = ?", "last_seen = ?"], [state, time.time()]
        if started_at is not None:
            sets.append("started_at = COALESCE(started_at, ?)")
            args.append(started_at)
        if ended_at is not None:
            sets.append("ended_at = ?")
            args.append(ended_at)
        args.append(jobid)
        with self._conn() as c:
            c.execute(f"UPDATE fasrc_jobs SET {', '.join(sets)} "
                      f"WHERE jobid = ?", args)

    def update_progress(self, jobid: str, step: int, total: int) -> None:
        with self._conn() as c:
            c.execute("UPDATE fasrc_jobs SET progress_step = ?, "
                      "progress_total = ?, last_seen = ? WHERE jobid = ?",
                      (int(step), int(total), time.time(), jobid))

    def update_runtime(self, jobid: str, runtime_seconds: float) -> None:
        """Record the script-reported wall time from a ``RUNTIME_SECONDS=...`` line."""
        with self._conn() as c:
            c.execute("UPDATE fasrc_jobs SET runtime_seconds = ?, "
                      "last_seen = ? WHERE jobid = ?",
                      (float(runtime_seconds), time.time(), jobid))

    def set_step_id(self, jobid: str, step_id: str) -> None:
        """Tag a job with its HST-pipeline step id (for per-step history)."""
        with self._conn() as c:
            c.execute("UPDATE fasrc_jobs SET step_id = ?, last_seen = ? "
                      "WHERE jobid = ?",
                      (str(step_id), time.time(), jobid))

    def get(self, jobid: str) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            r = c.execute("SELECT * FROM fasrc_jobs WHERE jobid = ?",
                          (jobid,)).fetchone()
        return dict(r) if r else None

    def list_recent(self, limit: int = 30) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM fasrc_jobs ORDER BY submitted_at DESC "
                "LIMIT ?", (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_completed(self, limit: int = 10) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM fasrc_jobs "
                "WHERE state IN ('COMPLETED', 'DONE', 'TIMEOUT', 'FAILED', "
                "                'CANCELLED') "
                "  AND started_at IS NOT NULL AND ended_at IS NOT NULL "
                "ORDER BY submitted_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]


DB = JobDB()
#: Module-level singleton; tests can swap in their own log via
#: ``monkeypatch.setattr(fasrc_jobs, "JOBLOG", JobLog(tmp_csv))``.
JOBLOG = JobLog(JOB_LOG_PATH)


# ---------------------------------------------------------------------------
# ETA heuristic
# ---------------------------------------------------------------------------

def _params_of(row: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return json.loads(row.get("params_json") or "{}")
    except json.JSONDecodeError:
        return {}


def secs_per_step_history(n: int = 8) -> Optional[float]:
    """Median wall-second-per-training-step across the last ``n`` finished jobs.

    ``steps`` can arrive as a string because submit forms write everything
    as text into ``params_json``; ``runtime`` is built from floats but
    either timestamp can be ``None``. Coerce both to numbers up front and
    skip rows where the coercion fails — silently dropping a bad row is
    better than 500-ing the training-status endpoint (which would freeze
    the live job ticker on every page in the UI).
    """
    samples = []
    for row in DB.list_completed(limit=n):
        params = _params_of(row)
        steps_raw = params.get("steps") or row.get("progress_total") or 0
        try:
            steps = float(steps_raw)
        except (TypeError, ValueError):
            continue
        try:
            runtime = float((row.get("ended_at") or 0)
                            - (row.get("started_at") or 0))
        except (TypeError, ValueError):
            continue
        if runtime <= 0 or steps <= 0:
            continue
        samples.append(runtime / steps)
    if not samples:
        return None
    return statistics.median(samples)


def eta_for_submission(steps: int) -> Optional[float]:
    """Rough wall-time ETA in seconds for a fresh job of ``steps`` steps."""
    spt = secs_per_step_history()
    if spt is None:
        return None
    return spt * steps


def eta_for_running(row: Dict[str, Any]) -> Optional[float]:
    """Live ETA for a job that has emitted a ``step X/Y`` line.

    Falls back to the historical heuristic if the log has nothing yet.
    """
    step  = row.get("progress_step")  or 0
    total = row.get("progress_total") or 0
    started = row.get("started_at")
    if step and total and started:
        elapsed = time.time() - started
        if step > 0:
            return elapsed * (total - step) / float(step)
    params = _params_of(row)
    return eta_for_submission(params.get("steps", 0))


# ---------------------------------------------------------------------------
# Progress-line parsing
# ---------------------------------------------------------------------------

#  tqdm formats step counters as ``  12345/400000 [12:34<…]`` — match the
#  ``step/total`` pair and ignore the rest. Falls through if absent.
_TQDM_PROGRESS_RE = re.compile(r"(\d{1,8})\s*/\s*(\d{1,8})")

#  HST pipeline scripts emit a final ``RUNTIME_SECONDS=12345.6`` line — this
#  lets the UI store actual wall time without having to compute from
#  started_at/ended_at (which can be wrong if SLURM was queued).
_RUNTIME_RE = re.compile(r"RUNTIME_SECONDS=([\d.]+)")
_STEP_ID_RE = re.compile(r"STEP_ID=([A-Za-z0-9_\-]+)")


def parse_progress(line: str) -> Optional[tuple[int, int]]:
    """Pull (step, total) out of a single log line, or None."""
    m = _TQDM_PROGRESS_RE.search(line)
    if not m:
        return None
    step, total = int(m.group(1)), int(m.group(2))
    # Guard against false positives like "shape 4/4" in module prints.
    if total < 50 or step > total:
        return None
    return step, total


def parse_runtime_seconds(text: str) -> Optional[float]:
    """Extract the last ``RUNTIME_SECONDS=...`` line from ``text`` (None if absent)."""
    last: Optional[float] = None
    for m in _RUNTIME_RE.finditer(text or ""):
        try:
            last = float(m.group(1))
        except ValueError:
            continue
    return last


def parse_step_id(text: str) -> Optional[str]:
    """Extract the last ``STEP_ID=...`` line — set by HST-pipeline sbatch banners."""
    last: Optional[str] = None
    for m in _STEP_ID_RE.finditer(text or ""):
        last = m.group(1)
    return last


def runtime_history_for_step(step_id: str, *, limit: int = 5) -> List[float]:
    """Return the last ``limit`` completed runtimes for a given HST step.

    Used by the UI to render "last runtime: ~3 min" hints next to each
    pipeline step's submit button. Returns newest-first; empty if no
    history yet.
    """
    with DB._conn() as c:
        rows = c.execute(
            "SELECT runtime_seconds FROM fasrc_jobs "
            "WHERE step_id = ? AND runtime_seconds IS NOT NULL "
            "  AND runtime_seconds > 0 "
            "ORDER BY submitted_at DESC LIMIT ?",
            (str(step_id), int(limit)),
        ).fetchall()
    return [float(r["runtime_seconds"]) for r in rows]


def median_runtime_for_step(step_id: str) -> Optional[float]:
    """Median of the last few runtimes for ``step_id``; None if no data."""
    hist = runtime_history_for_step(step_id, limit=5)
    if not hist:
        return None
    return statistics.median(hist)


# ---------------------------------------------------------------------------
# Submission presets — view of the legacy ``run_pipeline.py`` step classes
# ---------------------------------------------------------------------------
#
# The form's preset dropdown still expects a flat ``{name: {label,
# partition, n_gpus, …, skip_flags, needs_train_knobs}}`` dict. Build
# it from the :class:`RunPipelineStep` instances in the central registry
# so resource specs live in exactly one place (the step subclasses).


def _build_presets() -> Dict[str, Dict[str, Any]]:
    # Local import — :mod:`fasrc_pipeline` already imports
    # :mod:`fasrc_config` from this package; pulling it at module
    # top would close the cycle.
    from euclid_polish.web.fasrc_pipeline import REGISTRY, RunPipelineStep

    out: Dict[str, Dict[str, Any]] = {}
    for step in REGISTRY.all():
        if not isinstance(step, RunPipelineStep):
            continue
        out[step.step_id] = {
            "label":             step.label,
            "partition":         step.defaults.partition,
            "n_gpus":            int(step.defaults.n_gpus),
            "n_cpus":            int(step.defaults.n_cpus),
            "memory":            step.defaults.memory,
            "time_limit":        step.defaults.time_limit,
            "skip_flags":        " ".join(step.skip_flags),
            "needs_train_knobs": bool(step.needs_train_knobs),
        }
    return out


PRESETS: Dict[str, Dict[str, Any]] = _build_presets()


def resolve_preset(name: str) -> Dict[str, Any]:
    """Return the preset dict for ``name`` (falls back to ``custom``)."""
    return PRESETS.get(name) or PRESETS["custom"]


# ---------------------------------------------------------------------------
# Submission helper — single SSH-write + sbatch + parse + DB.insert flow
# ---------------------------------------------------------------------------

# Marker for the heredoc that streams the script over SSH. Single source
# of truth — both submit handlers used to hard-code their own copy.
_HEREDOC_EOF = "__EUCLID_POLISH_EOF__"

_SBATCH_JOBID_RE = re.compile(r"Submitted batch job (\d+)")


def submit_sbatch_script(
    ssh,
    *,
    cfg:      fasrc_config.FasrcConfig,
    built:    Dict[str, str],
    label:    str,
    params:   Dict[str, Any],
    step_id:  Optional[str] = None,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Write a rendered sbatch script to FASRC, ``sbatch`` it, record it.

    Parameters
    ----------
    ssh :
        A connected SSH session (e.g. ``STATE.ssh``). Must expose
        ``run(cmd, timeout=…) -> (rc, stdout, stderr)``.
    built :
        Return value of :func:`fasrc_pipeline.render_sbatch_body` —
        the dict with ``body``, ``script``, ``out``, ``err``, ``name``.
    step_id :
        If given, the row is tagged in :class:`JobDB` via ``set_step_id``
        so the per-step runtime history queries pick it up.

    Returns
    -------
    (slurm_id, payload) :
        ``slurm_id`` is ``None`` on failure; in that case ``payload`` is
        the ``jsonify``-able error dict the caller can return directly.
        On success ``payload`` has ``ok=True`` and the fields the JS
        client expects.
    """
    remote_script = f"{cfg.repo_path}/{built['script']}"
    write_cmd = (
        f"mkdir -p {cfg.repo_path}/{os.path.dirname(built['script'])} && "
        f"cat > {remote_script} <<'{_HEREDOC_EOF}'\n"
        f"{built['body']}"
        f"{_HEREDOC_EOF}\n"
        f"chmod +x {remote_script}"
    )
    rc, _out, err = ssh.run(write_cmd, timeout=20)
    if rc != 0:
        return None, {"ok": False,
                      "error": f"failed to write script: {err.strip()}"}

    rc, out, err = ssh.run(
        f"cd {cfg.repo_path} && sbatch {built['script']}",
        timeout=20,
    )
    if rc != 0:
        return None, {"ok": False,
                      "error": f"sbatch failed: {err.strip()}"}
    m = _SBATCH_JOBID_RE.search(out)
    if not m:
        return None, {"ok": False,
                      "error": f"unparseable sbatch output: {out}"}
    slurm_id    = m.group(1)
    log_path    = f"{cfg.repo_path}/{built['out']}"
    err_path    = f"{cfg.repo_path}/{built['err']}"
    # ``built`` may be missing ``events`` for callers that built their
    # script through an older path; tolerate either shape.
    events_rel  = built.get("events")
    events_path = f"{cfg.repo_path}/{events_rel}" if events_rel else None
    DB.insert(
        slurm_id,
        label=label,
        params=params,
        script_path=remote_script,
        log_path=log_path,
        err_path=err_path,
        events_path=events_path,
    )
    if step_id:
        DB.set_step_id(slurm_id, step_id)

    # Mirror into the CSV submission log. Resource fields come straight
    # from ``params`` (the form values, post-validation by
    # :class:`StepResources.from_form` → ``to_dict()``) so the log
    # matches what SLURM saw on the ``#SBATCH`` lines. Script-specific
    # params (n_stars, n_tiles, hst_fraction, …) are JSON-encoded into
    # the ``params_json`` column. Errors here must not break the
    # submit response, so swallow + log instead of raising.
    try:
        JOBLOG.record_submission(JobRecord(
            jobid=slurm_id,
            step_id=step_id or "",
            label=label,
            partition=str(params.get("partition", "")),
            req_cpus=int(params.get("n_cpus") or 0),
            req_gpus=int(params.get("n_gpus") or 0),
            req_memory=str(params.get("memory", "")),
            req_time_limit=str(params.get("time_limit", "")),
            # ``sort_keys`` is load-bearing: the per-step history lookup
            # filters by exact ``params_json`` string equality, so two
            # submissions with the same task params must serialise
            # identically regardless of form-field insertion order.
            params_json=json.dumps(
                {k: v for k, v in params.items()
                 if k not in ("partition", "n_cpus", "n_gpus", "memory",
                              "time_limit", "confirm", "label", "preset")},
                ensure_ascii=False, separators=(",", ":"),
                sort_keys=True,
            ),
            script_path=remote_script,
            log_path=log_path,
            err_path=err_path,
            events_path=events_path or "",
        ))
    except Exception:
        # The CSV log is a side channel; never let it fail the submit.
        pass

    payload: Dict[str, Any] = {
        "ok":          True,
        "jobid":       slurm_id,
        "label":       label,
        "log_path":    log_path,
        "events_path": events_path,
        "params":      params,
    }
    if step_id:
        payload["step_id"] = step_id
    return slurm_id, payload


# ---------------------------------------------------------------------------
# Parsing remote `squeue` output → row dicts the UI can render
# ---------------------------------------------------------------------------

def parse_squeue(text: str) -> List[Dict[str, str]]:
    """Parse our fixed-format ``squeue`` output.

    We invoke squeue with ``--format`` and a pipe separator. The literal
    ``\\t`` in modern SLURM's format string is NOT expanded — it shows up
    as the two characters ``\t`` in the output, which silently broke the
    earlier tab-based split. Pipes are passed through literally.
    """
    rows: List[Dict[str, str]] = []
    keys = ["jobid", "name", "state", "time", "time_limit",
            "nodes", "reason", "start_time"]
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("JOBID"):
            continue
        # Tolerate both pipe and tab separation so old runs of the helper
        # (or anyone pasting squeue output directly) still parse.
        if "|" in line:
            parts = line.split("|")
        elif "\t" in line:
            parts = line.split("\t")
        else:
            parts = line.split()
        if len(parts) < len(keys):
            parts += [""] * (len(keys) - len(parts))
        rows.append(dict(zip(keys, parts[: len(keys)])))
    return rows


SQUEUE_FMT = "%i|%j|%T|%M|%l|%D|%R|%S"


# Terminal states — once a row reaches any of these we stop reconciling
# it against squeue. ``UNKNOWN`` is here too: it means "this job was
# tracked but disappeared from squeue without ever showing started_at",
# so we treat it as a failure mode and leave it alone.
TERMINAL_STATES = frozenset({
    "COMPLETED", "DONE", "FAILED", "CANCELLED", "TIMEOUT", "UNKNOWN",
})


def sync_pending_on_connect(
    ssh: Any,
    *,
    db: Optional["JobDB"] = None,
    job_log: Optional[JobLog] = None,
    recent_limit: int = 50,
) -> Dict[str, str]:
    """Re-sync every non-terminal DB row against SLURM, fill missing post-mortems.

    Called whenever the SSH session becomes available — at server
    startup (auto-connect succeeded) and at the manual
    ``/api/fasrc/connect`` POST handler. Catches the case where a job
    ran to completion while the server was offline: ``squeue`` no
    longer lists it, so :func:`reconcile_with_squeue` marks it
    ``DONE`` and (because ``ssh`` is passed) fetches its ``sacct``
    accounting row into the CSV log.

    Returns the same change dict as :func:`reconcile_with_squeue`.
    Silently returns ``{}`` if SSH is missing, disconnected, or the
    squeue call fails — startup sync must never break the connect
    handshake.
    """
    if ssh is None:
        return {}
    try:
        if not ssh.is_connected():
            return {}
        rc, out, _err = ssh.run(
            f"squeue -h -u $USER --format='{SQUEUE_FMT}'", timeout=15,
        )
    except Exception:
        return {}
    if rc != 0:
        return {}
    squeue_rows = parse_squeue(out)
    return reconcile_with_squeue(
        squeue_rows, db=db, job_log=job_log,
        recent_limit=recent_limit, ssh=ssh,
    )


def reconcile_with_squeue(squeue_rows: List[Dict[str, Any]],
                          *, db: Optional["JobDB"] = None,
                          recent_limit: int = 50,
                          ssh: Optional[Any] = None,
                          job_log: Optional[JobLog] = None) -> Dict[str, str]:
    """Cross-check the JobDB against a live ``squeue`` snapshot.

    For every non-terminal DB row:

      * if its jobid IS in ``squeue_rows`` → set the DB state to whatever
        squeue says it is (RUNNING / PENDING / FAILED / …);
      * if its jobid is NOT in ``squeue_rows`` and the row has a
        ``started_at`` → the job ran and has since finished, mark
        ``DONE`` with ``ended_at = now``;
      * if its jobid is NOT in ``squeue_rows`` and ``started_at`` is
        missing → we never saw it start *and* it isn't queued anywhere
        we can ask about, so mark ``UNKNOWN``.

    Side effect: when a job transitions to a terminal state *and* an
    SSH handle is provided, ``sacct`` is queried for that job and its
    post-mortem stats are folded into :data:`JOBLOG`. The CSV row was
    already created at submit time; this fills in the actuals columns
    (elapsed time, max RSS, exit code, …). A sacct failure is silent —
    we never want to break the reconcile pass over a missing accounting
    row.

    Returns a dict ``{jobid: new_state}`` for every row whose state we
    changed, which the caller can use to log/debug.

    ``db`` defaults to the module-level :data:`DB` singleton;
    ``job_log`` defaults to :data:`JOBLOG`. Tests pass isolated ones.
    """
    target_db  = db if db is not None else DB
    target_log = job_log if job_log is not None else JOBLOG
    live_state: Dict[str, str] = {r["jobid"]: r.get("state", "?")
                                  for r in squeue_rows}
    changes: Dict[str, str] = {}
    just_finalised: List[str] = []

    for stored in target_db.list_recent(recent_limit):
        jobid = stored["jobid"]
        cur   = stored.get("state") or ""
        if cur in TERMINAL_STATES:
            continue
        if jobid in live_state:
            live = live_state[jobid]
            if live == "RUNNING":
                target_db.update_state(
                    jobid, state="RUNNING",
                    started_at=time.time() - parse_slurm_time(
                        next((r.get("time") for r in squeue_rows
                              if r["jobid"] == jobid), "")
                    ),
                )
            else:
                target_db.update_state(jobid, state=live)
            if live != cur:
                changes[jobid] = live
                if live in TERMINAL_STATES:
                    just_finalised.append(jobid)
            continue

        # jobid is not in live squeue → finalise it.
        if stored.get("started_at"):
            target_db.update_state(jobid, state="DONE",
                                   ended_at=time.time())
            changes[jobid] = "DONE"
            just_finalised.append(jobid)
        else:
            target_db.update_state(jobid, state="UNKNOWN",
                                   ended_at=time.time())
            changes[jobid] = "UNKNOWN"
            just_finalised.append(jobid)

    # Post-mortem accounting fetch. One ``sacct`` call per newly-
    # finalised job — typically zero or one per reconcile pass.
    if ssh is not None and just_finalised:
        for jobid in just_finalised:
            try:
                stats = fetch_sacct_stats(ssh, jobid)
            except Exception:
                stats = None
            if stats:
                try:
                    target_log.record_post_mortem(jobid, stats)
                except Exception:
                    pass

    return changes


def parse_slurm_time(t: Optional[str]) -> float:
    """SLURM ``d-hh:mm:ss`` / ``hh:mm:ss`` / ``mm:ss`` → seconds.

    Returns 0.0 on anything we can't parse rather than raising — the
    elapsed field can be blank for pending jobs.
    """
    if not t:
        return 0.0
    s = 0.0
    if "-" in t:
        days, t = t.split("-", 1)
        try:
            s += int(days) * 86400
        except ValueError:
            return 0.0
    parts = [int(x) for x in t.split(":") if x.isdigit()]
    while len(parts) < 3:
        parts.insert(0, 0)
    s += parts[0] * 3600 + parts[1] * 60 + parts[2]
    return s
