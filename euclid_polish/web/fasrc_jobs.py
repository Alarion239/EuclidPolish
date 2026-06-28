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

import contextlib
import json
import os
import re
import shlex
import sqlite3
import statistics
import time
from typing import Any

from euclid_polish.observability import JobLog, JobRecord
from euclid_polish.tracking import default_store
from euclid_polish.web import fasrc_config
from euclid_polish.web.job_status import fold_events
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
        params:      dict[str, Any],
        script_path: str,
        log_path:    str,
        err_path:    str,
        events_path: str | None = None,
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
                     started_at: float | None = None,
                     ended_at: float | None = None,
                     clear_ended: bool = False) -> None:
        sets, args = ["state = ?", "last_seen = ?"], [state, time.time()]
        if started_at is not None:
            sets.append("started_at = COALESCE(started_at, ?)")
            args.append(started_at)
        if clear_ended:
            # Resurrecting a wrongly-finalised job back to a live state — drop
            # the stale ended_at so the row reads as genuinely in-flight.
            sets.append("ended_at = NULL")
        elif ended_at is not None:
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

    def set_step_id(self, jobid: str, step_id: str) -> None:
        """Tag a job with its HST-pipeline step id (for per-step history)."""
        with self._conn() as c:
            c.execute("UPDATE fasrc_jobs SET step_id = ?, last_seen = ? "
                      "WHERE jobid = ?",
                      (str(step_id), time.time(), jobid))

    def get(self, jobid: str) -> dict[str, Any] | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM fasrc_jobs WHERE jobid = ?",
                          (jobid,)).fetchone()
        return dict(r) if r else None

    def list_recent(self, limit: int = 30) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM fasrc_jobs ORDER BY submitted_at DESC "
                "LIMIT ?", (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_completed(self, limit: int = 10) -> list[dict[str, Any]]:
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

def _params_of(row: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(row.get("params_json") or "{}")
    except json.JSONDecodeError:
        return {}


def secs_per_step_history(n: int = 8) -> float | None:
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


def eta_for_submission(steps: int) -> float | None:
    """Rough wall-time ETA in seconds for a fresh job of ``steps`` steps."""
    spt = secs_per_step_history()
    if spt is None:
        return None
    return spt * steps


def eta_for_running(row: dict[str, Any]) -> float | None:
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

_STEP_ID_RE = re.compile(r"STEP_ID=([A-Za-z0-9_\-]+)")


def parse_progress(line: str) -> tuple[int, int] | None:
    """Pull (step, total) out of a single log line, or None."""
    m = _TQDM_PROGRESS_RE.search(line)
    if not m:
        return None
    step, total = int(m.group(1)), int(m.group(2))
    # Guard against false positives like "shape 4/4" in module prints.
    if total < 50 or step > total:
        return None
    return step, total


# ---------------------------------------------------------------------------
# Submission helper — single SSH-write + sbatch + parse + DB.insert flow
# ---------------------------------------------------------------------------

# Marker for the heredoc that streams the script over SSH. Single source
# of truth — both submit handlers used to hard-code their own copy.
_HEREDOC_EOF = "__EUCLID_POLISH_EOF__"

_SBATCH_JOBID_RE = re.compile(r"Submitted batch job (\d+)")


def _conda_activate_snippet(env_path: str, load_cuda: bool = False) -> str:
    """Return a zero-indented bash block that loads modules and activates a conda env.

    Generates ``module load python`` (plus ``module load cuda`` when
    ``load_cuda=True``) followed by a ``CONDA_SHLVL``-gated conda/mamba
    initialization and ``mamba activate``. Used by both sbatch scripts
    (:func:`fasrc_pipeline.render_sbatch_body`) and login-node helper
    commands (:func:`jobs_impl._login_node_generate_cmd`) to eliminate
    duplicate inline shell snippets.
    """
    env = shlex.quote(env_path)
    cuda_line = "\nmodule load cuda" if load_cuda else ""
    lines = [
        f"module load python{cuda_line}",
        "",
        'if [ -z "${CONDA_SHLVL:-}" ]; then',
        '  CONDA_BASE="$(conda info --base 2>/dev/null || true)"',
        '  if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then',
        '    source "$CONDA_BASE/etc/profile.d/conda.sh"',
        '  fi',
        '  if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/mamba.sh" ]; then',
        '    source "$CONDA_BASE/etc/profile.d/mamba.sh"',
        '  fi',
        'fi',
        f"mamba activate {env}",
    ]
    return "\n".join(lines)


def build_remote_python_command(
    cfg: fasrc_config.FasrcConfig, argv: list[str],
) -> str:
    """Build the ``bash -lc '…'`` string that runs a project script on the
    FASRC **login node** (no SLURM).

    Activates the conda/mamba env exactly like the sbatch template's setup
    block (so it matches what already works on the cluster), exports
    ``EUCLID_POLISH_DATA_DIR`` / ``EUCLID_POLISH_CKPT_DIR`` so the script
    reads/writes the shared netscratch paths, ``cd``s into the remote repo,
    and runs ``python -u <argv>``. ``argv`` is repo-relative (e.g.
    ``["scripts/query_brightest_stars.py", "--num-stars", "200"]``).

    A login shell (``bash -l``) is used so the user's conda init runs; the
    explicit source-conda block is a fallback for non-init shells. The
    whole inner command is single-quoted via :func:`shlex.quote`, so the
    embedded ``$(…)`` / ``$CONDA_BASE`` are evaluated remotely, not locally.
    """
    env = shlex.quote(cfg.conda_env_path)
    py_cmd = "python -u " + " ".join(shlex.quote(a) for a in argv)
    inner = (
        # ``cd`` is the one strict step — bail loudly (127) if the repo
        # path is wrong rather than running python in the wrong place.
        f"cd {shlex.quote(cfg.repo_path)} || exit 127; "
        f"export EUCLID_POLISH_DATA_DIR={shlex.quote(cfg.data_dir)}; "
        f"export EUCLID_POLISH_CKPT_DIR={shlex.quote(cfg.ckpt_dir)}; "
        # Match the sbatch template: load python, then source conda+mamba
        # *unconditionally* (NOT gated on CONDA_SHLVL). A login shell often
        # has conda's base already active — which defines ``conda`` but not
        # ``mamba`` — so gating on CONDA_SHLVL left ``mamba`` undefined and
        # ``mamba activate`` exited 127. Sourcing every time fixes that.
        "module load python >/dev/null 2>&1 || true; "
        'CB="$(conda info --base 2>/dev/null || true)"; '
        'if [ -n "$CB" ] && [ -f "$CB/etc/profile.d/conda.sh" ]; then . "$CB/etc/profile.d/conda.sh"; fi; '
        'if [ -n "$CB" ] && [ -f "$CB/etc/profile.d/mamba.sh" ]; then . "$CB/etc/profile.d/mamba.sh"; fi; '
        # Prefer mamba (matches sbatch) but fall back to conda if the
        # install doesn't ship the mamba shell function.
        f"if command -v mamba >/dev/null 2>&1; then mamba activate {env}; else conda activate {env}; fi && "
        f"{py_cmd}"
    )
    return "bash -lc " + shlex.quote(inner)


def run_remote_python(
    ssh, *, cfg: fasrc_config.FasrcConfig, argv: list[str], timeout: int = 300,
) -> tuple[int, str, str]:
    """Run a project Python script on the FASRC login node over SSH.

    Synchronous (blocks up to ``timeout`` s) — meant for quick work like a
    catalog archive query, not heavy compute (that goes through
    :func:`submit_sbatch_script`). Returns ``(rc, stdout, stderr)``.
    """
    cmd = build_remote_python_command(cfg, argv)
    return ssh.run(cmd, timeout=timeout)


def submit_sbatch_script(
    ssh,
    *,
    cfg:      fasrc_config.FasrcConfig,
    built:    dict[str, str],
    label:    str,
    params:   dict[str, Any],
    step_id:  str | None = None,
) -> tuple[str | None, dict[str, Any]]:
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
    try:  # noqa: SIM105 — suppress() would awkwardly wrap a 25-line JobRecord(...)
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

    # Record the submission in the active tracking campaign's job log so the
    # "lab notebook" captures every job + its params (req: all FASRC jobs
    # logged with parameters). Best-effort: a missing campaign falls back to
    # tracking/unassigned_fasrc_jobs.jsonl, and any failure is swallowed so
    # tracking never breaks a submit.
    with contextlib.suppress(Exception):
        default_store().log_fasrc_job({
            "jobid":       slurm_id,
            "step_id":     step_id or "",
            "label":       label,
            "params":      params,
            "script_path": remote_script,
            "log_path":    log_path,
            "err_path":    err_path,
            "events_path": events_path or "",
        })

    payload: dict[str, Any] = {
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

def parse_squeue(text: str) -> list[dict[str, str]]:
    """Parse our fixed-format ``squeue`` output.

    We invoke squeue with ``--format`` and a pipe separator. The literal
    ``\\t`` in modern SLURM's format string is NOT expanded — it shows up
    as the two characters ``\t`` in the output, which silently broke the
    earlier tab-based split. Pipes are passed through literally.
    """
    rows: list[dict[str, str]] = []
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
        rows.append(dict(zip(keys, parts[: len(keys)], strict=False)))
    return rows


SQUEUE_FMT = "%i|%j|%T|%M|%l|%D|%R|%S"


# Terminal states — once a row reaches any of these we stop reconciling
# it against squeue. ``UNKNOWN`` is here too: it means "this job was
# tracked but disappeared from squeue without ever showing started_at",
# so we treat it as a failure mode and leave it alone.
TERMINAL_STATES = frozenset({
    "COMPLETED", "DONE", "FAILED", "CANCELLED", "TIMEOUT", "UNKNOWN",
})

#: Terminal states reconcile assigns *speculatively* from "absent from squeue"
#: (as opposed to the authoritative FAILED/CANCELLED/TIMEOUT/COMPLETED reported
#: by squeue/sacct). These are revocable: if such a job reappears in squeue
#: alive, squeue is the live source of truth and we un-finalise it. Without
#: this, a job that briefly vanished from one squeue snapshot (a controller
#: hiccup, a slow/empty squeue) stays terminal forever — the DB-driven
#: "current submission" view drops it while the squeue-driven sidebar still
#: shows it RUNNING.
SPECULATIVE_TERMINAL = frozenset({"DONE", "UNKNOWN"})

#: Grace period (seconds) before a submitted-but-never-seen-in-squeue job is
#: flagged UNKNOWN. ``sbatch`` returns before the controller reliably lists the
#: job, so a fresh job briefly absent from squeue is normal, not lost.
SUBMIT_GRACE_S = 120.0


def sync_pending_on_connect(
    ssh: Any,
    *,
    db: JobDB | None = None,
    job_log: JobLog | None = None,
    recent_limit: int = 50,
) -> dict[str, str]:
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


def fetch_resource_summary(ssh: Any, events_path: str | None) -> dict[str, Any]:
    """Fold a job's ``resource`` samples into post-mortem summary fields.

    ``cat``s the remote events file, folds it (reusing the same
    :func:`fold_events` the live status uses) and projects the
    :class:`ResourceUsage` aggregates onto :class:`JobRecord`'s
    ``gpu_util_*`` / ``gpu_mem_peak`` / ``cpu_util_*`` columns. Returns
    ``{}`` when SSH is down, the file is missing, or the job emitted no
    samples — so a job from before this feature simply leaves the columns
    blank. Percentages are rounded for a tidy CSV.
    """
    if not events_path or ssh is None or not ssh.is_connected():
        return {}
    try:
        rc, out, _err = ssh.run(
            f"cat {events_path} 2>/dev/null || true", timeout=10,
        )
    except Exception:
        return {}
    if rc != 0 or not out:
        return {}
    res = fold_events(out).resources
    if res is None or res.n_samples == 0:
        return {}

    def _r(x: float | None) -> str | None:
        return None if x is None else f"{x:.1f}"

    stats = {
        "gpu_util_mean": _r(res.gpu_mean),
        "gpu_util_peak": _r(res.gpu_peak),
        "gpu_mem_peak":  _r(res.gpu_mem_peak),
        "cpu_util_mean": _r(res.cpu_mean),
        "cpu_util_peak": _r(res.cpu_peak),
    }
    return {k: v for k, v in stats.items() if v is not None}


def refresh_all_post_mortems(
    ssh: Any, *, job_log: JobLog | None = None,
) -> dict[str, Any]:
    """Re-pull sacct for every finalised job in the CSV log and re-record.

    One-shot maintenance: when the way a post-mortem field is *computed*
    changes (e.g. cpu_efficiency switching from the always-1.0 CPUTimeRAW
    to real TotalCPU), rows already written hold the stale value and the
    normal reconcile won't re-fetch them (it only retries rows with a blank
    state). This re-queries sacct — authoritative — for each terminal job
    and overwrites the actuals. Jobs sacct has since purged are skipped
    (their old values stay). Returns ``{ok, updated, total}``.
    """
    target_log = job_log if job_log is not None else JOBLOG
    if ssh is None or not ssh.is_connected():
        return {"ok": False, "error": "not connected", "updated": 0, "total": 0}
    candidates = [
        r for r in target_log.list_all()
        if r.get("jobid") and (r.get("state") or "").strip() in TERMINAL_STATES
    ]
    updated = 0
    for r in candidates:
        jobid = r["jobid"]
        try:
            stats = fetch_sacct_stats(ssh, jobid)
        except Exception:
            stats = None
        if stats:
            # Fold the events stream's resource samples in alongside the
            # sacct actuals so the history panel keeps GPU/CPU util.
            stats.update(fetch_resource_summary(ssh, r.get("events_path")))
            if target_log.record_post_mortem(jobid, stats):
                updated += 1
    return {"ok": True, "updated": updated, "total": len(candidates)}


def reconcile_with_squeue(squeue_rows: list[dict[str, Any]],
                          *, db: JobDB | None = None,
                          recent_limit: int = 50,
                          ssh: Any | None = None,
                          job_log: JobLog | None = None) -> dict[str, str]:
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
    live_state: dict[str, str] = {r["jobid"]: r.get("state", "?")
                                  for r in squeue_rows}
    changes: dict[str, str] = {}
    just_finalised: list[str] = []

    db_rows = list(target_db.list_recent(recent_limit))
    db_state_before = {s["jobid"]: (s.get("state") or "") for s in db_rows}
    for stored in db_rows:
        jobid = stored["jobid"]
        cur   = stored.get("state") or ""
        live  = live_state.get(jobid)
        alive = live in ("RUNNING", "PENDING")

        if cur in TERMINAL_STATES:
            # A speculatively-finalised job (DONE/UNKNOWN, assigned from a
            # prior "absent from squeue" pass) that is alive again in squeue
            # was finalised in error — squeue is the live truth, so fall
            # through and re-sync it. Authoritative terminal states stay put.
            if not (cur in SPECULATIVE_TERMINAL and alive):
                continue
            resurrecting = True
        else:
            resurrecting = False

        if jobid in live_state:
            if live == "RUNNING":
                target_db.update_state(
                    jobid, state="RUNNING",
                    started_at=time.time() - parse_slurm_time(
                        next((r.get("time") for r in squeue_rows
                              if r["jobid"] == jobid), "")
                    ),
                    clear_ended=resurrecting,
                )
            else:
                target_db.update_state(jobid, state=live,
                                       clear_ended=resurrecting)
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
            # A job we never saw start may simply not be in squeue *yet*
            # (sbatch returns before the controller reliably lists it). Only
            # flag UNKNOWN once it's been missing well past submission, so a
            # fresh submission isn't killed in the UI the instant it lands.
            submitted = stored.get("submitted_at") or 0.0
            if (time.time() - submitted) < SUBMIT_GRACE_S:
                continue
            target_db.update_state(jobid, state="UNKNOWN",
                                   ended_at=time.time())
            changes[jobid] = "UNKNOWN"
            just_finalised.append(jobid)

    # Post-mortem accounting. We fetch sacct for jobs that JUST finalised
    # this pass AND (re)try any recent job that is terminal in the DB but
    # whose CSV log row still has no recorded state. sacct accounting lags
    # the job's actual end by minutes — especially for CANCELLED / TIMEOUT
    # — so a single fetch at finalisation often returns nothing and would
    # leave the per-step history panel stuck on "pending" forever. Since
    # reconcile runs on every dashboard poll, retrying here fills the row
    # in as soon as sacct catches up.
    needs_pm: list[str] = list(just_finalised)
    for jid, stt in db_state_before.items():
        if jid in needs_pm or stt not in TERMINAL_STATES:
            continue
        row = target_log.get(jid)
        if row is not None and not (row.get("state") or "").strip():
            needs_pm.append(jid)

    if ssh is not None and needs_pm:
        for jobid in needs_pm:
            try:
                stats = fetch_sacct_stats(ssh, jobid)
            except Exception:
                stats = None
            if not stats:
                continue
            # Merge the events stream's resource summary (GPU/CPU util)
            # into the sacct actuals before recording.
            pm_row = target_log.get(jobid)
            stats.update(fetch_resource_summary(
                ssh, pm_row.get("events_path") if pm_row else None))
            # record_post_mortem fills the CSV row's actuals AND its real
            # terminal ``state`` (e.g. CANCELLED / TIMEOUT) from sacct —
            # squeue can't report those, so without this the per-step
            # history panel would show the job stuck on "pending".
            with contextlib.suppress(Exception):
                target_log.record_post_mortem(jobid, stats)

    return changes


def parse_slurm_time(t: str | None) -> float:
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
