"""FASRC job tracking, sbatch templating, ETA heuristics.

Two layers:

  * :class:`JobDB` — persistent record of every job we've submitted from
    this UI. Sqlite at ``~/.euclid_polish/fasrc_jobs.db``. Survives
    Flask restarts so the ETA model has history to draw from.

  * :func:`build_sbatch_script` — assembles a one-shot SLURM script from
    the user's parameter form. We submit a fresh script per job
    (heredoc-streamed over SSH) rather than reusing ``fasrc_train.sh``
    so the UI's job vs. on-disk script never drift.

The ETA model is intentionally simple: median seconds-per-step across
the user's last N completed jobs, multiplied by their requested step
count. If the in-flight job's log emits ``step X/Y`` we refine the ETA
live.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sqlite3
import statistics
import textwrap
import time
from typing import Any, Dict, List, Optional

from euclid_polish.web import fasrc_config

DB_DIR  = fasrc_config.CONFIG_DIR
DB_PATH = os.path.join(DB_DIR, "fasrc_jobs.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS fasrc_jobs (
    jobid           TEXT PRIMARY KEY,
    submitted_at    REAL,
    label           TEXT,
    params_json     TEXT,
    script_path     TEXT,
    log_path        TEXT,
    err_path        TEXT,
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
    ):
        col_name = col_def.split()[0]
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

    def insert(self, jobid: str, *, label: str, params: Dict[str, Any],
               script_path: str, log_path: str, err_path: str) -> None:
        with self._conn() as c:
            c.execute(
                """
                INSERT OR REPLACE INTO fasrc_jobs
                  (jobid, submitted_at, label, params_json, script_path,
                   log_path, err_path, state, last_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)
                """,
                (jobid, time.time(), label, json.dumps(params),
                 script_path, log_path, err_path, time.time()),
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


# ---------------------------------------------------------------------------
# ETA heuristic
# ---------------------------------------------------------------------------

def _params_of(row: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return json.loads(row.get("params_json") or "{}")
    except json.JSONDecodeError:
        return {}


def secs_per_step_history(n: int = 8) -> Optional[float]:
    """Median wall-second-per-training-step across the last ``n`` finished jobs."""
    samples = []
    for row in DB.list_completed(limit=n):
        params = _params_of(row)
        steps = params.get("steps") or row.get("progress_total") or 0
        runtime = (row.get("ended_at") or 0) - (row.get("started_at") or 0)
        if runtime <= 0 or steps <= 0:
            continue
        samples.append(runtime / float(steps))
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
# sbatch template
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Submission presets
# ---------------------------------------------------------------------------
#
# The UI lets the user pick one of these instead of hand-tuning every
# resource knob. The CPU-only presets skip --gres=gpu entirely (some
# SLURM configs reject ``gpu:0``) and append the right ``--skip-*`` flag
# so they only run the stages they need. Resource defaults are tuned
# from the OOM that hit the user at 510² × 4-channel × 6400 images.

PRESETS: Dict[str, Dict[str, Any]] = {
    "gen_convolve": {
        "label":          "Generate + convolve (CPU)",
        "partition":      "shared",
        "n_gpus":         0,
        "n_cpus":         16,
        "memory":         "64G",
        "time_limit":     "6:00:00",
        "skip_flags":     "--skip-train",
        "needs_train_knobs": False,
    },
    "convolve_only": {
        "label":          "Convolve existing clean → dirty (CPU)",
        "partition":      "shared",
        "n_gpus":         0,
        "n_cpus":         8,
        "memory":         "32G",
        "time_limit":     "2:00:00",
        "skip_flags":     "--skip-generate --skip-train",
        "needs_train_knobs": False,
    },
    "train_only": {
        "label":          "Train (GPU)",
        "partition":      "gpu",
        "n_gpus":         1,
        "n_cpus":         4,
        "memory":         "32G",
        "time_limit":     "24:00:00",
        "skip_flags":     "--skip-generate --skip-convolve",
        "needs_train_knobs": True,
    },
    "custom": {
        "label":          "Custom (use form values, no auto --skip-* flags)",
        "skip_flags":     "",
        "needs_train_knobs": True,
    },
}


def resolve_preset(name: str) -> Dict[str, Any]:
    """Return the preset dict for ``name`` (falls back to ``custom``)."""
    return PRESETS.get(name) or PRESETS["custom"]


def build_sbatch_script(*, label: str, params: Dict[str, Any],
                        cfg: fasrc_config.FasrcConfig,
                        relative_log_dir: str = "logs/jobs") -> Dict[str, str]:
    """Return the sbatch script body + the in-repo log/script paths.

    ``params`` keys (all required; UI supplies defaults from ``cfg``):
      - n_gpus, n_cpus, memory, time_limit, partition
      - n_train, n_valid, image_size, batch_size, steps
      - extra_flags (free-form string appended to run_pipeline.py)
    """
    ts = time.strftime("%Y%m%d-%H%M%S")
    job_name  = f"euclid-{ts}"
    script_rel = f"{relative_log_dir}/{job_name}.sh"
    out_rel    = f"{relative_log_dir}/{job_name}.out"
    err_rel    = f"{relative_log_dir}/{job_name}.err"

    p = params
    extra = (p.get("extra_flags") or "").strip()
    safe_label = label.replace("\n", " ").replace("'", "")[:200]

    # Omit --gres entirely when the user asked for 0 GPUs. SLURM configs
    # vary on whether ``--gres=gpu:0`` is accepted; the safe form is no
    # --gres line at all.
    n_gpus = int(p['n_gpus'])
    gres_line = f"#SBATCH --gres=gpu:{n_gpus}\n        " if n_gpus > 0 else ""

    body = textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name={shlex.quote(job_name)}
        #SBATCH --partition={shlex.quote(p['partition'])}
        {gres_line}#SBATCH --cpus-per-task={int(p['n_cpus'])}
        #SBATCH --mem={p['memory']}
        #SBATCH --time={p['time_limit']}
        #SBATCH --output={out_rel}
        #SBATCH --error={err_rel}

        set -euo pipefail
        cd "$SLURM_SUBMIT_DIR"
        mkdir -p {relative_log_dir}

        echo "============================================================"
        echo "Web-submitted job: {safe_label}"
        echo "Job id:   ${{SLURM_JOB_ID:-local}}"
        echo "Host:     $(hostname)"
        echo "Started:  $(date)"
        echo "Workdir:  $(pwd)"
        echo "GPUs:"
        nvidia-smi --query-gpu=name,driver_version,memory.total \\
                   --format=csv 2>/dev/null || true
        echo "============================================================"

        export EUCLID_POLISH_DATA_DIR={shlex.quote(cfg.data_dir)}
        export EUCLID_POLISH_CKPT_DIR={shlex.quote(cfg.ckpt_dir)}
        mkdir -p "$EUCLID_POLISH_DATA_DIR" "$EUCLID_POLISH_CKPT_DIR"

        module purge
        module load python
        module load cuda

        if [ -z "${{CONDA_SHLVL:-}}" ]; then
          CONDA_BASE="$(conda info --base 2>/dev/null || true)"
          if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
            # shellcheck disable=SC1091
            source "$CONDA_BASE/etc/profile.d/conda.sh"
          fi
          if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/mamba.sh" ]; then
            # shellcheck disable=SC1091
            source "$CONDA_BASE/etc/profile.d/mamba.sh"
          fi
        fi
        mamba activate {shlex.quote(cfg.conda_env_path)}

        echo "Python:  $(which python)"
        python -u scripts/run_pipeline.py \\
          --ntrain {int(p['n_train'])} \\
          --nvalid {int(p['n_valid'])} \\
          --image-size {int(p['image_size'])} \\
          --batch-size {int(p['batch_size'])} \\
          --steps {int(p['steps'])} {extra}

        echo "============================================================"
        echo "Finished: $(date)"
        echo "============================================================"
    """)
    return {
        "body":     body,
        "script":   script_rel,
        "out":      out_rel,
        "err":      err_rel,
        "name":     job_name,
    }


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


def reconcile_with_squeue(squeue_rows: List[Dict[str, Any]],
                          *, db: Optional["JobDB"] = None,
                          recent_limit: int = 50) -> Dict[str, str]:
    """Cross-check the JobDB against a live ``squeue`` snapshot.

    For every non-terminal DB row:

      * if its jobid IS in ``squeue_rows`` → set the DB state to whatever
        squeue says it is (RUNNING / PENDING / FAILED / …);
      * if its jobid is NOT in ``squeue_rows`` and the row has a
        ``started_at`` → the job ran and has since finished, mark
        ``DONE`` with ``ended_at = now``;
      * if its jobid is NOT in ``squeue_rows`` and ``started_at`` is
        missing → we never saw it start *and* it isn't queued anywhere
        we can ask about, so mark ``UNKNOWN``. That's the failure mode
        the user complained about — DB said RUNNING but squeue had
        never heard of the job.

    Returns a dict ``{jobid: new_state}`` for every row whose state we
    changed, which the caller can use to log/debug.

    ``db`` defaults to the module-level :data:`DB` singleton; tests
    pass an isolated JobDB.
    """
    target_db = db if db is not None else DB
    live_state: Dict[str, str] = {r["jobid"]: r.get("state", "?")
                                  for r in squeue_rows}
    changes: Dict[str, str] = {}

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
            continue

        # jobid is not in live squeue → finalise it.
        if stored.get("started_at"):
            target_db.update_state(jobid, state="DONE",
                                   ended_at=time.time())
            changes[jobid] = "DONE"
        else:
            # Never seen running, and squeue doesn't know about it now —
            # we have no information, treat as a failure so the user
            # sees the row instead of it lingering as RUNNING forever.
            target_db.update_state(jobid, state="UNKNOWN",
                                   ended_at=time.time())
            changes[jobid] = "UNKNOWN"

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
