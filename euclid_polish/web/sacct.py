"""SLURM ``sacct`` accounting client.

Pulls per-job resource usage from SLURM's accounting database via the
``sacct`` command over SSH, then folds the multi-row output into one
flat dict that :class:`euclid_polish.observability.JobLog` can persist.

Why this is non-trivial
-----------------------
``sacct`` emits one row per *job step*, not per job. A typical batch
job ``12345`` will have three rows:

  * ``12345`` — top-level job record (Start / End / Elapsed / State /
    ExitCode / ReqMem / AllocCPUS / AllocTRES live here).
  * ``12345.batch`` — the script body's step (MaxRSS / AveRSS /
    CPUTimeRAW typically live here, not on the top row).
  * ``12345.extern`` — SLURM-internal accounting step, usually empty.

We need fields from both the main row (for state / timing /
allocation) and the ``.batch`` row (for peak memory / CPU seconds), so
the parser folds the rows together and returns the union.

Memory formatting
-----------------
SLURM reports memory with suffixes (``K`` / ``M`` / ``G`` / ``T``) and
sometimes per-core (``Mc``) or per-node (``Mn``). :func:`_parse_mem_mb`
normalises everything to megabytes for storage.
"""

from __future__ import annotations

import re
import shlex
from typing import Any, Dict, List, Optional, Protocol, Tuple


#: Fields fetched from ``sacct``. Order is enforced via ``-o`` so we
#: can index reliably. Pipe-separated (``-P``) so values containing
#: spaces (like start/end timestamps) don't break parsing.
_SACCT_FIELDS: Tuple[str, ...] = (
    "JobID",
    "State",
    "ExitCode",
    "Start",
    "End",
    "ElapsedRaw",
    "CPUTimeRAW",
    "TotalCPU",
    "MaxRSS",
    "ReqMem",
    "ReqCPUS",
    "ReqTRES",
    "AllocCPUS",
    "AllocTRES",
    "Timelimit",
)


class _SSHRunner(Protocol):
    def run(self, cmd: str, *, timeout: float = ...) -> Tuple[int, str, str]: ...
    def is_connected(self) -> bool: ...


def build_sacct_command(jobid: str) -> str:
    """Return the exact shell command we run for one job's accounting."""
    fields = ",".join(_SACCT_FIELDS)
    # ``-P`` pipes, ``--noheader`` skips the header row, ``-j`` filters
    # to one job. ``-X`` would give us only the top-level row but we
    # need the ``.batch`` step's MaxRSS / CPUTimeRAW too — fold below.
    return f"sacct -j {shlex.quote(str(jobid))} -P --noheader -o {fields}"


def fetch_sacct_stats(ssh: _SSHRunner, jobid: str) -> Optional[Dict[str, Any]]:
    """Run ``sacct`` for ``jobid`` and return the folded post-mortem dict.

    Returns ``None`` when SSH is unavailable, when ``sacct`` errors, or
    when no rows come back (job never made it into the accounting DB,
    e.g. ``sbatch`` rejected it). The caller should leave the CSV row's
    post-mortem columns blank in that case rather than overwriting them
    with placeholders.
    """
    if ssh is None or not ssh.is_connected():
        return None
    rc, out, _err = ssh.run(build_sacct_command(jobid), timeout=15)
    if rc != 0 or not out.strip():
        return None
    return parse_sacct_output(out)


# ---------------------------------------------------------------------------
# Parser (pure, unit-testable)
# ---------------------------------------------------------------------------

def parse_sacct_output(text: str) -> Dict[str, Any]:
    """Fold one job's multi-row ``sacct -P`` output into a flat dict.

    The returned dict's keys map directly to :class:`JobRecord`
    post-mortem field names so :meth:`JobLog.record_post_mortem` can
    consume it without further translation.
    """
    rows = [
        dict(zip(_SACCT_FIELDS, line.split("|")))
        for line in text.splitlines()
        if line.strip() and "|" in line
    ]
    if not rows:
        return {}

    # The main job row's JobID is the bare numeric id; step rows append
    # ``.batch`` / ``.extern`` etc. Find both classes.
    main_row  = next((r for r in rows if "." not in r["JobID"]), rows[0])
    batch_row = next((r for r in rows if r["JobID"].endswith(".batch")), None)

    elapsed_sec = _parse_int(main_row.get("ElapsedRaw")) or 0
    # CPU time actually CONSUMED (user+system across all cores) — TotalCPU,
    # NOT CPUTimeRAW. CPUTimeRAW = Elapsed × NCPUS is merely the *allocated*
    # CPU time, so using it makes efficiency a constant 1.0. TotalCPU is
    # what ``seff`` divides by (Elapsed × NCPUS) to get real utilisation.
    # Prefer the batch step (where the work runs); fall back to the main row.
    cpu_seconds = (
        _parse_slurm_duration_secs((batch_row or {}).get("TotalCPU"))
        or _parse_slurm_duration_secs(main_row.get("TotalCPU"))
        or 0.0
    )
    alloc_cpus = _parse_int(main_row.get("AllocCPUS")) or 0
    cpu_efficiency: Optional[float] = None
    if elapsed_sec > 0 and alloc_cpus > 0:
        cpu_efficiency = cpu_seconds / (elapsed_sec * alloc_cpus)

    alloc_tres  = main_row.get("AllocTRES", "")
    alloc_gpus  = _gres_count_from_tres(alloc_tres, "gres/gpu")
    alloc_mem_mb = _tres_mem_mb(alloc_tres)

    # MaxRSS typically lives on the batch step.
    max_rss_mb = _parse_mem_mb((batch_row or {}).get("MaxRSS"))
    req_mem_mb = _parse_mem_mb(main_row.get("ReqMem"))

    return {
        "state":           _clean_state(main_row.get("State")),
        "exit_code":       (main_row.get("ExitCode") or "").strip(),
        "started_at":      _normalise_sacct_time(main_row.get("Start")),
        "ended_at":        _normalise_sacct_time(main_row.get("End")),
        "elapsed_seconds": float(elapsed_sec),
        "cpu_seconds":     float(cpu_seconds),
        "cpu_efficiency":  cpu_efficiency,
        "max_rss_mb":      max_rss_mb,
        "alloc_cpus":      alloc_cpus,
        "alloc_gpus":      alloc_gpus,
        "alloc_memory_mb": alloc_mem_mb if alloc_mem_mb is not None else req_mem_mb,
    }


# ---------------------------------------------------------------------------
# Field parsers
# ---------------------------------------------------------------------------

_STATE_TAIL_RE = re.compile(r"\s+by\s+\S+$")


def _clean_state(s: Optional[str]) -> str:
    """``"CANCELLED by 1234"`` → ``"CANCELLED"``; everything else passes through."""
    if not s:
        return ""
    return _STATE_TAIL_RE.sub("", s.strip())


def _parse_int(s: Any) -> Optional[int]:
    if s is None or s == "":
        return None
    try:
        return int(str(s).strip())
    except (TypeError, ValueError):
        return None


def _parse_slurm_duration_secs(s: Any) -> Optional[float]:
    """Parse a SLURM duration (e.g. ``TotalCPU``) into seconds.

    Formats: ``"SS.mmm"``, ``"MM:SS.mmm"``, ``"HH:MM:SS"``,
    ``"DD-HH:MM:SS"``. Returns None on empty / unparseable input."""
    if s is None:
        return None
    raw = str(s).strip()
    if not raw or raw in ("Unknown", "None", "N/A"):
        return None
    days = 0.0
    if "-" in raw:
        d, raw = raw.split("-", 1)
        try:
            days = float(d)
        except ValueError:
            return None
    try:
        parts = [float(p) for p in raw.split(":")]
    except ValueError:
        return None
    if len(parts) == 3:
        h, m, sec = parts
    elif len(parts) == 2:
        h, m, sec = 0.0, parts[0], parts[1]
    elif len(parts) == 1:
        h, m, sec = 0.0, 0.0, parts[0]
    else:
        return None
    return days * 86400.0 + h * 3600.0 + m * 60.0 + sec


_MEM_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([KMGT]?)\s*([cn]?)\s*$",
                     re.IGNORECASE)
_MEM_UNIT_MB = {"": 1.0 / 1024.0 / 1024.0,   # bytes → MB (rare)
                "K": 1.0 / 1024.0,           # KB
                "M": 1.0,                    # MB
                "G": 1024.0,                 # GB
                "T": 1024.0 * 1024.0}        # TB


def _parse_mem_mb(s: Any) -> Optional[float]:
    """Parse a SLURM memory string into MB.

    Handles ``"1234K"`` (KB), ``"5.5G"`` (GB), ``"512M"`` (MB),
    ``"8000Mc"`` (per-core suffix — we drop the suffix, the value is
    still per-allocated-unit), and bare numbers (assumed bytes).
    """
    if not s:
        return None
    m = _MEM_RE.match(str(s))
    if not m:
        return None
    val   = float(m.group(1))
    unit  = m.group(2).upper()
    return val * _MEM_UNIT_MB.get(unit, 1.0)


def _normalise_sacct_time(s: Any) -> str:
    """``sacct`` emits ``"2026-05-26T14:33:21"`` (local time) or
    ``"Unknown"``. Pass through the ISO-looking ones, drop the rest."""
    if not s:
        return ""
    raw = str(s).strip()
    if raw in ("Unknown", "None", "N/A"):
        return ""
    return raw


_TRES_GPU_RE = re.compile(r"(?:^|,)\s*gres/gpu(?:[^=,]*)=(\d+)")
_TRES_MEM_RE = re.compile(r"(?:^|,)\s*mem=([0-9]+\.?[0-9]*)([KMGT]?)",
                          re.IGNORECASE)


def _gres_count_from_tres(tres: Optional[str], key_prefix: str) -> int:
    """Count GPUs from an ``AllocTRES`` string.

    SLURM emits ``"cpu=4,gres/gpu=1,mem=8000M"`` for a one-GPU job.
    Some sites tag the GPU type as ``gres/gpu:a100=1`` — accept any
    suffix after ``gres/gpu``.
    """
    if not tres or key_prefix not in tres:
        return 0
    m = _TRES_GPU_RE.search(tres)
    return int(m.group(1)) if m else 0


def _tres_mem_mb(tres: Optional[str]) -> Optional[float]:
    """Parse the ``mem=`` field from a TRES string into MB."""
    if not tres:
        return None
    m = _TRES_MEM_RE.search(tres)
    if not m:
        return None
    val, unit = float(m.group(1)), m.group(2).upper()
    return val * _MEM_UNIT_MB.get(unit, 1.0)
