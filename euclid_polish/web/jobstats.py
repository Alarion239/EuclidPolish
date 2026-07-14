"""FASRC ``jobstats`` client and text-report parser.

FASRC exposes Jobstats as a login-node command rather than an HTTP API.  The
command produces a human-readable report with the authoritative job-scoped
CPU, CPU-memory, GPU, and GPU-memory aggregates.  This module keeps the
format-specific parsing isolated from the job lifecycle code and returns the
same flat-stat shape that :mod:`euclid_polish.web.sacct` uses.

The parser is deliberately tolerant: Jobstats output varies slightly between
versions and cluster configurations, and a missing optional section should
not discard the overall utilization values.
"""

from __future__ import annotations

import json
import re
import shlex
from datetime import UTC, datetime
from typing import Any, Protocol


class _SSHRunner(Protocol):
    def run(self, cmd: str, *, timeout: float = ...) -> tuple[int, str, str]: ...
    def is_connected(self) -> bool: ...


def build_jobstats_command(jobid: str, *, cluster: str | None = None) -> str:
    """Return the shell command used to request one Jobstats report."""
    command = f"jobstats {shlex.quote(str(jobid))}"
    if cluster:
        command += f" -c {shlex.quote(str(cluster))}"
    return command


def fetch_jobstats_stats(
    ssh: _SSHRunner,
    jobid: str,
    *,
    cluster: str | None = None,
    timeout: float = 30,
) -> dict[str, Any] | None:
    """Fetch and parse a Jobstats report, or return ``None`` on failure."""
    if ssh is None or not ssh.is_connected():
        return None
    try:
        rc, out, _err = ssh.run(
            build_jobstats_command(jobid, cluster=cluster), timeout=timeout,
        )
    except Exception:
        return None
    if rc != 0 or not out.strip():
        return None
    stats = parse_jobstats_output(out)
    if not stats:
        return None
    stats["accounting_source"] = "jobstats"
    stats["jobstats_collected_at"] = _utc_now_iso()
    return stats


def parse_jobstats_output(text: str) -> dict[str, Any]:
    """Parse a Jobstats text report into normalized post-mortem fields.

    The returned keys are intentionally namespaced with ``jobstats_`` for
    utilization-specific values.  Lifecycle fields use the same names as
    :class:`~euclid_polish.observability.job_log.JobRecord` so the result can
    be merged with ``sacct`` data without another translation layer.
    """
    if not text or "Slurm Job Statistics" not in text:
        return {}

    lines = text.splitlines()
    out: dict[str, Any] = {}

    scalar_patterns: tuple[tuple[str, str, str], ...] = (
        ("state", r"^\s*State:\s*(.+?)\s*$", "text"),
        ("jobstats_cluster", r"^\s*Cluster:\s*(.+?)\s*$", "text"),
        ("jobstats_job_name", r"^\s*Job Name:\s*(.+?)\s*$", "text"),
        ("jobstats_qos_partition", r"^\s*QOS/Partition:\s*(.+?)\s*$", "text"),
        ("jobstats_started_at", r"^\s*Start Time:\s*(.+?)\s*$", "text"),
    )
    for key, pattern, _kind in scalar_patterns:
        match = _search_line(lines, pattern)
        if match:
            out[key] = match.group(1).strip()

    for key, pattern in (
        ("alloc_cpus", r"^\s*CPU Cores:\s*(\d+)"),
        ("alloc_gpus", r"^\s*GPUs:\s*(\d+)"),
        ("jobstats_nodes", r"^\s*Nodes:\s*(\d+)"),
    ):
        match = _search_line(lines, pattern)
        if match:
            out[key] = int(match.group(1))

    memory_line = _search_line(lines, r"^\s*CPU Memory:\s*(.+?)\s*$")
    if memory_line:
        memory_match = re.match(r"\s*([^\s(]+)", memory_line.group(1))
        if memory_match:
            out["alloc_memory_mb"] = _parse_size_mb(memory_match.group(1))

    runtime_line = _search_line(lines, r"^\s*Run Time:\s*(.+?)\s*$")
    if runtime_line:
        out["elapsed_seconds"] = _parse_duration_seconds(runtime_line.group(1))
    limit_line = _search_line(lines, r"^\s*Time Limit:\s*(.+?)\s*$")
    if limit_line:
        out["jobstats_time_limit_seconds"] = _parse_duration_seconds(limit_line.group(1))

    overall_patterns = (
        ("jobstats_cpu_util", "CPU utilization"),
        ("jobstats_cpu_memory_util", "CPU memory usage"),
        ("jobstats_gpu_util", "GPU utilization"),
        ("jobstats_gpu_memory_util", "GPU memory usage"),
    )
    for key, label in overall_patterns:
        match = _search_line(
            lines,
            rf"^\s*{re.escape(label)}\s*\[[^\]]*?([0-9]+(?:\.[0-9]+)?)%\]",
        )
        if match:
            out[key] = float(match.group(1))

    cpu_nodes = _parse_cpu_nodes(lines)
    if cpu_nodes:
        out["jobstats_cpu_nodes_json"] = json.dumps(
            cpu_nodes, ensure_ascii=False, separators=(",", ":"),
        )
        used = [n["efficiency_percent"] for n in cpu_nodes
                if n.get("efficiency_percent") is not None]
        if "jobstats_cpu_util" not in out and used:
            out["jobstats_cpu_util"] = sum(used) / len(used)

    cpu_memory_nodes = _parse_cpu_memory_nodes(lines)
    if cpu_memory_nodes:
        out["jobstats_cpu_memory_nodes_json"] = json.dumps(
            cpu_memory_nodes, ensure_ascii=False, separators=(",", ":"),
        )
        total_used = sum(n["used_mb"] for n in cpu_memory_nodes
                          if n.get("used_mb") is not None)
        total_alloc = sum(n["allocated_mb"] for n in cpu_memory_nodes
                          if n.get("allocated_mb") is not None)
        if total_alloc > 0:
            out.setdefault("jobstats_cpu_memory_used_mb", total_used)
            out.setdefault("jobstats_cpu_memory_alloc_mb", total_alloc)
            out.setdefault("jobstats_cpu_memory_util",
                           100.0 * total_used / total_alloc)

    gpu_nodes = _parse_gpu_util_nodes(lines)
    gpu_memory_nodes = _parse_gpu_memory_nodes(lines)
    if gpu_nodes:
        out["jobstats_gpu_nodes_json"] = json.dumps(
            gpu_nodes, ensure_ascii=False, separators=(",", ":"),
        )
        values = [n["util_percent"] for n in gpu_nodes
                  if n.get("util_percent") is not None]
        if "jobstats_gpu_util" not in out and values:
            out["jobstats_gpu_util"] = sum(values) / len(values)
    if gpu_memory_nodes:
        out["jobstats_gpu_memory_nodes_json"] = json.dumps(
            gpu_memory_nodes, ensure_ascii=False, separators=(",", ":"),
        )
        used = sum(n["used_mb"] for n in gpu_memory_nodes
                   if n.get("used_mb") is not None)
        total = sum(n["total_mb"] for n in gpu_memory_nodes
                    if n.get("total_mb") is not None)
        if total > 0:
            out.setdefault("jobstats_gpu_memory_used_mb", used)
            out.setdefault("jobstats_gpu_memory_total_mb", total)
            out.setdefault("jobstats_gpu_memory_util", 100.0 * used / total)

    notes = _parse_notes(lines)
    if notes:
        out["jobstats_notes_json"] = json.dumps(
            notes, ensure_ascii=False, separators=(",", ":"),
        )

    # A valid report with only lifecycle data is still useful, but reject a
    # shell error or an unrelated banner that happens to contain the title.
    return out if out.get("state") or "jobstats_cpu_util" in out else {}


def _search_line(lines: list[str], pattern: str) -> re.Match[str] | None:
    compiled = re.compile(pattern)
    return next((match for line in lines if (match := compiled.match(line))), None)


def _parse_cpu_nodes(lines: list[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    in_section = False
    for line in lines:
        if line.strip().startswith("CPU utilization per node"):
            in_section = True
            continue
        if in_section and line.strip().startswith("CPU memory usage per node"):
            break
        if not in_section:
            continue
        match = re.match(
            r"^\s*(\S+):\s*(\S+)/(\S+)\s+\(efficiency=([0-9.]+)%\)",
            line,
        )
        if not match:
            continue
        result.append({
            "node": match.group(1),
            "used_seconds": _parse_duration_seconds(match.group(2)),
            "allocated_seconds": _parse_duration_seconds(match.group(3)),
            "efficiency_percent": float(match.group(4)),
        })
    return result


def _parse_cpu_memory_nodes(lines: list[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    in_section = False
    for line in lines:
        if line.strip().startswith("CPU memory usage per node"):
            in_section = True
            continue
        if in_section and line.strip().startswith("GPU utilization per node"):
            break
        if not in_section:
            continue
        match = re.match(r"^\s*(\S+):\s*(\S+)/(\S+)\s+\(", line)
        if not match:
            continue
        result.append({
            "node": match.group(1),
            "used_mb": _parse_size_mb(match.group(2)),
            "allocated_mb": _parse_size_mb(match.group(3)),
        })
    return result


def _parse_gpu_util_nodes(lines: list[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    in_section = False
    for line in lines:
        if line.strip().startswith("GPU utilization per node"):
            in_section = True
            continue
        if in_section and line.strip().startswith("GPU memory usage per node"):
            break
        if not in_section:
            continue
        match = re.match(r"^\s*(\S+)\s+\((GPU\s+[^)]+)\):\s*([0-9.]+)%", line)
        if match:
            result.append({
                "node": match.group(1),
                "gpu": match.group(2),
                "util_percent": float(match.group(3)),
            })
    return result


def _parse_gpu_memory_nodes(lines: list[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    in_section = False
    for line in lines:
        if line.strip().startswith("GPU memory usage per node"):
            in_section = True
            continue
        if in_section and line.strip().startswith("Notes"):
            break
        if not in_section:
            continue
        match = re.match(
            r"^\s*(\S+)\s+\((GPU\s+[^)]+)\):\s*(\S+)/(\S+)\s+\(([0-9.]+)%\)",
            line,
        )
        if match:
            result.append({
                "node": match.group(1),
                "gpu": match.group(2),
                "used_mb": _parse_size_mb(match.group(3)),
                "total_mb": _parse_size_mb(match.group(4)),
                "util_percent": float(match.group(5)),
            })
    return result


def _parse_notes(lines: list[str]) -> list[str]:
    notes: list[str] = []
    in_section = False
    for line in lines:
        if line.strip() == "Notes":
            in_section = True
            continue
        if not in_section:
            continue
        stripped = line.strip()
        if stripped.startswith("*"):
            notes.append(stripped[1:].strip())
        elif notes and stripped and not set(stripped) <= {"="}:
            notes[-1] += " " + stripped
    return notes


_SIZE_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?i?B?)?\s*$", re.I)
_SIZE_MB = {"": 1 / 1024 / 1024, "B": 1 / 1024 / 1024,
            "K": 1 / 1024, "KB": 1 / 1024, "KIB": 1 / 1024,
            "M": 1, "MB": 1, "MIB": 1,
            "G": 1024, "GB": 1024, "GIB": 1024,
            "T": 1024 * 1024, "TB": 1024 * 1024, "TIB": 1024 * 1024}


def _parse_size_mb(value: str) -> float | None:
    match = _SIZE_RE.match(str(value))
    if not match:
        return None
    return float(match.group(1)) * _SIZE_MB.get((match.group(2) or "").upper(), 1.0)


def _parse_duration_seconds(value: str) -> float | None:
    raw = str(value).strip()
    if not raw or raw in {"Unknown", "None", "N/A"}:
        return None
    days = 0.0
    if "-" in raw:
        day_text, raw = raw.split("-", 1)
        try:
            days = float(day_text)
        except ValueError:
            return None
    try:
        parts = [float(part) for part in raw.split(":")]
    except ValueError:
        return None
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours, minutes, seconds = 0.0, parts[0], parts[1]
    elif len(parts) == 1:
        hours, minutes, seconds = 0.0, 0.0, parts[0]
    else:
        return None
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
