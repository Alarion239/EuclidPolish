"""Tests for the ``sacct`` parser + fetcher.

Verifies the multi-row fold (top-level + .batch step) and the various
SLURM string formats (memory suffixes, TRES, time stamps, CANCELLED-by).
"""

from __future__ import annotations

import pytest

from euclid_polish.web.sacct import (
    _SACCT_FIELDS,
    _gres_count_from_tres,
    _parse_mem_mb,
    build_sacct_command,
    fetch_sacct_stats,
    parse_sacct_output,
)

# ---------------------------------------------------------------------------
# Memory parser
# ---------------------------------------------------------------------------

class TestParseMemMB:

    @pytest.mark.parametrize("raw,mb", [
        ("0",       0.0),                # blank / zero
        ("512K",    0.5),
        ("1024K",   1.0),
        ("256M",    256.0),
        ("8000Mc",  8000.0),             # per-core suffix dropped
        ("8000Mn",  8000.0),             # per-node suffix dropped
        ("4G",      4096.0),
        ("1.5G",    1536.0),
        ("1T",      1048576.0),
    ])
    def test_canonical_forms(self, raw, mb):
        assert _parse_mem_mb(raw) == pytest.approx(mb)

    @pytest.mark.parametrize("bad", ["", None, "abc", "junk"])
    def test_unparseable_returns_none(self, bad):
        assert _parse_mem_mb(bad) is None


# ---------------------------------------------------------------------------
# TRES parser
# ---------------------------------------------------------------------------

class TestGresCount:

    def test_cpu_only_tres_returns_zero(self):
        assert _gres_count_from_tres("cpu=4,mem=8000M", "gres/gpu") == 0

    def test_one_gpu(self):
        assert _gres_count_from_tres(
            "cpu=4,gres/gpu=1,mem=8000M", "gres/gpu",
        ) == 1

    def test_two_gpus(self):
        assert _gres_count_from_tres(
            "cpu=8,gres/gpu=2,mem=64000M", "gres/gpu",
        ) == 2

    def test_typed_gpu_string(self):
        """Some sites tag the GPU type: ``gres/gpu:a100=1``."""
        assert _gres_count_from_tres(
            "cpu=8,gres/gpu:a100=1,mem=64000M", "gres/gpu",
        ) == 1

    def test_none_string_returns_zero(self):
        assert _gres_count_from_tres(None, "gres/gpu") == 0
        assert _gres_count_from_tres("", "gres/gpu") == 0


# ---------------------------------------------------------------------------
# Full output parser
# ---------------------------------------------------------------------------

#   Field order matches euclid_polish.web.sacct._SACCT_FIELDS:
#   JobID | State | ExitCode | Start | End | ElapsedRaw | CPUTimeRAW |
#   TotalCPU | MaxRSS | ReqMem | ReqCPUS | ReqTRES | AllocCPUS | AllocTRES |
#   Timelimit | TRESUsageInTot (GPU usage only in _SACCT_GPU)
# Every row of these module-level fixtures carries all 16 columns, so a row
# without GPU usage ends in an empty TRESUsageInTot (trailing ``|``). The
# inline single-row cases in TestParseSacctOutput omit that last column,
# which the parser tolerates.
# TotalCPU is the CPU time actually CONSUMED (duration string); CPUTimeRAW
# is merely Elapsed × NCPUS (allocated). Efficiency uses TotalCPU.
# Main rows leave MaxRSS empty (it sits on the .batch step); the .batch rows
# here leave ReqMem / ReqCPUS / ReqTRES / AllocCPUS / Timelimit empty but
# repeat AllocTRES. The parser reads the request and allocation fields from
# the job (or array-task) rows only, never from .batch.
# This job used all 4 cores fully → TotalCPU 09:32 = 572 s = elapsed × 4.
_SACCT_COMPLETED = """\
12345|COMPLETED|0:0|2026-05-26T14:33:21|2026-05-26T14:35:44|143|572|09:32||8000Mc|4|cpu=4,mem=8000M|4|cpu=4,gres/gpu=1,mem=8000M|2:00:00|
12345.batch|COMPLETED|0:0|2026-05-26T14:33:21|2026-05-26T14:35:44|143|572|09:32|2048M|||||cpu=4,gres/gpu=1,mem=8000M||"""


# A single-threaded job on 4 cores: TotalCPU 01:40 = 100 s ≈ elapsed (one
# core busy) → efficiency 100/(100×4) = 0.25, NOT 1.0.
_SACCT_PARTIAL = """\
222|COMPLETED|0:0|2026-05-26T18:00:00|2026-05-26T18:01:40|100|400|01:40||8000Mc|4|cpu=4,mem=8000M|4|cpu=4,mem=8000M|1:00:00|
222.batch|COMPLETED|0:0|2026-05-26T18:00:00|2026-05-26T18:01:40|100|400|01:40|512M|||||cpu=4,mem=8000M||"""


_SACCT_OOM = """\
67890|OUT_OF_MEMORY|0:9|2026-05-26T15:01:00|2026-05-26T15:01:42|42|168|01:24||8000Mc|4|cpu=4,mem=8000M|4|cpu=4,mem=8000M|1:00:00|
67890.batch|OUT_OF_MEMORY|0:9|2026-05-26T15:01:00|2026-05-26T15:01:42|42|168|01:24|8500M|||||cpu=4,mem=8000M||"""


_SACCT_CANCELLED = """\
11111|CANCELLED by 5550|0:15|2026-05-26T16:00:00|2026-05-26T16:00:09|9|9|00:09||8000Mc|1|cpu=1,mem=8000M|1|cpu=1,mem=8000M|0:10:00|
11111.batch|CANCELLED|0:15|2026-05-26T16:00:00|2026-05-26T16:00:09|9|9|00:09|256M|||||cpu=1,mem=8000M||"""


# Two array tasks (no parent ``123`` row); task 1 fails.
_SACCT_ARRAY = """\
123_0|COMPLETED|0:0|2026-05-26T14:00:00|2026-05-26T14:01:00|60|120|01:00||8G|2|cpu=2,mem=8G|2|cpu=2,gres/gpu=1,mem=8G|1:00:00|
123_0.batch|COMPLETED|0:0|2026-05-26T14:00:00|2026-05-26T14:01:00|60|120|01:00|1G|||||cpu=2,gres/gpu=1,mem=8G||
123_1|FAILED|1:0|2026-05-26T14:00:00|2026-05-26T14:00:30|30|60|00:20||8G|2|cpu=2,mem=8G|2|cpu=2,gres/gpu=1,mem=8G|1:00:00|
123_1.batch|FAILED|1:0|2026-05-26T14:00:00|2026-05-26T14:00:30|30|60|00:20|2G|||||cpu=2,gres/gpu=1,mem=8G||
"""


# One-GPU job: NVML accounting puts mean GPU util (gres/gpuutil, %) and peak
# GPU memory (gres/gpumem) in the .batch step's TRESUsageInTot.
# TotalCPU 2:00:00 = 7200 s over 3600 s × 4 cores → efficiency 0.5.
_SACCT_GPU = """\
777|COMPLETED|0:0|2026-05-26T19:00:00|2026-05-26T20:00:00|3600|14400|2:00:00||64000M|4|cpu=4,gres/gpu=1,mem=64000M|4|cpu=4,gres/gpu=1,mem=64000M|4:00:00|
777.batch|COMPLETED|0:0|2026-05-26T19:00:00|2026-05-26T20:00:00|3600|14400|2:00:00|20G|||||cpu=4,gres/gpu=1,mem=64000M||cpu=02:00:00,gres/gpumem=79638M,gres/gpuutil=42,mem=20G"""


_FIXTURES = {
    "completed": _SACCT_COMPLETED,
    "partial": _SACCT_PARTIAL,
    "oom": _SACCT_OOM,
    "cancelled": _SACCT_CANCELLED,
    "array": _SACCT_ARRAY,
    "gpu": _SACCT_GPU,
}


@pytest.mark.parametrize("name", sorted(_FIXTURES))
def test_fixture_rows_follow_sacct_field_order(name):
    """Each fixture row has one value per ``_SACCT_FIELDS`` slot, laid out as
    the comment above says: the allocation lives in AllocTRES on both rows,
    and the .batch row leaves the job-row request/limit fields empty."""
    for line in _FIXTURES[name].splitlines():
        values = line.split("|")
        assert len(values) == len(_SACCT_FIELDS), line
        row = dict(zip(_SACCT_FIELDS, values, strict=True))
        assert row["AllocTRES"].startswith("cpu="), line
        if row["JobID"].endswith(".batch"):
            assert row["MaxRSS"], line
            for field in ("ReqMem", "ReqCPUS", "ReqTRES", "AllocCPUS", "Timelimit"):
                assert row[field] == "", (field, line)
        else:
            assert row["MaxRSS"] == "", line
            assert row["Timelimit"], line


class TestParseSacctOutput:

    def test_array_fails_parent_when_any_task_fails(self):
        stats = parse_sacct_output(_SACCT_ARRAY)
        assert stats["state"] == "FAILED"
        assert stats["exit_code"] == "1:0"
        assert stats["cpu_seconds"] == pytest.approx(80.0)
        assert stats["max_rss_mb"] == 2048.0

    def test_completed_job_round_trip(self):
        stats = parse_sacct_output(_SACCT_COMPLETED)
        assert stats["state"] == "COMPLETED"
        assert stats["exit_code"] == "0:0"
        assert stats["started_at"] == "2026-05-26T14:33:21"
        assert stats["ended_at"]   == "2026-05-26T14:35:44"
        assert stats["elapsed_seconds"] == 143.0
        assert stats["cpu_seconds"]     == 572.0
        # MaxRSS on .batch row = 2048M = 2048 MB.
        assert stats["max_rss_mb"] == 2048.0
        assert stats["alloc_cpus"] == 4
        assert stats["alloc_gpus"] == 1
        # mem=8000M from AllocTRES on the main row.
        assert stats["alloc_memory_mb"] == 8000.0
        # 572 CPU-s / (143 elapsed × 4 cores) = 1.0 — fully used.
        assert stats["cpu_efficiency"] == pytest.approx(1.0)

    def test_cpu_efficiency_uses_totalcpu_not_allocated(self):
        """A single-threaded job on 4 cores must report ~25%, NOT 100%.

        Regression: efficiency was computed from CPUTimeRAW (= Elapsed ×
        NCPUS, the *allocated* CPU-time), making it a constant 1.0. It must
        use TotalCPU (CPU-time actually consumed)."""
        stats = parse_sacct_output(_SACCT_PARTIAL)
        # TotalCPU 01:40 = 100 s consumed (not CPUTimeRAW = 400).
        assert stats["cpu_seconds"] == pytest.approx(100.0)
        # 100 / (100 elapsed × 4 cores) = 0.25.
        assert stats["cpu_efficiency"] == pytest.approx(0.25)

    def test_oom_job_recorded_with_state_and_peak(self):
        stats = parse_sacct_output(_SACCT_OOM)
        assert stats["state"] == "OUT_OF_MEMORY"
        assert stats["exit_code"] == "0:9"
        # OOM peaked above the allocation — captures the over-shoot.
        assert stats["max_rss_mb"] == 8500.0
        assert stats["alloc_memory_mb"] == 8000.0

    def test_cancelled_by_user_strips_uid_suffix(self):
        stats = parse_sacct_output(_SACCT_CANCELLED)
        # ``CANCELLED by 5550`` → ``CANCELLED`` so downstream filters
        # match a known set.
        assert stats["state"] == "CANCELLED"

    def test_gpu_usage_read_from_batch_tres_usage(self):
        stats = parse_sacct_output(_SACCT_GPU)
        assert stats["gpu_util_mean"] == pytest.approx(42.0)
        assert stats["gpu_mem_peak_mb"] == pytest.approx(79638.0)
        assert stats["gpu_mem_peak"] == pytest.approx(79638.0)
        assert stats["alloc_gpus"] == 1
        assert stats["alloc_memory_mb"] == 64000.0
        # MaxRSS 20G on the .batch row.
        assert stats["max_rss_mb"] == 20480.0
        assert stats["cpu_efficiency"] == pytest.approx(0.5)

    def test_no_gpu_keys_without_tres_usage(self):
        """A job with an empty TRESUsageInTot leaves the GPU columns blank."""
        stats = parse_sacct_output(_SACCT_COMPLETED)
        assert "gpu_util_mean" not in stats
        assert "gpu_mem_peak_mb" not in stats

    def test_empty_input_returns_empty(self):
        assert parse_sacct_output("") == {}

    def test_no_batch_row_still_works(self):
        """Some short-lived jobs only emit the main row. Parser must not
        crash and must report what it can."""
        text = (
            "99|COMPLETED|0:0|2026-05-26T17:00:00|2026-05-26T17:00:05|"
            "5|10|00:05||100M|1|cpu=1,mem=100M|1|cpu=1,mem=100M|0:01:00"
        )
        stats = parse_sacct_output(text)
        assert stats["state"] == "COMPLETED"
        assert stats["elapsed_seconds"] == 5.0
        # No batch row → max_rss_mb is None (unknown).
        assert stats["max_rss_mb"] is None

    def test_unknown_timestamps_drop_to_empty(self):
        text = (
            "1|PENDING|0:0|Unknown|Unknown|0|0|00:00|||0|cpu=0,mem=0M|0|cpu=0,mem=0M|0:10:00"
        )
        stats = parse_sacct_output(text)
        assert stats["started_at"] == ""
        assert stats["ended_at"] == ""

    def test_cpu_efficiency_none_when_no_elapsed(self):
        """A job that never ran has elapsed=0 → efficiency must not divide."""
        text = (
            "1|CANCELLED|0:0|Unknown|Unknown|0|0|00:00|||0|cpu=0,mem=0M|0|cpu=0,mem=0M|0:10:00"
        )
        stats = parse_sacct_output(text)
        assert stats["cpu_efficiency"] is None


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

class _SSHStub:
    def __init__(self, *, connected=True, rc=0, out="", err=""):
        self.connected = connected
        self.rc = rc; self.out = out; self.err = err
        self.calls: list[str] = []

    def is_connected(self) -> bool:
        return self.connected

    def run(self, cmd: str, *, timeout: float = 15) -> tuple[int, str, str]:
        self.calls.append(cmd)
        return self.rc, self.out, self.err


class TestFetchSacctStats:

    def test_returns_none_when_disconnected(self):
        assert fetch_sacct_stats(_SSHStub(connected=False), "1") is None

    def test_returns_none_when_ssh_is_none(self):
        assert fetch_sacct_stats(None, "1") is None

    def test_returns_none_on_nonzero_rc(self):
        assert fetch_sacct_stats(_SSHStub(rc=1, err="oops"), "1") is None

    def test_returns_none_on_empty_output(self):
        """An sbatch-rejected job never enters the accounting DB, so
        sacct returns an empty body (rc=0, no rows). Must not crash."""
        assert fetch_sacct_stats(_SSHStub(rc=0, out="\n"), "1") is None

    def test_happy_path_returns_parsed_dict(self):
        stub = _SSHStub(rc=0, out=_SACCT_COMPLETED)
        stats = fetch_sacct_stats(stub, "12345")
        assert stats["state"] == "COMPLETED"
        assert stats["alloc_gpus"] == 1
        # And it asked for exactly that job id.
        assert "12345" in stub.calls[0]


# ---------------------------------------------------------------------------
# Command shape
# ---------------------------------------------------------------------------

class TestBuildSacctCommand:

    def test_quotes_jobid(self):
        cmd = build_sacct_command("12345; rm -rf /")
        # ``shlex.quote`` wraps anything dangerous in single quotes;
        # nothing should escape the quotation.
        assert "rm -rf /" in cmd            # only inside the quoted arg
        assert "'12345; rm -rf /'" in cmd

    def test_uses_pipe_separator_and_no_header(self):
        cmd = build_sacct_command("1")
        assert " -P " in cmd
        assert "--noheader" in cmd
