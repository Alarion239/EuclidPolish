"""Tests for the FASRC Jobstats client and report parser."""

from __future__ import annotations

import pytest

from euclid_polish.web.jobstats import (
    build_jobstats_command,
    fetch_jobstats_stats,
    parse_jobstats_output,
)

REPORT = """\
================================================================================
                              Slurm Job Statistics
================================================================================
Job ID: 39798795
User/Account: aturing/math
Job Name: sys_logic_ordinals
State: COMPLETED
Nodes: 2
CPU Cores: 48
CPU Memory: 256GB (5.3GB per CPU-core)
GPUs: 4
QOS/Partition: della-gpu/gpu
Cluster: della
Start Time: Fri Mar 4, 2022 at 1:56 AM
Run Time: 18:41:56
Time Limit: 4-00:00:00
                              Overall Utilization
================================================================================
CPU utilization  [|||||                                          10%]
CPU memory usage [|||                                             6%]
GPU utilization  [||||||||||||||||||||||||||||||||||             68%]
GPU memory usage [|||||||||||||||||||||||||||||||||              66%]
                              Detailed Utilization
================================================================================
CPU utilization per node (CPU time used/run time)
    della-i14g2: 1-21:41:20/18-16:46:24 (efficiency=10.2%)
    della-i14g3: 1-18:48:55/18-16:46:24 (efficiency=9.5%)
CPU memory usage per node - used/allocated
    della-i14g2: 7.9GB/128.0GB (335.5MB/5.3GB per core of 24)
    della-i14g3: 7.8GB/128.0GB (334.6MB/5.3GB per core of 24)
GPU utilization per node
    della-i14g2 (GPU 0): 65.7%
    della-i14g2 (GPU 1): 64.5%
    della-i14g3 (GPU 0): 72.9%
    della-i14g3 (GPU 1): 67.5%
GPU memory usage per node - maximum used/total
    della-i14g2 (GPU 0): 26.5GB/40.0GB (66.2%)
    della-i14g2 (GPU 1): 26.5GB/40.0GB (66.2%)
    della-i14g3 (GPU 0): 26.5GB/40.0GB (66.2%)
    della-i14g3 (GPU 1): 26.5GB/40.0GB (66.2%)
Notes
================================================================================
* This job only used 6% of the 256GB of total allocated CPU memory. For
  future jobs, please allocate less memory.
* See the URL below for various job metrics plotted as a function of time:
  https://example.invalid/jobstats/39798795
"""


class _SSHStub:
    def __init__(self, *, connected: bool = True, rc: int = 0, out: str = ""):
        self.connected = connected
        self.rc = rc
        self.out = out
        self.calls: list[str] = []

    def is_connected(self) -> bool:
        return self.connected

    def run(self, cmd: str, *, timeout: float = 30):
        self.calls.append(cmd)
        return self.rc, self.out, ""


def test_parse_jobstats_report():
    stats = parse_jobstats_output(REPORT)

    assert stats["state"] == "COMPLETED"
    assert stats["alloc_cpus"] == 48
    assert stats["alloc_gpus"] == 4
    assert stats["alloc_memory_mb"] == pytest.approx(256 * 1024)
    assert stats["elapsed_seconds"] == pytest.approx(18 * 3600 + 41 * 60 + 56)
    assert stats["jobstats_cpu_util"] == pytest.approx(10.0)
    assert stats["jobstats_cpu_memory_util"] == pytest.approx(6.0)
    assert stats["jobstats_gpu_util"] == pytest.approx(68.0)
    assert stats["jobstats_gpu_memory_util"] == pytest.approx(66.0)
    assert stats["jobstats_gpu_memory_used_mb"] == pytest.approx(4 * 26.5 * 1024)
    assert stats["jobstats_gpu_memory_total_mb"] == pytest.approx(4 * 40 * 1024)
    assert "future jobs" in stats["jobstats_notes_json"]
    assert "della-i14g2" in stats["jobstats_gpu_nodes_json"]


def test_parse_rejects_non_report_output():
    assert parse_jobstats_output("jobstats: No data was found") == {}


def test_build_command_quotes_jobid_and_cluster():
    command = build_jobstats_command("123; echo bad", cluster="odyssey;bad")
    assert "'123; echo bad'" in command
    assert "'odyssey;bad'" in command


def test_fetch_adds_source_and_timestamp():
    ssh = _SSHStub(out=REPORT)
    stats = fetch_jobstats_stats(ssh, "39798795")
    assert stats["accounting_source"] == "jobstats"
    assert stats["jobstats_collected_at"].endswith("Z")
    assert ssh.calls == ["jobstats 39798795"]


def test_fetch_returns_none_when_unavailable():
    assert fetch_jobstats_stats(_SSHStub(connected=False), "1") is None
    assert fetch_jobstats_stats(_SSHStub(rc=1), "1") is None
