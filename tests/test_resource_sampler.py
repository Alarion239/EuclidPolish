"""Tests for the CPU/GPU resource sampler.

The thread + nvidia-smi/psutil probes are thin glue; the testable pieces
are the nvidia-smi CSV parser, the single-shot payload builder, and the
fact that the sampler emits ``resource`` events through a Reporter that
``fold_events`` can read back.
"""

from __future__ import annotations

import json

from euclid_polish.observability import ResourceSampler
from euclid_polish.observability import resource_sampler as rs
from euclid_polish.observability.reporter import Reporter
from euclid_polish.web.job_status import fold_events

# ---------------------------------------------------------------------------
# nvidia-smi CSV parsing
# ---------------------------------------------------------------------------

class TestParseNvidiaSmi:

    def test_single_gpu(self):
        out = "47, 60, 12000, 40960\n"
        util, mem_util, used, total = rs.parse_nvidia_smi(out)
        assert util == 47.0 and mem_util == 60.0
        assert used == 12000.0 and total == 40960.0

    def test_multi_gpu_averages_util_sums_memory(self):
        out = "40, 50, 1000, 40960\n60, 70, 3000, 40960\n"
        util, mem_util, used, total = rs.parse_nvidia_smi(out)
        assert util == 50.0           # (40+60)/2
        assert mem_util == 60.0       # (50+70)/2
        assert used == 4000.0         # summed
        assert total == 81920.0       # summed

    def test_not_supported_rows_dropped(self):
        assert rs.parse_nvidia_smi("[N/A], [N/A], [N/A], [N/A]\n") is None
        assert rs.parse_nvidia_smi("") is None


# ---------------------------------------------------------------------------
# sample_once — payload assembly (probes monkeypatched)
# ---------------------------------------------------------------------------

class TestSampleOnce:

    def test_gpu_and_cpu_present(self, monkeypatch):
        monkeypatch.setattr(rs, "read_cpu_percent", lambda: 83.4)
        monkeypatch.setattr(rs, "read_gpu", lambda: (47.0, 60.0, 12000.0, 40960.0))
        p = rs.sample_once()
        assert p["cpu"] == 83.4
        assert p["gpu"] == 47.0 and p["gpu_mem"] == 60.0
        assert p["gpu_mem_used_mb"] == 12000.0 and p["gpu_mem_total_mb"] == 40960.0

    def test_cpu_only_node_omits_gpu(self, monkeypatch):
        monkeypatch.setattr(rs, "read_cpu_percent", lambda: 50.0)
        monkeypatch.setattr(rs, "read_gpu", lambda: None)
        p = rs.sample_once()
        assert p == {"cpu": 50.0}

    def test_empty_when_both_probes_blank(self, monkeypatch):
        monkeypatch.setattr(rs, "read_cpu_percent", lambda: None)
        monkeypatch.setattr(rs, "read_gpu", lambda: None)
        assert rs.sample_once() == {}


# ---------------------------------------------------------------------------
# End-to-end: sampler → events file → fold_events
# ---------------------------------------------------------------------------

def test_sampler_emits_resource_events_foldable(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "read_cpu_percent", lambda: 70.0)
    monkeypatch.setattr(rs, "read_gpu", lambda: (55.0, 40.0, 9000.0, 40960.0))
    events = tmp_path / "ev.jsonl"
    reporter = Reporter(events_path=str(events))

    # Tiny interval so a couple of samples land quickly; stop deterministically.
    sampler = ResourceSampler(reporter, interval_s=0.05).start()
    import time
    time.sleep(0.3)
    sampler.stop()

    text = events.read_text()
    assert '"kind":"resource"' in text
    # Every emitted line is valid JSON with the resource payload.
    kinds = [json.loads(l)["kind"] for l in text.splitlines() if l.strip()]
    assert "resource" in kinds

    status = fold_events(text)
    assert status.resources is not None
    assert status.resources.gpu_percent == 55.0
    assert status.resources.cpu_percent == 70.0
    assert status.resources.n_samples >= 1


def test_reporter_sample_resources_noop_without_path():
    """A no-op reporter (no events path) silently drops samples."""
    Reporter(events_path=None).sample_resources({"gpu": 50.0})  # no raise


def test_reporter_drops_empty_sample(tmp_path):
    events = tmp_path / "ev.jsonl"
    r = Reporter(events_path=str(events))
    r.sample_resources({})
    assert not events.exists() or events.read_text() == ""
