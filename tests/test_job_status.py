"""Tests for the structured job-status aggregator.

Covers:

  * ``fold_events`` — parsing & folding the JSONL stream into JobStatus.
  * Resilience to partial lines / unknown kinds / malformed values.
  * ``JobStatusFetcher`` — wiring an SSH stub through to ``fold_events``.
"""

from __future__ import annotations

import json
from typing import Tuple

import pytest

from euclid_polish.config import Config
from euclid_polish.web.job_status import (
    Event,
    JobStatus,
    JobStatusFetcher,
    StepProgress,
    fold_events,
)


def _jsonl(*events) -> str:
    """Render a list of event dicts to JSONL text the way Reporter would."""
    return "\n".join(json.dumps(e) for e in events) + "\n"


def _w(wid, pid, cur, tot, ts):
    return {"ts": ts, "kind": "worker",
            "value": {"worker_id": str(wid), "pid": pid,
                      "current": cur, "total": tot}}


# ---------------------------------------------------------------------------
# fold_events — parallel worker aggregation
# ---------------------------------------------------------------------------

class TestParallelFold:

    def test_cumulative_and_active_workers(self):
        text = _jsonl(
            {"ts": 100.0, "kind": "stage", "value": "generate+forward train"},
            {"ts": 100.1, "kind": "parallel", "value": {"total": 10, "workers": 4}},
            _w("0", 11, 0, 3, 100.2),
            _w("1", 12, 0, 3, 100.2),
            _w("0", 11, 3, 3, 105.0),   # shard 0 done (pid 11 frees up)
            _w("2", 11, 1, 4, 106.0),   # pid 11 now on shard 2
            _w("1", 12, 2, 3, 106.0),   # pid 12 still on shard 1
        )
        s = fold_events(text)
        # cumulative = shard0(3) + shard1(2) + shard2(1) = 6 of 10.
        assert s.step.current == 6 and s.step.total == 10
        assert s.parallel is not None
        assert s.parallel.current == 6 and s.parallel.total == 10
        assert s.parallel.n_workers == 4
        # Active = distinct pids with work left + recent: shard0 is done
        # (excluded), shard2/pid11 and shard1/pid12 in progress → 2 procs.
        assert s.parallel.active_workers == 2

    def test_all_done_means_zero_active(self):
        text = _jsonl(
            {"ts": 100.0, "kind": "parallel", "value": {"total": 3, "workers": 2}},
            _w("0", 1, 2, 2, 200.0),
            _w("1", 2, 1, 1, 200.0),
        )
        s = fold_events(text)
        assert s.step.current == 3 and s.step.total == 3
        assert s.parallel.active_workers == 0   # nothing left to do

    def test_stale_worker_not_counted_active(self):
        text = _jsonl(
            {"ts": 100.0, "kind": "parallel", "value": {"total": 10, "workers": 3}},
            _w("0", 1, 1, 5, 900.0),     # last heard 100 s before the newest
            _w("1", 2, 1, 5, 1000.0),    # recent
        )
        s = fold_events(text)
        assert s.parallel.active_workers == 1   # only the recent process

    def test_new_parallel_phase_resets(self):
        text = _jsonl(
            {"ts": 100.0, "kind": "parallel", "value": {"total": 6400, "workers": 16}},
            _w("0", 1, 256, 256, 200.0),   # a finished train shard
            {"ts": 300.0, "kind": "parallel", "value": {"total": 100, "workers": 16}},
            _w("0", 1, 4, 50, 301.0),      # validate phase, fresh
        )
        s = fold_events(text)
        # Cumulative reflects only the validate phase, not the train carry-over.
        assert s.step.current == 4 and s.step.total == 100
        assert s.parallel.total == 100


# ---------------------------------------------------------------------------
# fold_events — happy paths
# ---------------------------------------------------------------------------

class TestFoldEvents:

    def test_empty_input_returns_empty_status(self):
        s = fold_events("")
        assert s.stage is None
        assert s.step is None
        assert s.warnings == ()
        assert s.errors == ()
        assert s.has_events is False

    def test_last_stage_event_wins(self):
        text = _jsonl(
            {"ts": 1.0, "kind": "stage", "value": "Phase 1"},
            {"ts": 2.0, "kind": "stage", "value": "Phase 2"},
            {"ts": 3.0, "kind": "stage", "value": "Phase 3"},
        )
        s = fold_events(text)
        assert s.stage == "Phase 3"
        assert s.has_events is True

    def test_last_step_event_wins(self):
        text = _jsonl(
            {"ts": 1.0, "kind": "step",
             "value": {"current": 1, "total": 10, "label": ""}},
            {"ts": 2.0, "kind": "step",
             "value": {"current": 5, "total": 10, "label": "halfway"}},
        )
        s = fold_events(text)
        assert s.step == StepProgress(current=5, total=10, label="halfway")
        assert s.step.fraction == 0.5

    def test_all_warnings_and_errors_accumulate(self):
        text = _jsonl(
            {"ts": 1.0, "kind": "warn",  "value": "w1"},
            {"ts": 2.0, "kind": "error", "value": "e1"},
            {"ts": 3.0, "kind": "warn",  "value": "w2"},
            {"ts": 4.0, "kind": "error", "value": "e2"},
        )
        s = fold_events(text)
        assert s.warnings == (Event(ts=1.0, msg="w1"),
                              Event(ts=3.0, msg="w2"))
        assert s.errors   == (Event(ts=2.0, msg="e1"),
                              Event(ts=4.0, msg="e2"))

    def test_mixed_kinds_folded_independently(self):
        text = _jsonl(
            {"ts": 1.0, "kind": "stage", "value": "start"},
            {"ts": 2.0, "kind": "step",
             "value": {"current": 3, "total": 25, "label": "tile 3"}},
            {"ts": 3.0, "kind": "warn",  "value": "checksum"},
            {"ts": 4.0, "kind": "stage", "value": "next"},
            {"ts": 5.0, "kind": "error", "value": "boom"},
        )
        s = fold_events(text)
        assert s.stage == "next"
        assert s.step  == StepProgress(current=3, total=25, label="tile 3")
        assert [w.msg for w in s.warnings] == ["checksum"]
        assert [e.msg for e in s.errors]   == ["boom"]


# ---------------------------------------------------------------------------
# fold_events — step-rate / ETA computation
# ---------------------------------------------------------------------------

class TestFoldEventsRate:

    def test_rate_and_eta_within_current_stage(self):
        """Step events within the current stage drive ``step_rate_per_s``
        and ``step_eta_s``. Rate = Δsteps / Δt; ETA = (total − current) /
        rate. Two valid in-stage points are enough.
        """
        lines = [
            '{"ts":100.0,"kind":"stage","value":"train"}',
            '{"ts":110.0,"kind":"step","value":{"current":100,"total":2000,"label":""}}',
            '{"ts":120.0,"kind":"step","value":{"current":300,"total":2000,"label":""}}',
        ]
        s = fold_events("\n".join(lines))
        # 200 steps over 10 s → 20 steps/s. 2000−300 = 1700 left → ETA 85 s.
        assert s.step_rate_per_s == pytest.approx(20.0)
        assert s.step_eta_s == pytest.approx(85.0)

    def test_rate_resets_at_stage_boundary(self):
        """A new ``set_stage`` clears the per-stage history — a quick
        stage's rate must not pollute a slow following stage's ETA."""
        lines = [
            '{"ts":  0.0,"kind":"stage","value":"prep"}',
            '{"ts":  1.0,"kind":"step","value":{"current":10,"total":10,"label":""}}',
            '{"ts": 10.0,"kind":"stage","value":"train"}',
            '{"ts": 20.0,"kind":"step","value":{"current": 50,"total":1000,"label":""}}',
            '{"ts": 30.0,"kind":"step","value":{"current":100,"total":1000,"label":""}}',
        ]
        s = fold_events("\n".join(lines))
        # In ``train`` stage: 50 steps over 10 s → 5 steps/s.
        # 1000−100 = 900 left → ETA 180 s.
        assert s.stage == "train"
        assert s.step_rate_per_s == pytest.approx(5.0)
        assert s.step_eta_s == pytest.approx(180.0)

    def test_rate_undefined_with_single_step_event(self):
        """One step event isn't enough — need two timestamps to measure
        a rate. Both fields stay ``None``."""
        lines = [
            '{"ts":0.0,"kind":"stage","value":"train"}',
            '{"ts":5.0,"kind":"step","value":{"current":1,"total":100,"label":""}}',
        ]
        s = fold_events("\n".join(lines))
        assert s.step_rate_per_s is None
        assert s.step_eta_s is None

    def test_rate_undefined_when_steps_rewind(self):
        """If ``current`` goes backwards (a script restart, or two
        workers reporting different sub-ranges), Δsteps ≤ 0 — punt
        rather than emit a negative ETA."""
        lines = [
            '{"ts": 0.0,"kind":"stage","value":"train"}',
            '{"ts": 5.0,"kind":"step","value":{"current":100,"total":1000,"label":""}}',
            '{"ts":10.0,"kind":"step","value":{"current": 80,"total":1000,"label":""}}',
        ]
        s = fold_events("\n".join(lines))
        assert s.step_rate_per_s is None
        assert s.step_eta_s is None

    def test_ema_weights_recent_intervals(self):
        """A slow first interval then steady fast intervals: the reported
        rate is the EMA of per-step duration, which moves OFF the slow
        start toward the recent fast pace as more fast intervals arrive
        (it does not converge instantly — that's the point of the decay).
        Each interval advances 10 steps: the first over 100 s (spp 10),
        the rest over 10 s (spp 1)."""
        alpha = Config.WebFetch.STEP_RATE_EMA_ALPHA
        lines = ['{"ts":0.0,"kind":"stage","value":"train"}']
        ts, cur = 0.0, 0
        lines.append(f'{{"ts":{ts},"kind":"step",'
                     f'"value":{{"current":{cur},"total":1000,"label":""}}}}')
        ts, cur = 100.0, 10            # interval 0: spp = 100/10 = 10
        lines.append(f'{{"ts":{ts},"kind":"step",'
                     f'"value":{{"current":{cur},"total":1000,"label":""}}}}')
        for _ in range(5):             # intervals 1..5: spp = 10/10 = 1
            ts += 10.0
            cur += 10
            lines.append(f'{{"ts":{ts},"kind":"step",'
                         f'"value":{{"current":{cur},"total":1000,"label":""}}}}')

        # Reproduce the EMA the implementation computes.
        ema = 10.0
        for _ in range(5):
            ema = (1.0 - alpha) * ema + alpha * 1.0
        expected_rate = 1.0 / ema

        s = fold_events("\n".join(lines))
        assert s.step_rate_per_s == pytest.approx(expected_rate, rel=1e-6)
        # Recency sanity: the estimate has moved off the slow-start rate
        # (0.1 step/s) toward the fast pace (1 step/s), without reaching it.
        assert 0.1 < s.step_rate_per_s < 1.0
        # ETA divides remaining steps by the same EMA rate.
        assert s.step_eta_s == pytest.approx(
            (1000 - cur) / s.step_rate_per_s, rel=1e-6)


# ---------------------------------------------------------------------------
# fold_events — resilience
# ---------------------------------------------------------------------------

class TestFoldEventsResilience:

    def test_partial_last_line_is_skipped(self):
        """A job that crashes mid-write can leave a torn last line.
        The fold must consume what's there and ignore the leftover."""
        good = json.dumps({"ts": 1.0, "kind": "stage", "value": "Phase 1"})
        torn = '{"ts": 2.0, "kind": "step", "value": {"current": 5'
        text = good + "\n" + torn
        s = fold_events(text)
        assert s.stage == "Phase 1"
        assert s.step is None  # torn line silently dropped

    def test_unknown_kind_is_silently_dropped(self):
        """A future Reporter version might emit ``kind=metric`` etc. —
        older consumers must not 500 on unknown kinds."""
        text = _jsonl(
            {"ts": 1.0, "kind": "stage",      "value": "ok"},
            {"ts": 2.0, "kind": "metric",     "value": 0.7},
            {"ts": 3.0, "kind": "checkpoint", "value": "/path/ckpt.h5"},
        )
        s = fold_events(text)
        assert s.stage == "ok"
        assert s.has_events is True

    def test_malformed_step_value_skipped(self):
        text = _jsonl(
            {"ts": 1.0, "kind": "step", "value": "not a dict"},
            {"ts": 2.0, "kind": "step",
             "value": {"current": 5, "total": 10}},
        )
        s = fold_events(text)
        # First skipped, second accepted.
        assert s.step == StepProgress(current=5, total=10, label="")

    def test_blank_lines_are_ignored(self):
        text = (
            "\n"
            + json.dumps({"ts": 1.0, "kind": "stage", "value": "x"})
            + "\n\n\n"
        )
        s = fold_events(text)
        assert s.stage == "x"

    def test_non_dict_jsonl_payload_is_skipped(self):
        text = "42\n\"a string\"\n" + json.dumps(
            {"ts": 1.0, "kind": "stage", "value": "ok"}
        ) + "\n"
        s = fold_events(text)
        assert s.stage == "ok"


# ---------------------------------------------------------------------------
# JobStatus serialisation
# ---------------------------------------------------------------------------

class TestToDict:

    def test_to_dict_shape(self):
        s = JobStatus(
            stage="Phase 1",
            step=StepProgress(current=3, total=25, label="tile 3"),
            warnings=(Event(ts=1.0, msg="w"),),
            errors=(Event(ts=2.0, msg="e"),),
            has_events=True,
        )
        d = s.to_dict()
        assert d["stage"] == "Phase 1"
        assert d["step"] == {"current": 3, "total": 25, "label": "tile 3"}
        assert d["warnings"] == [{"ts": 1.0, "msg": "w"}]
        assert d["errors"]   == [{"ts": 2.0, "msg": "e"}]
        assert d["has_events"] is True
        assert isinstance(d["last_fetched"], float)
        # Metric fields default to empty/None when no metric events seen.
        assert d["latest_metrics"] is None
        assert d["metrics"] == []
        assert d["last_checkpoint"] is None


class TestMetricFold:
    """``metric`` events fold into latest_metrics / metrics history /
    last_checkpoint — the events-native training progress the WebUI reads
    instead of parsing the .out log."""

    @staticmethod
    def _events(*rows) -> str:
        return "\n".join(
            json.dumps({"ts": float(i), "kind": "metric", "value": r})
            for i, r in enumerate(rows)
        )

    def test_latest_metric_wins_and_history_accumulates(self):
        s = fold_events(self._events(
            {"step": 100, "total": 1000, "loss": 0.5, "psnr_stretched": 30.0},
            {"step": 200, "total": 1000, "loss": 0.3, "psnr_stretched": 33.0},
        ))
        assert s.latest_metrics["step"] == 200
        assert s.latest_metrics["loss"] == 0.3
        assert len(s.metrics) == 2
        assert [m["step"] for m in s.metrics] == [100, 200]
        # to_dict surfaces them for the UI.
        assert s.to_dict()["latest_metrics"]["psnr_stretched"] == 33.0

    def test_saved_flag_sets_last_checkpoint(self):
        s = fold_events(self._events(
            {"step": 100, "loss": 0.5},
            {"step": 200, "loss": 0.3, "saved": True},
            {"step": 300, "loss": 0.2},
        ))
        assert s.last_checkpoint == "step 200"

    def test_history_capped_to_newest(self):
        from euclid_polish.web.job_status import _METRICS_CAP
        rows = [{"step": i, "loss": 0.0} for i in range(_METRICS_CAP + 25)]
        s = fold_events(self._events(*rows))
        assert len(s.metrics) == _METRICS_CAP
        assert s.metrics[-1]["step"] == _METRICS_CAP + 24   # newest kept
        assert s.latest_metrics["step"] == _METRICS_CAP + 24


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

class _SSHStub:
    """In-memory SSH stand-in. ``files[path]`` is what ``cat`` returns."""

    def __init__(self, files: dict, connected: bool = True) -> None:
        self.files     = files
        self.connected = connected
        self.calls: list[str] = []

    def is_connected(self) -> bool:
        return self.connected

    def run(self, cmd: str, *, timeout: float = 10) -> Tuple[int, str, str]:
        self.calls.append(cmd)
        # Strip the ``cat <path> 2>/dev/null || true`` wrapper so the
        # stub keeps looking like a filesystem.
        if cmd.startswith("cat "):
            path = cmd.split(" ", 2)[1]
            if path in self.files:
                return 0, self.files[path], ""
            return 0, "", ""  # || true → rc=0 even if cat missed
        return 1, "", f"unhandled stub command: {cmd}"


class TestJobStatusFetcher:

    def test_no_events_path_returns_empty(self):
        f = JobStatusFetcher(ssh=_SSHStub(files={}))
        s = f.fetch(events_path=None)
        assert s == JobStatus(last_fetched=s.last_fetched)
        assert s.has_events is False

    def test_no_ssh_returns_empty(self):
        f = JobStatusFetcher(ssh=None)
        s = f.fetch(events_path="/tmp/foo.events")
        assert s.has_events is False

    def test_disconnected_ssh_returns_empty(self):
        f = JobStatusFetcher(ssh=_SSHStub(files={}, connected=False))
        s = f.fetch(events_path="/tmp/foo.events")
        assert s.has_events is False

    def test_missing_remote_file_returns_empty(self):
        f = JobStatusFetcher(ssh=_SSHStub(files={}))
        s = f.fetch(events_path="/tmp/nope.events")
        assert s.has_events is False

    def test_round_trips_a_real_events_file(self):
        text = _jsonl(
            {"ts": 1.0, "kind": "stage", "value": "Phase 1"},
            {"ts": 2.0, "kind": "warn",  "value": "be careful"},
        )
        stub = _SSHStub(files={"/remote/job.events": text})
        f = JobStatusFetcher(ssh=stub)
        s = f.fetch(events_path="/remote/job.events")
        assert s.stage == "Phase 1"
        assert len(s.warnings) == 1
        assert s.warnings[0].msg == "be careful"
        # Single round-trip per fetch.
        assert len(stub.calls) == 1
        assert "/remote/job.events" in stub.calls[0]


# ---------------------------------------------------------------------------
# fold_events — resource utilisation (GPU/CPU) samples
# ---------------------------------------------------------------------------

def _res(ts, **kw):
    return {"ts": ts, "kind": "resource", "value": kw}


class TestResourceFold:

    def test_no_resource_events_leaves_resources_none(self):
        s = fold_events(_jsonl({"ts": 1.0, "kind": "stage", "value": "x"}))
        assert s.resources is None

    def test_smoothed_live_value_is_recent_mean(self):
        # 10 samples; the smoothed gauge averages only the last few, so a
        # low early run doesn't drag the live reading down.
        evs = [_res(float(i), gpu=10.0) for i in range(5)]
        evs += [_res(float(5 + i), gpu=90.0) for i in range(6)]
        s = fold_events(_jsonl(*evs))
        assert s.resources is not None
        # Last 6 samples are all 90 → smoothed ≈ 90, not the 10s.
        assert s.resources.gpu_percent == pytest.approx(90.0, abs=1e-6)
        # Aggregates span the whole run.
        assert s.resources.gpu_peak == pytest.approx(90.0)
        assert s.resources.gpu_mean < 90.0
        assert s.resources.n_samples == 11

    def test_cpu_only_job_has_no_gpu(self):
        s = fold_events(_jsonl(_res(1.0, cpu=42.0), _res(2.0, cpu=44.0)))
        assert s.resources is not None
        assert s.resources.cpu_percent == pytest.approx(43.0)
        assert s.resources.gpu_percent is None
        assert s.resources.gpu_peak is None

    def test_series_capped_and_in_dict(self):
        evs = [_res(float(i), gpu=float(i % 100), cpu=50.0) for i in range(200)]
        s = fold_events(_jsonl(*evs))
        d = s.to_dict()["resources"]
        assert d is not None
        assert len(d["gpu_series"]) == 60          # _RESOURCE_SERIES_CAP
        assert d["cpu_percent"] == pytest.approx(50.0)

    def test_malformed_sample_skipped(self):
        s = fold_events(_jsonl(
            _res(1.0, gpu="not-a-number"),
            _res(2.0, gpu=70.0),
        ))
        assert s.resources is not None
        assert s.resources.gpu_peak == pytest.approx(70.0)
        assert s.resources.n_samples == 1
