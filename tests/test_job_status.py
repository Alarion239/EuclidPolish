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
